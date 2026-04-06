"""Sheaf Diffusion + Neural ODE layers for directed hypergraph neural networks.

Implements continuous-time diffusion on a sheaf Laplacian over the bipartite
node-net graph. Each node-net incidence gets a learnable diagonal restriction
map, and node/net features evolve via an ODE: dx/dt = -σ·L_F·x + MLP(x,t).
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.utils import scatter


class RestrictionMapMLP(nn.Module):
    """Amortized restriction map generator.

    Instead of per-edge parameters (which would depend on graph size),
    this MLP generates diagonal restriction maps from endpoint features
    and edge type. Separate MLPs for source and sink edges.
    """

    def __init__(self, emb_dim):
        super().__init__()
        # Input: [node_feat || net_feat || edge_type_embed]
        input_dim = emb_dim * 2 + 1
        self.source_mlp = Seq(
            Linear(input_dim, emb_dim),
            LeakyReLU(),
            Linear(emb_dim, emb_dim),
        )
        self.sink_mlp = Seq(
            Linear(input_dim, emb_dim),
            LeakyReLU(),
            Linear(emb_dim, emb_dim),
        )

    def forward(self, x_node, x_net, edge_index, edge_type):
        """Compute diagonal restriction maps for all edges.

        Args:
            x_node: (N, d) node features
            x_net: (M, d) net features
            edge_index: (2, E) node-to-net edge indices
            edge_type: (E,) bool, True=source, False=sink

        Returns:
            restriction_maps: (E, d) diagonal restriction vectors
        """
        src_nodes = x_node[edge_index[0]]  # (E, d)
        dst_nets = x_net[edge_index[1]]    # (E, d)
        edge_type_feat = edge_type.float().unsqueeze(-1)  # (E, 1)
        inp = torch.cat([src_nodes, dst_nets, edge_type_feat], dim=-1)  # (E, 2d+1)

        maps = torch.zeros(edge_index.shape[1], x_node.shape[1], device=x_node.device)
        source_mask = edge_type == 1
        sink_mask = ~source_mask

        if source_mask.any():
            maps[source_mask] = self.source_mlp(inp[source_mask])
        if sink_mask.any():
            maps[sink_mask] = self.sink_mlp(inp[sink_mask])

        return maps


class SheafODEFunc(nn.Module):
    """ODE right-hand side: f(t, x) = -sigma * L_F * x + g(x, t).

    The Sheaf Laplacian L_F is applied via sparse scatter operations
    (never materialized as a dense matrix).
    """

    def __init__(self, emb_dim, num_nodes, num_nets, edge_index, edge_type,
                 restriction_maps):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_nodes = num_nodes
        self.num_nets = num_nets

        # Stored graph topology (fixed during ODE integration)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type', edge_type)
        self.restriction_maps = restriction_maps  # (E, d) — precomputed

        # Learnable diffusivity
        self.sigma = nn.Parameter(torch.tensor(1.0))

        # Nonlinear source term with time embedding
        # Time embedding: [sin(ωt), cos(ωt)] for ω in {1, 2, 4, 8}
        time_dim = 8
        self.source_term = Seq(
            Linear(emb_dim + time_dim, emb_dim),
            LeakyReLU(),
            Linear(emb_dim, emb_dim),
        )
        self.time_freqs = nn.Parameter(
            torch.tensor([1.0, 2.0, 4.0, 8.0]), requires_grad=False
        )

    def _time_embed(self, t):
        """Sinusoidal time embedding."""
        # t is a scalar
        freqs = self.time_freqs * t  # (4,)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)])  # (8,)

    def _sheaf_laplacian(self, x):
        """Apply Sheaf Laplacian via scatter operations.

        For each edge (node_i, net_j) with restriction map F_ij:
            msg_ij = F_ij * x_node[i] - F_ij * x_net[j]
        Then scatter:
            L_F[x]_node[i] += F_ij * msg_ij
            L_F[x]_net[j]  -= F_ij * msg_ij
        """
        x_node = x[:self.num_nodes]
        x_net = x[self.num_nodes:]

        node_idx = self.edge_index[0]  # (E,)
        net_idx = self.edge_index[1]   # (E,)
        F = self.restriction_maps      # (E, d)

        # Message: F * (x_node[i] - x_net[j])
        msg = F * (x_node[node_idx] - x_net[net_idx])  # (E, d)

        # Scatter to nodes and nets
        Lx_node = scatter(F * msg, node_idx, dim=0,
                          dim_size=self.num_nodes, reduce='add')
        Lx_net = scatter(-F * msg, net_idx, dim=0,
                         dim_size=self.num_nets, reduce='add')

        return torch.cat([Lx_node, Lx_net], dim=0)

    def forward(self, t, x):
        """ODE right-hand side.

        Args:
            t: scalar, current time
            x: (N+M, d) concatenated node and net features

        Returns:
            dxdt: (N+M, d)
        """
        # Sheaf Laplacian diffusion term
        Lx = self._sheaf_laplacian(x)

        # Nonlinear source term
        t_emb = self._time_embed(t)  # (8,)
        t_emb = t_emb.unsqueeze(0).expand(x.shape[0], -1)  # (N+M, 8)
        g = self.source_term(torch.cat([x, t_emb], dim=-1))

        dxdt = -self.sigma * Lx + g
        return dxdt


class SheafODEBlock(nn.Module):
    """Wraps the Sheaf ODE into a single forward pass.

    Replaces the discrete layer loop in GNN_node with a single
    continuous-time ODE integration using torchdiffeq.
    """

    def __init__(self, emb_dim, ode_T=1.0, ode_tol=1e-3, ode_method='dopri5'):
        super().__init__()
        self.emb_dim = emb_dim
        self.T = nn.Parameter(torch.tensor(ode_T))
        self.ode_tol = ode_tol
        self.ode_method = ode_method

        # Restriction map generator (graph-size independent)
        self.restriction_mlp = RestrictionMapMLP(emb_dim)

    def forward(self, h_node, h_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """Run ODE integration.

        Args:
            h_node: (N, d) encoded node features
            h_net: (M, d) encoded net features
            edge_index_node_to_net: (2, E)
            edge_weight_node_to_net: (E,)
            edge_type_node_to_net: (E,) bool
            edge_index_net_to_node: (2, E)
            edge_weight_net_to_node: (E,)
            device: torch device

        Returns:
            h_node_out: (N, d)
            h_net_out: (M, d)
        """
        from torchdiffeq import odeint_adjoint as odeint

        num_nodes = h_node.shape[0]
        num_nets = h_net.shape[0]

        # Compute restriction maps from current features
        restriction_maps = self.restriction_mlp(
            h_node, h_net,
            edge_index_node_to_net.to(device),
            edge_type_node_to_net.to(device)
        )

        # Build ODE function with current graph structure
        odefunc = SheafODEFunc(
            emb_dim=self.emb_dim,
            num_nodes=num_nodes,
            num_nets=num_nets,
            edge_index=edge_index_node_to_net.to(device),
            edge_type=edge_type_node_to_net.to(device),
            restriction_maps=restriction_maps,
        ).to(device)
        # Copy learnable parameters
        odefunc.sigma = self.odefunc_sigma if hasattr(self, 'odefunc_sigma') else odefunc.sigma
        odefunc.source_term = self.odefunc_source_term if hasattr(self, 'odefunc_source_term') else odefunc.source_term

        # Concatenate state
        x0 = torch.cat([h_node, h_net], dim=0)  # (N+M, d)
        t_span = torch.tensor([0.0, self.T.abs()], device=device)

        # Solve ODE
        x_out = odeint(
            odefunc, x0, t_span,
            method=self.ode_method,
            atol=self.ode_tol,
            rtol=self.ode_tol,
        )[-1]  # Take final state

        h_node_out = x_out[:num_nodes]
        h_net_out = x_out[num_nodes:]

        return h_node_out, h_net_out

    def _initialize_odefunc_params(self):
        """Initialize shared ODE function parameters.
        Called once during __init__ to create persistent learnable params.
        """
        self.odefunc_sigma = nn.Parameter(torch.tensor(1.0))
        time_dim = 8
        self.odefunc_source_term = Seq(
            Linear(self.emb_dim + time_dim, self.emb_dim),
            LeakyReLU(),
            Linear(self.emb_dim, self.emb_dim),
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class SheafODEBlockV2(nn.Module):
    """Improved version with persistent ODE function parameters.

    The ODE function's learnable parameters (sigma, source_term MLP)
    are owned by this module, ensuring they're properly tracked by
    the optimizer. The graph topology is set dynamically per forward call.
    """

    def __init__(self, emb_dim, ode_T=1.0, ode_tol=1e-3, ode_method='dopri5'):
        super().__init__()
        self.emb_dim = emb_dim
        self.T = nn.Parameter(torch.tensor(ode_T))
        self.ode_tol = ode_tol
        self.ode_method = ode_method

        # Restriction map generator
        self.restriction_mlp = RestrictionMapMLP(emb_dim)

        # Persistent ODE function parameters
        self.sigma = nn.Parameter(torch.tensor(1.0))
        time_dim = 8
        self.source_term = Seq(
            Linear(emb_dim + time_dim, emb_dim),
            LeakyReLU(),
            Linear(emb_dim, emb_dim),
        )
        self.time_freqs = torch.tensor([1.0, 2.0, 4.0, 8.0])

    def forward(self, h_node, h_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        from torchdiffeq import odeint_adjoint as odeint

        num_nodes = h_node.shape[0]
        num_nets = h_net.shape[0]

        # Compute restriction maps
        restriction_maps = self.restriction_mlp(
            h_node, h_net,
            edge_index_node_to_net.to(device),
            edge_type_node_to_net.to(device)
        )

        # Edge indices on device
        edge_index = edge_index_node_to_net.to(device)
        time_freqs = self.time_freqs.to(device)

        # Closure-based ODE function to share this module's parameters
        sigma = self.sigma
        source_term = self.source_term
        emb_dim = self.emb_dim

        def odefunc(t, x):
            x_node = x[:num_nodes]
            x_net = x[num_nodes:]

            node_idx = edge_index[0]
            net_idx = edge_index[1]
            F = restriction_maps

            # Sheaf Laplacian
            msg = F * (x_node[node_idx] - x_net[net_idx])
            Lx_node = scatter(F * msg, node_idx, dim=0,
                              dim_size=num_nodes, reduce='add')
            Lx_net = scatter(-F * msg, net_idx, dim=0,
                             dim_size=num_nets, reduce='add')
            Lx = torch.cat([Lx_node, Lx_net], dim=0)

            # Time embedding
            freqs = time_freqs * t
            t_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)])
            t_emb = t_emb.unsqueeze(0).expand(x.shape[0], -1)

            # Source term
            g = source_term(torch.cat([x, t_emb], dim=-1))

            return -sigma * Lx + g

        # Integrate
        x0 = torch.cat([h_node, h_net], dim=0)
        t_span = torch.tensor([0.0, self.T.abs().item()], device=device)

        x_out = odeint(
            odefunc, x0, t_span,
            method=self.ode_method,
            atol=self.ode_tol,
            rtol=self.ode_tol,
        )[-1]

        return x_out[:num_nodes], x_out[num_nodes:]
