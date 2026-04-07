"""
Sheaf-augmented Bipartite Hypergraph Convolution Layer for VLSI netlists.

V2: Real Sheaf Laplacian diffusion instead of simple transform+aggregate.

Key differences from V1:
  1. stalk_dim defaults to emb_dim (no bottleneck)
  2. Actual sheaf Laplacian diffusion: H' = H - sigma * L_sheaf @ H
  3. Low-rank restriction maps (rank r << d) to control memory
  4. Sheaf consistency loss for regularization
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.utils import scatter


class SheafBipartiteConv(nn.Module):
    """
    Sheaf Laplacian diffusion on bipartite node↔net hypergraph.

    Instead of transform → aggregate → MLP (≈ attention),
    this layer performs actual sheaf Laplacian diffusion:

        H_node' = H_node - sigma * L_sheaf_node @ H_node
        H_net'  = H_net  - sigma * L_sheaf_net  @ H_net

    where L_sheaf is computed from learned restriction maps F_{v←e}.

    Restriction maps use low-rank factorization F = U @ V^T (rank r)
    to control memory: O(E * d * r) instead of O(E * d * d).

    Args:
        in_channels: Input/output feature dimension (= stalk dimension)
        out_channels: Output feature dimension
        rank: Rank of restriction map factorization (controls capacity vs memory)
        aggr: Aggregation type ('add' or 'mean')
    """

    def __init__(self, in_channels, out_channels, rank=8, aggr='add'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = out_channels  # stalk = full embedding dim
        self.rank = rank
        self.aggr = aggr
        d = self.stalk_dim

        # Linear transforms with residual (same as DEHNN)
        self.lin_node = Linear(in_channels, out_channels)
        self.lin_net = Linear(in_channels, out_channels)

        # Low-rank restriction map generators: F = U @ V^T where U,V ∈ R^{d×r}
        # Separate for sink and source edges
        # Input: [h_node; h_net] → output: U and V vectors
        self.map_sink_U = Linear(out_channels * 2, d * rank)
        self.map_sink_V = Linear(out_channels * 2, d * rank)
        self.map_source_U = Linear(out_channels * 2, d * rank)
        self.map_source_V = Linear(out_channels * 2, d * rank)

        # Learnable diffusion step sizes (separate for node and net)
        self.sigma_node = nn.Parameter(torch.tensor(0.5))
        self.sigma_net = nn.Parameter(torch.tensor(0.5))

        # Post-diffusion transform
        self.post_node = Seq(Linear(out_channels, out_channels), ReLU(),
                             Linear(out_channels, out_channels))
        self.post_net = Seq(Linear(out_channels, out_channels), ReLU(),
                            Linear(out_channels, out_channels))

    def _compute_low_rank_maps(self, h_node, h_net, edge_index, edge_type, device):
        """
        Compute low-rank restriction maps F = U @ V^T for each edge.

        Returns:
            U: [E, d, r] left factor
            V: [E, d, r] right factor
            (F = U @ V^T is [E, d, d] but never materialized)
        """
        d = self.stalk_dim
        r = self.rank
        node_idx, net_idx = edge_index[0], edge_index[1]

        pair_feat = torch.cat([h_node[node_idx], h_net[net_idx]], dim=-1)  # [E, 2d]

        source_mask = edge_type == 1
        sink_mask = ~source_mask
        E = edge_index.size(1)

        U = torch.zeros(E, d, r, device=device)
        V = torch.zeros(E, d, r, device=device)

        if sink_mask.any():
            U[sink_mask] = self.map_sink_U(pair_feat[sink_mask]).view(-1, d, r)
            V[sink_mask] = self.map_sink_V(pair_feat[sink_mask]).view(-1, d, r)
        if source_mask.any():
            U[source_mask] = self.map_source_U(pair_feat[source_mask]).view(-1, d, r)
            V[source_mask] = self.map_source_V(pair_feat[source_mask]).view(-1, d, r)

        return U, V

    def _apply_map(self, U, V, h):
        """Apply F @ h = U @ (V^T @ h) using low-rank factorization. [E,d]"""
        # V^T @ h: [E, r, d] @ [E, d, 1] → [E, r, 1]
        Vth = torch.bmm(V.transpose(1, 2), h.unsqueeze(-1))  # [E, r, 1]
        # U @ (V^T @ h): [E, d, r] @ [E, r, 1] → [E, d, 1]
        return torch.bmm(U, Vth).squeeze(-1)  # [E, d]

    def _apply_map_transpose(self, U, V, h):
        """Apply F^T @ h = V @ (U^T @ h) using low-rank factorization. [E,d]"""
        Uth = torch.bmm(U.transpose(1, 2), h.unsqueeze(-1))  # [E, r, 1]
        return torch.bmm(V, Uth).squeeze(-1)  # [E, d]

    def forward(self, x, x_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """
        Args: Same interface as HyperConvLayer for drop-in compatibility.

        Returns:
            h: [N, out_channels] updated node features
            h_net: [M, out_channels] updated net features
            consistency_loss: scalar, sheaf consistency regularization
        """
        # Move all inputs to device
        edge_index_node_to_net = edge_index_node_to_net.to(device)
        edge_weight_node_to_net = edge_weight_node_to_net.to(device)
        edge_type_node_to_net = edge_type_node_to_net.to(device)
        edge_index_net_to_node = edge_index_net_to_node.to(device)
        edge_weight_net_to_node = edge_weight_net_to_node.to(device)

        num_nodes = x.size(0)
        num_nets = x_net.size(0)

        # Linear transform with residual
        h = self.lin_node(x) + x       # [N, d]
        h_net = self.lin_net(x_net) + x_net  # [M, d]

        node_idx = edge_index_node_to_net[0]  # [E]
        net_idx = edge_index_node_to_net[1]   # [E]

        # Compute low-rank restriction maps
        U, V = self._compute_low_rank_maps(
            h, h_net, edge_index_node_to_net,
            edge_type_node_to_net, device
        )  # U, V: [E, d, r]

        # ═══════════════════════════════════════════════════════
        # Sheaf Laplacian Diffusion: Node side
        # L_node @ H_node = Σ_{e∋v} w_e/|e| * (F^T F H_v - F^T avg_e)
        # ═══════════════════════════════════════════════════════

        # Step 1: F @ h_node for each edge
        Fh = self._apply_map(U, V, h[node_idx])  # [E, d]
        Fh_w = Fh * edge_weight_node_to_net.unsqueeze(-1)

        # Step 2: Aggregate F@h per net (= sum of projected node features per net)
        net_sum = scatter(Fh_w, net_idx, dim=0, dim_size=num_nets,
                          reduce='add')  # [M, d]

        # Step 3: Net degree for normalization
        ones = edge_weight_node_to_net.abs()
        net_deg = scatter(ones, net_idx, dim=0, dim_size=num_nets,
                          reduce='add').clamp(min=1.0)  # [M]
        net_avg = net_sum / net_deg.unsqueeze(-1)  # [M, d]

        # Step 4: F^T @ net_avg back to nodes
        FT_avg = self._apply_map_transpose(U, V, net_avg[net_idx])  # [E, d]
        FT_avg_w = FT_avg * edge_weight_node_to_net.unsqueeze(-1)
        off_diag_node = scatter(FT_avg_w, node_idx, dim=0,
                                dim_size=num_nodes, reduce='add')  # [N, d]

        # Step 5: Diagonal term: F^T @ F @ h_v per edge, aggregated
        FtFh = self._apply_map_transpose(U, V, Fh)  # [E, d]
        w_norm = (edge_weight_node_to_net / net_deg[net_idx]).unsqueeze(-1)
        diag_node = scatter(FtFh * w_norm, node_idx, dim=0,
                            dim_size=num_nodes, reduce='add')  # [N, d]

        # Step 6: Laplacian diffusion
        LH_node = diag_node - off_diag_node
        h = h - self.sigma_node * LH_node  # [N, d]
        h = self.post_node(h) + x  # post-diffusion MLP + residual

        # ═══════════════════════════════════════════════════════
        # Sheaf Laplacian Diffusion: Net side
        # L_net @ H_net = Σ_{v∈e} w_e/|v_nets| * (F F^T H_e - F avg_v)
        # ═══════════════════════════════════════════════════════

        # Recompute maps with updated node features
        U2, V2 = self._compute_low_rank_maps(
            h, h_net, edge_index_node_to_net,
            edge_type_node_to_net, device
        )

        net_idx_bwd = edge_index_net_to_node[0]
        node_idx_bwd = edge_index_net_to_node[1]

        # Step 1: F^T @ h_net for each edge
        FTh_net = self._apply_map_transpose(U2, V2, h_net[net_idx])  # [E, d]
        FTh_net_w = FTh_net * edge_weight_net_to_node.unsqueeze(-1)

        # Step 2: Aggregate per node
        node_sum = scatter(FTh_net_w, node_idx, dim=0, dim_size=num_nodes,
                           reduce='add')  # [N, d]

        node_deg = scatter(edge_weight_net_to_node.abs(), node_idx, dim=0,
                           dim_size=num_nodes, reduce='add').clamp(min=1.0)
        node_avg = node_sum / node_deg.unsqueeze(-1)  # [N, d]

        # Step 3: F @ node_avg back to nets
        F_avg = self._apply_map(U2, V2, node_avg[node_idx])  # [E, d]
        F_avg_w = F_avg * edge_weight_net_to_node.unsqueeze(-1)
        off_diag_net = scatter(F_avg_w, net_idx, dim=0,
                               dim_size=num_nets, reduce='add')

        # Step 4: Diagonal: F @ F^T @ h_net
        FFTh = self._apply_map(U2, V2, FTh_net)  # [E, d]
        w_norm2 = (edge_weight_net_to_node / node_deg[node_idx]).unsqueeze(-1)
        diag_net = scatter(FFTh * w_norm2, net_idx, dim=0,
                           dim_size=num_nets, reduce='add')

        # Step 5: Diffusion
        LH_net = diag_net - off_diag_net
        h_net = h_net - self.sigma_net * LH_net
        h_net = self.post_net(h_net) + x_net

        # ═══════════════════════════════════════════════════════
        # Sheaf Consistency Loss
        # ||F_{v←e} @ h_v - avg_{u∈e} F_{u←e} @ h_u||^2
        # ═══════════════════════════════════════════════════════
        Fh_new = self._apply_map(U, V, h[node_idx])  # [E, d]
        net_mean = scatter(Fh_new, net_idx, dim=0, dim_size=num_nets,
                           reduce='mean')  # [M, d]
        consistency_loss = ((Fh_new - net_mean[net_idx]) ** 2).mean()

        return h, h_net, consistency_loss
