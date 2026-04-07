"""
Sheaf-augmented Bipartite Hypergraph Convolution Layer for VLSI netlists.

Extends the DEHNN bipartite node↔net message passing with learnable
Sheaf restriction maps on the incidence structure.

Key idea: each (node, net) incidence pair gets a d×d restriction map F_{v←e}
that transforms node features before aggregation. This allows the model
to learn direction-aware, role-specific (sink vs source) transformations.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn.conv import SimpleConv


class SheafBipartiteConv(nn.Module):
    """
    One layer of Sheaf-augmented bipartite hypergraph convolution.

    Unlike standard HyperConvLayer which uses SimpleConv (sum aggregation),
    this layer learns restriction maps for each edge type (sink/source)
    and applies sheaf-aware message passing.

    Forward pass:
        1. Node → Net: For each (node, net) edge, apply restriction map F
           to node features, then aggregate per net
        2. Net → Node: Apply transposed restriction maps to net features,
           then aggregate per node
        3. Residual connections throughout

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        stalk_dim: Dimension of the sheaf stalk (restriction map size)
        aggr: Aggregation type ('add' or 'mean')
    """

    def __init__(self, in_channels, out_channels, stalk_dim=4, aggr='add'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = stalk_dim
        self.aggr = aggr

        # Project to stalk dimension
        self.node_to_stalk = Linear(in_channels, stalk_dim)
        self.net_to_stalk = Linear(in_channels, stalk_dim)

        # Sheaf map generators: edge features → d×d restriction map
        # Separate maps for sink and source edges
        self.sheaf_sink = nn.Sequential(
            Linear(stalk_dim * 2, stalk_dim * stalk_dim),
        )
        self.sheaf_source = nn.Sequential(
            Linear(stalk_dim * 2, stalk_dim * stalk_dim),
        )

        # Linear transforms with residual
        self.lin_node = Linear(in_channels, out_channels)
        self.lin_net = Linear(in_channels, out_channels)

        # Combining MLPs (same structure as DEHNN)
        self.psi = Seq(Linear(out_channels + stalk_dim * 2, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp = Seq(Linear(out_channels + stalk_dim * 2, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        # Learnable diffusion step size
        self.sigma = nn.Parameter(torch.tensor(0.5))

    def _compute_sheaf_maps(self, h_node_stalk, h_net_stalk,
                            edge_index, edge_type, device):
        """
        Compute restriction maps for each edge.

        Args:
            h_node_stalk: [N, d] node features in stalk space
            h_net_stalk: [M, d] net features in stalk space
            edge_index: [2, E] node→net edges
            edge_type: [E] bool, True=source, False=sink

        Returns:
            maps: [E, d, d] restriction maps
        """
        d = self.stalk_dim
        node_idx, net_idx = edge_index[0], edge_index[1]

        # Concatenate node and net stalk features for each edge
        pair_feat = torch.cat([
            h_node_stalk[node_idx],
            h_net_stalk[net_idx]
        ], dim=-1)  # [E, 2d]

        # Generate maps based on edge type
        source_mask = edge_type == 1
        sink_mask = ~source_mask

        maps = torch.zeros(edge_index.size(1), d, d, device=device)
        if sink_mask.any():
            maps[sink_mask] = self.sheaf_sink(pair_feat[sink_mask]).view(-1, d, d)
        if source_mask.any():
            maps[source_mask] = self.sheaf_source(pair_feat[source_mask]).view(-1, d, d)

        return maps

    def forward(self, x, x_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """
        Args: Same interface as HyperConvLayer for drop-in compatibility.
        """
        num_nodes = x.size(0)
        num_nets = x_net.size(0)
        d = self.stalk_dim

        # Linear transform with residual
        h = self.lin_node(x) + x
        h_net = self.lin_net(x_net) + x_net

        # Project to stalk space
        h_stalk = self.node_to_stalk(x)   # [N, d]
        h_net_stalk = self.net_to_stalk(x_net)  # [M, d]

        # === Forward: Node → Net (sheaf-aware) ===
        maps_fwd = self._compute_sheaf_maps(
            h_stalk, h_net_stalk,
            edge_index_node_to_net, edge_type_node_to_net, device
        )  # [E, d, d]

        node_idx_fwd = edge_index_node_to_net[0]
        net_idx_fwd = edge_index_node_to_net[1]

        # Apply restriction maps: F @ h_node for each edge
        Fh = torch.bmm(
            maps_fwd,
            h_stalk[node_idx_fwd].unsqueeze(-1)
        ).squeeze(-1)  # [E, d]

        # Weight by edge weight
        Fh_weighted = Fh * edge_weight_node_to_net.unsqueeze(-1).to(device)

        # Aggregate per net, split by edge type
        source_mask = edge_type_node_to_net == 1
        sink_mask = ~source_mask

        h_net_sink_sheaf = scatter(
            Fh_weighted[sink_mask], net_idx_fwd[sink_mask],
            dim=0, dim_size=num_nets, reduce=self.aggr
        )  # [M, d]
        h_net_source_sheaf = scatter(
            Fh_weighted[source_mask], net_idx_fwd[source_mask],
            dim=0, dim_size=num_nets, reduce=self.aggr
        )  # [M, d]

        # Combine: [h_net, sheaf_sink, sheaf_source] → MLP
        h_net = self.psi(torch.cat([h_net, h_net_sink_sheaf, h_net_source_sheaf], dim=1)) + x_net

        # === Backward: Net → Node (sheaf-aware) ===
        # Recompute net stalk after update
        h_net_stalk_updated = self.net_to_stalk(h_net)

        maps_bwd = self._compute_sheaf_maps(
            h_stalk, h_net_stalk_updated,
            edge_index_node_to_net, edge_type_node_to_net, device
        )  # [E, d, d]

        # Apply transposed maps: F^T @ h_net for each edge
        net_idx_bwd = edge_index_net_to_node[0]
        node_idx_bwd = edge_index_net_to_node[1]

        FTh = torch.bmm(
            maps_bwd.transpose(1, 2),
            h_net_stalk_updated[net_idx_bwd].unsqueeze(-1)
        ).squeeze(-1)  # [E, d]

        FTh_weighted = FTh * edge_weight_net_to_node.unsqueeze(-1).to(device)

        # Aggregate per node, split by edge type
        h_sink_sheaf = scatter(
            FTh_weighted[sink_mask], node_idx_bwd[sink_mask],
            dim=0, dim_size=num_nodes, reduce=self.aggr
        )
        h_source_sheaf = scatter(
            FTh_weighted[source_mask], node_idx_bwd[source_mask],
            dim=0, dim_size=num_nodes, reduce=self.aggr
        )

        h = self.mlp(torch.cat([h, h_sink_sheaf, h_source_sheaf], dim=1)) + x

        return h, h_net
