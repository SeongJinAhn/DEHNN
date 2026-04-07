"""
Sheaf-augmented Bipartite Hypergraph Convolution Layer for VLSI netlists.

V3: Full-dimensional diagonal restriction maps + directional maps +
    sheaf disagreement attention.

Key ideas (based on 2025-2026 papers):
  1. Diagonal maps in full emb_dim (no stalk bottleneck)
     - Directional Sheaf Hypergraph Networks (arXiv 2510.04727)
  2. Separate forward/backward maps (not F and F^T)
     - CoED GNN (ICLR 2025)
  3. Sheaf disagreement as attention signal
     - Edge-Based Message Passing (arXiv 2510.13615)

Preserves DEHNN's aggregate+MLP structure for compatibility.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.utils import scatter


class SheafBipartiteConv(nn.Module):
    """
    Sheaf-augmented bipartite hypergraph convolution with:
    - Full-dimensional diagonal restriction maps (no stalk bottleneck)
    - Directional: separate learned maps for node→net and net→node
    - Sheaf disagreement attention: edges with higher disagreement
      from the net mean get higher attention weights

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation type ('add' or 'mean')
    """

    def __init__(self, in_channels, out_channels, aggr='add'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        d = out_channels

        # Linear transforms with residual (same as DEHNN)
        self.lin_node = Linear(in_channels, out_channels)
        self.lin_net = Linear(in_channels, out_channels)

        # Diagonal restriction maps: [h_node; h_net] → sigmoid → f ∈ R^d
        # 4 maps: forward/backward × sink/source (directional)
        self.fwd_sink = Linear(d * 2, d)
        self.fwd_source = Linear(d * 2, d)
        self.bwd_sink = Linear(d * 2, d)
        self.bwd_source = Linear(d * 2, d)

        # Disagreement → attention weight
        # Input: |Fh - mean(Fh)| ∈ R^d → scalar attention
        self.attn_fwd = Seq(Linear(d, 1), nn.Sigmoid())
        self.attn_bwd = Seq(Linear(d, 1), nn.Sigmoid())

        # Combining MLPs: 3 * out_channels input (no bottleneck!)
        self.psi = Seq(Linear(d * 3, d), ReLU(), Linear(d, d))
        self.mlp = Seq(Linear(d * 3, d), ReLU(), Linear(d, d))

    def forward(self, x, x_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """
        Args: Same interface as HyperConvLayer for drop-in compatibility.
        Returns: (h, h_net) — same as DEHNN.
        """
        edge_index_node_to_net = edge_index_node_to_net.to(device)
        edge_weight_node_to_net = edge_weight_node_to_net.to(device)
        edge_type_node_to_net = edge_type_node_to_net.to(device)
        edge_index_net_to_node = edge_index_net_to_node.to(device)
        edge_weight_net_to_node = edge_weight_net_to_node.to(device)

        num_nodes = x.size(0)
        num_nets = x_net.size(0)

        # Linear transform with residual
        h = self.lin_node(x) + x
        h_net = self.lin_net(x_net) + x_net

        node_idx_fwd = edge_index_node_to_net[0]
        net_idx_fwd = edge_index_node_to_net[1]
        source_mask = edge_type_node_to_net == 1
        sink_mask = ~source_mask

        # ═══════════════════════════════════════════════════════
        # Forward: Node → Net (with diagonal sheaf maps)
        # ═══════════════════════════════════════════════════════

        # Compute pair features for all edges
        pair_fwd = torch.cat([h[node_idx_fwd], h_net[net_idx_fwd]], dim=-1)  # [E, 2d]

        # Diagonal maps (sigmoid-bounded)
        f_all = torch.zeros(node_idx_fwd.size(0), self.out_channels, device=device)
        if sink_mask.any():
            f_all[sink_mask] = torch.sigmoid(self.fwd_sink(pair_fwd[sink_mask]))
        if source_mask.any():
            f_all[source_mask] = torch.sigmoid(self.fwd_source(pair_fwd[source_mask]))
        del pair_fwd

        # Apply diagonal map: f * h (element-wise, no bmm)
        Fh = f_all * h[node_idx_fwd]  # [E, d]
        del f_all

        # Disagreement attention: how much does this edge differ from net mean?
        net_mean = scatter(Fh, net_idx_fwd, dim=0, dim_size=num_nets, reduce='mean')
        disagree = (Fh - net_mean[net_idx_fwd]).abs()  # [E, d]
        attn = self.attn_fwd(disagree)  # [E, 1]
        del disagree, net_mean

        # Attention-weighted aggregation, split by edge type
        Fh_attn = attn * Fh * edge_weight_node_to_net.unsqueeze(-1)
        del Fh, attn

        agg_sink = scatter(Fh_attn[sink_mask], net_idx_fwd[sink_mask],
                           dim=0, dim_size=num_nets, reduce=self.aggr)
        agg_source = scatter(Fh_attn[source_mask], net_idx_fwd[source_mask],
                             dim=0, dim_size=num_nets, reduce=self.aggr)
        del Fh_attn

        # MLP combine (DEHNN structure)
        h_net = self.psi(torch.cat([h_net, agg_sink, agg_source], dim=1)) + x_net
        del agg_sink, agg_source

        # ═══════════════════════════════════════════════════════
        # Backward: Net → Node (separate directional maps)
        # ═══════════════════════════════════════════════════════

        net_idx_bwd = edge_index_net_to_node[0]
        node_idx_bwd = edge_index_net_to_node[1]

        # Pair features for backward (updated h_net)
        pair_bwd = torch.cat([h[node_idx_fwd], h_net[net_idx_fwd]], dim=-1)  # [E, 2d]

        # Backward diagonal maps (separate from forward — directional)
        g_all = torch.zeros(node_idx_fwd.size(0), self.out_channels, device=device)
        if sink_mask.any():
            g_all[sink_mask] = torch.sigmoid(self.bwd_sink(pair_bwd[sink_mask]))
        if source_mask.any():
            g_all[source_mask] = torch.sigmoid(self.bwd_source(pair_bwd[source_mask]))
        del pair_bwd

        # Apply backward map to net features
        Gh = g_all * h_net[net_idx_bwd]  # [E, d]
        del g_all

        # Disagreement attention (backward)
        node_mean = scatter(Gh, node_idx_bwd, dim=0, dim_size=num_nodes, reduce='mean')
        disagree_bwd = (Gh - node_mean[node_idx_bwd]).abs()
        attn_bwd = self.attn_bwd(disagree_bwd)  # [E, 1]
        del disagree_bwd, node_mean

        # Attention-weighted aggregation
        Gh_attn = attn_bwd * Gh * edge_weight_net_to_node.unsqueeze(-1)
        del Gh, attn_bwd

        agg_sink_bwd = scatter(Gh_attn[sink_mask], node_idx_bwd[sink_mask],
                               dim=0, dim_size=num_nodes, reduce=self.aggr)
        agg_source_bwd = scatter(Gh_attn[source_mask], node_idx_bwd[source_mask],
                                 dim=0, dim_size=num_nodes, reduce=self.aggr)
        del Gh_attn

        h = self.mlp(torch.cat([h, agg_sink_bwd, agg_source_bwd], dim=1)) + x
        del agg_sink_bwd, agg_source_bwd

        return h, h_net
