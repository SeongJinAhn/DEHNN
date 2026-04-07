"""
Sheaf-augmented Bipartite Hypergraph Convolution Layer for VLSI netlists.

V2: Real Sheaf Laplacian diffusion with diagonal restriction maps.

Key design:
  1. stalk_dim = emb_dim (no bottleneck)
  2. Actual sheaf Laplacian diffusion: H' = H - sigma * L_sheaf @ H
  3. Diagonal restriction maps: F = diag(f), so F@h = f * h
     Memory: O(E * d) instead of O(E * d * d) or O(E * d * r)
  4. Sheaf consistency loss for regularization
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.utils import scatter


class SheafBipartiteConv(nn.Module):
    """
    Sheaf Laplacian diffusion on bipartite node-net hypergraph.

    Uses diagonal restriction maps F = diag(f) for memory efficiency.
    F@h = f * h (element-wise), F^T@h = f * h (diagonal is symmetric).

    Diffusion: H' = H - sigma * L_sheaf @ H

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        rank: Unused, kept for API compatibility
        aggr: Aggregation type ('add' or 'mean')
    """

    def __init__(self, in_channels, out_channels, rank=8, aggr='add'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = out_channels
        self.aggr = aggr
        d = self.stalk_dim

        # Linear transforms with residual
        self.lin_node = Linear(in_channels, out_channels)
        self.lin_net = Linear(in_channels, out_channels)

        # Diagonal restriction map generators: [h_node; h_net] → f ∈ R^d
        # f is the diagonal of F, so F@h = f * h
        self.map_sink = Linear(out_channels * 2, d)
        self.map_source = Linear(out_channels * 2, d)

        # Learnable diffusion step sizes
        self.sigma_node = nn.Parameter(torch.tensor(0.1))
        self.sigma_net = nn.Parameter(torch.tensor(0.1))

        # Post-diffusion transform
        self.post_node = Seq(Linear(out_channels, out_channels), ReLU(),
                             Linear(out_channels, out_channels))
        self.post_net = Seq(Linear(out_channels, out_channels), ReLU(),
                            Linear(out_channels, out_channels))

    def _compute_diag_maps(self, h_node, h_net, edge_index, edge_type, device):
        """
        Compute diagonal restriction maps for each edge.

        Returns:
            f: [E, d] diagonal entries of restriction maps
        """
        d = self.stalk_dim
        node_idx, net_idx = edge_index[0], edge_index[1]

        pair_feat = torch.cat([h_node[node_idx], h_net[net_idx]], dim=-1)  # [E, 2d]

        source_mask = edge_type == 1
        sink_mask = ~source_mask
        E = edge_index.size(1)

        f = torch.zeros(E, d, device=device)

        if sink_mask.any():
            f[sink_mask] = torch.sigmoid(self.map_sink(pair_feat[sink_mask]))
        if source_mask.any():
            f[source_mask] = torch.sigmoid(self.map_source(pair_feat[source_mask]))

        del pair_feat
        return f

    def forward(self, x, x_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """
        Returns:
            h: [N, out_channels] updated node features
            h_net: [M, out_channels] updated net features
            consistency_loss: scalar, sheaf consistency regularization
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

        node_idx = edge_index_node_to_net[0]
        net_idx = edge_index_node_to_net[1]

        # Diagonal restriction maps: f[e] ∈ R^d
        f = self._compute_diag_maps(
            h, h_net, edge_index_node_to_net,
            edge_type_node_to_net, device
        )  # [E, d]

        # ═══════════════════════════════════════════════════════
        # Sheaf Laplacian Diffusion: Node side
        # With diagonal F: F@h = f*h, F^T@h = f*h
        # ═══════════════════════════════════════════════════════

        # Step 1: F @ h_node = f * h_node for each edge
        Fh = f * h[node_idx]  # [E, d]
        Fh_w = Fh * edge_weight_node_to_net.unsqueeze(-1)

        # Step 2: Aggregate per net
        net_sum = scatter(Fh_w, net_idx, dim=0, dim_size=num_nets, reduce='add')

        # Step 3: Net degree for normalization
        net_deg = scatter(edge_weight_node_to_net.abs(), net_idx, dim=0,
                          dim_size=num_nets, reduce='add').clamp(min=1.0)
        net_avg = net_sum / net_deg.unsqueeze(-1)

        # Step 4: F^T @ net_avg = f * net_avg back to nodes
        FT_avg = f * net_avg[net_idx]
        FT_avg_w = FT_avg * edge_weight_node_to_net.unsqueeze(-1)
        off_diag_node = scatter(FT_avg_w, node_idx, dim=0,
                                dim_size=num_nodes, reduce='add')
        del FT_avg, FT_avg_w

        # Step 5: Diagonal term: F^T @ F @ h = f * f * h = f^2 * h
        FtFh = f * Fh  # [E, d] — f^2 * h
        w_norm = (edge_weight_node_to_net / net_deg[net_idx]).unsqueeze(-1)
        diag_node = scatter(FtFh * w_norm, node_idx, dim=0,
                            dim_size=num_nodes, reduce='add')
        del FtFh, w_norm

        # Step 6: Laplacian diffusion
        LH_node = diag_node - off_diag_node
        h = h - self.sigma_node * LH_node
        h = self.post_node(h) + x
        del diag_node, off_diag_node, LH_node

        # ═══════════════════════════════════════════════════════
        # Sheaf Laplacian Diffusion: Net side
        # ═══════════════════════════════════════════════════════

        # Recompute maps with updated node features
        f2 = self._compute_diag_maps(
            h, h_net, edge_index_node_to_net,
            edge_type_node_to_net, device
        )

        # Step 1: F^T @ h_net = f * h_net for each edge
        FTh_net = f2 * h_net[net_idx]  # [E, d]
        FTh_net_w = FTh_net * edge_weight_net_to_node.unsqueeze(-1)

        # Step 2: Aggregate per node
        node_sum = scatter(FTh_net_w, node_idx, dim=0, dim_size=num_nodes,
                           reduce='add')
        del FTh_net_w

        node_deg = scatter(edge_weight_net_to_node.abs(), node_idx, dim=0,
                           dim_size=num_nodes, reduce='add').clamp(min=1.0)
        node_avg = node_sum / node_deg.unsqueeze(-1)
        del node_sum

        # Step 3: F @ node_avg = f * node_avg back to nets
        F_avg = f2 * node_avg[node_idx]
        F_avg_w = F_avg * edge_weight_net_to_node.unsqueeze(-1)
        off_diag_net = scatter(F_avg_w, net_idx, dim=0,
                               dim_size=num_nets, reduce='add')
        del F_avg, F_avg_w

        # Step 4: Diagonal: F @ F^T @ h_net = f^2 * h_net
        FFTh = f2 * FTh_net  # [E, d]
        w_norm2 = (edge_weight_net_to_node / node_deg[node_idx]).unsqueeze(-1)
        diag_net = scatter(FFTh * w_norm2, net_idx, dim=0,
                           dim_size=num_nets, reduce='add')
        del FFTh, FTh_net, w_norm2, f2

        # Step 5: Diffusion
        LH_net = diag_net - off_diag_net
        h_net = h_net - self.sigma_net * LH_net
        h_net = self.post_net(h_net) + x_net
        del diag_net, off_diag_net, LH_net

        # ═══════════════════════════════════════════════════════
        # Sheaf Consistency Loss (detached maps to save memory)
        # ||f*h_v - avg_{u∈e}(f*h_u)||^2
        # ═══════════════════════════════════════════════════════
        with torch.no_grad():
            f_det = f.detach()
        Fh_new = f_det * h[node_idx]  # [E, d]
        net_mean = scatter(Fh_new, net_idx, dim=0, dim_size=num_nets,
                           reduce='mean')
        consistency_loss = ((Fh_new - net_mean[net_idx]) ** 2).mean()
        del f, f_det, Fh, Fh_new, net_mean

        return h, h_net, consistency_loss
