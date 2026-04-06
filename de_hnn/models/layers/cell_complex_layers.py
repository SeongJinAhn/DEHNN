"""Cell Complex Network layers for directed hypergraph neural networks.

Extends the bipartite node-net graph to a cell complex:
  - 0-cells: circuit cells (nodes)
  - 1-cells: source→sink connections within a net (driver-load pairs)
  - 2-cells: nets (hyperedges)

Message passing occurs at all three levels and between adjacent levels.
"""

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn.conv import GATv2Conv, SimpleConv
from torch_geometric.utils import scatter


class CellComplexConvLayer(nn.Module):
    """Cell complex convolution layer with 3-level message passing.

    Follows the CWN (Bodnar et al., NeurIPS 2021) pattern adapted for
    the circuit netlist bipartite structure with source/sink distinction.

    Message passing:
        1. 0-cell update: aggregate from 1-cells (B1^T) + from 2-cells (net→node)
        2. 1-cell update: aggregate from boundary 0-cells (B1) + from co-boundary 2-cells (B2^T)
        3. 2-cell update: aggregate from boundary 1-cells (B2) + from 0-cells (node→net)
    """

    def __init__(self, in_channels, out_channels, aggr='add', att=False):
        super().__init__()
        self.att = att

        # --- 0-cell (node) update ---
        # Aggregation from 1-cells via B1^T
        if att:
            self.conv_1cell_to_node = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_1cell_to_node = SimpleConv()
        # Aggregation from 2-cells (net→node), source/sink split
        if att:
            self.conv_net_to_node_src = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
            self.conv_net_to_node_snk = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_net_to_node_src = SimpleConv()
            self.conv_net_to_node_snk = SimpleConv()

        self.node_update = Seq(
            Linear(out_channels * 4, out_channels),
            LeakyReLU(),
            Linear(out_channels, out_channels),
        )

        # --- 1-cell (edge) update ---
        # From boundary 0-cells (B1: node→1-cell)
        if att:
            self.conv_node_to_1cell = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_node_to_1cell = SimpleConv()
        # From co-boundary 2-cells (B2^T: net→1-cell)
        if att:
            self.conv_net_to_1cell = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_net_to_1cell = SimpleConv()

        self.edge_update = Seq(
            Linear(out_channels * 3, out_channels),
            LeakyReLU(),
            Linear(out_channels, out_channels),
        )

        # --- 2-cell (net) update ---
        # From boundary 1-cells (B2: 1-cell→net)
        if att:
            self.conv_1cell_to_net = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_1cell_to_net = SimpleConv()
        # From 0-cells (node→net), source/sink split
        if att:
            self.conv_node_to_net_src = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
            self.conv_node_to_net_snk = GATv2Conv(
                in_channels, out_channels, heads=2, concat=False, add_self_loops=False
            )
        else:
            self.conv_node_to_net_src = SimpleConv()
            self.conv_node_to_net_snk = SimpleConv()

        self.net_update = Seq(
            Linear(out_channels * 4, out_channels),
            LeakyReLU(),
            Linear(out_channels, out_channels),
        )

        # Linear projections for residual alignment
        self.lin_node = Linear(in_channels, out_channels)
        self.lin_edge = Linear(in_channels, out_channels)
        self.lin_net = Linear(in_channels, out_channels)

    def forward(self, x_node, x_1cell, x_net,
                B1_index, B1_weight,
                B2_index, B2_weight,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device):
        """Forward pass.

        Args:
            x_node: (N, d) 0-cell features
            x_1cell: (E1, d) 1-cell features
            x_net: (M, d) 2-cell features
            B1_index: (2, B1_E) boundary operator edges (1-cell → 0-cell)
                      [0]: 0-cell indices, [1]: 1-cell indices
            B1_weight: (B1_E,) boundary weights
            B2_index: (2, B2_E) boundary operator edges (2-cell → 1-cell)
                      [0]: 1-cell indices, [1]: 2-cell indices
            B2_weight: (B2_E,) boundary weights
            edge_index_node_to_net: (2, E) original bipartite edges
            edge_weight_node_to_net: (E,)
            edge_type_node_to_net: (E,) bool
            edge_index_net_to_node: (2, E) reverse bipartite edges
            edge_weight_net_to_node: (E,)
            device: torch device

        Returns:
            x_node_out: (N, d)
            x_1cell_out: (E1, d)
            x_net_out: (M, d)
        """
        # --- Prepare residuals ---
        res_node = self.lin_node(x_node) + x_node
        res_edge = self.lin_edge(x_1cell) + x_1cell
        res_net = self.lin_net(x_net) + x_net

        source_mask = edge_type_node_to_net == 1
        sink_mask = ~source_mask

        # B1^T: 1-cell → 0-cell (flip B1_index for reverse direction)
        B1_T_index = B1_index.flip(0)  # (2, B1_E): [1-cell, 0-cell] → [0-cell src, 1-cell dst] reversed
        # B2^T: 2-cell → 1-cell
        B2_T_index = B2_index.flip(0)

        # =============================================
        # 1. UPDATE 0-CELLS (nodes)
        # =============================================
        # Aggregate from 1-cells via B1^T
        if self.att:
            h_from_1cell = self.conv_1cell_to_node(
                (x_1cell, x_node), B1_index.to(device)
            ) + x_node
        else:
            h_from_1cell = self.conv_1cell_to_node(
                (x_1cell, x_node), B1_index.to(device), B1_weight.to(device)
            ) + x_node

        # Aggregate from 2-cells (net→node) with source/sink split
        if self.att:
            h_from_net_src = self.conv_net_to_node_src(
                (x_net, x_node),
                edge_index_net_to_node[:, source_mask].to(device)
            ) + x_node
            h_from_net_snk = self.conv_net_to_node_snk(
                (x_net, x_node),
                edge_index_net_to_node[:, sink_mask].to(device)
            ) + x_node
        else:
            h_from_net_src = self.conv_net_to_node_src(
                (x_net, x_node),
                edge_index_net_to_node[:, source_mask].to(device),
                edge_weight_net_to_node[source_mask].to(device)
            ) + x_node
            h_from_net_snk = self.conv_net_to_node_snk(
                (x_net, x_node),
                edge_index_net_to_node[:, sink_mask].to(device),
                edge_weight_net_to_node[sink_mask].to(device)
            ) + x_node

        x_node_out = self.node_update(
            torch.cat([res_node, h_from_1cell, h_from_net_src, h_from_net_snk], dim=-1)
        ) + x_node

        # =============================================
        # 2. UPDATE 1-CELLS (edges)
        # =============================================
        # From boundary 0-cells via B1
        if self.att:
            h_from_node = self.conv_node_to_1cell(
                (x_node, x_1cell), B1_T_index.to(device)
            ) + x_1cell
        else:
            h_from_node = self.conv_node_to_1cell(
                (x_node, x_1cell), B1_T_index.to(device), B1_weight.to(device)
            ) + x_1cell

        # From co-boundary 2-cells via B2^T
        if self.att:
            h_from_net_edge = self.conv_net_to_1cell(
                (x_net, x_1cell), B2_index.to(device)
            ) + x_1cell
        else:
            h_from_net_edge = self.conv_net_to_1cell(
                (x_net, x_1cell), B2_index.to(device), B2_weight.to(device)
            ) + x_1cell

        x_1cell_out = self.edge_update(
            torch.cat([res_edge, h_from_node, h_from_net_edge], dim=-1)
        ) + x_1cell

        # =============================================
        # 3. UPDATE 2-CELLS (nets)
        # =============================================
        # From boundary 1-cells via B2
        if self.att:
            h_from_1cell_net = self.conv_1cell_to_net(
                (x_1cell, x_net), B2_T_index.to(device)
            ) + x_net
        else:
            h_from_1cell_net = self.conv_1cell_to_net(
                (x_1cell, x_net), B2_T_index.to(device), B2_weight.to(device)
            ) + x_net

        # From 0-cells (node→net) with source/sink split
        if self.att:
            h_from_node_src = self.conv_node_to_net_src(
                (x_node, x_net),
                edge_index_node_to_net[:, source_mask].to(device)
            ) + x_net
            h_from_node_snk = self.conv_node_to_net_snk(
                (x_node, x_net),
                edge_index_node_to_net[:, sink_mask].to(device)
            ) + x_net
        else:
            h_from_node_src = self.conv_node_to_net_src(
                (x_node, x_net),
                edge_index_node_to_net[:, source_mask].to(device),
                edge_weight_node_to_net[source_mask].to(device)
            ) + x_net
            h_from_node_snk = self.conv_node_to_net_snk(
                (x_node, x_net),
                edge_index_node_to_net[:, sink_mask].to(device),
                edge_weight_node_to_net[sink_mask].to(device)
            ) + x_net

        x_net_out = self.net_update(
            torch.cat([res_net, h_from_1cell_net, h_from_node_src, h_from_node_snk], dim=-1)
        ) + x_net

        return x_node_out, x_1cell_out, x_net_out
