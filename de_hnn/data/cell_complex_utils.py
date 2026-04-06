"""Utilities for constructing a cell complex from a circuit netlist.

Converts the bipartite node-net graph into a cell complex:
  - 0-cells: circuit cells (nodes)
  - 1-cells: source→sink pairs within each net
  - 2-cells: nets (hyperedges)

Computes boundary operators B1 and B2 as sparse edge_index tensors.
"""

import torch
import numpy as np
from collections import defaultdict


def build_cell_complex(edge_index_node_to_net, edge_type, num_nodes, num_nets):
    """Build cell complex from bipartite netlist graph.

    For each net, creates 1-cells connecting source (driver) nodes to
    sink (load) nodes. Typically each net has 1 source and many sinks,
    so the number of 1-cells ≈ number of sink edges.

    Args:
        edge_index_node_to_net: (2, E) tensor, [0]=node indices, [1]=net indices
        edge_type: (E,) bool tensor, True=source, False=sink
        num_nodes: int, number of 0-cells
        num_nets: int, number of 2-cells

    Returns:
        dict with:
            'num_1cells': int
            'B1_index': (2, B1_E) tensor — boundary (0-cell ↔ 1-cell)
                        [0]: 0-cell index, [1]: 1-cell index
            'B1_weight': (B1_E,) tensor — boundary weights (+1 for src, -1 for snk)
            'B2_index': (2, B2_E) tensor — boundary (1-cell ↔ 2-cell)
                        [0]: 1-cell index, [1]: 2-cell index
            'B2_weight': (B2_E,) tensor — all ones
    """
    node_idx = edge_index_node_to_net[0].numpy()
    net_idx = edge_index_node_to_net[1].numpy()
    edge_type_np = edge_type.numpy() if isinstance(edge_type, torch.Tensor) else edge_type

    # Group edges by net
    net_to_sources = defaultdict(list)  # net_id → [node_ids]
    net_to_sinks = defaultdict(list)    # net_id → [node_ids]

    for i in range(len(node_idx)):
        nid = int(net_idx[i])
        vid = int(node_idx[i])
        if edge_type_np[i]:  # source
            net_to_sources[nid].append(vid)
        else:  # sink
            net_to_sinks[nid].append(vid)

    # Create 1-cells: one per (source, sink) pair within each net
    one_cells = []  # list of (source_node, sink_node, net_id)
    for nid in range(num_nets):
        sources = net_to_sources.get(nid, [])
        sinks = net_to_sinks.get(nid, [])
        if not sources or not sinks:
            continue
        for src in sources:
            for snk in sinks:
                one_cells.append((src, snk, nid))

    num_1cells = len(one_cells)

    if num_1cells == 0:
        return {
            'num_1cells': 0,
            'B1_index': torch.zeros(2, 0, dtype=torch.long),
            'B1_weight': torch.zeros(0),
            'B2_index': torch.zeros(2, 0, dtype=torch.long),
            'B2_weight': torch.zeros(0),
        }

    # Build B1: boundary operator (0-cell ↔ 1-cell)
    # Each 1-cell has exactly 2 boundary 0-cells (source and sink)
    B1_src_0cell = []  # 0-cell indices
    B1_src_1cell = []  # 1-cell indices
    B1_weights = []

    for cell_id, (src, snk, nid) in enumerate(one_cells):
        # Source endpoint: +1 weight
        B1_src_0cell.append(src)
        B1_src_1cell.append(cell_id)
        B1_weights.append(1.0)
        # Sink endpoint: -1 weight
        B1_src_0cell.append(snk)
        B1_src_1cell.append(cell_id)
        B1_weights.append(-1.0)

    B1_index = torch.tensor([B1_src_0cell, B1_src_1cell], dtype=torch.long)
    B1_weight = torch.tensor(B1_weights, dtype=torch.float)

    # Build B2: boundary operator (1-cell ↔ 2-cell)
    # Each 1-cell belongs to exactly one 2-cell (net)
    B2_src_1cell = []  # 1-cell indices
    B2_src_2cell = []  # 2-cell (net) indices

    for cell_id, (src, snk, nid) in enumerate(one_cells):
        B2_src_1cell.append(cell_id)
        B2_src_2cell.append(nid)

    B2_index = torch.tensor([B2_src_1cell, B2_src_2cell], dtype=torch.long)
    B2_weight = torch.ones(len(B2_src_1cell), dtype=torch.float)

    return {
        'num_1cells': num_1cells,
        'B1_index': B1_index,
        'B1_weight': B1_weight,
        'B2_index': B2_index,
        'B2_weight': B2_weight,
    }


def estimate_1cell_count(edge_index_node_to_net, edge_type, num_nets):
    """Estimate the number of 1-cells without building them.

    Useful for checking memory feasibility before construction.

    Returns:
        int: estimated number of 1-cells
    """
    net_idx = edge_index_node_to_net[1].numpy()
    edge_type_np = edge_type.numpy() if isinstance(edge_type, torch.Tensor) else edge_type

    src_count = defaultdict(int)
    snk_count = defaultdict(int)

    for i in range(len(net_idx)):
        nid = int(net_idx[i])
        if edge_type_np[i]:
            src_count[nid] += 1
        else:
            snk_count[nid] += 1

    total = 0
    for nid in range(num_nets):
        total += src_count.get(nid, 0) * snk_count.get(nid, 0)

    return total
