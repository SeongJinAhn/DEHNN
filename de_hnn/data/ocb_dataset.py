"""
OCB (Open Circuit Benchmark) Dataset Loader for Sheaf Hypergraph Model.

Loads CktGNN's Ckt-Bench-101/301 data (igraph format) and converts circuit DAGs
into hypergraph representations suitable for Sheaf Hypergraph diffusion.

Data format (from CktGNN):
  Each sample = (g_subgraph, g_full) where:
    - g_subgraph: igraph DAG of subcircuit blocks (type, subg_ntypes, r, c, gm)
    - g_full: igraph DAG of individual devices (type, feat)

Hypergraph construction:
  - Nodes = devices from g_full
  - Hyperedges = subcircuit blocks from g_subgraph (grouping constituent devices)
  - Falls back to star expansion when block membership can't be determined
"""

import os
import pickle
import csv
import subprocess
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader


CKTGNN_REPO = "https://github.com/zehao-dong/CktGNN.git"

# Number of device types in the full node-level DAG
NUM_DEVICE_TYPES = 10  # types 0-9 observed in CktGNN data


def _clone_cktgnn(data_root):
    """Clone CktGNN repo if not already present."""
    cktgnn_dir = os.path.join(data_root, "CktGNN")
    if not os.path.exists(cktgnn_dir):
        print(f"Cloning CktGNN repo to {cktgnn_dir} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", CKTGNN_REPO, cktgnn_dir],
            check=True,
        )
    return cktgnn_dir


def _load_performance(csv_path):
    """Load performance CSV → list of dicts with gain, bw, pm, fom, valid."""
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "gain": float(row["gain"]),
                "bw": float(row["bw"]),
                "pm": float(row["pm"]),
                "fom": float(row["fom"]),
                "valid": int(float(row["valid"])),
            })
    return records


def _igraph_to_edge_index(g):
    """Convert igraph edges to PyG edge_index [2, E] tensor."""
    edges = g.get_edgelist()
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def _build_node_features(g_full):
    """
    Build node feature matrix from the full node-level igraph.
    Features: one-hot device type (10 dims) + normalized feat value (1 dim) = 11 dims.
    """
    num_nodes = g_full.vcount()
    x = torch.zeros(num_nodes, NUM_DEVICE_TYPES + 1, dtype=torch.float)
    for v in g_full.vs:
        t = v["type"]
        if 0 <= t < NUM_DEVICE_TYPES:
            x[v.index, t] = 1.0
        feat_val = float(v["feat"]) if v["feat"] is not None else 0.0
        x[v.index, NUM_DEVICE_TYPES] = feat_val / 100.0  # normalize
    return x


def _build_hypergraph_from_subgraph(g_sub, g_full):
    """
    Build hypergraph incidence using subcircuit block membership.

    Each subcircuit block in g_sub expands into constituent devices in g_full.
    We reconstruct block→device membership by matching the ordered expansion:
    the subgraph DAG nodes expand sequentially into the full DAG nodes.

    Falls back to star expansion if the mapping fails.
    """
    num_full = g_full.vcount()
    num_sub = g_sub.vcount()

    # Reconstruct block membership: each subgraph node has subg_ntypes
    # listing the device types it contains. The full graph nodes appear
    # in the same order as the expansion of subgraph nodes.
    node_list = []
    hedge_list = []

    # Count how many devices each subgraph block produces
    block_sizes = []
    for v in g_sub.vs:
        ntypes = v["subg_ntypes"]
        block_sizes.append(len(ntypes) if isinstance(ntypes, (list, tuple)) else 1)

    total_from_blocks = sum(block_sizes)

    if total_from_blocks == num_full:
        # Perfect match: assign devices to blocks sequentially
        offset = 0
        for block_id, size in enumerate(block_sizes):
            for i in range(size):
                node_list.append(offset + i)
                hedge_list.append(block_id)
            offset += size
    else:
        # Fallback: star expansion on the full DAG
        return _star_expansion(g_full)

    node_indices = torch.tensor(node_list, dtype=torch.long)
    hedge_indices = torch.tensor(hedge_list, dtype=torch.long)
    num_hedges = num_sub
    return node_indices, hedge_indices, num_hedges


def _star_expansion(g):
    """Star expansion: each node + its neighbors = one hyperedge."""
    num_nodes = g.vcount()
    adj = [set() for _ in range(num_nodes)]
    for e in g.get_edgelist():
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    node_list = []
    hedge_list = []
    for v in range(num_nodes):
        members = [v] + sorted(adj[v])
        for m in members:
            node_list.append(m)
            hedge_list.append(v)

    return (torch.tensor(node_list, dtype=torch.long),
            torch.tensor(hedge_list, dtype=torch.long),
            num_nodes)


def _convert_single(sample, perf, idx):
    """
    Convert a single igraph sample + performance record into a
    hypergraph Data object.

    Args:
        sample: tuple (g_subgraph, g_full) of igraph graphs
        perf: dict with gain, bw, pm, fom, valid
        idx: circuit index
    """
    g_sub, g_full = sample

    # Node features from full DAG
    x = _build_node_features(g_full)
    num_nodes = x.size(0)

    # Original DAG edges (for baseline models)
    edge_index = _igraph_to_edge_index(g_full)

    # Make edges undirected for GCN baseline
    edge_index_undir = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Hypergraph incidence from subcircuit blocks
    node_indices, hedge_indices, num_hedges = _build_hypergraph_from_subgraph(
        g_sub, g_full
    )

    # Targets
    y = torch.tensor(
        [perf["gain"], perf["bw"], perf["pm"], perf["fom"]],
        dtype=torch.float,
    )
    valid = torch.tensor([perf["valid"]], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index_undir,  # undirected for baseline GCN
        edge_index_directed=edge_index,  # original DAG edges
        node_indices=node_indices,
        hedge_indices=hedge_indices,
        num_hedges=num_hedges,
        y=y,
        valid=valid,
        num_nodes=num_nodes,
        circuit_idx=idx,
    )
    return data


class OCBDataset(InMemoryDataset):
    """
    Open Circuit Benchmark dataset wrapped as a PyG InMemoryDataset.

    Uses the igraph-format pickle (ckt_bench_XXX.pkl) which is compatible
    across PyG versions.

    Args:
        root: Root directory for data storage.
        bench: '101' or '301' (Ckt-Bench variant).
        valid_only: If True, keep only circuits with valid=1.
        transform, pre_transform: Standard PyG transforms.
    """

    def __init__(self, root, bench="101", valid_only=True,
                 transform=None, pre_transform=None):
        self.bench = bench
        self.valid_only = valid_only
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"ckt_bench_{self.bench}.pkl",
                f"perform{self.bench}.csv"]

    @property
    def processed_file_names(self):
        suffix = "valid" if self.valid_only else "all"
        return [f"ocb_{self.bench}_{suffix}.pt"]

    def download(self):
        cktgnn_dir = _clone_cktgnn(self.root)
        bench_dir = os.path.join(cktgnn_dir, "OCB", f"CktBench{self.bench}")
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in self.raw_file_names:
            src = os.path.abspath(os.path.join(bench_dir, fname))
            dst = os.path.join(self.raw_dir, fname)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    def process(self):
        pkl_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        csv_path = os.path.join(self.raw_dir, self.raw_file_names[1])

        with open(pkl_path, "rb") as f:
            all_datasets = pickle.load(f)

        # all_datasets = (train_set, test_set)
        all_samples = []
        for split in all_datasets:
            all_samples.extend(split)

        perf_records = _load_performance(csv_path)

        data_list = []
        for idx, sample in enumerate(all_samples):
            if idx >= len(perf_records):
                break
            perf = perf_records[idx]
            if self.valid_only and perf["valid"] != 1:
                continue
            data = _convert_single(sample, perf, idx)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        print(f"Processed {len(data_list)} circuits "
              f"(filtered from {len(all_samples)} total)")
        self.save(data_list, self.processed_paths[0])


def load_ocb_splits(root, bench="101", valid_only=True, train_ratio=0.8,
                    val_ratio=0.1, batch_size=64, seed=42):
    """
    Convenience function: load OCB data and return train/val/test DataLoaders.
    """
    dataset = OCBDataset(root=root, bench=bench, valid_only=valid_only)
    n = len(dataset)
    print(f"OCB-{bench}: {n} circuits loaded")

    # Deterministic shuffle
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    dataset = dataset[perm]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ds = dataset[:n_train]
    val_ds = dataset[n_train:n_train + n_val]
    test_ds = dataset[n_train + n_val:]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset
