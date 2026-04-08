import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.utils import scatter

import time
import wandb
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.model_att import GNN_node
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Function to compute accuracy, precision, and recall
def compute_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Precision
    precision = precision_score(true_labels, predicted_labels, average='binary')

    # Recall
    recall = recall_score(true_labels, predicted_labels, average='binary')

    return accuracy, precision, recall


def build_h_dataset(data_dir, save_path):
    """Load raw Superblue data and convert to HeteroData list."""
    dataset = NetlistDataset(
        data_dir=data_dir, load_pe=True, pl=True,
        processed=True, load_indices=None
    )

    h_dataset = []
    for data in tqdm(dataset, desc="Processing designs"):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances

        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]]
        data.edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        mask_edges = mask[data.edge_index_sink_to_net[1]]
        data.edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['net'].x = data.net_features

        edge_index = torch.concat([data.edge_index_sink_to_net, data.edge_index_source_to_net], dim=1)
        h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([torch.zeros(data.edge_index_sink_to_net.shape[1]), torch.ones(data.edge_index_source_to_net.shape[1])]).bool()
        h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)

        h_data['design_name'] = data['design_name']
        h_data.num_instances = data.node_features.shape[0]

        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl

        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([global_mean_pool(h_data['node'].x, batch),
                global_max_pool(h_data['node'].x, batch)], dim=1)

        node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        net_demand = (net_demand - torch.mean(net_demand)) / torch.std(net_demand)

        h_data['variant_data_lst'] = [(node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)]
        h_dataset.append(h_data)

    torch.save(h_dataset, save_path)
    print(f"Saved {len(h_dataset)} designs to {save_path}")
    return h_dataset


def build_h_dataset_partition(data_dir, save_path):
    """Split each design into METIS partitions as separate training samples."""
    dataset = NetlistDataset(
        data_dir=data_dir, load_pe=True, pl=True,
        processed=True, load_indices=None
    )

    h_dataset = []
    for design_idx, data in enumerate(tqdm(dataset, desc="Processing designs")):
        num_instances = data.node_features.shape[0]
        data.num_instances = num_instances
        data.edge_index_sink_to_net[1] = data.edge_index_sink_to_net[1] - num_instances
        data.edge_index_source_to_net[1] = data.edge_index_source_to_net[1] - num_instances

        out_degrees = data.net_features[:, 1]
        mask = (out_degrees < 3000)
        mask_edges = mask[data.edge_index_source_to_net[1]]
        data.edge_index_source_to_net = data.edge_index_source_to_net[:, mask_edges]
        mask_edges = mask[data.edge_index_sink_to_net[1]]
        data.edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_edges]

        sink_edges = data.edge_index_sink_to_net
        source_edges = data.edge_index_source_to_net

        part_ids = data.batch
        unique_pids = torch.unique(part_ids)

        for pid in unique_pids:
            node_mask = (part_ids == pid)
            node_indices = node_mask.nonzero(as_tuple=False).squeeze(1)

            sink_node_mask = node_mask[sink_edges[0]]
            sub_sink = sink_edges[:, sink_node_mask]

            src_node_mask = node_mask[source_edges[0]]
            sub_source = source_edges[:, src_node_mask]

            relevant_net_ids = torch.cat([sub_sink[1], sub_source[1]]).unique()

            if len(node_indices) == 0 or len(relevant_net_ids) == 0:
                continue

            node_remap = torch.full((num_instances,), -1, dtype=torch.long)
            node_remap[node_indices] = torch.arange(len(node_indices))

            num_nets = data.net_features.shape[0]
            net_remap = torch.full((num_nets,), -1, dtype=torch.long)
            net_remap[relevant_net_ids] = torch.arange(len(relevant_net_ids))

            sub_sink_remapped = torch.stack([
                node_remap[sub_sink[0]],
                net_remap[sub_sink[1]]
            ])
            sub_source_remapped = torch.stack([
                node_remap[sub_source[0]],
                net_remap[sub_source[1]]
            ])

            h_data = HeteroData()
            h_data['node'].x = data.node_features[node_indices]
            h_data['net'].x = data.net_features[relevant_net_ids]

            edge_index = torch.cat([sub_sink_remapped, sub_source_remapped], dim=1)
            h_data['node', 'to', 'net'].edge_index, \
                h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
            h_data['node', 'to', 'net'].edge_type = torch.cat([
                torch.zeros(sub_sink_remapped.shape[1]),
                torch.ones(sub_source_remapped.shape[1])
            ]).bool()
            h_data['net', 'to', 'node'].edge_index, \
                h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)

            h_data['design_name'] = data['design_name']
            h_data['design_idx'] = design_idx
            h_data.num_instances = len(node_indices)

            node_demand = data.node_demand[node_indices]
            net_demand = data.net_demand[relevant_net_ids]
            net_hpwl = data.net_hpwl[relevant_net_ids]

            std_nd = torch.std(node_demand)
            node_demand = (node_demand - torch.mean(node_demand)) / max(std_nd, 1e-6)
            std_nh = torch.std(net_hpwl)
            net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / max(std_nh, 1e-6)
            std_netd = torch.std(net_demand)
            net_demand = (net_demand - torch.mean(net_demand)) / max(std_netd, 1e-6)

            batch = torch.zeros(len(node_indices), dtype=torch.long)
            num_vn = 1
            vn_node = torch.cat([
                global_mean_pool(h_data['node'].x, batch),
                global_max_pool(h_data['node'].x, batch)
            ], dim=1)

            h_data['variant_data_lst'] = [
                (node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)
            ]
            h_dataset.append(h_data)

    torch.save(h_dataset, save_path)
    print(f"Saved {len(h_dataset)} partition subgraphs to {save_path}")
    return h_dataset


### hyperparameter ###
test = False # if only test but not train
restart = False # if restart training
reload_dataset = False # if reload already processed h_dataset
mode = "full"  # "full" or "partition"

if test:
    restart = True

model_type = "dehnn" #this can be one of ["dehnn", "dehnn_att", "digcn", "digat"] "dehnn_att" might need large memory usage
num_layer = 3 #large number will cause OOM
num_dim = 32 #large number will cause OOM
vn = False #use virtual node or not
trans = False #use transformer or not
aggr = "add" #use aggregation as one of ["add", "max"]
device = "cuda" #use cuda or cpu
learning_rate = 0.001

# ── Data ──
full_cache = "h_dataset.pt"
partition_cache = "h_dataset_partition.pt"

if mode == "partition":
    # Train: partition subgraphs
    if reload_dataset and os.path.exists(partition_cache):
        h_dataset_train = torch.load(partition_cache)
    else:
        h_dataset_train = build_h_dataset_partition("data/superblue", partition_cache)

    # Val/Test: full graphs
    if reload_dataset and os.path.exists(full_cache):
        h_dataset_val = torch.load(full_cache)
    else:
        h_dataset_val = build_h_dataset("data/superblue", full_cache)

    all_train_indices = list(range(len(h_dataset_train)))
    all_valid_indices = list(range(len(h_dataset_val)))
    all_test_indices = all_valid_indices

    print(f"Train: {len(h_dataset_train)} partition subgraphs")
    print(f"Val/Test: {len(h_dataset_val)} full graphs")
else:
    if reload_dataset and os.path.exists(full_cache):
        h_dataset_train = torch.load(full_cache)
    else:
        h_dataset_train = build_h_dataset("data/superblue", full_cache)
    h_dataset_val = h_dataset_train

    load_data_indices = list(range(len(h_dataset_train)))
    all_train_indices = load_data_indices[:10]
    all_valid_indices = load_data_indices[10:]
    all_test_indices = load_data_indices[10:]

sys.path.append("models/layers/")

# ── Model ──
h_data = h_dataset_train[0]
model_name = f"{model_type}_{mode}_{num_layer}_{num_dim}_{vn}_{trans}_model.pt"

if restart:
    model = torch.load(model_name)
else:
    model = GNN_node(num_layer, num_dim, 1, 1, node_dim = h_data['node'].x.shape[1], net_dim = h_data['net'].x.shape[1], gnn_type=model_type, vn=vn, trans=trans, aggr=aggr, JK="Normal").to(device)

criterion_node = nn.MSELoss()
criterion_net = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
best_total_val = None

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*70}")
print(f"Model: {model_type.upper()}, Mode: {mode}, Layers: {num_layer}, Dim: {num_dim}, VN: {vn}, Trans: {trans}")
print(f"Parameters: {n_params:,}")
print(f"Train samples: {len(all_train_indices)}, Val samples: {len(all_valid_indices)}")
print(f"Device: {device}")
print(f"{'='*70}")

if not test:
    print(f"\n{'Epoch':>5} | {'Train Node':>11} | {'Train Net':>11} | "
          f"{'Val Node':>11} | {'Val Net':>11} | {'Best':>6}")
    print("-" * 70)

    for epoch in range(500):
        np.random.shuffle(all_train_indices)
        loss_node_all = 0
        loss_net_all = 0
        val_loss_node_all = 0
        val_loss_net_all = 0

        all_train_idx = 0
        for data_idx in all_train_indices:
            data = h_dataset_train[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                optimizer.zero_grad()
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)

                loss_node = criterion_node(node_representation, target_node.to(device))
                loss_net = criterion_net(net_representation, target_net_demand.to(device))
                loss = loss_node + loss_net
                loss.backward()
                optimizer.step()

                loss_node_all += loss_node.item()
                loss_net_all += loss_net.item()
                all_train_idx += 1

        all_valid_idx = 0
        for data_idx in all_valid_indices:
            data = h_dataset_val[data_idx]
            for inner_data_idx in range(len(data.variant_data_lst)):
                target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
                data.batch = batch
                data.num_vn = num_vn
                data.vn = vn_node
                node_representation, net_representation = model(data, device)
                node_representation = torch.squeeze(node_representation)
                net_representation = torch.squeeze(net_representation)

                val_loss_node = criterion_node(node_representation, target_node.to(device))
                val_loss_net = criterion_net(net_representation, target_net_demand.to(device))
                val_loss_node_all +=  val_loss_node.item()
                val_loss_net_all += val_loss_net.item()
                all_valid_idx += 1

        train_node = loss_node_all / all_train_idx
        train_net = loss_net_all / all_train_idx
        val_node = val_loss_node_all / all_valid_idx
        val_net = val_loss_net_all / all_valid_idx
        is_best = (best_total_val is None) or (train_node < best_total_val)

        print(f"{epoch+1:5d} | {train_node:11.6f} | {train_net:11.6f} | "
              f"{val_node:11.6f} | {val_net:11.6f} | {'  *' if is_best else '':>6}", flush=True)

        if is_best:
            best_total_val = train_node
            torch.save(model, model_name)

    print(f"\n{'='*70}")
    print(f"Training complete. Best train node loss: {best_total_val:.6f}")
    print(f"Model saved to: {model_name}")
    print(f"{'='*70}")
else:
    all_test_idx = 0
    test_loss_node_all = 0
    test_loss_net_all = 0
    for data_idx in all_test_indices:
        data = h_dataset_val[data_idx]
        for inner_data_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, batch, num_vn, vn_node = data.variant_data_lst[inner_data_idx]
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node
            node_representation, net_representation = model(data, device)
            node_representation = torch.squeeze(node_representation)
            net_representation = torch.squeeze(net_representation)

            test_loss_node = criterion_node(node_representation, target_node.to(device))
            test_loss_net = criterion_net(net_representation, target_net_demand.to(device))
            test_loss_node_all +=  test_loss_node.item()
            test_loss_net_all += test_loss_net.item()
            all_test_idx += 1

    print(f"\n{'='*70}")
    print(f"Test Set Evaluation")
    print(f"{'='*70}")
    print(f"  Node demand MSE: {test_loss_node_all/all_test_idx:.6f}")
    print(f"  Net demand MSE:  {test_loss_net_all/all_test_idx:.6f}")
    print(f"{'='*70}")
