#!/usr/bin/env python3 -u
"""
Training script for Sheaf Hypergraph GNN on Superblue VLSI netlists.

Based on train_all_cross.py but uses SheafGNN_node instead of GNN_node.
Same data loading, same targets (node_demand, net_demand), same eval.

Usage:
    # First run: process dataset from raw pkl files
    python train_sheaf_superblue.py

    # Subsequent runs: reload cached dataset
    python train_sheaf_superblue.py --reload

    # Test only
    python train_sheaf_superblue.py --test --reload
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import global_mean_pool, global_max_pool
from tqdm import tqdm

sys.path.insert(1, 'data/')
from pyg_dataset import NetlistDataset

sys.path.append("models/layers/")
from models.sheaf_model import SheafGNN_node


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
        data.edge_index_sink_to_net[1] -= num_instances
        data.edge_index_source_to_net[1] -= num_instances

        # Filter high-degree nets (>3000 connections)
        out_degrees = data.net_features[:, 1]
        mask = out_degrees < 3000
        mask_src = mask[data.edge_index_source_to_net[1]]
        data.edge_index_source_to_net = data.edge_index_source_to_net[:, mask_src]
        mask_sink = mask[data.edge_index_sink_to_net[1]]
        data.edge_index_sink_to_net = data.edge_index_sink_to_net[:, mask_sink]

        h_data = HeteroData()
        h_data['node'].x = data.node_features
        h_data['net'].x = data.net_features

        edge_index = torch.concat([
            data.edge_index_sink_to_net,
            data.edge_index_source_to_net
        ], dim=1)
        h_data['node', 'to', 'net'].edge_index, \
            h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
        h_data['node', 'to', 'net'].edge_type = torch.concat([
            torch.zeros(data.edge_index_sink_to_net.shape[1]),
            torch.ones(data.edge_index_source_to_net.shape[1])
        ]).bool()
        h_data['net', 'to', 'node'].edge_index, \
            h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)

        h_data['design_name'] = data['design_name']
        h_data.num_instances = num_instances

        # Targets (normalized)
        node_demand = data.node_demand
        net_demand = data.net_demand
        net_hpwl = data.net_hpwl
        batch = data.batch
        num_vn = len(np.unique(batch))
        vn_node = torch.concat([
            global_mean_pool(h_data['node'].x, batch),
            global_max_pool(h_data['node'].x, batch)
        ], dim=1)

        node_demand = (node_demand - torch.mean(node_demand)) / torch.std(node_demand)
        net_hpwl = (net_hpwl - torch.mean(net_hpwl)) / torch.std(net_hpwl)
        net_demand = (net_demand - torch.mean(net_demand)) / torch.std(net_demand)

        h_data['variant_data_lst'] = [
            (node_demand, net_hpwl, net_demand, batch, num_vn, vn_node)
        ]
        h_dataset.append(h_data)

    torch.save(h_dataset, save_path)
    print(f"Saved {len(h_dataset)} designs to {save_path}")
    return h_dataset


def train_epoch(model, h_dataset, train_indices, optimizer,
                criterion_node, criterion_net, device):
    """Train one epoch over all training designs."""
    model.train()
    np.random.shuffle(train_indices)
    loss_node_total = 0.0
    loss_net_total = 0.0
    count = 0

    for data_idx in tqdm(train_indices, desc="Train", leave=False):
        data = h_dataset[data_idx]
        for inner_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, \
                batch, num_vn, vn_node = data.variant_data_lst[inner_idx]

            optimizer.zero_grad()
            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node

            node_pred, net_pred = model(data, device)
            node_pred = torch.squeeze(node_pred)
            net_pred = torch.squeeze(net_pred)

            loss_node = criterion_node(node_pred, target_node.to(device))
            loss_net = criterion_net(net_pred, target_net_demand.to(device))
            loss = loss_node + loss_net
            loss.backward()
            optimizer.step()

            loss_node_total += loss_node.item()
            loss_net_total += loss_net.item()
            count += 1

    return loss_node_total / count, loss_net_total / count


@torch.no_grad()
def eval_epoch(model, h_dataset, indices, criterion_node, criterion_net, device):
    """Evaluate on a set of designs."""
    model.eval()
    loss_node_total = 0.0
    loss_net_total = 0.0
    count = 0

    for data_idx in tqdm(indices, desc="Eval", leave=False):
        data = h_dataset[data_idx]
        for inner_idx in range(len(data.variant_data_lst)):
            target_node, target_net_hpwl, target_net_demand, \
                batch, num_vn, vn_node = data.variant_data_lst[inner_idx]

            data.batch = batch
            data.num_vn = num_vn
            data.vn = vn_node

            node_pred, net_pred = model(data, device)
            node_pred = torch.squeeze(node_pred)
            net_pred = torch.squeeze(net_pred)

            loss_node = criterion_node(node_pred, target_node.to(device))
            loss_net = criterion_net(net_pred, target_net_demand.to(device))

            loss_node_total += loss_node.item()
            loss_net_total += loss_net.item()
            count += 1

    return loss_node_total / count, loss_net_total / count


def main():
    parser = argparse.ArgumentParser(description="Train Sheaf GNN on Superblue")
    parser.add_argument("--data_dir", type=str, default="data/superblue")
    parser.add_argument("--cache", type=str, default="h_dataset_sheaf.pt")
    parser.add_argument("--reload", action="store_true",
                        help="Reload cached h_dataset instead of reprocessing")
    parser.add_argument("--test", action="store_true",
                        help="Test mode (load trained model, evaluate only)")
    parser.add_argument("--restart", action="store_true",
                        help="Resume training from saved model")

    # Model hyperparameters
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--num_dim", type=int, default=32)
    parser.add_argument("--stalk_dim", type=int, default=4)
    parser.add_argument("--vn", action="store_true", help="Use virtual node")
    parser.add_argument("--trans", action="store_true", help="Use transformer for VN")
    parser.add_argument("--aggr", type=str, default="add", choices=["add", "max"])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    if args.reload and os.path.exists(args.cache):
        print(f"Loading cached dataset from {args.cache}")
        h_dataset = torch.load(args.cache)
    else:
        print(f"Processing dataset from {args.data_dir}")
        h_dataset = build_h_dataset(args.data_dir, args.cache)

    print(f"Loaded {len(h_dataset)} designs")
    for i, d in enumerate(h_dataset):
        print(f"  [{i}] {d['design_name']}: "
              f"nodes={d['node'].x.shape[0]:,}, "
              f"nets={d['net'].x.shape[0]:,}, "
              f"edges={d['node', 'to', 'net'].edge_index.shape[1]:,}")

    # ── Splits ──
    all_indices = list(range(len(h_dataset)))
    train_indices = all_indices[:10]
    val_indices = all_indices[10:]
    test_indices = all_indices[10:]

    # ── Model ──
    h_data = h_dataset[0]
    model_name = f"sheaf_{args.num_layer}_{args.num_dim}_{args.stalk_dim}_{args.vn}_{args.trans}_model.pt"

    if args.test or args.restart:
        print(f"Loading model from {model_name}")
        model = torch.load(model_name, map_location=device)
    else:
        model = SheafGNN_node(
            num_layer=args.num_layer,
            emb_dim=args.num_dim,
            out_node_dim=1,
            out_net_dim=1,
            stalk_dim=args.stalk_dim,
            node_dim=h_data['node'].x.shape[1],
            net_dim=h_data['net'].x.shape[1],
            vn=args.vn,
            trans=args.trans,
            aggr=args.aggr,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion_node = nn.MSELoss()
    criterion_net = nn.MSELoss()

    # ── Test mode ──
    if args.test:
        test_node, test_net = eval_epoch(
            model, h_dataset, test_indices,
            criterion_node, criterion_net, device
        )
        print(f"Test node demand MSE: {test_node:.6f}")
        print(f"Test net demand MSE:  {test_net:.6f}")
        return

    # ── Train ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_train_loss = None

    print(f"\n{'Epoch':>5} | {'Train Node':>11} | {'Train Net':>11} | "
          f"{'Val Node':>11} | {'Val Net':>11}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_node, train_net = train_epoch(
            model, h_dataset, train_indices,
            optimizer, criterion_node, criterion_net, device
        )

        val_node, val_net = eval_epoch(
            model, h_dataset, val_indices,
            criterion_node, criterion_net, device
        )

        print(f"{epoch:5d} | {train_node:11.6f} | {train_net:11.6f} | "
              f"{val_node:11.6f} | {val_net:11.6f}", flush=True)

        if best_train_loss is None or train_node < best_train_loss:
            best_train_loss = train_node
            torch.save(model, model_name)

    print(f"\nBest train node loss: {best_train_loss:.6f}")
    print(f"Model saved to: {model_name}")


if __name__ == "__main__":
    main()
