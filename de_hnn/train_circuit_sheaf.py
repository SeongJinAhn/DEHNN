#!/usr/bin/env python3
"""
Training script for Physics-Structured Sheaf Hypergraph on AnalogGenie circuits.

Ablation modes (--sheaf_mode):
  B0: identity     — no sheaf (baseline)
  B1: typed        — one R[tau] per terminal role
  B2: physics      — hard physics constraints (gate=proj, d-s tie)
  B3: generic      — MLP-learned per incidence (Duta'23)
  B4: physics_soft — soft physics + regularizer

Usage:
    python train_circuit_sheaf.py --data_dir Dataset --sheaf_mode typed
    python train_circuit_sheaf.py --data_dir Dataset --sheaf_mode physics --stalk_dim 4
    python train_circuit_sheaf.py --data_dir Dataset --sheaf_mode generic --model_type gnn
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.insert(1, 'data/')
from analoggenie_dataset import (
    load_analoggenie_dataset, NUM_ROLES, NUM_DEVICE_TYPES, CIRCUIT_LABELS
)

sys.path.append("models/layers/")
from models.sheaf_circuit_model import SheafCircuitModel, GNNBaseline


def train_val_test_split(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Stratified-ish split by shuffling."""
    rng = np.random.RandomState(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    return train_set, val_set, test_set


def train_epoch(model, loader, optimizer, criterion, device, lambda_phys=0.01):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if isinstance(model, GNNBaseline):
            pred = model(batch, device)
            reg = torch.tensor(0.0, device=device)
        else:
            pred, reg = model(batch, device)

        loss = criterion(pred, batch.y.squeeze()) + lambda_phys * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.y.size(0)
        pred_labels = pred.argmax(dim=1)
        total_correct += (pred_labels == batch.y.squeeze()).sum().item()
        total_samples += batch.y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        if isinstance(model, GNNBaseline):
            pred = model(batch, device)
        else:
            pred, _ = model(batch, device)

        loss = criterion(pred, batch.y.squeeze())
        total_loss += loss.item() * batch.y.size(0)

        pred_labels = pred.argmax(dim=1)
        all_preds.extend(pred_labels.cpu().tolist())
        all_labels.extend(batch.y.squeeze().cpu().tolist())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return total_loss / n, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train Sheaf HyperGNN on AnalogGenie")
    parser.add_argument("--data_dir", type=str, default="Dataset")
    parser.add_argument("--cache", type=str, default="analoggenie_cache.pt")
    parser.add_argument("--reload", action="store_true")

    # Model
    parser.add_argument("--model_type", type=str, default="sheaf",
                        choices=["sheaf", "gnn"])
    parser.add_argument("--sheaf_mode", type=str, default="generic",
                        choices=["identity", "typed", "physics", "physics_soft", "generic"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--stalk_dim", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--gate_rank", type=int, default=None)
    parser.add_argument("--readout", type=str, default="mean", choices=["mean", "add"])

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_phys", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)

    # Data subset (for quick experiments)
    parser.add_argument("--id_start", type=int, default=1)
    parser.add_argument("--id_end", type=int, default=3502)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──
    if args.reload and os.path.exists(args.cache):
        print(f"Loading cached dataset from {args.cache}")
        dataset = torch.load(args.cache)
    else:
        print(f"Loading AnalogGenie from {args.data_dir} (IDs {args.id_start}-{args.id_end})")
        dataset = load_analoggenie_dataset(
            args.data_dir, id_range=(args.id_start, args.id_end)
        )
        torch.save(dataset, args.cache)

    print(f"Total circuits: {len(dataset)}")
    labels = [d.y.item() for d in dataset]
    num_classes = len(set(labels))
    print(f"Classes: {num_classes}")
    inv_labels = {v: k for k, v in CIRCUIT_LABELS.items()}
    for label_id, count in sorted(Counter(labels).items()):
        print(f"  {inv_labels.get(label_id, '?'):20s}: {count}")

    # ── Split ──
    train_set, val_set, test_set = train_val_test_split(
        dataset, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # ── Model ──
    input_dim = NUM_ROLES + NUM_DEVICE_TYPES

    if args.model_type == "gnn":
        model = GNNBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=num_classes,
            dropout=args.dropout,
        ).to(device)
        model_name = f"circuit_gnn_{args.num_layers}_{args.hidden_dim}.pt"
    else:
        model = SheafCircuitModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            stalk_dim=args.stalk_dim,
            num_layers=args.num_layers,
            output_dim=num_classes,
            dropout=args.dropout,
            readout=args.readout,
            sheaf_mode=args.sheaf_mode,
            num_roles=NUM_ROLES,
            gate_rank=args.gate_rank,
        ).to(device)
        model_name = f"circuit_sheaf_{args.sheaf_mode}_{args.num_layers}_{args.stalk_dim}.pt"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_type} (sheaf_mode={args.sheaf_mode})")
    print(f"Parameters: {n_params:,}")

    # ── Train ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    print(f"\n{'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>7} | {'Best':>5}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.lambda_phys
        )
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)

        print(f"{epoch:5d} | {train_loss:8.4f} | {train_acc:6.1%} | "
              f"{val_loss:8.4f} | {val_acc:6.1%} | {val_f1:7.4f} | "
              f"{'*' if is_best else '':>5}", flush=True)

    # ── Test ──
    print(f"\nLoading best model from {model_name}")
    model.load_state_dict(torch.load(model_name, map_location=device))
    test_loss, test_acc, test_f1 = eval_epoch(model, test_loader, criterion, device)
    print(f"Test:  loss={test_loss:.4f}  acc={test_acc:.1%}  f1={test_f1:.4f}")
    print(f"Best val acc: {best_val_acc:.1%}")


if __name__ == "__main__":
    main()
