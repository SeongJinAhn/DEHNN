#!/usr/bin/env python3 -u
"""
Training script for Sheaf Hypergraph model on OCB (Open Circuit Benchmark).

Usage:
    python train_ocb.py --bench 101 --epochs 50 --model sheaf
    python train_ocb.py --bench 101 --epochs 50 --model gcn   # baseline

Targets: [gain, bw, pm, fom]  — primary metric is FoM prediction.
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

from data.ocb_dataset import load_ocb_splits
from models.sheaf_circuit_model import SheafCircuitModel, GNNBaseline


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    """Compute MAE, RMSE, R² for each target and overall."""
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    mae = (preds - targets).abs().mean(dim=0)  # [4]
    mse = ((preds - targets) ** 2).mean(dim=0)
    rmse = mse.sqrt()

    # R² per target
    ss_res = ((targets - preds) ** 2).sum(dim=0)
    ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)

    target_names = ["gain", "bw", "pm", "fom"]
    metrics = {}
    for i, name in enumerate(target_names):
        metrics[f"{name}_mae"] = mae[i].item()
        metrics[f"{name}_rmse"] = rmse[i].item()
        metrics[f"{name}_r2"] = r2[i].item()

    metrics["avg_mae"] = mae.mean().item()
    metrics["avg_rmse"] = rmse.mean().item()
    metrics["avg_r2"] = r2.mean().item()
    metrics["fom_mae"] = mae[3].item()
    return metrics


# ── Train / Eval ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch, device)
        loss = criterion(pred, batch.y.view(-1, 4))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * batch.y.size(0) // 4
        n_samples += batch.y.size(0) // 4

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, device)
        targets = batch.y.view(-1, 4)
        loss = criterion(pred, targets)
        total_loss += loss.item() * targets.size(0)
        n_samples += targets.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = avg_loss
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Sheaf Hypergraph on OCB")
    parser.add_argument("--bench", type=str, default="101", choices=["101", "301"])
    parser.add_argument("--data_root", type=str, default="./ocb_data")
    parser.add_argument("--model", type=str, default="sheaf", choices=["sheaf", "gcn"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--stalk_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--hypergraph_strategy", type=str, default="block_and_bridge",
                        choices=["block", "block_and_bridge", "star"],
                        help="Hypergraph construction strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data
    print(f"\n{'='*60}")
    print(f"Loading OCB Ckt-Bench-{args.bench} ...")
    print(f"{'='*60}")
    train_loader, val_loader, test_loader, dataset = load_ocb_splits(
        root=args.data_root, bench=args.bench,
        strategy=args.hypergraph_strategy,
        batch_size=args.batch_size, seed=args.seed,
    )

    # Determine input dimension from first sample
    sample = dataset[0]
    input_dim = sample.x.size(1)
    print(f"Input dim: {input_dim}, Num circuits: {len(dataset)}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Model
    print(f"\nModel: {args.model.upper()}")
    if args.model == "sheaf":
        model = SheafCircuitModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            stalk_dim=args.stalk_dim,
            num_layers=args.num_layers,
            output_dim=4,
            dropout=args.dropout,
            sigma=args.sigma,
        )
    else:
        model = GNNBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            output_dim=4,
            dropout=args.dropout,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Training setup
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'FoM MAE':>8} | {'FoM R²':>8} | {'Time':>6}")
    print("-" * 60)

    best_val_loss = float("inf")
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics
            # Save best model
            save_path = os.path.join(args.data_root, f"best_{args.model}_{args.bench}.pt")
            torch.save(model.state_dict(), save_path)

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_metrics['loss']:10.4f} | "
              f"{val_metrics['fom_mae']:8.4f} | {val_metrics['fom_r2']:8.4f} | "
              f"{elapsed:5.1f}s", flush=True)

    # Final test evaluation
    print(f"\n{'='*60}")
    print("Test Set Evaluation (best model)")
    print(f"{'='*60}")

    save_path = os.path.join(args.data_root, f"best_{args.model}_{args.bench}.pt")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))

    test_metrics = evaluate(model, test_loader, criterion, device)

    target_names = ["gain", "bw", "pm", "fom"]
    print(f"\n{'Metric':>10} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8}")
    print("-" * 42)
    for name in target_names:
        print(f"{name:>10} | {test_metrics[f'{name}_mae']:8.4f} | "
              f"{test_metrics[f'{name}_rmse']:8.4f} | "
              f"{test_metrics[f'{name}_r2']:8.4f}")
    print("-" * 42)
    print(f"{'Average':>10} | {test_metrics['avg_mae']:8.4f} | "
          f"{test_metrics['avg_rmse']:8.4f} | "
          f"{test_metrics['avg_r2']:8.4f}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
