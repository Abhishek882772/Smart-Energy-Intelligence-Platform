"""Standalone Training & Evaluation CLI for Multi-Output NILM Baseline Models."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ML.src.models.seq2point import MultiOutputSeq2Point
from ML.src.models.dataset import REFITShardDataset
from ML.src.losses.masked_multitask_loss import MaskedMultiTaskLoss
from ML.src.training import (
    Trainer,
    Evaluator,
    CheckpointManager,
    TrainingHistory,
    get_device,
    get_dataloader_kwargs,
    log_environment_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Unified Multi-Output Seq2Point NILM Baseline")
    parser.add_argument(
        "--config",
        type=str,
        default="ML/configs/model_seq2point.yaml",
        help="Path to model architecture configuration YAML",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="ML/configs/refit_normalization_stats.yaml",
        help="Path to normalization statistics YAML",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="ML/data/processed/REFIT",
        help="Path to materialized processed REFIT shards directory",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="ML/results/experiments/m4_5_seq2point_baseline",
        help="Directory where checkpoints, history, and evaluation results are saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training and evaluation (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Optimizer weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs (default: 3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or None for auto-detect)",
    )
    return parser.parse_args()


def run_training(args):
    """Execute authoritative M4.5 training and evaluation pipeline."""
    # 1. Device and Environment Diagnostics
    device = get_device(args.device)
    env_info = log_environment_info(device)

    # 2. Paths
    cfg_path = repo_root / args.config
    stats_path = repo_root / args.stats
    data_dir = repo_root / args.data_dir
    exp_dir = repo_root / args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = yaml.safe_load(f)["channel_statistics"]

    # 3. Training and Validation Shard Paths
    train_houses = [2, 3, 5, 7, 9]
    val_houses = [1, 8, 15]
    test_houses = [6, 10, 11, 20]
    domain_shift_houses = [21]

    train_shards = [data_dir / "train" / f"house_{h}.npz" for h in train_houses]
    val_shards = [data_dir / "validation" / f"house_{h}.npz" for h in val_houses]

    for s in train_shards + val_shards:
        if not s.exists():
            raise FileNotFoundError(f"Required shard not found: {s.resolve()}")

    print(f"\nLoading Training Dataset ({len(train_shards)} shards: Houses {train_houses})...")
    train_ds = REFITShardDataset(train_shards)
    print(f"Loading Validation Dataset ({len(val_shards)} shards: Houses {val_houses})...")
    val_ds = REFITShardDataset(val_shards)

    loader_kwargs = get_dataloader_kwargs(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # 4. Model, Criterion, Optimizer, Scheduler
    model = MultiOutputSeq2Point.from_config(cfg_path)
    criterion = MaskedMultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-6
    )

    # 5. Checkpoint Manager & History Logger
    ckpt_mgr = CheckpointManager(exp_dir, monitor="val_loss", mode="min")
    history = TrainingHistory(exp_dir)
    evaluator = Evaluator(criterion, normalization_stats=norm_stats, device=device)

    # Save Experiment Configuration Snapshot
    config_snapshot = {
        "milestone": "M4.5",
        "experiment_name": exp_dir.name,
        "model": "MultiOutputSeq2Point",
        "parameters": model.get_parameter_count()["total_trainable"],
        "seed": 42,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "train_households": train_houses,
        "val_households": val_houses,
        "test_households": test_houses,
        "domain_shift_households": domain_shift_houses,
        "sampling_rate_seconds": 6,
        "window_length": 599,
        "stride": 30,
        "system_info": env_info,
    }
    with open(exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_snapshot, f, sort_keys=False)

    # 6. Execute Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        checkpoint_manager=ckpt_mgr,
        history=history,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=5.0,
        early_stopping_patience=args.patience,
    )

    t_start = time.time()
    fit_summary = trainer.fit(max_epochs=args.epochs)
    total_train_time = time.time() - t_start

    print(f"\nTraining completed in {total_train_time / 60:.2f} minutes.")
    print(f"Best Validation Loss: {fit_summary['best_val_loss']:.6f} at Epoch {fit_summary['best_epoch']}.")

    # 7. Evaluate Best Checkpoint on Test and Domain-Shift Partitions
    best_ckpt_path = exp_dir / "best_model.pt"
    if best_ckpt_path.exists():
        print("\nLoading Best Checkpoint for Final Test Evaluation...")
        ckpt_mgr.load(best_ckpt_path, model=model, device=device)

        # Test Partition Evaluation
        test_shards = [data_dir / "test" / f"house_{h}.npz" for h in test_houses]
        if all(s.exists() for s in test_shards):
            print(f"\nEvaluating on Test Partition (Houses {test_houses})...")
            test_ds = REFITShardDataset(test_shards)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            test_res = evaluator.evaluate(model, test_loader, compute_detailed_metrics=True)

            with open(exp_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_res, f, indent=2, default=str)
            print(f"  [Test Result] Loss: {test_res['total_loss']:.6f} | Macro MAE: {test_res.get('macro_mae', 0.0):.2f}W | Macro F1: {test_res.get('macro_f1', 0.0):.4f}")

        # Domain Shift Partition Evaluation
        ds_shards = [data_dir / "domain_shift" / f"house_{h}.npz" for h in domain_shift_houses]
        if all(s.exists() for s in ds_shards):
            print(f"\nEvaluating on Domain Shift Partition (House 21)...")
            ds_dataset = REFITShardDataset(ds_shards)
            ds_loader = DataLoader(ds_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            ds_res = evaluator.evaluate(model, ds_loader, compute_detailed_metrics=True)

            with open(exp_dir / "domain_shift_metrics.json", "w", encoding="utf-8") as f:
                json.dump(ds_res, f, indent=2, default=str)
            print(f"  [Domain-Shift Result] Loss: {ds_res['total_loss']:.6f} | Macro MAE: {ds_res.get('macro_mae', 0.0):.2f}W | Macro F1: {ds_res.get('macro_f1', 0.0):.4f}")

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED: Artifacts saved to {exp_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
