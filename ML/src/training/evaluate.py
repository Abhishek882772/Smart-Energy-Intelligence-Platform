"""Standalone Evaluation CLI for Multi-Output NILM Baseline Models.

Evaluates a saved checkpoint (e.g. best_model.pt) on Test (Houses 6, 10, 11, 20)
and Domain-Shift (House 21) partitions without retraining or modifying weights.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

# Resolve repository root (4 levels up from ML/src/training/evaluate.py)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

cwd_path = Path.cwd().resolve()
if str(cwd_path) not in sys.path:
    sys.path.insert(0, str(cwd_path))

from ML.src.models.seq2point import MultiOutputSeq2Point
from ML.src.models.dataset import REFITShardDataset
from ML.src.losses.masked_multitask_loss import MaskedMultiTaskLoss
from ML.src.training import (
    Evaluator,
    CheckpointManager,
    get_device,
    get_dataloader_kwargs,
    log_environment_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Output Seq2Point NILM Baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ML/results/experiments/m4_5_seq2point_baseline/best_model.pt",
        help="Path to model checkpoint (.pt file) to evaluate",
    )
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
        help="Directory where evaluation metrics JSON files will be saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--eval_val",
        action="store_true",
        help="Also evaluate on validation partition (Houses 1, 8, 15)",
    )
    return parser.parse_args()


def print_metric_table(partition_name: str, res: Dict[str, Any], targets: List[str]):
    """Pretty-print evaluation metrics in a structured console table."""
    print(f"\n{'=' * 88}")
    print(f"PARTITION EVALUATION RESULTS: {partition_name.upper()} ({res.get('total_samples', 0):,} windows)")
    print(f"{'=' * 88}")
    print(f"Total Masked Loss : {res['total_loss']:.6f}")
    print(f"Macro Average MAE : {res.get('macro_mae', 0.0):.2f} W")
    print(f"Macro Average F1  : {res.get('macro_f1', 0.0):.4f}")
    print(f"{'-' * 88}")
    print(f"{'Appliance':<18} | {'Loss':<8} | {'MAE (W)':<8} | {'RMSE (W)':<9} | {'NDE':<7} | {'SAE':<7} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print(f"{'-' * 88}")
    
    app_metrics = res.get("appliance_metrics", {})
    app_losses = res.get("appliance_losses", {})
    for app in targets:
        m = app_metrics.get(app, {})
        loss_val = app_losses.get(app, 0.0)
        mae = m.get("mae", 0.0)
        rmse = m.get("rmse", 0.0)
        nde = m.get("nde", 0.0)
        sae = m.get("sae", 0.0)
        prec = m.get("precision", 0.0)
        rec = m.get("recall", 0.0)
        f1 = m.get("f1_score", 0.0)
        print(f"{app:<18} | {loss_val:<8.4f} | {mae:<8.2f} | {rmse:<9.2f} | {nde:<7.4f} | {sae:<7.4f} | {prec:<6.4f} | {rec:<6.4f} | {f1:<6.4f}")
    print(f"{'=' * 88}\n")


def run_evaluation(args):
    """Execute standalone evaluation for test and domain-shift partitions."""
    # 1. Device and Environment Diagnostics
    device = get_device(args.device)
    log_environment_info(device)

    # 2. Paths
    ckpt_path = repo_root / args.checkpoint
    cfg_path = repo_root / args.config
    stats_path = repo_root / args.stats
    data_dir = repo_root / args.data_dir
    exp_dir = repo_root / args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path.resolve()}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path.resolve()}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path.resolve()}")

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = yaml.safe_load(f)["channel_statistics"]

    # 3. Instantiate Architecture & Load Checkpoint
    print(f"\n[1/4] Instantiating Model Architecture from {cfg_path.name}...")
    model = MultiOutputSeq2Point.from_config(cfg_path)
    model.to(device)
    model.eval()

    print(f"[2/4] Loading Weights from Best Checkpoint: {ckpt_path.name}...")
    ckpt_mgr = CheckpointManager(exp_dir, monitor="val_loss", mode="min")
    ckpt_data = ckpt_mgr.load(ckpt_path, model=model, device=device)
    best_epoch = ckpt_data.get("best_epoch", ckpt_data.get("epoch", "N/A"))
    best_score = ckpt_data.get("best_score", ckpt_data.get("current_score", "N/A"))
    print(f"      Checkpoint Metadata -> Epoch: {best_epoch}, Val Loss: {best_score}")

    # 4. Setup Criterion & Evaluator
    criterion = MaskedMultiTaskLoss()
    target_names = ["refrigerator", "washing_machine", "television"]
    evaluator = Evaluator(
        criterion=criterion,
        normalization_stats=norm_stats,
        target_names=target_names,
        device=device,
    )
    loader_kwargs = get_dataloader_kwargs(device)

    test_houses = [6, 10, 11, 20]
    domain_shift_houses = [21]
    val_houses = [1, 8, 15]

    # Optional Validation Partition Evaluation
    if args.eval_val:
        val_shards = [data_dir / "validation" / f"house_{h}.npz" for h in val_houses]
        if all(s.exists() for s in val_shards):
            print(f"\n[EVAL] Evaluating Validation Partition (Houses {val_houses})...")
            val_ds = REFITShardDataset(val_shards)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
            val_res = evaluator.evaluate(model, val_loader, compute_detailed_metrics=True)
            print_metric_table("Validation Partition (Houses 1, 8, 15)", val_res, target_names)
            with open(exp_dir / "val_metrics_rechecked.json", "w", encoding="utf-8") as f:
                json.dump(val_res, f, indent=2, default=str)

    # 5. Test Partition Evaluation
    test_shards = [data_dir / "test" / f"house_{h}.npz" for h in test_houses]
    for s in test_shards:
        if not s.exists():
            raise FileNotFoundError(f"Required test shard missing: {s.resolve()}")

    print(f"\n[3/4] Evaluating Test Partition (Houses {test_houses})...")
    t0 = time.time()
    test_ds = REFITShardDataset(test_shards)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_res = evaluator.evaluate(model, test_loader, compute_detailed_metrics=True)
    t_test = time.time() - t0
    test_res["evaluation_time_seconds"] = round(t_test, 2)
    test_res["checkpoint_evaluated"] = str(ckpt_path.name)
    test_res["best_epoch"] = best_epoch

    print_metric_table(f"Test Partition (Houses {test_houses})", test_res, target_names)
    test_out_path = exp_dir / "test_metrics.json"
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_res, f, indent=2, default=str)
    print(f"  [SAVED] Test evaluation metrics saved to -> {test_out_path.resolve()}")

    # 6. Domain-Shift Partition Evaluation
    ds_shards = [data_dir / "domain_shift" / f"house_{h}.npz" for h in domain_shift_houses]
    for s in ds_shards:
        if not s.exists():
            raise FileNotFoundError(f"Required domain-shift shard missing: {s.resolve()}")

    print(f"\n[4/4] Evaluating Domain-Shift Partition (House 21)...")
    t0_ds = time.time()
    ds_dataset = REFITShardDataset(ds_shards)
    ds_loader = DataLoader(ds_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    ds_res = evaluator.evaluate(model, ds_loader, compute_detailed_metrics=True)
    t_ds = time.time() - t0_ds
    ds_res["evaluation_time_seconds"] = round(t_ds, 2)
    ds_res["checkpoint_evaluated"] = str(ckpt_path.name)
    ds_res["best_epoch"] = best_epoch

    print_metric_table(f"Domain-Shift Partition (House {domain_shift_houses})", ds_res, target_names)
    ds_out_path = exp_dir / "domain_shift_metrics.json"
    with open(ds_out_path, "w", encoding="utf-8") as f:
        json.dump(ds_res, f, indent=2, default=str)
    print(f"  [SAVED] Domain-shift evaluation metrics saved to -> {ds_out_path.resolve()}")

    print("\n" + "=" * 88)
    print("ALL EVALUATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
