#!/usr/bin/env python3
"""Smart-Energy-Intelligence-Platform — NILM Seq2Point Model Demonstration.

This interactive presentation script demonstrates the multi-output Sequence-to-Point (Seq2Point)
neural network for Non-Intrusive Load Monitoring (NILM).

It loads a trained PyTorch model checkpoint from Milestone 4.4, feeds a real 1-hour
aggregate power window (599 samples @ 6s cadence) from the REFIT House 2 dataset,
and disaggregates the individual active power consumption for:
  1. Refrigerator
  2. Washing Machine
  3. Television
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Resolve repository root
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ML.src.models.seq2point import MultiOutputSeq2Point
from ML.src.models.dataset import REFITShardDataset


def main():
    parser = argparse.ArgumentParser(description="Run Seq2Point NILM Disaggregation Presentation Demo")
    parser.add_argument(
        "--index",
        type=int,
        default=4510,
        help="Sliding window sample index in House 2 (default: 4510 - active fridge & TV)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ML/results/experiments/m4_4_tiny_overfit/tiny_overfit_checkpoint.pt",
        help="Path to trained PyTorch checkpoint",
    )
    parser.add_argument(
        "--output_figure",
        type=str,
        default="ML/demo/figures/demo_disaggregation_presentation.png",
        help="Destination path for generated presentation figure",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SMART ENERGY INTELLIGENCE PLATFORM - NILM SEQ2POINT DEMONSTRATION")
    print("=" * 80)

    # 1. Verify and Load Paths
    ckpt_path = repo_root / args.checkpoint
    if not ckpt_path.exists():
        print(f"\n[ERROR] Checkpoint not found at: {ckpt_path}")
        print("Please ensure Milestone 4.4 has been run and generated tiny_overfit_checkpoint.pt.")
        sys.exit(1)

    model_cfg_path = repo_root / "ML/configs/model_seq2point.yaml"
    if not model_cfg_path.exists():
        print(f"\n[ERROR] Model configuration not found at: {model_cfg_path}")
        sys.exit(1)

    stats_path = repo_root / "ML/configs/refit_normalization_stats.yaml"
    if not stats_path.exists():
        print(f"\n[ERROR] Normalization stats not found at: {stats_path}")
        sys.exit(1)

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = yaml.safe_load(f)["channel_statistics"]

    shard_path = repo_root / "ML/data/processed/REFIT/train/house_2.npz"
    if not shard_path.exists():
        print(f"\n[ERROR] Materialized REFIT House 2 shard not found at: {shard_path}")
        sys.exit(1)

    # 2. Load Materialized REFIT House 2 Dataset
    print(f"\n1. Loading Materialized Dataset:")
    print(f"   - Shard Source : {shard_path.relative_to(repo_root)}")
    dataset = REFITShardDataset(shard_path)
    print(f"   - Total Windows: {len(dataset):,} available sliding windows")

    window_idx = args.index
    if window_idx < 0 or window_idx >= len(dataset):
        print(f"[ERROR] Window index {window_idx} is out of range [0 .. {len(dataset)-1}]")
        sys.exit(1)

    x_tensor, y_tensor, mask_tensor, h_id = dataset[window_idx]
    timestamp_str = str(dataset.timestamps[window_idx])

    # 3. Load Model and Weights
    print(f"\n2. Loading Model & Checkpoint:")
    print(f"   - Architecture : MultiOutputSeq2Point (30,710,299 parameters)")
    print(f"   - Checkpoint   : {ckpt_path.relative_to(repo_root)}")
    model = MultiOutputSeq2Point.from_config(model_cfg_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"   - Training Mode: Evaluation Mode (torch.no_grad)")

    # 4. Prepare Input and Execute Inference
    # Input tensor shape: (599, 1) -> (1, 599, 1)
    x_input = x_tensor.unsqueeze(0)
    with torch.no_grad():
        pred_norm_tensor = model(x_input)  # Output shape: (1, 3)

    pred_norm = pred_norm_tensor.squeeze(0).numpy()  # (3,)
    true_norm = y_tensor.numpy()                     # (3,)
    masks = mask_tensor.numpy().astype(bool)         # (3,)

    # 5. Inverse-Transform to Physical Power (Watts) & Clamp Non-Negative
    def to_watts(val_norm: float, channel: str) -> float:
        st = norm_stats[channel]
        raw_w = (val_norm * st["std"]) + st["mean"]
        return float(np.maximum(0.0, raw_w))

    # De-normalize input aggregate sequence for time-series display
    agg_st = norm_stats["aggregate"]
    agg_watts = np.maximum(0.0, (x_tensor.squeeze(-1).numpy() * agg_st["std"]) + agg_st["mean"])

    channels = ["refrigerator", "washing_machine", "television"]
    preds_watts = [to_watts(pred_norm[i], ch) for i, ch in enumerate(channels)]
    trues_watts = [to_watts(true_norm[i], ch) for i, ch in enumerate(channels)]

    # 6. Display Structured Terminal Output
    print(f"\n3. Selected Input Context:")
    print(f"   - Target Household : House {h_id} (UK REFIT Longitudinal Dataset)")
    print(f"   - Window Index     : #{window_idx:,} (Sequence length: 599 samples @ 6s = 59.9 minutes)")
    print(f"   - Midpoint Index   : Sample #{window_idx + 299:,} (Step 299 within context window)")
    print(f"   - Midpoint Time    : {timestamp_str}")
    print(f"   - Aggregate Power  : Mean = {np.mean(agg_watts):.1f} W | Peak = {np.max(agg_watts):.1f} W | Midpoint = {agg_watts[299]:.1f} W")

    print("\n4. Disaggregation Results (Ground Truth vs Model Prediction):")
    print("   +" + "-" * 20 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 14 + "+")
    print("   | Appliance Channel  | Ground Truth (W) | Predicted (W)    | Abs Error (W)|")
    print("   +" + "-" * 20 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 14 + "+")
    for i, ch in enumerate(channels):
        diff = abs(preds_watts[i] - trues_watts[i])
        disp_name = ch.replace("_", " ").title()
        print(f"   | {disp_name:<18} | {trues_watts[i]:>14.2f} W | {preds_watts[i]:>14.2f} W | {diff:>10.2f} W |")
    print("   +" + "-" * 20 + "+" + "-" * 18 + "+" + "-" * 18 + "+" + "-" * 14 + "+")

    print("\n" + "=" * 80)
    print("METHODOLOGICAL NOTE:")
    print("This M4.4 checkpoint was trained on the tiny overfit subset and therefore this")
    print("is a MODEL LEARNABILITY DEMO, NOT a generalization/accuracy demonstration.")
    print("=" * 80)

    # 7. Generate Clean, Publication-Ready Presentation Figure
    print(f"\n5. Generating Presentation Figure...")
    plt.style.use("default")
    fig = plt.figure(figsize=(14, 8), tight_layout=True)
    fig.patch.set_facecolor("#ffffff")

    # Grid layout: Top = Aggregate Time-Series, Bottom = 3 Appliance Comparison Panels
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0])

    # Subplot 1 (Top Span): Aggregate Power Context Window
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor("#fbfbfb")
    time_steps = np.arange(len(agg_watts))
    ax_top.plot(time_steps, agg_watts, color="#1f77b4", linewidth=2.0, label="Household Aggregate Power (Watts)")
    ax_top.axvline(299, color="#d62728", linestyle="--", linewidth=2.0, label=f"Disaggregation Target Midpoint (t=299 @ {timestamp_str})")
    ax_top.scatter([299], [agg_watts[299]], color="#d62728", s=80, zorder=5)

    ax_top.set_title(
        f"Input Sequence Context X: 1-Hour Aggregate Load Profile (House {h_id}, Window #{window_idx})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax_top.set_xlabel("Time Steps within Window (6-Second Intervals | 0 to 598 = ~59.9 Minutes)", fontsize=10, fontweight="bold")
    ax_top.set_ylabel("Power (Watts)", fontsize=10, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    # Bottom Subplots: 3 Appliance Target vs Prediction Comparison Panels
    app_meta = [
        ("refrigerator", "Refrigerator", "#2ca02c", 0),
        ("washing_machine", "Washing Machine", "#d62728", 1),
        ("television", "Television", "#9467bd", 2),
    ]

    for ch_key, ch_title, ch_color, col_idx in app_meta:
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor("#fbfbfb")
        
        gt_val = trues_watts[col_idx]
        pr_val = preds_watts[col_idx]
        
        bars = ax.bar(
            ["Ground Truth", "Predicted"],
            [gt_val, pr_val],
            color=["#7f7f7f", ch_color],
            width=0.45,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.85,
        )
        
        # Value annotations on bars
        y_max = max(gt_val, pr_val, 10.0) * 1.3
        ax.set_ylim(0, y_max)
        
        for bar, val in zip(bars, [gt_val, pr_val]):
            h_val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h_val + (y_max * 0.03),
                f"{val:.1f} W",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        err = abs(pr_val - gt_val)
        ax.set_title(f"{ch_title}\n(Abs Error: {err:.1f} W)", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Power (Watts)", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Super Title and Footer Disclaimer
    fig.suptitle(
        f"Smart Energy Intelligence Platform — Seq2Point Disaggregation Presentation Demo\n[Household 2 | Midpoint Timestamp: {timestamp_str}]",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    fig_dest = repo_root / args.output_figure
    fig_dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dest, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"   [SUCCESS] Presentation figure saved to:")
    print(f"   -> {fig_dest.relative_to(repo_root)}")
    print("\n" + "=" * 80)
    print("DEMO EXECUTION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
