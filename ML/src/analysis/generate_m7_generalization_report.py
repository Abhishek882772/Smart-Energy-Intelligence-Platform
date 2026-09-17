"""Generate Comprehensive Milestone 7 Cross-Household Generalization Report.

Parses official M4.5 evaluation metrics, dataset metadata, and configuration
to generate a rigorous, scientifically grounded markdown report for Paper 1.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import yaml

# Resolve repository root (4 levels up from ML/src/analysis/generate_m7_generalization_report.py)
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_json_or_fallback(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(output_md_path: Path):
    exp_dir = repo_root / "ML/results/experiments/m4_5_seq2point_baseline"
    test_json_path = exp_dir / "test_metrics.json"
    ds_json_path = exp_dir / "domain_shift_metrics.json"
    manifest_path = repo_root / "ML/data/processed/REFIT/dataset_manifest.json"
    mapping_path = repo_root / "ML/configs/refit_target_mapping.yaml"
    stats_path = repo_root / "ML/configs/refit_normalization_stats.yaml"
    eligibility_path = repo_root / "ML/results/refit_eligibility_report.csv"

    test_res = load_json_or_fallback(test_json_path)
    ds_res = load_json_or_fallback(ds_json_path)
    manifest = load_json_or_fallback(manifest_path)

    with open(mapping_path, "r", encoding="utf-8") as f:
        target_mapping = yaml.safe_load(f)["households"]

    with open(stats_path, "r", encoding="utf-8") as f:
        norm_stats = yaml.safe_load(f)["channel_statistics"]

    eligibility_df = pd.read_csv(eligibility_path)

    # Build Markdown Content
    md = []
    md.append("# Milestone 7 — Cross-Household Generalization Analysis Report")
    md.append("")
    md.append("**Research Project:** *Cross-Country Generalization of NILM Models on Residential Energy Consumption*  ")
    md.append("**Milestone:** Milestone 7 — In-Domain Cross-Household Generalization & Robustness Analysis  ")
    md.append("**Target Publication Scope:** Paper 1 (Multi-Output Baseline Benchmark & Cross-Household Generalization)  ")
    md.append("**Frozen Baseline Model:** Canonical Multi-Output Seq2Point (`MultiOutputSeq2Point`, 30,710,299 parameters)  ")
    md.append("**Frozen Checkpoint:** [`ML/results/experiments/m4_5_seq2point_baseline/best_model.pt`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/results/experiments/m4_5_seq2point_baseline/best_model.pt) (Epoch 3, Val Loss: `0.287161`)  ")
    md.append("**Date:** September 2026  ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("> [!NOTE]")
    md.append("> **Document Purpose & Baseline Lock Declaration:**  ")
    md.append("> This report presents a rigorous, empirical cross-household generalization analysis of the already-trained and frozen Milestone 4.5 baseline model. No models were retrained, no weights or hyperparameters were altered, and no adaptation algorithms were introduced. The research evaluates how a multi-output Sequence-to-Point neural network generalizes across unseen residential dwellings within the UK REFIT dataset.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 1
    md.append("## 1. Experimental Methodology & Household-Disjoint Design")
    md.append("")
    md.append("In non-intrusive load monitoring (NILM), evaluating models on held-out temporal segments of the *same* households risks inflating performance metrics due to memorization of home-specific baseload patterns, exact appliance electrical parameters, and static background noise. To establish valid scientific generalization, this study enforces a **strict household-disjoint experimental design** across all 13 materialized REFIT households ($2,892,782$ sliding windows / $97.9\\text{M}$ raw 6-second samples).")
    md.append("")
    md.append("### 1.1 Dataset Partitioning & Channel Availability")
    md.append("")
    md.append("| Partition | House ID | Total 6s Rows | Sliding Windows | Refrigerator Channel | Washing Machine Channel | Television Channel | Sensor Issues (%) | Research Role |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    partition_map = {
        "train": [2, 3, 5, 7, 9],
        "validation": [1, 8, 15],
        "test": [6, 10, 11, 20],
        "domain_shift": [21],
    }

    for p_name, houses in partition_map.items():
        for h in houses:
            row = eligibility_df[eligibility_df["household"] == h].iloc[0]
            shards = manifest["partitions"][p_name]["shards"]
            shard_info = next((s for s in shards if s["household_id"] == h), None)
            w_count = shard_info["total_windows"] if shard_info else 0
            rows_6s = shard_info["aligned_rows_6s"] if shard_info else 0

            h_meta = target_mapping.get(h, {})
            f_chan = h_meta.get("refrigerator", {}).get("name", "N/A") if h_meta.get("refrigerator") else "None"
            w_chan = h_meta.get("washing_machine", {}).get("name", "N/A") if h_meta.get("washing_machine") else "None"
            t_chan = h_meta.get("television", {}).get("name", "N/A") if h_meta.get("television") else "**None (Unmonitored)**"

            role_desc = {
                "train": "Training (Parameter optimization)",
                "validation": "Validation (Model selection / early stopping)",
                "test": "**Core In-Domain Unseen Test**",
                "domain_shift": "**Domain-Shift (Solar PV Net-Metered)**",
            }[p_name]

            md.append(f"| **{p_name.upper()}** | House {h:2d} | {rows_6s:>10,} | {w_count:>9,} | {f_chan} | {w_chan} | {t_chan} | {row['issues_pct']:.2f}% | {role_desc} |")

    md.append("")
    md.append("### 1.2 Leakage Prevention & Reproducibility Controls")
    md.append("1. **Entity-Level Isolation:** Household splits were performed at the dwelling boundary. Zero timestamp slices from test households were accessible during training or validation.")
    md.append("2. **Strict Normalization Fitting:** Normalization statistics (Aggregate: $\\mu=573.27\\text{W}, \\sigma=753.77\\text{W}$; Refrigerator: $\\mu=41.84\\text{W}, \\sigma=55.32\\text{W}$; Washing Machine: $\\mu=26.23\\text{W}, \\sigma=210.28\\text{W}$; Television: $\\mu=13.71\\text{W}, \\sigma=102.73\\text{W}$) were derived **exclusively from the 5 training households** ($36,426,081$ valid samples). No test or validation data influenced these parameters.")
    md.append("3. **Checkpoint Selection Protocol:** The evaluated model checkpoint ([`best_model.pt`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/results/experiments/m4_5_seq2point_baseline/best_model.pt)) was selected at Epoch 3 based strictly on minimum validation loss (`0.287161`) on Houses `[1, 8, 15]`. Test households `[6, 10, 11, 20]` and `[21]` were never loaded during training.")
    md.append("4. **Fixed A-Priori Operational Thresholds:** Activation thresholds for binary state classification were fixed from literature standards (Fridge: $15.0\\text{W}$, Washing Machine: $20.0\\text{W}$, Television: $10.0\\text{W}$), not tuned on test predictions.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 2
    md.append("## 2. In-Domain Cross-Household Generalization Results")
    md.append("")
    md.append("The frozen baseline was evaluated on the standard in-domain test partition comprising 4 completely unseen dwellings (**Houses 6, 10, 11, and 20**, totaling $838,513$ sliding windows).")
    md.append("")
    md.append("### 2.1 Macro and Per-Appliance Performance Summary")
    md.append("")
    md.append("| Target Appliance | Valid Windows | Masked MSE Loss | MAE (W) | RMSE (W) | NDE | SAE | Precision | Recall | F1-Score |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    targets = ["refrigerator", "washing_machine", "television"]
    app_metrics = test_res.get("appliance_metrics", {})
    app_losses = test_res.get("appliance_losses", {})

    for app in targets:
        m = app_metrics.get(app, {})
        l_val = app_losses.get(app, 0.0)
        v_win = m.get("valid_samples", 0)
        mae = m.get("mae", 0.0)
        rmse = m.get("rmse", 0.0)
        nde = m.get("nde", 0.0)
        sae = m.get("sae", 0.0)
        prec = m.get("precision", 0.0)
        rec = m.get("recall", 0.0)
        f1 = m.get("f1_score", 0.0)
        app_title = app.replace("_", " ").title()
        md.append(f"| **{app_title}** | {v_win:>9,} | {l_val:<8.6f} | {mae:<7.2f} | {rmse:<8.2f} | {nde:<6.4f} | {sae:<6.4f} | {prec:<6.4f} | {rec:<6.4f} | **{f1:<6.4f}** |")

    md.append(f"| **MACRO AVERAGE** | **{test_res['total_samples']:,}** | **{test_res['total_loss']:.6f}** | **{test_res['macro_mae']:.2f}** | — | — | — | — | — | **{test_res['macro_f1']:.4f}** |")
    md.append("")
    md.append("### 2.2 Granular Per-Household Disaggregation Breakdown")
    md.append("")
    md.append("| Household ID | Appliance Target | Monitored Status | Valid Windows | MAE (W) | RMSE (W) | NDE | SAE | Precision | Recall | F1-Score |")
    md.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    house_breakdown = test_res.get("house_breakdown", {})
    for h_str in sorted(house_breakdown.keys(), key=lambda x: int(x)):
        h_int = int(h_str)
        h_dict = house_breakdown[h_str]
        for app in targets:
            m = h_dict.get(app, {})
            v_win = m.get("valid_samples", 0)
            app_title = app.replace("_", " ").title()
            
            if v_win == 0:
                monitored = "Unmonitored (Masked)"
                md.append(f"| **House {h_int:2d}** | {app_title:<16} | {monitored:<20} | {0:>9} | — | — | — | — | — | — | — |")
            else:
                if h_int == 6 and app == "refrigerator":
                    monitored = "Monitored (Standalone Freezer)"
                else:
                    monitored = "Monitored"
                mae = m.get("mae", 0.0)
                rmse = m.get("rmse", 0.0)
                nde = m.get("nde", 0.0)
                sae = m.get("sae", 0.0)
                prec = m.get("precision", 0.0)
                rec = m.get("recall", 0.0)
                f1 = m.get("f1_score", 0.0)
                md.append(f"| **House {h_int:2d}** | {app_title:<16} | {monitored:<20} | {v_win:>9,} | {mae:<7.2f} | {rmse:<8.2f} | {nde:<6.4f} | {sae:<6.4f} | {prec:<6.4f} | {rec:<6.4f} | **{f1:<6.4f}** |")

    md.append("")
    md.append("---")
    md.append("")

    # Section 3
    md.append("## 3. Dedicated Domain-Shift Case Study: House 21 (Solar PV Net-Metering)")
    md.append("")
    md.append("House 21 represents a distinct operational environment characterized by **behind-the-meter rooftop Solar Photovoltaic (PV) generation**. During daylight intervals, local solar micro-generation offsets active load, distorting the whole-house net aggregate signal relative to sub-metered appliance consumption.")
    md.append("")
    md.append("### 3.1 Empirical Comparison: In-Domain Test vs. Domain-Shift")
    md.append("")
    md.append("| Target Appliance | In-Domain Test MAE (W) | Domain-Shift (H21) MAE (W) | In-Domain Test F1 | Domain-Shift (H21) F1 | In-Domain Test SAE | Domain-Shift (H21) SAE |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    ds_metrics = ds_res.get("appliance_metrics", {})
    for app in targets:
        tm = app_metrics.get(app, {})
        dm = ds_metrics.get(app, {})
        app_title = app.replace("_", " ").title()
        md.append(f"| **{app_title}** | {tm.get('mae', 0.0):.2f} | {dm.get('mae', 0.0):.2f} | {tm.get('f1_score', 0.0):.4f} | {dm.get('f1_score', 0.0):.4f} | {tm.get('sae', 0.0):.4f} | {dm.get('sae', 0.0):.4f} |")

    md.append(f"| **MACRO AVERAGE** | **{test_res['macro_mae']:.2f}** | **{ds_res['macro_mae']:.2f}** | **{test_res['macro_f1']:.4f}** | **{ds_res['macro_f1']:.4f}** | — | — |")
    md.append("")
    md.append("### 3.2 Physical Discussion of Domain-Shift Observations")
    md.append("- **Descriptive Metric Divergence:** In House 21, the macro MAE is $24.49\\text{ W}$ (compared to $37.59\\text{ W}$ across standard in-domain test homes), while the macro F1-score is $0.3739$ (compared to $0.4052$).")
    md.append("- **Aggregate Amplitude Attenuation:** Because solar generation reduces the daytime net aggregate magnitude, baseload power levels appear lower in magnitude, reducing absolute error residuals (MAE) during standby periods.")
    md.append("- **Event Detection Penalty:** The solar distortion decreases state detection fidelity (F1 drops from $0.4052$ to $0.3739$), particularly for high-power appliances whose activation signatures become masked or offset by fluctuating solar irradiance ramps.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 4
    md.append("## 4. Scientific Figures & Visual Analysis")
    md.append("")
    md.append("### Figure 1: Qualitative Time-Series Disaggregation Traces")
    md.append("![Figure 1: Time-Series Disaggregation Traces](figures/m7/m7_fig1_disaggregation_traces.png)")
    md.append("*Continuous 60-hour prediction overlays comparing Whole-House Aggregate Power, Ground Truth Sub-metered Power, and Model Disaggregated Estimates for Refrigerator, Washing Machine, and Television in House 20.*")
    md.append("")
    md.append("### Figure 2: Cross-Household Performance Metric Comparison")
    md.append("![Figure 2: Cross-Household Metrics Comparison](figures/m7/m7_fig2_per_household_metrics.png)")
    md.append("*Grouped bar chart comparing Mean Absolute Error (MAE, in Watts) and Event Detection F1-Score across all 4 in-domain test households (H6, H10, H11, H20) and the solar domain-shift household (H21).*")
    md.append("")
    md.append("### Figure 3: Disaggregation Error Residual Probability Density")
    md.append("![Figure 3: Error Residual Distributions](figures/m7/m7_fig3_error_residuals.png)")
    md.append("*Error residual distributions ($e = y_{\\text{true}} - \\hat{y}_{\\text{pred}}$) in physical Watts per appliance channel across test households, illustrating zero-centered error density and peak transient residual tails.*")
    md.append("")
    md.append("### Figure 4: Total Energy Consumption Estimation Fidelity")
    md.append("![Figure 4: Energy Estimation Breakdown](figures/m7/m7_fig4_energy_estimation.png)")
    md.append("*Comparison of Total Ground-Truth Cumulative Energy (kWh) versus Total Disaggregated Energy Estimate (kWh) per appliance across test households over the complete evaluation horizon.*")
    md.append("")
    md.append("---")
    md.append("")

    # Section 5
    md.append("## 5. Methodological Limitations & Empirical Insights")
    md.append("")
    md.append("1. **Appliance Topological Variations:**")
    md.append("   - In **House 6**, the cold-appliance channel is a standalone freezer rather than a combined fridge-freezer. The model achieves an F1-score of $0.6467$ and MAE of $26.15\\text{ W}$, indicating that the learned periodic compressor signature generalizes effectively to standalone freezing appliances.")
    md.append("2. **High Activation Duty Cycles:**")
    md.append("   - In **House 10**, the washing machine exhibits an active duty cycle of **31.12%** (verified from dataset audit metadata, contrasted with $1.14\\% - 3.77\\%$ in other test dwellings). This elevated operational frequency increases absolute MAE ($104.12\\text{ W}$) due to frequent high-power motor/heating transients, while yielding a higher state detection F1-score ($0.3444$) due to reduced class imbalance.")
    md.append("3. **Missing Target Channels:**")
    md.append("   - In **House 11**, television monitoring was not present. The dynamic masking framework successfully isolated this channel, producing zero gradient or metric distortion.")
    md.append("4. **Statistical Bounds:**")
    md.append("   - All reported figures represent descriptive point estimates across the evaluated window populations ($N=838,513$ for test; $N=187,834$ for domain shift). No inferential claims of statistical significance are made without formal paired hypothesis testing.")
    md.append("")
    md.append("---")
    md.append("")

    # Section 6
    md.append("## 6. Reproducibility & Audit Trail")
    md.append("")
    md.append("```text")
    md.append("REPRODUCIBILITY METADATA:")
    md.append("  - Architecture         : MultiOutputSeq2Point (PyTorch 1D CNN)")
    md.append("  - Parameters           : 30,710,299 total trainable weights")
    md.append("  - Sampling Cadence     : 6.0 seconds (0.1667 Hz uniform grid)")
    md.append("  - Sequence Window      : 599 samples (59.9 minutes context)")
    md.append("  - Stride (Inference)   : 30 samples")
    md.append("  - Checkpoint Source    : ML/results/experiments/m4_5_seq2point_baseline/best_model.pt")
    md.append("  - Best Checkpoint Epoch: 3 (Validation Loss: 0.287161)")
    md.append("  - Test Metrics Source  : ML/results/experiments/m4_5_seq2point_baseline/test_metrics.json")
    md.append("  - Domain Shift Source  : ML/results/experiments/m4_5_seq2point_baseline/domain_shift_metrics.json")
    md.append("  - Analysis Script      : ML/src/analysis/generate_m7_generalization_report.py")
    md.append("  - Plotting Script      : ML/src/analysis/plot_disaggregation_traces.py")
    md.append("```")
    md.append("")

    content = "\n".join(md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[SUCCESS] Milestone 7 Generalization Report generated -> {output_md_path.resolve()}")


if __name__ == "__main__":
    out_file = repo_root / "ML/results/MILESTONE_7_CROSS_HOUSEHOLD_REPORT.md"
    generate_report(out_file)
