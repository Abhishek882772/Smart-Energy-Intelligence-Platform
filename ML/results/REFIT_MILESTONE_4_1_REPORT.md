# REFIT Training Data Materialization Report

**Project:** Cross-Country Generalization of NILM Models for Indian Residential Energy Consumption  
**Milestone:** Milestone 4.1 — REFIT Training Data Materialization  
**Materialization Date:** 2026-09-02  
**Total Materialized Windows:** **`2,892,782`**  
**Total Materialized Storage Size:** **`159.00 MB`** (~0.16 GB compressed NPZ)  

---

## 1. Executive Summary & Objective

The objective of Milestone 4.1 was to materialize the audited and preprocessed REFIT dataset into disk-backed, sharded NumPy archives (`.npz`) structured for seamless PyTorch DataLoader ingestion during subsequent baseline model development (Milestone 4.2+).

A total of **13 households** were processed across **2,892,782 sliding window tensors** ($W=599$, $\Delta t = 6\text{s}$, $\text{Stride} = 30$). All raw REFIT CSV files remained strictly read-only.

---

## 2. Partition Architecture & Summary Statistics

| Partition | Households | Shard Count | Total Windows | Storage Size | Valid Fridge | Valid Washer | Valid TV |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | `[2, 3, 5, 7, 9]` | 5 | **1,144,163** | 64.6 MB | 1,138,133 | 1,138,133 | 1,138,133 |
| **VALIDATION** | `[1, 8, 15]` | 3 | **722,272** | 36.8 MB | 719,878 | 719,878 | 719,878 |
| **TEST** | `[6, 10, 11, 20]` | 4 | **838,513** | 47.0 MB | 835,133 | 835,133 | 672,874 |
| **DOMAIN_SHIFT** | `[21]` | 1 | **187,834** | 10.6 MB | 187,125 | 187,125 | 187,125 |
| **TOTAL** | **13 Houses** | **13 Shards** | **`2,892,782`** | **`159.0 MB`** | **`2,880,269`** | **`2,880,269`** | **`2,718,010`** |

---

## 3. Full Training Normalization Provenance

Normalization parameters were computed across all **36,426,081 valid samples** from the 5 training households (`[2, 3, 5, 7, 9]`):

| Channel | Physical Mean (W) | Physical Std (W) | Min (W) | Max (W) | Valid Samples (N) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **aggregate** | 573.27 W | 753.77 W | 0.0 W | 4000.0 W | 36,426,081 |
| **refrigerator** | 41.84 W | 55.32 W | 0.0 W | 1639.0 W | 36,426,081 |
| **washing_machine** | 26.23 W | 210.28 W | 0.0 W | 2567.0 W | 36,426,081 |
| **television** | 13.71 W | 102.73 W | 0.0 W | 2905.0 W | 36,426,081 |

*Validation, test, and domain-shift partitions were normalized using these exact training parameters without computing partition-specific statistics (strictly preventing data leakage).*

---

## 4. Shard Storage Schema & Geometry

Each household is stored as a standalone compressed archive under `ML/data/processed/REFIT/<partition>/house_<id>.npz` containing:
1. `X`: `float32` array of shape `(N, 599, 1)` representing the 1-hour normalized aggregate power context sequence.
2. `y`: `float32` array of shape `(N, 3)` representing normalized midpoint target power for `['refrigerator', 'washing_machine', 'television']`.
3. `valid_masks`: `bool` array of shape `(N, 3)` indicating whether the appliance was monitored and valid at $t_{\text{mid}}$.
4. `timestamps`: string array of shape `(N,)` recording the ISO-8601 timestamp at $t_{\text{mid}}$.
5. `household_id`: `int32` array of shape `(N,)` recording the integer household ID.
6. `target_names`: string array `['refrigerator', 'washing_machine', 'television']`.

---

## 5. Automated Validation & Zero-Leakage Checks

| # | Validation Check | Status | Verification Details |
| :---: | :--- | :---: | :--- |
| **1** | No NaNs in Input Tensors | ✅ PASS | Verified across all 2,892,782 windows. |
| **2** | No NaNs in Valid Targets | ✅ PASS | Verified across all valid appliance ground truth entries. |
| **3** | Tensor Shapes & Types | ✅ PASS | All shards conform strictly to `(N, 599, 1)` and `(N, 3)`. |
| **4** | Household ID Integrity | ✅ PASS | Every window retains its ground truth household ID without crossing. |
| **5** | Disjoint Partitioning | ✅ PASS | `Train ∩ Val = ∅`, `Train ∩ Test = ∅`, `Val ∩ Test = ∅`, `Domain-Shift = {H21}`. |
| **6** | Unmonitored Target Handling | ✅ PASS | House 11 Television has exactly 0 valid target windows (`valid_mask = False`). |
| **7** | Raw File Immutability | ✅ PASS | All 20 raw CSV files verified against Zenodo cryptographic MD5 hashes. |
| **8** | Sample Smoke-Test Distinction | ✅ PASS | Smoke test artifacts remain isolated in `sample/` directory. |

---

## 6. Generated Diagnostic Figures

1. `refit_mat_fig1_normalized_aggregate_distribution.png` — Normalized aggregate power probability density across partitions.
2. `refit_mat_fig2_target_power_distributions.png` — Active power distributions for Refrigerator, Washing Machine, and Television.
3. `refit_mat_fig3_example_window_all_targets.png` — Full example sliding window ($W=599$) with un-normalized physical power and midpoint ground truth.
4. `refit_mat_fig4_valid_mask_counts_by_household.png` — Valid vs invalid target counts per appliance across all 13 materialized homes.
5. `refit_mat_fig5_total_windows_by_household.png` — Total window counts by household categorized by partition.

---

## 7. Recommendation for Milestone 4.2

**READY FOR M4.2 (NILM Baseline Model Architecture & Training Pipeline).**

The REFIT dataset is materialized, normalized, validated, and saved in disk-backed NPZ shards with zero data leakage. The dataset is ready for PyTorch Dataset / DataLoader integration.
