# Smart Energy Intelligence Platform — NILM Seq2Point Demonstration

This directory contains a self-contained demonstration for presenting the Non-Intrusive Load Monitoring (NILM) multi-output **Sequence-to-Point (Seq2Point)** neural network.

---

## 1. Quickstart & Execution Command

Run the demonstration using the Python virtual environment:

```powershell
.\.venv\Scripts\python.exe ML/demo/run_seq2point_demo.py
```

### Optional Arguments:
* `--index <INT>`: Specify a custom sliding window index from House 2 (default: `4510`, an active morning window with concurrent refrigerator cooling and television activity).
* `--checkpoint <PATH>`: Specify model checkpoint path (default: `ML/results/experiments/m4_4_tiny_overfit/tiny_overfit_checkpoint.pt`).
* `--output_figure <PATH>`: Specify destination for presentation plot (default: `ML/demo/figures/demo_disaggregation_presentation.png`).

---

## 2. Demonstration Methodology & Pipeline Flow

```text
1. Load Materialized REFIT House 2 Dataset (ML/data/processed/REFIT/train/house_2.npz)
                           │
2. Extract 599-Sample Aggregate Window X (1-hour load profile @ 6s cadence)
                           │
3. Load Trained MultiOutputSeq2Point Checkpoint (30,710,299 parameters)
                           │
4. Model Inference under torch.no_grad() -> Predictions: (1, 3) in normalized z-score space
                           │
5. Inverse-Transform to Physical Watts using Training Split Normalization (mu, sigma)
                           │
6. Non-Negative Clamping: P_watts = max(0.0, P_raw)
                           │
7. Terminal Output Table + Publication-Grade Multi-Panel Figure Generation
```

---

## 3. Disaggregation Output Example (Window #4,510)

* **Household:** House 2 (UK REFIT Dataset)
* **Timestamp:** `2013-09-28 09:29:00`
* **Input Sequence:** $W=599\text{ samples}$ (~$59.9\text{ minutes}$)

| Appliance Channel | Ground Truth (Watts) | Model Predicted (Watts) | Absolute Error (Watts) |
| :--- | :---: | :---: | :---: |
| **Refrigerator** | **85.00 W** | **87.90 W** | $2.90\text{ W}$ |
| **Washing Machine** | **0.00 W** | **0.00 W** | $0.00\text{ W}$ |
| **Television** | **46.00 W** | **36.85 W** | $9.15\text{ W}$ |

---

## 4. Generated Presentation Artifacts

* **Figure Path:** `ML/demo/figures/demo_disaggregation_presentation.png`
* **Figure Structure:**
  - **Top Panel:** Complete 1-hour aggregate load profile context with target midpoint highlighted at $t=299$.
  - **Bottom Panels:** Ground Truth vs Model Prediction bar charts for Refrigerator, Washing Machine, and Television with annotated physical power readings.

---

> [!NOTE]
> **Methodological Provenance:**  
> This M4.4 checkpoint was trained on the tiny overfit sanity subset to verify architecture and pipeline learnability. This demonstration illustrates model functionality, data ingestion, and inference capabilities prior to full M4.5 multi-epoch training.
