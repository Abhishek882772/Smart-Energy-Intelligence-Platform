# Milestone 4.3 — Unified Multi-Output Seq2Point Model Report

**Project:** Cross-Country Generalization of NILM Models for Indian Residential Energy Consumption  
**Milestone:** Milestone 4.3 — Unified Multi-Output Seq2Point Model Implementation & Sanity Verification  
**Date:** 2026-09-02  
**Framework:** PyTorch (CPU-compatible)  
**Model Name:** `MultiOutputSeq2Point`  

---

> [!NOTE]
> **Methodological Provenance & Baseline Declaration:**  
> This is an implementation of an established Sequence-to-Point (Seq2Point) NILM baseline (Zhang et al., AAAI 2018) adapted to a unified three-appliance multi-output formulation with masked multi-task regression loss. It is not being claimed as a novel architecture; rather, it serves as the foundational canonical deep learning baseline against which subsequent TCN, Transformer, and Indian domain-adaptation architectures will be benchmarked.

---

## 1. Executive Summary & Objective

In **Milestone 4.3**, we implemented a clean, modular PyTorch version of the canonical Seq2Point 1D Convolutional Neural Network, adapted into a unified multi-output architecture capable of simultaneous energy disaggregation for **Refrigerator, Washing Machine, and Television**.

All model code, loss functions, dataset loaders, configuration files, and unit/integration tests were created under [`ML/src/models/`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/models/), [`ML/src/losses/`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/losses/), [`ML/configs/`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/configs/), and [`ML/tests/`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/tests/).

All unit tests and end-to-end integration tests using real materialized REFIT training data (`house_2.npz`) passed.

---

## 2. Model Architecture & Multi-Output Adaptation

```text
                           Input Aggregate Sequence
                              X: (Batch, 599, 1)
                                      │
                         [Transposition: (B, 1, 599)]
                                      │
                         ┌────────────┴────────────┐
                         │ Shared 1D CNN Backbone  │
                         │ Layer 1: Conv1D(1->30)  │
                         │ Layer 2: Conv1D(30->30) │
                         │ Layer 3: Conv1D(30->40) │
                         │ Layer 4: Conv1D(40->50) │
                         │ Layer 5: Conv1D(50->50) │
                         │ Flatten -> (B, 29,950)  │
                         │ Dense(1024) + Dropout0.2│
                         └────────────┬────────────┘
                                      │
                         Shared Representation: (B, 1024)
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
      ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
      │ Refrigerator Head│  │  Washing Machine │  │ Television Head  │
      │ Linear(1024->1)  │  │ Linear(1024->1)  │  │ Linear(1024->1)  │
      └─────────┬────────┘  └─────────┬────────┘  └─────────┬────────┘
                ▼                     ▼                     ▼
         \hat{y}_fridge          \hat{y}_wm            \hat{y}_tv
                └─────────────────────┼─────────────────────┘
                                      ▼
                            Concatenated Output:
                             \hat{y}: (Batch, 3)
```

### 2.1 Key Design Decisions
1. **Explicit Dimension Transposition:** The materialized dataset stores inputs as `(B, 599, 1)` ($B \times \text{Length} \times \text{Channels}$). Inside `MultiOutputSeq2Point.forward()`, the tensor is transposed to `(B, 1, 599)` ($B \times \text{Channels} \times \text{Length}$) to conform with PyTorch `nn.Conv1d` conventions.
2. **Shared Feature Representation:** Rather than running 3 distinct networks (3x computational cost), a shared 5-layer 1D CNN backbone learns universal baseload and transient temporal features.
3. **Decoupled Task Heads:** 3 separate linear projections (`Linear(1024, 1)`) predict the midpoint power for Refrigerator, Washing Machine, and Television independently.
4. **Unconstrained Linear Output:** No non-linear activation is applied at the output heads, ensuring predictions remain unconstrained in standardized ($z$-score) space.
5. **Fixed Output Ordering:** Index 0 = Refrigerator, Index 1 = Washing Machine, Index 2 = Television.

---

## 3. Trainable Parameter Count Breakdown

| Layer / Component | Specification | Weights Formula | Bias | Trainable Parameters |
| :--- | :--- | :--- | :---: | :---: |
| **`conv1`** | `Conv1d(1 -> 30, k=10, pad='same')` | $30 \times 1 \times 10 = 300$ | $30$ | **330** |
| **`conv2`** | `Conv1d(30 -> 30, k=8, pad='same')` | $30 \times 30 \times 8 = 7,200$ | $30$ | **7,230** |
| **`conv3`** | `Conv1d(30 -> 40, k=6, pad='same')` | $40 \times 30 \times 6 = 7,200$ | $40$ | **7,240** |
| **`conv4`** | `Conv1d(40 -> 50, k=5, pad='same')` | $50 \times 40 \times 5 = 10,000$ | $50$ | **10,050** |
| **`conv5`** | `Conv1d(50 -> 50, k=5, pad='same')` | $50 \times 50 \times 5 = 12,500$ | $50$ | **12,550** |
| **`fc_shared`** | `Linear(29,950 -> 1024)` | $1024 \times 29,950 = 30,668,800$ | $1,024$ | **30,669,824** |
| **`head_refrigerator`** | `Linear(1024 -> 1)` | $1 \times 1024 = 1024$ | $1$ | **1,025** |
| **`head_washing_machine`**| `Linear(1024 -> 1)` | $1 \times 1024 = 1024$ | $1$ | **1,025** |
| **`head_television`** | `Linear(1024 -> 1)` | $1 \times 1024 = 1024$ | $1$ | **1,025** |
| **TOTAL** | **MultiOutputSeq2Point** | — | — | **`30,710,299`** |

*(Note: The vast majority of weights—99.87%—reside in the first dense layer `fc_shared` connecting the flattened 29,950-dimensional feature map to the 1024-dimensional shared embedding, matching the original Seq2Point publication).*

---

## 4. Masked Multi-Task Loss Formulation

Implemented in [`ML/src/losses/masked_multitask_loss.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/losses/masked_multitask_loss.py):

For each appliance channel $k \in \{0, 1, 2\}$:
$$S_k = \sum_{i=1}^B M_{i,k}$$
$$L_k = \begin{cases} \frac{\sum_{i=1}^B M_{i,k} \cdot (\hat{y}_{i,k} - y_{i,k})^2}{S_k + \epsilon}, & \text{if } S_k > 0 \\ 0.0, & \text{if } S_k = 0 \end{cases}$$

$$\mathcal{L}_{\text{total}} = \frac{\sum_{k: S_k > 0} L_k}{\max\left(1, \sum_{k=1}^3 \mathbb{I}(S_k > 0)\right)}$$

### Key Safety Guarantees:
* **Zero Division by Zero:** When an appliance is unmonitored across an entire batch (e.g. House 11 Television where $S_{\text{tv}} = 0$), $L_{\text{tv}}$ is set to $0.0$ and excluded from the active appliance denominator.
* **Zero Gradient Pollution:** The backward gradient for unmonitored target heads is strictly zero: $\frac{\partial \mathcal{L}}{\partial \hat{y}_{i,\text{tv}}} = 0$.

---

## 5. Automated Validation & Test Suite Results

### 5.1 Unit Tests (`ML/tests/test_seq2point_model.py`)
Executed with `.venv\Scripts\python.exe`:

| Test | Description | Result | Verification Notes |
| :---: | :--- | :---: | :--- |
| **1** | Parameter Count Verification | ✅ PASS | Verified layer-by-layer formulas totaling exactly 30,710,299 weights. |
| **2** | Forward Pass & Gradients | ✅ PASS | Shape `(4, 599, 1) -> (4, 3)`, float32, no NaNs/Infs, valid backward flow. |
| **3** | Loss Case A (All True) | ✅ PASS | Normal multi-task loss calculation. |
| **4** | Loss Case B (Partial False) | ✅ PASS | Correctly excludes invalid entries from MSE average. |
| **5** | Loss Case C (Unmonitored TV) | ✅ PASS | Loss finite, TV component = 0.0, TV gradients = 0.0. |
| **6** | Loss Case D (All False) | ✅ PASS | Loss returns 0.0 without division by zero. |

### 5.2 Real Materialized Dataset Integration Test (`ML/tests/test_dataset_integration.py`)
Executed using real REFIT training shard (`ML/data/processed/REFIT/train/house_2.npz`):
* **Batch Loaded:** $X \in \mathbb{R}^{32 \times 599 \times 1}$, $y \in \mathbb{R}^{32 \times 3}$, $M \in \{0, 1\}^{32 \times 3}$, $h = [2, \dots, 2]$.
* **Forward Pass Output:** $\hat{y} \in \mathbb{R}^{32 \times 3}$ (no NaNs, no Infs).
* **Initial Forward Loss:** Total Loss = `0.2030` (Fridge: `0.5725`, Washer: `0.0145`, TV: `0.0220`).
* **Backward Pass & Optimizer Step:** Successful gradient backpropagation through all 30.71M parameters and clean optimizer weight update.

---

## 6. Files Created & Modified

* [`ML/configs/model_seq2point.yaml`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/configs/model_seq2point.yaml) — Centralized architecture hyperparameters.
* [`ML/src/models/seq2point.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/models/seq2point.py) — `MultiOutputSeq2Point` PyTorch neural network class.
* [`ML/src/models/dataset.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/models/dataset.py) — `REFITShardDataset` PyTorch dataset loader.
* [`ML/src/models/__init__.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/models/__init__.py) — Models package exports.
* [`ML/src/losses/masked_multitask_loss.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/losses/masked_multitask_loss.py) — `MaskedMultiTaskLoss` implementation.
* [`ML/src/losses/__init__.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/src/losses/__init__.py) — Losses package exports.
* [`ML/tests/test_seq2point_model.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/tests/test_seq2point_model.py) — Unit test suite.
* [`ML/tests/test_dataset_integration.py`](file:///C:/Users/Mahi/OneDrive/Documents/Smart-Energy-Intelligence-Platform/ML/tests/test_dataset_integration.py) — Real REFIT integration test suite.

---

## 7. Exact Validation Commands Executed

```powershell
# 1. Run model architecture and loss unit tests
.\.venv\Scripts\python.exe ML/tests/test_seq2point_model.py

# 2. Run real materialized REFIT integration test
.\.venv\Scripts\python.exe ML/tests/test_dataset_integration.py

# 3. Verify git hygiene
git status
```

---

## 8. Final Status & Next Steps

```text
MILESTONE_4_3_STATUS: READY_FOR_M4_4
```
*(The baseline Seq2Point model and loss functions are implemented, parameter-verified, unit-tested, and validated against real materialized REFIT data. Standing by for instructions to begin Milestone 4.4.)*
