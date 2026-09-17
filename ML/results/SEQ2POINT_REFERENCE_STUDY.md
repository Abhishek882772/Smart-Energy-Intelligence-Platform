# Seq2Point Reference Architecture & Methodology Study
**Research Project:** Cross-Country Generalization of NILM Models for Indian Residential Energy Consumption  
**Milestone:** Milestone 4.2 — Seq2Point Reference Architecture Study  
**Date:** 2026-09-02  
**Baseline Focus:** Unified Multi-Output Sequence-to-Point (Seq2Point) 1D Convolutional Neural Network  

---

## 1. Executive Summary & Objective

Before implementing our first NILM neural network baseline in Milestone 4.3, this study examines the canonical **Sequence-to-Point (Seq2Point)** methodology published by Zhang et al. (AAAI 2018). We compare the reference methodology against our materialized REFIT dataset pipeline (`ML/data/processed/REFIT/`), specify the unified multi-output adaptation, formulate masked multi-task loss functions, and define standard NILM evaluation metrics.

---

## 2. Current Materialized Data Pipeline Characteristics

From Milestone 4.1 materialization, our disk-backed dataset provides the following exact tensor geometry:

* **Input Tensor ($X$):** Shape `(N, 599, 1)` representing a 1-hour sequence of aggregate active power measurements.
* **Target Tensor ($y$):** Shape `(N, 3)` corresponding to midpoint ground truth power for `['refrigerator', 'washing_machine', 'television']`.
* **Validity Mask Tensor ($M$):** Shape `(N, 3)` boolean array indicating sensor health and monitoring status.
* **Temporal Grid ($\Delta t$):** $6.0\text{ seconds}$ ($0.1667\text{ Hz}$), uniform across all households.
* **Window Length ($W$):** $599\text{ samples}$ ($599 \times 6\text{s} = 3594\text{s} \approx 59.9\text{ minutes}$).
* **Stride ($S$):** $30\text{ samples}$ ($3\text{ minutes}$) for training, validation, and test partitions.
* **Target Timestamp Position:** Exact window midpoint: $\tau = t + \lfloor W / 2 \rfloor = t + 299$.
* **Normalization:** Standardization ($z$-score) computed strictly across 36.4M valid samples from training households `[2, 3, 5, 7, 9]`.
* **Household Partitions:**
  - **Train (5 Houses):** `[2, 3, 5, 7, 9]` — $1,144,163\text{ windows}$ ($64.6\text{ MB}$)
  - **Validation (3 Houses):** `[1, 8, 15]` — $722,272\text{ windows}$ ($36.8\text{ MB}$)
  - **Test (4 Houses):** `[6, 10, 11, 20]` — $838,513\text{ windows}$ ($47.0\text{ MB}$)
  - **Domain Shift (1 House):** `[21]` — $187,834\text{ windows}$ ($10.6\text{ MB}$)
  - **Total Materialized:** **$2,892,782\text{ sliding windows}$** ($159.0\text{ MB}$ compressed NPZ).

---

## 3. Original Seq2Point Methodology Study

### 3.1 Literature Citation & Reference Implementations
* **Primary Conference Publication:**  
  Chaoyun Zhang, Mingjun Zhong, Zhaolong Wang, Nigel Goddard, Charles Sutton. *"Sequence-to-Point Learning with Neural Networks for Nonintrusive Load Monitoring"*. In **Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)**, New Orleans, USA, pp. 2606–2612, 2018.  
  URL: [arXiv:1612.09106](https://arxiv.org/abs/1612.09106)
* **Journal Expansion:**  
  Chaoyun Zhang, Nigel Goddard, Charles Sutton. *"Sequence-to-point learning using convolutional neural networks for nonintrusive load monitoring"*. **Neural Computing and Applications**, 33:1463–1477, 2021.  
  DOI: `10.1007/s00521-020-05047-w`
* **Official Reference Repositories:**  
  - Mingjun Zhong / Chaoyun Zhang: [https://github.com/MingjunZhong/Seq2Point-NILM](https://github.com/MingjunZhong/Seq2Point-NILM)
  - TransferNILM Suite: [https://github.com/MingjunZhong/transferNILM](https://github.com/MingjunZhong/transferNILM)

### 3.2 Mathematical Formulation
Traditional Sequence-to-Sequence (Seq2Seq) models map an input aggregate window $X_{t:t+W-1} \in \mathbb{R}^W$ to an appliance sequence $Y_{t:t+W-1} \in \mathbb{R}^W$. However, Seq2Seq models suffer from severe boundary fading and prediction smearing due to averaging multiple overlapping window outputs.

Seq2Point replaces sequence output with a non-linear regression function $f_\theta: \mathbb{R}^W \to \mathbb{R}$ that maps the aggregate surrounding window directly to the **midpoint power reading** $y_\tau$:
$$\hat{y}_\tau = f_\theta(X_{t:t+W-1}), \quad \text{where } \tau = t + \lfloor W / 2 \rfloor$$

The receptive field of the deep 1D CNN treats the preceding $\lfloor W/2 \rfloor$ steps as past context and succeeding $\lfloor W/2 \rfloor$ steps as future context, allowing the network to recognize the operational state signature around the central point.

### 3.3 Original Neural Network Architecture Specification
The canonical Seq2Point architecture uses a 5-layer 1D Convolutional feature extractor followed by two fully connected layers:

```text
Input: Aggregate Sequence (Batch_Size, 599, 1)
  │
  ├── Conv1D (Filters: 30, Kernel Size: 10, Stride: 1, Padding: 'same') + ReLU
  ├── Conv1D (Filters: 30, Kernel Size: 8,  Stride: 1, Padding: 'same') + ReLU
  ├── Conv1D (Filters: 40, Kernel Size: 6,  Stride: 1, Padding: 'same') + ReLU
  ├── Conv1D (Filters: 50, Kernel Size: 5,  Stride: 1, Padding: 'same') + ReLU
  ├── Conv1D (Filters: 50, Kernel Size: 5,  Stride: 1, Padding: 'same') + ReLU
  │
  ├── Flatten (50 * 599 = 29,950 units)
  ├── Dense (Units: 1024) + ReLU + Dropout(p = 0.2)
  │
  └── Dense Output (Units: 1) + Linear Activation -> \hat{y}
```

* **Loss Function:** Mean Squared Error (MSE): $\mathcal{L} = \frac{1}{B}\sum_{i=1}^B (\hat{y}_i - y_i)^2$.
* **Training Strategy:** In original Seq2Point, **separate isolated models** are trained per appliance (one model for Fridge, another for Washing Machine).

---

## 4. Architectural Comparison: Original Seq2Point vs Our Pipeline

| Component | Original Seq2Point (Zhang et al. 2018) | Our Research Implementation | Compatibility | Action / Rationale |
| :--- | :--- | :--- | :---: | :--- |
| **Input Window Length ($W$)** | 599 samples (Washing Machine, Microwave) or 99 samples (Fridge) | **599 samples** (~59.9 min at 6s) unified across all appliances | ✅ Exact Match | Reproduce $W=599$; provides sufficient context for slow thermostatic cycles and multi-phase wash cycles. |
| **Sampling Cadence ($\Delta t$)** | 6.0 seconds (UK-DALE) / 8.0 seconds (REFIT) | **6.0 seconds** ($0.1667\text{ Hz}$) uniform grid | ✅ Compatible | Uniform 6s grid aligns REFIT with UK-DALE and future iAWE Indian evaluation. |
| **Target Formulation** | Central midpoint $y(t + W//2)$ | Central midpoint $y(t + 299)$ | ✅ Exact Match | Reproduce exact midpoint regression. |
| **Normalization** | Per-appliance Standardization ($z$-score) or Min-Max | **Zero-Leakage Training Split Standardization** ($\mu, \sigma$ from H2, H3, H5, H7, H9) | ✅ Compatible | Normalization parameters computed strictly on training split; applied to Val/Test/Domain-Shift. |
| **Model Structure** | Single-Appliance Model (1 model = 1 appliance) | **Unified Multi-Task Model** (Shared 1D CNN + 3 Appliance Heads) | 🔄 Adapted | Single forward-pass disaggregation; shared representation enhances baseload filtering. |
| **Sensor Validity Masking** | None (dropped corrupted rows or imputed blindly) | **Masked Multi-Task MSE Loss** | 🔄 Adapted | Prevents unmonitored (e.g. H11 TV) or sensor-glitched samples from polluting gradients. |
| **Loss Function** | Standard MSE: $\frac{1}{B}\sum(\hat{y}_i - y_i)^2$ | **Masked Multi-Task MSE**: $\sum_k \frac{\sum_i M_{ik}(\hat{y}_{ik} - y_{ik})^2}{\sum_i M_{ik} + \epsilon}$ | 🔄 Adapted | Multi-task loss weighted by validity mask tensor. |
| **Household Partitioning** | Random time-slice or 1 held-out house | **Strict House-Level Disjoint Split** (5 Train, 3 Val, 4 Test, 1 Domain-Shift) | ✅ Compatible | Prevents temporal data leakage and benchmark overfitting. |

---

## 5. Proposed Unified Multi-Output Architecture Specification

To build an efficient, scalable baseline suitable for cross-country transfer, we adapt Seq2Point into a **Unified Multi-Output Architecture**:

```text
                               Input Aggregate Window
                                   (B, 599, 1)
                                        │
                         ┌──────────────┴──────────────┐
                         │   Shared 1D CNN Backbone    │
                         │  Layer 1: Conv1D(30, k=10)  │
                         │  Layer 2: Conv1D(30, k=8)   │
                         │  Layer 3: Conv1D(40, k=6)   │
                         │  Layer 4: Conv1D(50, k=5)   │
                         │  Layer 5: Conv1D(50, k=5)   │
                         │  Flatten -> (B, 29950)      │
                         │  Dense(1024, ReLU, p=0.2)   │
                         └──────────────┬──────────────┘
                                        │
                         Shared Feature Representation (B, 1024)
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
      ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
      │ Refrigerator Head│    │  Washing Machine │    │ Television Head  │
      │  Linear(1024->1) │    │  Linear(1024->1) │    │  Linear(1024->1) │
      └─────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
                ▼                       ▼                       ▼
          \hat{y}_fridge           \hat{y}_wm              \hat{y}_tv
```

### 5.1 Masked Multi-Task Objective Function
Let $B$ be the batch size, $\hat{y}_{i,k}$ be the prediction for sample $i$ and appliance $k \in \{\text{refrigerator}, \text{washing\_machine}, \text{television}\}$, $y_{i,k}$ be the normalized ground truth, and $M_{i,k} \in \{0, 1\}$ be the boolean validity mask.

$$\mathcal{L}_{\text{total}}(\theta) = \sum_{k=1}^3 \lambda_k \cdot \frac{\sum_{i=1}^B M_{i,k} \cdot (\hat{y}_{i,k} - y_{i,k})^2}{\sum_{i=1}^B M_{i,k} + \epsilon}$$

Where:
* $\lambda_k = 1.0$ (equal multi-task weighting).
* $\epsilon = 10^{-8}$ prevents division by zero if all samples in a batch are invalid for an unmonitored appliance (e.g. House 11 TV).
* Gradients only backpropagate through valid targets ($M_{i,k} = 1$).

---

## 6. Standard NILM Evaluation Metrics Formulation

To rigorously assess disaggregation performance, we define two classes of metrics:

### 6.1 Continuous Power Estimation Metrics
Calculated on physical power values (Watts) after inverse transformation: $y_t = z_t \cdot \sigma + \mu$.

1. **Mean Absolute Error (MAE) [Watts]:**
   $$\text{MAE} = \frac{1}{N}\sum_{t=1}^N |\hat{y}_t - y_t|$$
   *Measures the average magnitude of absolute error at each 6-second interval.*

2. **Root Mean Squared Error (RMSE) [Watts]:**
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^N (\hat{y}_t - y_t)^2}$$
   *Penalizes large transient estimation spikes (e.g. missed motor activations).*

3. **Normalized Disaggregation Error (NDE) [Dimensionless]:**
   $$\text{NDE} = \frac{\sum_{t=1}^N (\hat{y}_t - y_t)^2}{\sum_{t=1}^N y_t^2}$$
   *Scale-invariant normalized metric standard in NILM benchmarking.*

4. **Signal Aggregate Error (SAE) [Dimensionless]:**
   $$\text{SAE} = \frac{|\hat{E} - E|}{E} = \frac{\left|\sum_{t=1}^N \hat{y}_t - \sum_{t=1}^N y_t\right|}{\sum_{t=1}^N y_t}$$
   *Measures total energy discrepancy over an extended monitoring window (vital for billing and consumer energy feedback).*

### 6.2 State Classification & Event Detection Metrics
Appliance operational state is binarized using empirical ON-power thresholds $P_{\text{on}}$:
$$s_t = \mathbb{I}(y_t \ge P_{\text{on}}), \quad \hat{s}_t = \mathbb{I}(\hat{y}_t \ge P_{\text{on}})$$

* **Thresholds ($P_{\text{on}}$):** Refrigerator: $15\text{W}$ | Washing Machine: $20\text{W}$ | Television: $10\text{W}$.
* Contingency table: True Positive ($TP$), False Positive ($FP$), False Negative ($FN$).

1. **Precision:** $\text{Precision} = \frac{TP}{TP + FP}$
2. **Recall:** $\text{Recall} = \frac{TP}{TP + FN}$
3. **F1-Score:** $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 7. Baseline Philosophy & Architectural Scope

The first neural model in Milestone 4.3 must serve as a **clean, transparent, and canonical baseline**:
* **Included:** 5-layer 1D CNN feature extractor, 1024-unit dense layer with dropout, 3 linear task heads, Adam optimizer, masked MSE loss.
* **Explicitly Excluded from Baseline:**
  - Self-attention / Transformer blocks (reserved for Milestone 6)
  - Temporal Convolutional Networks / Dilated causal convolutions (reserved for Milestone 5)
  - Complex residual skip connections
  - Domain adaptation / Adversarial alignment layers (reserved for Milestone 12+)
  - Graph Neural Networks

*Purpose:* Establish the pure in-domain UK REFIT baseline performance ($\text{MAE}_{\text{REFIT}}$, $\text{F1}_{\text{REFIT}}$) to serve as the reference benchmark against which all future architectures (TCN, Transformer) and cross-country Indian transfer methods (iAWE adaptation) will be measured for Paper 1.

---

## 8. Open Questions & Implementation Decisions

1. **Dense Layer Parameter Count:**  
   $50\text{ filters} \times 599\text{ length} = 29,950\text{ flattened units}$. The first Dense layer ($29,950 \times 1024$) contains $\approx 30.67\text{ million parameters}$.  
   *Decision:* For CPU training feasibility, we will evaluate whether a stride-2 pooling or a lightweight global receptive field (e.g. MaxPool1D or strided convolution) is necessary to reduce parameter count from 30M to ~2M without compromising receptive field. This will be parameterized cleanly in `ML/configs/model_config.yaml`.
2. **PyTorch Dataset Architecture:**  
   Materialized NPZ shards can be indexed via a custom PyTorch `REFITDataset` that streams samples by house index without holding all shards in RAM simultaneously.

---

## 9. References

1. Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018). *Sequence-to-point learning with neural networks for nonintrusive load monitoring*. In **Proceedings of the AAAI Conference on Artificial Intelligence** (Vol. 32, No. 1). [https://arxiv.org/abs/1612.09106](https://arxiv.org/abs/1612.09106)
2. Zhang, C., Goddard, N., & Sutton, C. (2021). *Sequence-to-point learning using convolutional neural networks for nonintrusive load monitoring*. **Neural Computing and Applications**, 33, 1463–1477. [https://doi.org/10.1007/s00521-020-05047-w](https://doi.org/10.1007/s00521-020-05047-w)
3. Kelly, J., & Knottenbelt, W. (2015). *Neural NILM: Deep neural networks applied to energy disaggregation*. In **Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments** (pp. 55-64). [https://arxiv.org/abs/1507.06594](https://arxiv.org/abs/1507.06594)
4. Murray, D., Stankovic, L., & Stankovic, V. (2017). *An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study*. **Scientific Data**, 4(1), 1-12. [https://doi.org/10.1038/sdata.2017.122](https://doi.org/10.1038/sdata.2017.122)
5. Official Seq2Point GitHub Implementation: [https://github.com/MingjunZhong/Seq2Point-NILM](https://github.com/MingjunZhong/Seq2Point-NILM)

---

## 10. Final Recommendation

```text
MILESTONE_4_2_STATUS: READY_FOR_M4_3
```
*(The Seq2Point reference architecture, mathematical formulations, multi-output adaptation, and evaluation metrics are fully documented. We are ready to proceed to baseline PyTorch model development in Milestone 4.3.)*
