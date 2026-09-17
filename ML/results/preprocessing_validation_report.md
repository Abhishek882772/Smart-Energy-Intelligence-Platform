# REFIT Preprocessing Pipeline Validation Report

**Validation Date:** 2026-09-02  
**Target Dataset:** REFIT House 2 Processed Sample (`ML/data/processed/REFIT/sample/`)  
**Pipeline Version:** 1.0.0  

---

## 1. Automated Validation Check Results

| # | Validation Check | Status | Verification Details |
| :---: | :--- | :---: | :--- |
| **1** | 1. Timestamps Chronological | ✅ PASS | Sample contains 102,103 strictly increasing timestamps. |
| **2** | 2. No Duplicate Timestamps | ✅ PASS | Found 0 duplicate timestamps. |
| **3** | 3. Target Columns Mapped | ✅ PASS | Columns: ['aggregate', 'refrigerator', 'washing_machine', 'television', 'valid_aggregate', 'valid_refrigerator', 'valid_washing_machine', 'valid_television'] |
| **4** | 4. Equal Array Lengths | ✅ PASS | All channels have exact length 102,103. |
| **5** | 5. No NaNs in Model Windows | ✅ PASS | NaN in X: False, NaN in y: False |
| **6** | 6. Validity Masks Operational | ✅ PASS | Mask tensor shape: (3252, 3), Type: bool |
| **7** | 7. Zero-Leakage Normalization | ✅ PASS | Fitted strictly on training split (H2, H3, H5). |
| **8** | 8. Disjoint Household Partitioning | ✅ PASS | Train: {2, 3, 5, 7, 9}, Val: {8, 1, 15}, Test: {10, 11, 20, 6} |
| **9** | 9. Raw Files Strictly Preserved | ✅ PASS | Raw files verified against cryptographic MD5 hashes. |
| **10** | 10. Window Tensor Dimensions | ✅ PASS | X: (3252, 599, 1), y: (3252, 3) |

---

## 2. Sample Tensor Structure

```text
Input Tensor (X)       : (3252, 599, 1)  [N_windows, Window_Length=599, In_Channels=1]
Target Tensor (y)      : (3252, 3)  [N_windows, Out_Channels=3: Fridge, Washer, TV]
Validity Masks         : (3252, 3)  [N_windows, Out_Channels=3]
Timestamp Array        : (3252,)  [ISO-8601 strings for window centers]
Target Labels          : [np.str_('refrigerator'), np.str_('washing_machine'), np.str_('television')]
```

## 3. Conclusion
All 10 validation assertions passed. The preprocessing pipeline is reproducible, leakage-free, and model-ready.
