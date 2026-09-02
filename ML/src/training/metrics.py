"""Standard NILM Evaluation Metrics Module (PyTorch and NumPy).

Implements mask-aware continuous disaggregation metrics (MAE, RMSE, NDE, SAE)
and event detection / state classification metrics (Precision, Recall, F1-Score).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch


def _to_numpy(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor or array to NumPy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_mae(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Mean Absolute Error (MAE) in physical Watts.

    MAE = sum(mask * |y_pred - y_true|) / (sum(mask) + eps)
    """
    p = _to_numpy(y_pred)
    t = _to_numpy(y_true)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.sum() == 0:
            return 0.0
        return float(np.sum(m * np.abs(p - t)) / (np.sum(m) + eps))
    return float(np.mean(np.abs(p - t)))


def compute_rmse(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Root Mean Squared Error (RMSE) in physical Watts.

    RMSE = sqrt( sum(mask * (y_pred - y_true)^2) / (sum(mask) + eps) )
    """
    p = _to_numpy(y_pred)
    t = _to_numpy(y_true)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.sum() == 0:
            return 0.0
        return float(np.sqrt(np.sum(m * (p - t) ** 2) / (np.sum(m) + eps)))
    return float(np.sqrt(np.mean((p - t) ** 2)))


def compute_nde(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Normalized Disaggregation Error (NDE, dimensionless).

    NDE = sum(mask * (y_pred - y_true)^2) / (sum(mask * y_true^2) + eps)
    """
    p = _to_numpy(y_pred)
    t = _to_numpy(y_true)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.sum() == 0:
            return 0.0
        num = np.sum(m * (p - t) ** 2)
        denom = np.sum(m * (t ** 2)) + eps
        return float(num / denom)
    num = np.sum((p - t) ** 2)
    denom = np.sum(t ** 2) + eps
    return float(num / denom)


def compute_sae(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Signal Aggregate Error (SAE / Relative Energy Discrepancy).

    SAE = |sum(mask * y_pred) - sum(mask * y_true)| / (sum(mask * y_true) + eps)
    """
    p = _to_numpy(y_pred)
    t = _to_numpy(y_true)
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.sum() == 0:
            return 0.0
        e_pred = np.sum(m * p)
        e_true = np.sum(m * t)
        return float(np.abs(e_pred - e_true) / (e_true + eps))
    e_pred = np.sum(p)
    e_true = np.sum(t)
    return float(np.abs(e_pred - e_true) / (e_true + eps))


def compute_classification_metrics(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    threshold_watts: float = 15.0,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compute state classification metrics (Precision, Recall, F1) at activation threshold.

    State binarization:
        s_true = (y_true >= threshold)
        s_pred = (y_pred >= threshold)
    """
    p = _to_numpy(y_pred)
    t = _to_numpy(y_true)
    
    s_pred = (p >= threshold_watts)
    s_true = (t >= threshold_watts)

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.sum() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}
        tp = np.sum(m & s_pred & s_true)
        fp = np.sum(m & s_pred & (~s_true))
        fn = np.sum(m & (~s_pred) & s_true)
        tn = np.sum(m & (~s_pred) & (~s_true))
        total = np.sum(m)
    else:
        tp = np.sum(s_pred & s_true)
        fp = np.sum(s_pred & (~s_true))
        fn = np.sum((~s_pred) & s_true)
        tn = np.sum((~s_pred) & (~s_true))
        total = len(p)

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2.0 * precision * recall / (precision + recall + eps))
    acc = float((tp + tn) / (total + eps))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
    }


def evaluate_appliance_metrics(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    threshold_watts: float = 15.0,
) -> Dict[str, float]:
    """Evaluate comprehensive metric suite for a single appliance channel in physical Watts."""
    p = np.maximum(0.0, _to_numpy(y_pred))  # Non-negative physical power clamp
    t = np.maximum(0.0, _to_numpy(y_true))
    m = _to_numpy(mask).astype(bool) if mask is not None else None

    valid_count = int(m.sum()) if m is not None else len(p)
    if valid_count == 0:
        return {
            "valid_samples": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "nde": 0.0,
            "sae": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    clf_metrics = compute_classification_metrics(p, t, threshold_watts=threshold_watts, mask=m)
    
    return {
        "valid_samples": valid_count,
        "mae": round(compute_mae(p, t, mask=m), 2),
        "rmse": round(compute_rmse(p, t, mask=m), 2),
        "nde": round(compute_nde(p, t, mask=m), 4),
        "sae": round(compute_sae(p, t, mask=m), 4),
        "precision": clf_metrics["precision"],
        "recall": clf_metrics["recall"],
        "f1_score": clf_metrics["f1_score"],
    }
