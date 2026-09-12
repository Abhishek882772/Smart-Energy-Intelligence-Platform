"""Evaluator Module for NILM Model Assessment on Validation and Test Partitions."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .device import get_device
from .metrics import evaluate_appliance_metrics


class Evaluator:
    """Evaluates multi-output NILM models with physical de-standardization and masked metric aggregation."""

    def __init__(
        self,
        criterion: nn.Module,
        normalization_stats: Dict[str, Dict[str, float]],
        target_names: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize Evaluator.

        Args:
            criterion: MaskedMultiTaskLoss instance.
            normalization_stats: Dict mapping channel name -> {mean, std}.
            target_names: Ordered list of target appliance names.
            thresholds: Dict mapping appliance name -> activation threshold in Watts.
            device: Target execution device ('cpu' or 'cuda', default: auto-resolved).
        """
        self.criterion = criterion
        self.norm_stats = normalization_stats
        self.target_names = target_names or ["refrigerator", "washing_machine", "television"]
        self.thresholds = thresholds or {
            "refrigerator": 15.0,
            "washing_machine": 20.0,
            "television": 10.0,
        }
        self.device = get_device(device)

    def to_watts(self, arr_norm: np.ndarray, channel_name: str) -> np.ndarray:
        """Inverse-transform normalized z-scores to physical power in Watts with non-negative clamp."""
        st = self.norm_stats[channel_name]
        raw_watts = (arr_norm * st["std"]) + st["mean"]
        return np.maximum(0.0, raw_watts)

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        compute_detailed_metrics: bool = True,
    ) -> Dict[str, Any]:
        """Execute evaluation loop on validation or test DataLoader.

        Returns:
            results_dict: Dictionary containing total_loss, per_appliance_loss, and physical metrics.
        """
        model.eval()
        total_loss = 0.0
        app_loss_accum = {name: 0.0 for name in self.target_names}
        total_samples = 0

        all_preds = []
        all_targets = []
        all_masks = []
        all_house_ids = []

        non_blocking = (self.device.type == "cuda")

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 4:
                    batch_X, batch_y, batch_mask, batch_h = batch
                else:
                    batch_X, batch_y, batch_mask = batch
                    batch_h = torch.zeros(len(batch_X), dtype=torch.int64)

                batch_X = batch_X.to(self.device, non_blocking=non_blocking)
                batch_y = batch_y.to(self.device, non_blocking=non_blocking)
                batch_mask = batch_mask.to(self.device, non_blocking=non_blocking)

                preds = model(batch_X)
                loss, components = self.criterion(preds, batch_y, batch_mask, return_components=True)

                bsz = len(batch_X)
                total_loss += loss.item() * bsz
                for name, val in components.items():
                    if name in app_loss_accum:
                        app_loss_accum[name] += val * bsz
                total_samples += bsz

                if compute_detailed_metrics:
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())
                    all_masks.append(batch_mask.cpu().numpy())
                    all_house_ids.append(batch_h.numpy())

        avg_total_loss = total_loss / max(1, total_samples)
        avg_app_losses = {name: val / max(1, total_samples) for name, val in app_loss_accum.items()}

        out = {
            "total_loss": round(avg_total_loss, 6),
            "appliance_losses": {k: round(v, 6) for k, v in avg_app_losses.items()},
            "total_samples": total_samples,
        }

        if not compute_detailed_metrics or total_samples == 0:
            return out

        # Concatenate full arrays for physical metric calculation
        preds_norm = np.concatenate(all_preds, axis=0)
        targets_norm = np.concatenate(all_targets, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        house_ids = np.concatenate(all_house_ids, axis=0)

        # Convert to physical Watts
        appliance_metrics = {}
        mae_list = []
        f1_list = []

        for i, app_name in enumerate(self.target_names):
            p_watts = self.to_watts(preds_norm[:, i], app_name)
            t_watts = self.to_watts(targets_norm[:, i], app_name)
            app_mask = masks[:, i]
            thresh = self.thresholds.get(app_name, 15.0)

            m = evaluate_appliance_metrics(p_watts, t_watts, mask=app_mask, threshold_watts=thresh)
            appliance_metrics[app_name] = m

            if m["valid_samples"] > 0:
                mae_list.append(m["mae"])
                f1_list.append(m["f1_score"])

        out["appliance_metrics"] = appliance_metrics
        out["macro_mae"] = round(float(np.mean(mae_list)), 2) if mae_list else 0.0
        out["macro_f1"] = round(float(np.mean(f1_list)), 4) if f1_list else 0.0

        # Per-household breakdown
        unique_houses = np.unique(house_ids)
        if len(unique_houses) > 1 or unique_houses[0] != 0:
            house_breakdown = {}
            for h in unique_houses:
                h_idx = (house_ids == h)
                h_dict = {}
                for i, app_name in enumerate(self.target_names):
                    p_w = self.to_watts(preds_norm[h_idx, i], app_name)
                    t_w = self.to_watts(targets_norm[h_idx, i], app_name)
                    m_w = masks[h_idx, i]
                    thresh = self.thresholds.get(app_name, 15.0)
                    h_dict[app_name] = evaluate_appliance_metrics(p_w, t_w, mask=m_w, threshold_watts=thresh)
                house_breakdown[int(h)] = h_dict
            out["house_breakdown"] = house_breakdown

        return out
