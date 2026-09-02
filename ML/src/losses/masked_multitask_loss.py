"""Masked Multi-Task Loss Module for NILM Energy Disaggregation."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class MaskedMultiTaskLoss(nn.Module):
    """Masked Multi-Task Mean Squared Error Loss.

    Designed for multi-appliance NILM where individual target channels may be unmonitored,
    defective, or corrupted by sensor glitches (e.g. House 11 with no television).

    Formulation:
    For each appliance k in {0, 1, 2}:
        valid_count_k = sum_i(M_i,k)
        if valid_count_k > 0:
            L_k = sum_i( M_i,k * (y_hat_i,k - y_i,k)^2 ) / valid_count_k
        else:
            L_k = 0.0  (ignored; does not contribute to loss or gradients)

    Total Loss:
        L_total = sum_{k: valid_count_k > 0}( L_k ) / num_active_appliances
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        eps: float = 1e-8,
        target_names: Optional[List[str]] = None,
    ):
        """Initialize Masked Multi-Task Loss.

        Args:
            weights: Optional per-appliance scalar weights (default: equal [1.0, 1.0, 1.0]).
            eps: Epsilon term preventing division by zero.
            target_names: List of appliance names for per-appliance logging.
        """
        super().__init__()
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]
        self.eps = eps
        self.target_names = target_names or ["refrigerator", "washing_machine", "television"]

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        valid_masks: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute the masked multi-task regression loss.

        Args:
            predictions: Model output tensor of shape (Batch_Size, Num_Appliances).
            targets: Ground-truth target tensor of shape (Batch_Size, Num_Appliances).
            valid_masks: Boolean validity mask tensor of shape (Batch_Size, Num_Appliances).
            return_components: If True, also returns dictionary of detached per-appliance losses.

        Returns:
            total_loss: Scalar PyTorch loss tensor.
            components (optional): Dict of floats mapping appliance name -> loss.
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        if targets.shape != valid_masks.shape:
            raise ValueError(f"Shape mismatch: targets {targets.shape} vs valid_masks {valid_masks.shape}")

        # Ensure correct types and device
        masks = valid_masks.to(dtype=predictions.dtype, device=predictions.device)
        targets = targets.to(dtype=predictions.dtype, device=predictions.device)

        num_appliances = predictions.shape[-1]
        appliance_losses = []
        components = {}
        active_appliance_count = 0

        # Element-wise squared error
        sq_errors = (predictions - targets) ** 2  # (B, C)

        for k in range(num_appliances):
            app_mask = masks[:, k]  # (B,)
            valid_sum = torch.sum(app_mask)  # scalar tensor

            if valid_sum > 0:
                # Masked mean for appliance k
                app_loss = torch.sum(sq_errors[:, k] * app_mask) / (valid_sum + self.eps)
                weighted_loss = self.weights[k] * app_loss
                appliance_losses.append(weighted_loss)
                active_appliance_count += 1
                app_name = self.target_names[k] if k < len(self.target_names) else f"appliance_{k}"
                components[app_name] = float(app_loss.detach().cpu().item())
            else:
                # No valid samples for this appliance in this batch
                app_name = self.target_names[k] if k < len(self.target_names) else f"appliance_{k}"
                components[app_name] = 0.0

        if active_appliance_count > 0:
            total_loss = torch.sum(torch.stack(appliance_losses)) / active_appliance_count
        else:
            # Fallback if entire batch has zero valid targets across all appliances
            total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        if return_components:
            return total_loss, components
        return total_loss
