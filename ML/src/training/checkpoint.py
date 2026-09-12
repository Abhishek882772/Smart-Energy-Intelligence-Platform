"""Checkpoint Management Module for NILM Model Weights and State."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer


class CheckpointManager:
    """Manages saving, loading, and recovery of model weights and optimizer states."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = "val_loss",
        mode: str = "min",
        save_latest: bool = True,
    ):
        """Initialize Checkpoint Manager.

        Args:
            checkpoint_dir: Directory where .pt files are stored.
            monitor: Name of validation metric used to select best checkpoint.
            mode: Optimization direction ('min' or 'max').
            save_latest: If True, saves latest_checkpoint.pt every epoch.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_latest = save_latest

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def is_better(self, current: float) -> bool:
        """Check if current score improves upon best observed score."""
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        current_score: float,
        scheduler: Optional[Any] = None,
        val_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save model and optimizer states.

        Returns:
            is_best: True if current checkpoint is the new best model.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "current_score": current_score,
            "best_score": self.best_score,
            "monitor": self.monitor,
            "val_metrics": val_metrics or {},
            "config": config or {},
        }

        # Save latest checkpoint
        if self.save_latest:
            latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
            torch.save(payload, latest_path)

        # Check and save best checkpoint
        is_best = False
        if self.is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch
            payload["best_score"] = self.best_score
            payload["best_epoch"] = self.best_epoch
            
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(payload, best_path)
            is_best = True

        return is_best

    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """Load state dictionary into model, optimizer, and scheduler.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            model: Model whose weights should be restored.
            optimizer: Optional optimizer whose state should be restored.
            scheduler: Optional LR scheduler whose state should be restored.
            device: Target device for loading weights and optimizer states (default: None, inferred from model or 'cpu').

        Returns:
            checkpoint: Loaded checkpoint dictionary.
        """
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at {p.resolve()}")

        target_device = (
            device
            if device is not None
            else (next(model.parameters()).device if list(model.parameters()) else "cpu")
        )

        checkpoint = torch.load(p, map_location=target_device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Ensure all optimizer state tensors reside on target_device
            target_torch_device = torch.device(target_device)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(target_torch_device)

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_score = checkpoint.get("best_score", self.best_score)
        self.best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", 0))

        return checkpoint
