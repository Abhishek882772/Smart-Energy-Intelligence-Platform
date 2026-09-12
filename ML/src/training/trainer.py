"""Trainer Module for Multi-Output NILM Baseline Models."""

import time
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .device import get_device, log_environment_info
from .evaluator import Evaluator
from .history import TrainingHistory


class Trainer:
    """Orchestrates model training, validation evaluation, checkpointing, and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        evaluator: Evaluator,
        checkpoint_manager: CheckpointManager,
        history: TrainingHistory,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        grad_clip_norm: Optional[float] = 5.0,
        early_stopping_patience: int = 5,
    ):
        """Initialize Trainer.

        Args:
            model: PyTorch model instance (MultiOutputSeq2Point).
            optimizer: Optimizer instance (e.g. Adam).
            criterion: MaskedMultiTaskLoss instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            evaluator: Evaluator instance.
            checkpoint_manager: CheckpointManager instance.
            history: TrainingHistory instance.
            scheduler: Optional learning rate scheduler.
            device: Computation device ('cpu', 'cuda', or None for auto-detection).
            grad_clip_norm: Maximum gradient norm for clipping.
            early_stopping_patience: Epochs without validation loss improvement before stopping.
        """
        self.device = get_device(device)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.checkpoint_manager = checkpoint_manager
        self.history = history
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.patience = early_stopping_patience

        self.model.to(self.device)

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        app_losses = {name: 0.0 for name in self.evaluator.target_names}
        non_blocking = (self.device.type == "cuda")

        t0 = time.time()
        for batch in self.train_loader:
            if len(batch) == 4:
                batch_X, batch_y, batch_mask, _ = batch
            else:
                batch_X, batch_y, batch_mask = batch

            batch_X = batch_X.to(self.device, non_blocking=non_blocking)
            batch_y = batch_y.to(self.device, non_blocking=non_blocking)
            batch_mask = batch_mask.to(self.device, non_blocking=non_blocking)

            self.optimizer.zero_grad()
            preds = self.model(batch_X)
            loss, components = self.criterion(preds, batch_y, batch_mask, return_components=True)

            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            bsz = len(batch_X)
            total_loss += loss.item() * bsz
            for name, val in components.items():
                if name in app_losses:
                    app_losses[name] += val * bsz
            total_samples += bsz

        elapsed = time.time() - t0
        throughput = total_samples / max(1e-6, elapsed)

        avg_loss = total_loss / max(1, total_samples)
        avg_apps = {name: val / max(1, total_samples) for name, val in app_losses.items()}
        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "train_loss": round(avg_loss, 6),
            "train_fridge_loss": round(avg_apps.get("refrigerator", 0.0), 6),
            "train_wm_loss": round(avg_apps.get("washing_machine", 0.0), 6),
            "train_tv_loss": round(avg_apps.get("television", 0.0), 6),
            "lr": current_lr,
            "epoch_time_seconds": round(elapsed, 2),
            "throughput_samples_per_sec": round(throughput, 1),
        }

    def fit(self, max_epochs: int = 20, start_epoch: int = 1) -> Dict[str, Any]:
        """Execute full training and validation loop with early stopping."""
        log_environment_info(self.device)

        print("=" * 80)
        print(f"STARTING FULL TRAINING RUN ({max_epochs} Epochs Max | Target Device: {self.device})")
        print("=" * 80)

        epochs_no_improve = 0
        best_val_loss = float("inf")

        for epoch in range(start_epoch, max_epochs + 1):
            print(f"\n--- Epoch [{epoch:02d}/{max_epochs}] ---")
            train_res = self.train_epoch(epoch)
            print(f"  [Train] Loss: {train_res['train_loss']:.6f} | Fridge: {train_res['train_fridge_loss']:.6f} | WM: {train_res['train_wm_loss']:.6f} | TV: {train_res['train_tv_loss']:.6f} | LR: {train_res['lr']:.6f} ({train_res['epoch_time_seconds']:.1f}s, {train_res['throughput_samples_per_sec']:.0f} spl/s)")

            # Validation Evaluation
            val_res = self.evaluator.evaluate(self.model, self.val_loader, compute_detailed_metrics=True)
            v_loss = val_res["total_loss"]
            v_apps = val_res["appliance_losses"]
            print(f"  [Val]   Loss: {v_loss:.6f} | Fridge: {v_apps.get('refrigerator', 0.0):.6f} | WM: {v_apps.get('washing_machine', 0.0):.6f} | TV: {v_apps.get('television', 0.0):.6f} | Macro MAE: {val_res.get('macro_mae', 0.0):.2f}W | Macro F1: {val_res.get('macro_f1', 0.0):.4f}")

            # LR Scheduler Step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

            # Save Checkpoint
            is_best = self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                current_score=v_loss,
                val_metrics=val_res,
            )

            if is_best:
                print(f"  --> [NEW BEST MODEL SAVED] Val Loss improved: {best_val_loss:.6f} -> {v_loss:.6f}")
                best_val_loss = v_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  --> No improvement for {epochs_no_improve}/{self.patience} epochs.")

            # Record Epoch
            epoch_log = {
                **train_res,
                "val_loss": v_loss,
                "val_fridge_loss": v_apps.get("refrigerator", 0.0),
                "val_wm_loss": v_apps.get("washing_machine", 0.0),
                "val_tv_loss": v_apps.get("television", 0.0),
                "val_macro_mae": val_res.get("macro_mae", 0.0),
                "val_macro_f1": val_res.get("macro_f1", 0.0),
                "is_best": is_best,
            }
            self.history.record_epoch(epoch, epoch_log)

            # Early Stopping Check
            if epochs_no_improve >= self.patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation loss did not improve for {self.patience} epochs.")
                break

        print("\n" + "=" * 80)
        best_ep, best_val = self.checkpoint_manager.best_epoch, self.checkpoint_manager.best_score
        print(f"TRAINING COMPLETED: Best Model at Epoch {best_ep} (Val Loss: {best_val:.6f})")
        print("=" * 80)

        return {
            "best_epoch": best_ep,
            "best_val_loss": best_val,
            "total_epochs": epoch,
            "history": self.history.to_dataframe(),
        }
