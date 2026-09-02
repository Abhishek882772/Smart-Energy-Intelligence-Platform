"""Unit tests for ML/src/training framework modules."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

repo_root = Path(r"C:\Users\Mahi\OneDrive\Documents\Smart-Energy-Intelligence-Platform")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ML.src.training.metrics import (
    compute_mae,
    compute_rmse,
    compute_nde,
    compute_sae,
    compute_classification_metrics,
    evaluate_appliance_metrics,
)
from ML.src.training.history import TrainingHistory
from ML.src.training.checkpoint import CheckpointManager
from ML.src.training.evaluator import Evaluator
from ML.src.training.trainer import Trainer
from ML.src.losses.masked_multitask_loss import MaskedMultiTaskLoss
from ML.src.models.seq2point import MultiOutputSeq2Point


def test_metrics_computations():
    """Verify continuous and discrete metric formulas with masking."""
    y_true = np.array([0.0, 10.0, 50.0, 100.0])
    y_pred = np.array([0.0, 15.0, 45.0, 90.0])
    mask = np.array([True, True, True, True])

    # MAE: (|0-0| + |15-10| + |45-50| + |90-100|) / 4 = (0 + 5 + 5 + 10) / 4 = 5.0
    mae = compute_mae(y_pred, y_true, mask=mask)
    assert abs(mae - 5.0) < 1e-4, f"MAE {mae} != 5.0"

    # RMSE: sqrt((0 + 25 + 25 + 100) / 4) = sqrt(150 / 4) = sqrt(37.5) = 6.1237
    rmse = compute_rmse(y_pred, y_true, mask=mask)
    assert abs(rmse - np.sqrt(37.5)) < 1e-4

    # NDE: 150 / (0 + 100 + 2500 + 10000) = 150 / 12600 = 0.0119
    nde = compute_nde(y_pred, y_true, mask=mask)
    assert abs(nde - (150.0 / 12600.0)) < 1e-4

    # SAE: |150 - 160| / 160 = 10 / 160 = 0.0625
    sae = compute_sae(y_pred, y_true, mask=mask)
    assert abs(sae - 0.0625) < 1e-4

    # Classification at threshold 15W:
    # true: [F, F, T, T], pred: [F, T, T, T]
    # TP: 2, FP: 1, FN: 0, TN: 1
    # Precision: 2/3 = 0.6667, Recall: 2/2 = 1.0, F1: 2*(2/3)*1 / (5/3) = 4/5 = 0.8
    clf = compute_classification_metrics(y_pred, y_true, threshold_watts=15.0, mask=mask)
    assert abs(clf["f1_score"] - 0.8) < 1e-3
    assert abs(clf["precision"] - (2.0/3.0)) < 1e-3
    assert abs(clf["recall"] - 1.0) < 1e-3

    # Test masking out an unmonitored channel (all False)
    mask_false = np.zeros(4, dtype=bool)
    res_unmonitored = evaluate_appliance_metrics(y_pred, y_true, mask=mask_false)
    assert res_unmonitored["valid_samples"] == 0
    assert res_unmonitored["mae"] == 0.0
    assert res_unmonitored["f1_score"] == 0.0

    print("[PASS] Metric computations and masking verified.")


def test_training_modules_flow():
    """Verify History, CheckpointManager, Evaluator, and Trainer flow on synthetic mini-dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        history = TrainingHistory(tmp_path)
        ckpt_mgr = CheckpointManager(tmp_path, monitor="val_loss", mode="min")

        # Fake normalization stats
        norm_stats = {
            "refrigerator": {"mean": 40.0, "std": 50.0},
            "washing_machine": {"mean": 25.0, "std": 200.0},
            "television": {"mean": 15.0, "std": 100.0},
        }

        criterion = MaskedMultiTaskLoss()
        evaluator = Evaluator(criterion, normalization_stats=norm_stats)

        # Synthetic tensors
        torch.manual_seed(42)
        X = torch.randn(16, 599, 1)
        y = torch.randn(16, 3)
        masks = torch.ones(16, 3, dtype=torch.bool)
        h_ids = torch.full((16,), 2, dtype=torch.int64)

        train_ds = TensorDataset(X, y, masks, h_ids)
        val_ds = TensorDataset(X[:8], y[:8], masks[:8], h_ids[:8])

        train_loader = DataLoader(train_ds, batch_size=4)
        val_loader = DataLoader(val_ds, batch_size=4)

        model = MultiOutputSeq2Point(window_length=599, input_channels=1, num_outputs=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            checkpoint_manager=ckpt_mgr,
            history=history,
            early_stopping_patience=2,
        )

        res = trainer.fit(max_epochs=2)
        assert res["best_epoch"] in [1, 2]
        assert (tmp_path / "best_model.pt").exists()
        assert (tmp_path / "training_history.csv").exists()

        print("[PASS] Full training loop, evaluator, history, and checkpoint manager verified.")


if __name__ == "__main__":
    test_metrics_computations()
    test_training_modules_flow()
    print("\nALL TRAINING FRAMEWORK UNIT TESTS PASSED!")
