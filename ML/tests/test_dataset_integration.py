"""Real REFIT dataset integration test with MultiOutputSeq2Point and MaskedMultiTaskLoss."""

import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

repo_root = Path(r"C:\Users\Mahi\OneDrive\Documents\Smart-Energy-Intelligence-Platform")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ML.src.models.seq2point import MultiOutputSeq2Point
from ML.src.models.dataset import REFITShardDataset
from ML.src.losses.masked_multitask_loss import MaskedMultiTaskLoss


def test_real_refit_batch_pipeline():
    """Verify end-to-end integration: Real Shard -> Dataset -> DataLoader -> Model -> Loss -> Backward."""
    torch.manual_seed(42)
    
    # 1. Locate real materialized training shard
    shard_path = repo_root / "ML/data/processed/REFIT/train/house_2.npz"
    assert shard_path.exists(), f"Materialized shard not found at {shard_path}"
    
    # 2. Instantiate Dataset and DataLoader
    dataset = REFITShardDataset(shard_path)
    assert len(dataset) > 0, "Dataset is empty"
    
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Instantiate Model, Loss, and Optimizer
    model = MultiOutputSeq2Point.from_config(repo_root / "ML/configs/model_seq2point.yaml")
    model.train()
    criterion = MaskedMultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Fetch one real batch
    X_batch, y_batch, mask_batch, h_ids = next(iter(loader))
    
    print(f"\nReal REFIT Batch Loaded:")
    print(f"  - X_batch shape   : {X_batch.shape}  [Expected: ({batch_size}, 599, 1)]")
    print(f"  - y_batch shape   : {y_batch.shape}  [Expected: ({batch_size}, 3)]")
    print(f"  - mask_batch shape: {mask_batch.shape}  [Expected: ({batch_size}, 3)]")
    print(f"  - h_ids shape     : {h_ids.shape}  [Household IDs: {h_ids[0].item()}]")
    
    assert X_batch.shape == (batch_size, 599, 1), f"Unexpected X_batch shape: {X_batch.shape}"
    assert y_batch.shape == (batch_size, 3), f"Unexpected y_batch shape: {y_batch.shape}"
    assert mask_batch.shape == (batch_size, 3), f"Unexpected mask_batch shape: {mask_batch.shape}"
    assert (h_ids == 2).all(), "All household IDs in house_2.npz should be 2"
    
    # 5. Model Forward Pass
    predictions = model(X_batch)
    assert predictions.shape == (batch_size, 3), f"Unexpected predictions shape: {predictions.shape}"
    assert not torch.isnan(predictions).any(), "Predictions contain NaN"
    
    # 6. Masked Multi-Task Loss Computation
    loss, components = criterion(predictions, y_batch, mask_batch, return_components=True)
    
    print(f"\nForward Pass Loss Results:")
    print(f"  - Total Loss: {loss.item():.4f}")
    for k, v in components.items():
        print(f"    * {k:<16}: {v:.4f}")
        
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0.0, "Loss should be positive"
    
    # 7. Backward Pass & Parameter Update Check
    optimizer.zero_grad()
    loss.backward()
    
    # Verify gradients exist before step
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' did not receive gradients"
        assert not torch.isnan(param.grad).any(), f"Gradients for '{name}' contain NaN"
        
    # Optimizer step
    optimizer.step()
    
    print("[PASS] Real REFIT Dataset Integration Test Successful (Data -> Model -> Loss -> Backward -> Step).")


if __name__ == "__main__":
    test_real_refit_batch_pipeline()
