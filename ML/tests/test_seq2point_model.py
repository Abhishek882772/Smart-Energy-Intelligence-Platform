"""Unit tests for MultiOutputSeq2Point model architecture and MaskedMultiTaskLoss."""

import sys
from pathlib import Path
import torch
import yaml

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ML.src.models.seq2point import MultiOutputSeq2Point
from ML.src.losses.masked_multitask_loss import MaskedMultiTaskLoss


def test_model_instantiation_and_parameter_counts():
    """Verify model initialization and calculate exact layer-by-layer parameter counts."""
    torch.manual_seed(42)
    config_path = repo_root / "ML/configs/model_seq2point.yaml"
    model = MultiOutputSeq2Point.from_config(config_path)
    
    counts = model.get_parameter_count()
    
    # Layer 1: Conv1D(1 -> 30, k=10) -> (30 * 1 * 10) + 30 = 330
    assert counts["conv1"] == 330, f"Conv1 params: {counts['conv1']} != 330"
    
    # Layer 2: Conv1D(30 -> 30, k=8) -> (30 * 30 * 8) + 30 = 7,230
    assert counts["conv2"] == 7230, f"Conv2 params: {counts['conv2']} != 7230"
    
    # Layer 3: Conv1D(30 -> 40, k=6) -> (40 * 30 * 6) + 40 = 7,240
    assert counts["conv3"] == 7240, f"Conv3 params: {counts['conv3']} != 7240"
    
    # Layer 4: Conv1D(40 -> 50, k=5) -> (50 * 40 * 5) + 50 = 10,050
    assert counts["conv4"] == 10050, f"Conv4 params: {counts['conv4']} != 10050"
    
    # Layer 5: Conv1D(50 -> 50, k=5) -> (50 * 50 * 5) + 50 = 12,550
    assert counts["conv5"] == 12550, f"Conv5 params: {counts['conv5']} != 12550"
    
    # Shared Dense: Linear(29950 -> 1024) -> (29950 * 1024) + 1024 = 30,669,824
    assert counts["fc_shared"] == 30669824, f"FC params: {counts['fc_shared']} != 30669824"
    
    # Heads: Linear(1024 -> 1) -> (1024 * 1) + 1 = 1,025 each
    assert counts["head_refrigerator"] == 1025
    assert counts["head_washing_machine"] == 1025
    assert counts["head_television"] == 1025
    
    # Total trainable parameters:
    # 330 + 7230 + 7240 + 10050 + 12550 + 30669824 + 1025 + 1025 + 1025 = 30,710,299
    total_expected = 30710299
    assert counts["total_trainable"] == total_expected, f"Total params: {counts['total_trainable']} != {total_expected}"
    print(f"\n[PASS] Model Parameter Count Verified: {counts['total_trainable']:,} trainable weights.")


def test_forward_pass_and_backward_gradients():
    """Verify forward pass tensor geometry, absence of NaNs, and backward gradient computation."""
    torch.manual_seed(42)
    model = MultiOutputSeq2Point(window_length=599, input_channels=1, num_outputs=3)
    model.train()
    
    batch_size = 4
    # Synthetic batch: (B=4, Length=599, Channels=1)
    X = torch.randn(batch_size, 599, 1, dtype=torch.float32, requires_grad=False)
    
    # Forward pass
    out = model(X)
    
    assert out.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32 output, got {out.dtype}"
    assert not torch.isnan(out).any(), "Model output contains NaN values"
    assert not torch.isinf(out).any(), "Model output contains Inf values"
    
    # Dummy loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Check that all trainable parameters received gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' did not receive gradients"
        assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaNs"
    
    print("[PASS] Forward pass and backward gradient flow verified.")


def test_masked_multitask_loss_cases():
    """Verify MaskedMultiTaskLoss across diverse mask validity cases."""
    torch.manual_seed(42)
    criterion = MaskedMultiTaskLoss()
    
    batch_size = 4
    preds = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0],
        [2.5, 3.5, 4.5]
    ], requires_grad=True)
    
    targets = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0]
    ])
    
    # Case A: All masks True
    mask_a = torch.ones(batch_size, 3, dtype=torch.bool)
    loss_a, comp_a = criterion(preds, targets, mask_a, return_components=True)
    assert torch.isfinite(loss_a), "Case A loss not finite"
    assert loss_a.item() > 0.0
    assert len(comp_a) == 3
    
    # Case B: Some refrigerator targets False
    mask_b = torch.tensor([
        [True, True, True],
        [False, True, True],
        [True, True, True],
        [False, True, True]
    ])
    loss_b = criterion(preds, targets, mask_b)
    assert torch.isfinite(loss_b), "Case B loss not finite"
    
    # Case C: All television masks False (e.g. House 11)
    mask_c = torch.tensor([
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False]
    ])
    loss_c, comp_c = criterion(preds, targets, mask_c, return_components=True)
    assert torch.isfinite(loss_c), "Case C loss not finite"
    assert comp_c["television"] == 0.0, "Television loss should be 0 when unmonitored"
    
    # Test gradient flow with television unmonitored: TV head should receive zero gradients
    loss_c.backward(retain_graph=True)
    assert preds.grad is not None
    assert (preds.grad[:, 2] == 0.0).all(), "Unmonitored TV targets must receive 0 gradients"
    
    # Case D: All masks False across an appliance
    mask_d = torch.zeros(batch_size, 3, dtype=torch.bool)
    loss_d = criterion(preds, targets, mask_d)
    assert torch.isfinite(loss_d), "Case D loss not finite"
    assert loss_d.item() == 0.0
    
    print("[PASS] All 4 MaskedMultiTaskLoss test cases passed.")


if __name__ == "__main__":
    test_model_instantiation_and_parameter_counts()
    test_forward_pass_and_backward_gradients()
    test_masked_multitask_loss_cases()
    print("\nALL MODEL AND LOSS UNIT TESTS PASSED!")
