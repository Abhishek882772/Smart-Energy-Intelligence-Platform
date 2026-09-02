"""Unified Multi-Output Sequence-to-Point (Seq2Point) 1D CNN Architecture.

Adapted from the canonical Seq2Point architecture (Zhang et al., AAAI 2018) for
simultaneous multi-appliance energy disaggregation across Refrigerator, Washing Machine,
and Television.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import yaml


class MultiOutputSeq2Point(nn.Module):
    """Unified Multi-Output Seq2Point Neural Network.

    Architecture Overview:
    1. Input Transposition: Converts input tensor (B, 599, 1) -> (B, 1, 599) for PyTorch Conv1D.
    2. Shared 1D CNN Backbone: 5 convolutional layers with increasing filter counts (30 -> 50)
       extracting hierarchical temporal patterns (thermostatic cycling, motor pulses).
    3. Shared Dense Representation: 1024-unit fully connected layer with ReLU and Dropout (p=0.2).
    4. Decoupled Appliance Heads: Three dedicated linear layers projecting the shared representation
       onto midpoint power estimates for:
         - Target 1: Refrigerator (Index 0)
         - Target 2: Washing Machine (Index 1)
         - Target 3: Television (Index 2)
    5. Linear Output: Unconstrained regression output (power values in standardized space).
    """

    def __init__(
        self,
        window_length: int = 599,
        input_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        dense_units: int = 1024,
        dropout_rate: float = 0.2,
        num_outputs: int = 3,
        target_names: Optional[List[str]] = None,
    ):
        """Initialize the MultiOutputSeq2Point model.

        Args:
            window_length: Length of aggregate sequence window (default: 599 samples = ~59.9 min at 6s).
            input_channels: Number of input channels (default: 1 for aggregate active power).
            conv_channels: List of output channels for the 5 Conv1D layers.
            kernel_sizes: List of kernel sizes for the 5 Conv1D layers.
            dense_units: Dimensionality of shared fully connected representation (default: 1024).
            dropout_rate: Dropout probability after shared dense layer (default: 0.2).
            num_outputs: Total number of target appliances (default: 3).
            target_names: Ordered list of target appliance names.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [30, 30, 40, 50, 50]
        if kernel_sizes is None:
            kernel_sizes = [10, 8, 6, 5, 5]
        if target_names is None:
            target_names = ["refrigerator", "washing_machine", "television"]

        self.window_length = window_length
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs
        self.target_names = target_names

        # ----------------------------------------------------------------------
        # 1. SHARED 1D CONVOLUTIONAL FEATURE EXTRACTOR
        # ----------------------------------------------------------------------
        # Layer 1: Conv1D (in=1 -> out=30, kernel=10, stride=1, padding='same')
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_channels[0],
            kernel_size=kernel_sizes[0],
            stride=1,
            padding="same",
        )
        self.act1 = nn.ReLU()

        # Layer 2: Conv1D (in=30 -> out=30, kernel=8, stride=1, padding='same')
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_sizes[1],
            stride=1,
            padding="same",
        )
        self.act2 = nn.ReLU()

        # Layer 3: Conv1D (in=30 -> out=40, kernel=6, stride=1, padding='same')
        self.conv3 = nn.Conv1d(
            in_channels=conv_channels[1],
            out_channels=conv_channels[2],
            kernel_size=kernel_sizes[2],
            stride=1,
            padding="same",
        )
        self.act3 = nn.ReLU()

        # Layer 4: Conv1D (in=40 -> out=50, kernel=5, stride=1, padding='same')
        self.conv4 = nn.Conv1d(
            in_channels=conv_channels[2],
            out_channels=conv_channels[3],
            kernel_size=kernel_sizes[3],
            stride=1,
            padding="same",
        )
        self.act4 = nn.ReLU()

        # Layer 5: Conv1D (in=50 -> out=50, kernel=5, stride=1, padding='same')
        self.conv5 = nn.Conv1d(
            in_channels=conv_channels[3],
            out_channels=conv_channels[4],
            kernel_size=kernel_sizes[4],
            stride=1,
            padding="same",
        )
        self.act5 = nn.ReLU()

        # ----------------------------------------------------------------------
        # 2. SHARED DENSE REPRESENTATION
        # ----------------------------------------------------------------------
        self.flatten_dim = conv_channels[4] * window_length  # 50 * 599 = 29,950
        self.fc_shared = nn.Linear(self.flatten_dim, dense_units)
        self.act_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # ----------------------------------------------------------------------
        # 3. INDEPENDENT MULTI-OUTPUT APPLIANCE HEADS
        # ----------------------------------------------------------------------
        # Head 1: Refrigerator Midpoint Estimate
        self.head_refrigerator = nn.Linear(dense_units, 1)

        # Head 2: Washing Machine Midpoint Estimate
        self.head_washing_machine = nn.Linear(dense_units, 1)

        # Head 3: Television Midpoint Estimate
        self.head_television = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Unified Multi-Output Seq2Point model.

        Args:
            x: Input aggregate power tensor. Accepts:
               - (Batch_Size, 599, 1)  [Standard materialized dataset shape]
               - (Batch_Size, 1, 599)  [PyTorch Conv1D shape]
               - (Batch_Size, 599)     [1D sequence shape]

        Returns:
            predictions: Tensor of shape (Batch_Size, 3) representing standardized power estimates:
                         [:, 0] -> Refrigerator
                         [:, 1] -> Washing Machine
                         [:, 2] -> Television
        """
        # Step 1: Ensure PyTorch Conv1D dimension ordering (B, Channels, Length)
        if x.ndim == 3:
            if x.shape[1] == self.window_length and x.shape[2] == self.input_channels:
                # Transpose (B, Length=599, Channels=1) -> (B, Channels=1, Length=599)
                x = x.transpose(1, 2)
            elif x.shape[1] == self.input_channels and x.shape[2] == self.window_length:
                # Already (B, 1, 599)
                pass
            else:
                raise ValueError(f"Unexpected 3D tensor shape: {x.shape}. Expected (B, {self.window_length}, 1) or (B, 1, {self.window_length}).")
        elif x.ndim == 2:
            # (B, Length=599) -> (B, Channels=1, Length=599)
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.ndim}D tensor of shape {x.shape}")

        # Ensure float32 dtype
        x = x.to(dtype=torch.float32)

        # Step 2: Shared 1D Convolutional Backbone
        h = self.act1(self.conv1(x))  # -> (B, 30, 599)
        h = self.act2(self.conv2(h))  # -> (B, 30, 599)
        h = self.act3(self.conv3(h))  # -> (B, 40, 599)
        h = self.act4(self.conv4(h))  # -> (B, 50, 599)
        h = self.act5(self.conv5(h))  # -> (B, 50, 599)

        # Step 3: Flatten and Shared Dense Representation
        h_flat = h.flatten(start_dim=1)  # -> (B, 29950)
        h_shared = self.dropout(self.act_fc(self.fc_shared(h_flat)))  # -> (B, 1024)

        # Step 4: Multi-Output Heads (Linear Projections)
        y_fridge = self.head_refrigerator(h_shared)      # -> (B, 1)
        y_wm = self.head_washing_machine(h_shared)        # -> (B, 1)
        y_tv = self.head_television(h_shared)            # -> (B, 1)

        # Step 5: Concatenate along channel dimension -> (B, 3)
        # Output order strictly fixed: [refrigerator, washing_machine, television]
        predictions = torch.cat([y_fridge, y_wm, y_tv], dim=-1)

        return predictions

    def get_parameter_count(self) -> Dict[str, int]:
        """Compute detailed layer-by-layer trainable parameter counts."""
        counts = {
            "conv1": sum(p.numel() for p in self.conv1.parameters() if p.requires_grad),
            "conv2": sum(p.numel() for p in self.conv2.parameters() if p.requires_grad),
            "conv3": sum(p.numel() for p in self.conv3.parameters() if p.requires_grad),
            "conv4": sum(p.numel() for p in self.conv4.parameters() if p.requires_grad),
            "conv5": sum(p.numel() for p in self.conv5.parameters() if p.requires_grad),
            "fc_shared": sum(p.numel() for p in self.fc_shared.parameters() if p.requires_grad),
            "head_refrigerator": sum(p.numel() for p in self.head_refrigerator.parameters() if p.requires_grad),
            "head_washing_machine": sum(p.numel() for p in self.head_washing_machine.parameters() if p.requires_grad),
            "head_television": sum(p.numel() for p in self.head_television.parameters() if p.requires_grad),
        }
        counts["total_trainable"] = sum(counts.values())
        return counts

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "MultiOutputSeq2Point":
        """Instantiate model from YAML configuration file or dictionary."""
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config

        arch = cfg.get("architecture", cfg)
        return cls(
            window_length=arch.get("window_length", 599),
            input_channels=arch.get("input_channels", 1),
            conv_channels=arch.get("conv_channels", [30, 30, 40, 50, 50]),
            kernel_sizes=arch.get("kernel_sizes", [10, 8, 6, 5, 5]),
            dense_units=arch.get("dense_units", 1024),
            dropout_rate=arch.get("dropout_rate", 0.2),
            num_outputs=arch.get("num_outputs", 3),
            target_names=arch.get("target_names", ["refrigerator", "washing_machine", "television"]),
        )
