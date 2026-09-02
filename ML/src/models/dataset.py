"""PyTorch Dataset classes for materialized REFIT sliding-window shards."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class REFITShardDataset(Dataset):
    """PyTorch Dataset loading materialized sliding-window NPZ shard(s)."""

    def __init__(self, shard_paths: Union[str, Path, List[Union[str, Path]]]):
        """Initialize Dataset from one or more NPZ shard paths.

        Args:
            shard_paths: Path or list of Paths to .npz shard files.
        """
        if isinstance(shard_paths, (str, Path)):
            shard_paths = [shard_paths]

        self.shard_paths = [Path(p) for p in shard_paths]
        
        # Load and concatenate arrays across provided shards
        X_list = []
        y_list = []
        mask_list = []
        ts_list = []
        h_list = []

        for p in self.shard_paths:
            if not p.exists():
                raise FileNotFoundError(f"Shard not found at {p.resolve()}")
            data = np.load(p, allow_pickle=True)
            X_list.append(data["X"])
            y_list.append(data["y"])
            mask_list.append(data["valid_masks"])
            ts_list.append(data["timestamps"])
            h_list.append(data["household_id"])

        self.X = np.concatenate(X_list, axis=0).astype(np.float32)
        self.y = np.concatenate(y_list, axis=0).astype(np.float32)
        self.masks = np.concatenate(mask_list, axis=0).astype(bool)
        self.timestamps = np.concatenate(ts_list, axis=0)
        self.household_ids = np.concatenate(h_list, axis=0).astype(np.int64)

        self.length = len(self.X)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return a single sliding window sample.

        Returns:
            X: Tensor of shape (599, 1), float32
            y: Tensor of shape (3,), float32
            mask: Tensor of shape (3,), bool
            household_id: int
        """
        x_tensor = torch.from_numpy(self.X[idx])
        y_tensor = torch.from_numpy(self.y[idx])
        mask_tensor = torch.from_numpy(self.masks[idx])
        h_id = int(self.household_ids[idx])

        return x_tensor, y_tensor, mask_tensor, h_id
