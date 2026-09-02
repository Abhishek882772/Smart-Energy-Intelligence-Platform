"""Sliding window generator for NILM sequence models (Seq2Point, TCN, Transformer)."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class SlidingWindowGenerator:
    """Configurable sliding-window generator for multi-appliance NILM time-series."""

    def __init__(
        self,
        window_length: int = 599,
        stride: int = 30,
        target_format: str = "midpoint",
        min_valid_ratio: float = 0.90,
    ):
        """Initialize Window Generator.

        Args:
            window_length: Sequence window length $W$ (default: 599 samples = ~59.9 min at 6s).
            stride: Step size $S$ between consecutive windows (default: 30 for train, 1 for eval).
            target_format: 'midpoint' (Seq2Point: midpoint single point $y(t + W//2)$) or
                           'sequence' (Seq2Seq / TCN: full sequence of length $W$).
            min_valid_ratio: Minimum fraction of valid samples required in an input window.
        """
        if window_length % 2 == 0 and target_format == "midpoint":
            # For midpoint, odd length provides a unique exact center index
            pass
        self.window_length = window_length
        self.stride = stride
        self.target_format = target_format
        self.min_valid_ratio = min_valid_ratio

    def generate_windows(
        self,
        df: pd.DataFrame,
        input_col: str = "aggregate",
        target_cols: Optional[List[str]] = None,
        household_id: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract sliding windows and targets from an aligned preprocessed DataFrame.

        Args:
            df: Aligned DataFrame with DatetimeIndex, power channels, and validity masks.
            input_col: Name of input feature column (default: 'aggregate').
            target_cols: List of target appliance columns (e.g. ['refrigerator', 'washing_machine']).
            household_id: Optional household integer ID to attach to windows.

        Returns:
            Dictionary containing:
            - 'X': Input tensor of shape (N, W, 1) or (N, W)
            - 'y': Target tensor of shape (N, C) [for midpoint] or (N, W, C) [for sequence]
            - 'valid_masks': Boolean mask array of shape (N, C) indicating validity of each target sample
            - 'timestamps': Array of midpoint ISO timestamps (N,)
            - 'household_id': Array of integer household IDs (N,)
            - 'target_names': List of appliance names [C]
        """
        if target_cols is None:
            target_cols = [c for c in ["refrigerator", "washing_machine", "television"] if c in df.columns]

        if input_col not in df.columns:
            raise KeyError(f"Input column '{input_col}' not found in DataFrame.")

        n_samples = len(df)
        w = self.window_length
        s = self.stride
        half_w = w // 2

        if n_samples < w:
            return {
                "X": np.empty((0, w, 1), dtype=np.float32),
                "y": np.empty((0, len(target_cols)), dtype=np.float32),
                "valid_masks": np.empty((0, len(target_cols)), dtype=bool),
                "timestamps": np.empty((0,), dtype=object),
                "household_id": np.empty((0,), dtype=np.int32),
                "target_names": np.array(target_cols),
            }

        # Extract arrays
        x_raw = df[input_col].to_numpy(dtype=np.float32)
        x_mask = df[f"valid_{input_col.lower()}"].to_numpy(dtype=bool) if f"valid_{input_col.lower()}" in df.columns else np.ones(n_samples, dtype=bool)

        y_raw = df[target_cols].to_numpy(dtype=np.float32)
        y_masks = np.column_stack([
            df[f"valid_{c.lower()}"].to_numpy(dtype=bool) if f"valid_{c.lower()}" in df.columns else np.ones(n_samples, dtype=bool)
            for c in target_cols
        ])

        if isinstance(df.index, pd.DatetimeIndex):
            times = df.index.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
        else:
            times = np.arange(n_samples)

        # Sliding window indices
        num_windows = (n_samples - w) // s + 1
        start_indices = np.arange(0, num_windows * s, s)

        X_list = []
        y_list = []
        mask_list = []
        ts_list = []

        for start in start_indices:
            end = start + w
            mid = start + half_w

            # Quality Check 1: Input window valid ratio
            window_x_mask = x_mask[start:end]
            if np.mean(window_x_mask) < self.min_valid_ratio:
                continue

            # Input slice
            x_win = x_raw[start:end]

            # Target slice & mask
            if self.target_format == "midpoint":
                y_point = y_raw[mid]
                m_point = y_masks[mid]
                # If target point is not valid, mask indicates it
                y_list.append(y_point)
                mask_list.append(m_point)
                ts_list.append(times[mid])
            elif self.target_format == "sequence":
                y_seq = y_raw[start:end]
                m_seq = y_masks[start:end]
                y_list.append(y_seq)
                mask_list.append(np.mean(m_seq, axis=0) >= self.min_valid_ratio)
                ts_list.append(times[mid])

            X_list.append(x_win)

        if not X_list:
            return {
                "X": np.empty((0, w, 1), dtype=np.float32),
                "y": np.empty((0, len(target_cols)), dtype=np.float32),
                "valid_masks": np.empty((0, len(target_cols)), dtype=bool),
                "timestamps": np.empty((0,), dtype=object),
                "household_id": np.empty((0,), dtype=np.int32),
                "target_names": np.array(target_cols),
            }

        X_arr = np.array(X_list, dtype=np.float32)
        # Reshape to (N, W, 1) for Conv1D / Attention layers
        X_arr = np.expand_dims(X_arr, axis=-1)

        y_arr = np.array(y_list, dtype=np.float32)
        masks_arr = np.array(mask_list, dtype=bool)
        ts_arr = np.array(ts_list)
        h_arr = np.full(len(X_arr), household_id if household_id is not None else -1, dtype=np.int32)

        return {
            "X": X_arr,
            "y": y_arr,
            "valid_masks": masks_arr,
            "timestamps": ts_arr,
            "household_id": h_arr,
            "target_names": np.array(target_cols),
        }

