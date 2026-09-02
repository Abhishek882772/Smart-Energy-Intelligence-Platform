"""Leakage-free normalization module for NILM power signals."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml


class NILMNormalizer:
    """Normalizer for power signals supporting Standardization (z-score) and Min-Max scaling.

    STRICT LEAKAGE PREVENTION:
    Statistics (mean, std, min, max) are calculated strictly on training split data
    and applied identically to validation, test, and downstream transfer datasets.
    """

    def __init__(
        self,
        strategy: str = "standardization",
        clip_zscore: float = 10.0,
        epsilon: float = 1e-6,
    ):
        """Initialize Normalizer.

        Args:
            strategy: 'standardization' ((x - mu)/sigma) or 'minmax' ((x - min)/(max - min)).
            clip_zscore: Max absolute normalized value for standardization to clip extreme spikes.
            epsilon: Safeguard against division by zero.
        """
        if strategy not in ["standardization", "minmax"]:
            raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'standardization' or 'minmax'.")
        self.strategy = strategy
        self.clip_zscore = clip_zscore
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, float]] = {}

    def fit(self, data_list: List[pd.DataFrame], channels: Optional[List[str]] = None) -> "NILMNormalizer":
        """Compute normalization statistics across a collection of training DataFrames.

        Only rows where valid_{channel} is True are included in parameter calculations.

        Args:
            data_list: List of aligned DataFrames from training households.
            channels: List of channel names to normalize (e.g. ['aggregate', 'refrigerator', 'washing_machine']).
        """
        if not data_list:
            raise ValueError("Cannot fit normalizer on empty data list.")

        if channels is None:
            first_df = data_list[0]
            channels = [c for c in first_df.columns if not c.startswith("valid_") and np.issubdtype(first_df[c].dtype, np.number)]

        self.stats = {}

        for ch in channels:
            valid_arrays = []
            for df in data_list:
                if ch in df.columns:
                    mask_col = f"valid_{ch.lower()}"
                    if mask_col in df.columns:
                        val_series = df.loc[df[mask_col], ch].dropna().to_numpy()
                    else:
                        val_series = df[ch].dropna().to_numpy()
                    if len(val_series) > 0:
                        valid_arrays.append(val_series)

            if not valid_arrays:
                # Default identity fallback if channel not observed in training set
                self.stats[ch] = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
                continue

            all_vals = np.concatenate(valid_arrays)
            mean_val = float(np.mean(all_vals))
            std_val = float(np.std(all_vals))
            min_val = float(np.min(all_vals))
            max_val = float(np.max(all_vals))

            # Ensure non-zero scale
            std_val = max(std_val, self.epsilon)
            range_val = max(max_val - min_val, self.epsilon)

            self.stats[ch] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "sample_count": int(len(all_vals)),
            }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization transformation to DataFrame."""
        if not self.stats:
            raise RuntimeError("Normalizer has not been fitted yet. Call .fit() or .load_stats().")

        df_out = df.copy()

        for ch, st in self.stats.items():
            if ch in df_out.columns:
                vals = df_out[ch].to_numpy(dtype=float)
                if self.strategy == "standardization":
                    normed = (vals - st["mean"]) / st["std"]
                    if self.clip_zscore is not None:
                        normed = np.clip(normed, -self.clip_zscore, self.clip_zscore)
                elif self.strategy == "minmax":
                    normed = (vals - st["min"]) / max(st["max"] - st["min"], self.epsilon)
                    normed = np.clip(normed, 0.0, 1.0)
                df_out[ch] = normed

        return df_out

    def inverse_transform(self, arr: np.ndarray, channel: str) -> np.ndarray:
        """Revert normalized NumPy array back to original Watts."""
        if channel not in self.stats:
            raise KeyError(f"Channel '{channel}' not found in normalizer statistics.")

        st = self.stats[channel]
        if self.strategy == "standardization":
            reverted = (arr * st["std"]) + st["mean"]
        elif self.strategy == "minmax":
            reverted = (arr * (st["max"] - st["min"])) + st["min"]
        
        # Physical power is non-negative
        return np.maximum(0.0, reverted)

    def save_stats(self, filepath: Union[str, Path]) -> None:
        """Save fitted normalization parameters to a YAML file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": self.strategy,
            "clip_zscore": self.clip_zscore,
            "epsilon": self.epsilon,
            "channel_statistics": self.stats,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)

    def load_stats(self, filepath: Union[str, Path]) -> "NILMNormalizer":
        """Load normalization parameters from YAML file."""
        in_path = Path(filepath)
        if not in_path.exists():
            raise FileNotFoundError(f"Stats file not found at {in_path.resolve()}")
        with open(in_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.strategy = data.get("strategy", "standardization")
        self.clip_zscore = data.get("clip_zscore", 10.0)
        self.epsilon = data.get("epsilon", 1e-6)
        self.stats = data.get("channel_statistics", {})
        return self
