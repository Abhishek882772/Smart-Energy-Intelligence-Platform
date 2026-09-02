"""Time-series grid resampling module for NILM power measurements."""

from typing import List, Optional
import numpy as np
import pandas as pd


def resample_to_uniform_grid(
    df: pd.DataFrame,
    target_interval_s: int = 6,
    max_forward_fill_s: int = 30,
    power_cols: Optional[List[str]] = None,
    mask_cols: Optional[List[str]] = None,
    agg_method: str = "mean",
) -> pd.DataFrame:
    """Resample irregular time series to a uniform temporal grid (default 6s).

    Methodology:
    1. Forward-fill short intervals (<= max_forward_fill_s) to accommodate packet transmission jitter.
    2. Bin into fixed target_interval_s windows (e.g. '6s').
    3. Aggregate power values with binned mean (preserving energy conservation: Watts * seconds).
    4. Compute resampled validity masks (valid if >= 50% of source samples in the bin were valid).
    5. Gaps longer than max_forward_fill_s are filled with 0.0 power and marked valid = False.

    Args:
        df: Input DataFrame with DatetimeIndex.
        target_interval_s: Target interval in seconds (default: 6s = 0.1667 Hz).
        max_forward_fill_s: Maximum gap to forward fill (default: 30s).
        power_cols: List of power measurement columns.
        mask_cols: List of boolean validity mask columns.
        agg_method: Aggregation function ('mean').

    Returns:
        DataFrame resampled onto regular 6-second frequency grid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex before resampling.")

    freq_str = f"{target_interval_s}s"

    if power_cols is None:
        power_cols = [c for c in df.columns if not c.startswith("valid_") and np.issubdtype(df[c].dtype, np.number)]
    if mask_cols is None:
        mask_cols = [c for c in df.columns if c.startswith("valid_")]

    # Step 1: Forward-fill power channels for short jitter up to max_forward_fill_s
    # In Pandas resample, forward filling at 1s resolution before 6s mean is clean and exact
    # Or 1s upsample -> ffill(limit) -> 6s mean
    df_ffill = df.copy()
    
    # 1-second grid alignment for exact time-weighted forward fill
    # To keep memory low, we apply limit directly on resampler
    # First, resample raw points onto 6s grid with mean
    resampled_power = df_ffill[power_cols].resample(freq_str).agg(agg_method)
    
    # Forward fill short gaps on the 6s grid (limit = 30s // 6s = 5 steps)
    limit_steps = max(1, max_forward_fill_s // target_interval_s)
    resampled_power_filled = resampled_power.ffill(limit=limit_steps).fillna(0.0)

    # Step 2: Resample validity masks
    if mask_cols:
        # Mean of boolean mask gives proportion of valid samples in the 6s bin
        mask_means = df_ffill[mask_cols].astype(float).resample(freq_str).mean()
        # Forward fill validity mask along with power
        mask_means_filled = mask_means.ffill(limit=limit_steps).fillna(0.0)
        resampled_masks = mask_means_filled >= 0.5
    else:
        # Derive validity from whether power was present (not filled with 0 from long gap)
        resampled_masks = pd.DataFrame(index=resampled_power.index)
        for col in power_cols:
            valid_col = f"valid_{col.lower()}"
            is_real = resampled_power[col].notna()
            resampled_masks[valid_col] = is_real.ffill(limit=limit_steps).fillna(False)

    # Combine into unified DataFrame
    resampled_df = pd.concat([resampled_power_filled, resampled_masks], axis=1)

    return resampled_df
