"""Timestamp processing and validation module for REFIT time series."""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def parse_and_validate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "Time",
    unix_col: str = "Unix",
    set_index: bool = True,
) -> pd.DataFrame:
    """Parse string timestamps into DateTime objects, validate monotonicity, and optionally set as index.

    Args:
        df: Input DataFrame containing timestamp columns.
        timestamp_col: Name of ISO timestamp column (e.g. 'Time').
        unix_col: Name of integer Unix timestamp column (e.g. 'Unix').
        set_index: Whether to set DateTime as the DataFrame index.

    Returns:
        DataFrame with parsed DateTime.
    """
    df_out = df.copy()

    if "DateTime" not in df_out.columns:
        if timestamp_col in df_out.columns:
            df_out["DateTime"] = pd.to_datetime(df_out[timestamp_col], errors="coerce")
        elif unix_col in df_out.columns:
            df_out["DateTime"] = pd.to_datetime(df_out[unix_col], unit="s")
        else:
            raise KeyError(f"Neither '{timestamp_col}' nor '{unix_col}' found in DataFrame columns.")

    # Drop any rows with unparseable timestamps
    df_out = df_out.dropna(subset=["DateTime"])

    # Ensure chronological sorting
    if not df_out["DateTime"].is_monotonic_increasing:
        df_out = df_out.sort_values("DateTime").reset_index(drop=True)

    # Check for duplicate timestamps
    if df_out["DateTime"].duplicated().any():
        # Keep first reading for identical timestamps
        df_out = df_out.drop_duplicates(subset=["DateTime"], keep="first").reset_index(drop=True)

    if set_index:
        df_out = df_out.set_index("DateTime")

    return df_out


def check_timestamp_monotonicity(df: pd.DataFrame, time_col: str = "DateTime") -> Tuple[bool, int, Dict[str, float]]:
    """Check if timestamps are strictly monotonic and compute delta statistics.

    Returns:
        is_monotonic: bool
        inversions_count: number of negative steps
        delta_stats: dict with mean, median, min, max interval in seconds.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series()
    else:
        dt_series = pd.to_datetime(df[time_col])

    diffs = dt_series.diff().dt.total_seconds().dropna().to_numpy()

    inversions = int(np.sum(diffs < 0))
    is_mono = inversions == 0

    stats = {
        "mean_interval_s": float(np.mean(diffs)) if len(diffs) > 0 else 0.0,
        "median_interval_s": float(np.median(diffs)) if len(diffs) > 0 else 0.0,
        "min_interval_s": float(np.min(diffs)) if len(diffs) > 0 else 0.0,
        "max_interval_s": float(np.max(diffs)) if len(diffs) > 0 else 0.0,
    }

    return is_mono, inversions, stats
