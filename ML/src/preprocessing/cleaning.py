"""Sensor issue handling and validity masking module."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def generate_validity_masks(
    df: pd.DataFrame,
    issues_col: str = "Issues",
    power_cols: Optional[List[str]] = None,
    min_power: float = 0.0,
    max_power: float = 4000.0,
    mask_issues_flags: bool = True,
) -> pd.DataFrame:
    """Generate explicit boolean validity masks for each power channel.

    Rules for validity (valid = True):
    1. issues_col == 0 (if mask_issues_flags=True).
    2. Power value is not NaN.
    3. min_power <= power <= max_power (IAM rated limit is 4000W).

    Args:
        df: Input DataFrame containing power channels and issues column.
        issues_col: Name of sensor anomaly column ('Issues').
        power_cols: List of power channels to generate masks for.
        min_power: Minimum physically valid power in Watts (default: 0W).
        max_power: Maximum sensor ceiling in Watts (default: 4000W).
        mask_issues_flags: If True, mark entire row invalid when Issues == 1.

    Returns:
        DataFrame containing original columns plus 'valid_{channel}' boolean columns.
    """
    df_out = df.copy()

    if power_cols is None:
        power_cols = [c for c in df.columns if "aggregate" in c.lower() or "appliance" in c.lower() or c in ["refrigerator", "washing_machine", "television"]]

    # Base issue mask (True = clean row, False = issue flag present)
    if mask_issues_flags and issues_col in df_out.columns:
        row_clean = (df_out[issues_col] == 0)
    else:
        row_clean = pd.Series(True, index=df_out.index)

    for col in power_cols:
        if col in df_out.columns:
            s = df_out[col]
            # Valid conditions
            is_not_nan = s.notna()
            in_range = (s >= min_power) & (s <= max_power)
            
            # Combine
            mask_col = f"valid_{col.lower()}"
            df_out[mask_col] = row_clean & is_not_nan & in_range

    return df_out


def clip_sensor_outliers(
    df: pd.DataFrame,
    power_cols: Optional[List[str]] = None,
    min_power: float = 0.0,
    max_power: float = 4000.0,
) -> pd.DataFrame:
    """Clip physical power measurements to [min_power, max_power] without modifying validity masks."""
    df_out = df.copy()
    if power_cols is None:
        power_cols = [c for c in df.columns if not c.startswith("valid_") and c not in ["Time", "Unix", "Issues", "DateTime"]]

    for col in power_cols:
        if col in df_out.columns and np.issubdtype(df_out[col].dtype, np.number):
            df_out[col] = df_out[col].clip(lower=min_power, upper=max_power)

    return df_out
