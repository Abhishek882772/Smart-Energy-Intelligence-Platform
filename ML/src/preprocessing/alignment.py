"""Channel extraction, target mapping, and multi-channel alignment module."""

from typing import Any, Dict, Optional, Union
import pandas as pd

from .timestamps import parse_and_validate_timestamps
from .cleaning import generate_validity_masks, clip_sensor_outliers
from .resampling import resample_to_uniform_grid


def align_household_data(
    raw_df: pd.DataFrame,
    house_id: int,
    target_mapping: Dict[str, Any],
    target_interval_s: int = 6,
    max_forward_fill_s: int = 30,
    mask_issues_flags: bool = True,
    max_power_cutoff: float = 4000.0,
) -> pd.DataFrame:
    """Extract and align Aggregate and Target Appliances for a given household onto a 6-second grid.

    Output Schema:
    - Index: DatetimeIndex (6-second cadence)
    - Power columns: 'aggregate', 'refrigerator', 'washing_machine', 'television' (if available)
    - Mask columns: 'valid_aggregate', 'valid_refrigerator', 'valid_washing_machine', 'valid_television' (if available)

    Args:
        raw_df: Raw DataFrame containing REFIT columns (Time, Aggregate, Appliance1..9, Issues).
        house_id: Household number (1-21, excluding 14).
        target_mapping: Dictionary loaded from refit_target_mapping.yaml.
        target_interval_s: Target resampling interval in seconds (default: 6s).
        max_forward_fill_s: Maximum gap to forward fill (default: 30s).
        mask_issues_flags: Whether to invalidate rows where Issues == 1.
        max_power_cutoff: Maximum physical sensor cutoff in Watts (default: 4000W).

    Returns:
        Aligned, resampled, and quality-masked DataFrame.
    """
    house_key = house_id if house_id in target_mapping.get("households", {}) else str(house_id)
    h_info = target_mapping.get("households", {}).get(house_key, None)

    if h_info is None:
        raise KeyError(f"Household {house_id} not found in target mapping configuration.")

    # 1. Parse timestamps
    df_timed = parse_and_validate_timestamps(raw_df, set_index=True)

    # 2. Extract and rename target channels
    rename_dict = {}
    power_cols = []

    # Aggregate
    agg_col = h_info.get("aggregate_channel", "Aggregate")
    if agg_col in df_timed.columns:
        rename_dict[agg_col] = "aggregate"
        power_cols.append("aggregate")

    # Refrigerator
    fridge_info = h_info.get("refrigerator")
    if fridge_info and fridge_info.get("channel") in df_timed.columns:
        col_name = fridge_info["channel"]
        rename_dict[col_name] = "refrigerator"
        power_cols.append("refrigerator")

    # Washing Machine
    wm_info = h_info.get("washing_machine")
    if wm_info and wm_info.get("channel") in df_timed.columns:
        col_name = wm_info["channel"]
        rename_dict[col_name] = "washing_machine"
        power_cols.append("washing_machine")

    # Television (Exploratory target)
    tv_info = h_info.get("television")
    if tv_info and tv_info.get("channel") in df_timed.columns:
        col_name = tv_info["channel"]
        rename_dict[col_name] = "television"
        power_cols.append("television")

    # Select required columns + Issues
    needed_raw_cols = list(rename_dict.keys())
    if "Issues" in df_timed.columns:
        needed_raw_cols.append("Issues")

    df_subset = df_timed[needed_raw_cols].rename(columns=rename_dict)

    # 3. Clip sensor outliers
    df_clipped = clip_sensor_outliers(df_subset, power_cols=power_cols, min_power=0.0, max_power=max_power_cutoff)

    # 4. Generate validity masks
    df_masked = generate_validity_masks(
        df_clipped,
        issues_col="Issues",
        power_cols=power_cols,
        min_power=0.0,
        max_power=max_power_cutoff,
        mask_issues_flags=mask_issues_flags,
    )

    # Drop raw Issues column before resampling
    if "Issues" in df_masked.columns:
        df_masked = df_masked.drop(columns=["Issues"])

    mask_cols = [f"valid_{c}" for c in power_cols]

    # 5. Resample to 6-second grid
    df_resampled = resample_to_uniform_grid(
        df_masked,
        target_interval_s=target_interval_s,
        max_forward_fill_s=max_forward_fill_s,
        power_cols=power_cols,
        mask_cols=mask_cols,
        agg_method="mean",
    )

    # 6. Ensure all standard targets exist (refrigerator, washing_machine, television)
    STANDARD_TARGETS = ["refrigerator", "washing_machine", "television"]
    for tgt in STANDARD_TARGETS:
        if tgt not in df_resampled.columns:
            df_resampled[tgt] = 0.0
            df_resampled[f"valid_{tgt}"] = False

    # Standard column ordering
    ordered_cols = ["aggregate", "refrigerator", "washing_machine", "television",
                    "valid_aggregate", "valid_refrigerator", "valid_washing_machine", "valid_television"]
    df_resampled = df_resampled[[c for c in ordered_cols if c in df_resampled.columns]]

    return df_resampled

