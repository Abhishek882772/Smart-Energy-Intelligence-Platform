"""REFIT Preprocessing Package for Non-Intrusive Load Monitoring (NILM).

Provides modular tools for raw REFIT data loading, timestamp grid resampling (6s),
sensor anomaly masking, target appliance channel alignment, leakage-free normalization,
and sliding-window sequence generation.
"""

from .load_refit import load_raw_refit_house, get_household_file_path
from .timestamps import parse_and_validate_timestamps, check_timestamp_monotonicity
from .cleaning import generate_validity_masks, clip_sensor_outliers
from .resampling import resample_to_uniform_grid
from .alignment import align_household_data
from .normalization import NILMNormalizer
from .windowing import SlidingWindowGenerator
from .pipeline import REFITPreprocessingPipeline

__all__ = [
    "load_raw_refit_house",
    "get_household_file_path",
    "parse_and_validate_timestamps",
    "check_timestamp_monotonicity",
    "generate_validity_masks",
    "clip_sensor_outliers",
    "resample_to_uniform_grid",
    "align_household_data",
    "NILMNormalizer",
    "SlidingWindowGenerator",
    "REFITPreprocessingPipeline",
]
