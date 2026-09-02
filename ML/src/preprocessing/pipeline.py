"""End-to-end preprocessing pipeline coordinator for the REFIT dataset."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

from .load_refit import load_raw_refit_house
from .alignment import align_household_data
from .normalization import NILMNormalizer
from .windowing import SlidingWindowGenerator


class REFITPreprocessingPipeline:
    """Orchestrates end-to-end raw data loading, cleaning, 6s resampling, normalization, and window generation."""

    def __init__(
        self,
        config_path: Union[str, Path] = "ML/configs/refit_preprocessing.yaml",
        target_mapping_path: Union[str, Path] = "ML/configs/refit_target_mapping.yaml",
    ):
        """Initialize Pipeline with YAML configurations."""
        self.config_path = Path(config_path)
        self.target_mapping_path = Path(target_mapping_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        with open(self.target_mapping_path, "r", encoding="utf-8") as f:
            self.target_mapping = yaml.safe_load(f)

        self.normalizer = NILMNormalizer(
            strategy=self.config["normalization"]["strategy"],
            clip_zscore=self.config["normalization"]["clip_zscore"],
            epsilon=self.config["normalization"]["epsilon"],
        )

    def process_household(
        self,
        house_id: int,
        nrows: Optional[int] = None,
        skiprows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Process a single household CSV into an aligned 6-second resampled DataFrame."""
        raw_dir = self.config["data"]["raw_dir"]
        raw_df = load_raw_refit_house(house_id, raw_dir=raw_dir, nrows=nrows, skiprows=skiprows)

        target_interval_s = self.config["sampling"]["target_sampling_interval_seconds"]
        max_ffill_s = self.config["sampling"]["max_forward_fill_seconds"]
        mask_issues = self.config["quality_control"]["mask_issues_flags"]
        max_power = self.config["quality_control"]["max_power_cutoff_watts"]

        aligned_df = align_household_data(
            raw_df=raw_df,
            house_id=house_id,
            target_mapping=self.target_mapping,
            target_interval_s=target_interval_s,
            max_forward_fill_s=max_ffill_s,
            mask_issues_flags=mask_issues,
            max_power_cutoff=max_power,
        )

        return aligned_df

    def fit_normalizer_on_training_split(
        self,
        train_house_ids: Optional[List[int]] = None,
        nrows_per_house: Optional[int] = None,
        save_stats: bool = True,
    ) -> NILMNormalizer:
        """Fit normalization parameters strictly on training households."""
        if train_house_ids is None:
            train_house_ids = self.config["splits"]["provisional_train"]

        train_dfs = []
        for h_id in train_house_ids:
            df_aligned = self.process_household(h_id, nrows=nrows_per_house)
            train_dfs.append(df_aligned)

        # Fit on all active power channels
        self.normalizer.fit(train_dfs)

        if save_stats:
            stats_file = self.config["data"]["normalization_stats_file"]
            self.normalizer.save_stats(stats_file)

        return self.normalizer

    def create_model_windows(
        self,
        house_id: int,
        nrows: Optional[int] = None,
        skiprows: Optional[int] = None,
        normalize: bool = True,
        window_length: Optional[int] = None,
        stride: Optional[int] = None,
        target_format: Optional[str] = None,
        target_appliances: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Generate model-ready sliding windows for a given household."""
        df_aligned = self.process_household(house_id, nrows=nrows, skiprows=skiprows)

        if normalize:
            if not self.normalizer.stats:
                # Try loading saved stats or fit on default
                stats_file = self.config["data"]["normalization_stats_file"]
                if Path(stats_file).exists():
                    self.normalizer.load_stats(stats_file)
                else:
                    self.fit_normalizer_on_training_split()
            df_aligned = self.normalizer.transform(df_aligned)

        w_len = window_length or self.config["windowing"]["default_window_length"]
        w_stride = stride or self.config["windowing"]["default_stride_train"]
        w_format = target_format or self.config["windowing"]["target_format"]
        min_valid = self.config["quality_control"]["min_valid_samples_ratio"]

        generator = SlidingWindowGenerator(
            window_length=w_len,
            stride=w_stride,
            target_format=w_format,
            min_valid_ratio=min_valid,
        )

        return generator.generate_windows(
            df_aligned,
            input_col="aggregate",
            target_cols=target_appliances,
            household_id=house_id,
        )

