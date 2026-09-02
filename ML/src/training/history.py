"""Structured Training History Logger for NILM Experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


class TrainingHistory:
    """Tracks, serializes, and analyzes training and validation metric history."""

    def __init__(self, output_dir: Union[str, Path], filename_prefix: str = "training_history"):
        """Initialize history logger.

        Args:
            output_dir: Destination directory for history files.
            filename_prefix: Base prefix for .csv and .json files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.history: List[Dict[str, Any]] = []

    def record_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for a completed epoch."""
        entry = {"epoch": epoch}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                entry[k] = v
            elif hasattr(v, "item"):
                entry[k] = v.item()
            else:
                entry[k] = str(v)
        self.history.append(entry)
        self.save()

    def save(self) -> None:
        """Atomically persist history to both CSV and JSON formats."""
        if not self.history:
            return

        df = pd.DataFrame(self.history)
        csv_path = self.output_dir / f"{self.filename_prefix}.csv"
        json_path = self.output_dir / f"{self.filename_prefix}.json"

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def get_best(self, metric: str = "val_loss", mode: str = "min") -> Tuple[int, float]:
        """Retrieve epoch index and value of best observed metric."""
        if not self.history:
            return (0, float("inf") if mode == "min" else float("-inf"))

        valid_entries = [e for e in self.history if metric in e and e[metric] is not None]
        if not valid_entries:
            return (0, float("inf") if mode == "min" else float("-inf"))

        if mode == "min":
            best_entry = min(valid_entries, key=lambda x: x[metric])
        else:
            best_entry = max(valid_entries, key=lambda x: x[metric])

        return int(best_entry["epoch"]), float(best_entry[metric])

    def to_dataframe(self) -> pd.DataFrame:
        """Return history as pandas DataFrame."""
        return pd.DataFrame(self.history)

    def state_dict(self) -> List[Dict[str, Any]]:
        """Return raw history records for checkpointing."""
        return self.history

    def load_state_dict(self, state: List[Dict[str, Any]]) -> None:
        """Load history records from checkpoint state."""
        self.history = state
