"""Raw REFIT data loader module with chunking and slicing capabilities."""

from pathlib import Path
from typing import Generator, List, Optional, Union
import pandas as pd

RAW_COLUMNS = [
    "Time", "Unix", "Aggregate",
    "Appliance1", "Appliance2", "Appliance3",
    "Appliance4", "Appliance5", "Appliance6",
    "Appliance7", "Appliance8", "Appliance9",
    "Issues"
]

COLUMN_DTYPES = {
    "Unix": "int64",
    "Aggregate": "float64",
    "Appliance1": "float64",
    "Appliance2": "float64",
    "Appliance3": "float64",
    "Appliance4": "float64",
    "Appliance5": "float64",
    "Appliance6": "float64",
    "Appliance7": "float64",
    "Appliance8": "float64",
    "Appliance9": "float64",
    "Issues": "int8"
}


def get_household_file_path(house_id: int, raw_dir: Union[str, Path] = "ML/data/raw/REFIT") -> Path:
    """Return the absolute/relative Path to the raw CSV file for a given household ID."""
    p = Path(raw_dir) / f"CLEAN_House{house_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Raw REFIT file for House {house_id} not found at {p.resolve()}")
    return p


def load_raw_refit_house(
    house_id: int,
    raw_dir: Union[str, Path] = "ML/data/raw/REFIT",
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[int] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, Generator[pd.DataFrame, None, None]]:
    """Load raw REFIT CSV data for a specific household.

    Args:
        house_id: Household number (1-21, excluding 14).
        raw_dir: Path to raw REFIT directory.
        usecols: List of column names to load (default: all 13 columns).
        nrows: Maximum number of rows to read.
        skiprows: Number of rows to skip from the beginning.
        chunksize: If specified, returns an iterator yielding DataFrames of size chunksize.

    Returns:
        pd.DataFrame or Generator of pd.DataFrame.
    """
    fpath = get_household_file_path(house_id, raw_dir)
    cols = usecols if usecols is not None else RAW_COLUMNS
    dtypes = {c: COLUMN_DTYPES[c] for c in cols if c in COLUMN_DTYPES}

    if skiprows is not None and skiprows > 0:
        # If skipping rows, we need to pass column names explicitly
        return pd.read_csv(
            fpath,
            names=RAW_COLUMNS,
            header=0,
            skiprows=skiprows,
            nrows=nrows,
            usecols=cols,
            dtype=dtypes,
            chunksize=chunksize,
        )
    else:
        return pd.read_csv(
            fpath,
            usecols=cols,
            nrows=nrows,
            dtype=dtypes,
            chunksize=chunksize,
        )
