"""Data processing and transformation module.

This module handles cleaning, transforming, and preparing
the raw ACN charging session data for analysis and modeling.
"""

import json
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_sessions(site: str = "caltech") -> pd.DataFrame:
    """Load raw charging sessions from JSON file.

    Args:
        site: Site identifier ('caltech', 'jpl', 'office001').

    Returns:
        DataFrame with raw session data.
    """
    filepath = RAW_DATA_DIR / f"acn_{site}_FULL_2018-2024.json"

    with open(filepath, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess charging session data.

    Args:
        df: Raw session DataFrame.

    Returns:
        Cleaned DataFrame with proper types and derived columns.
    """
    df = df.copy()

    # Convert timestamps
    time_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Calculate session duration (minutes)
    if "connectionTime" in df.columns and "disconnectTime" in df.columns:
        df["session_duration_min"] = (
            df["disconnectTime"] - df["connectionTime"]
        ).dt.total_seconds() / 60

    # Calculate charging duration (minutes)
    if "connectionTime" in df.columns and "doneChargingTime" in df.columns:
        df["charging_duration_min"] = (
            df["doneChargingTime"] - df["connectionTime"]
        ).dt.total_seconds() / 60

    return df


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save processed DataFrame to Parquet format.

    Args:
        df: Processed DataFrame.
        filename: Output filename (without extension).

    Returns:
        Path to saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / f"{filename}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # Process all sites
    sites = ["caltech", "jpl", "office001"]

    for site in sites:
        print(f"Processing {site}...")
        try:
            df = load_raw_sessions(site)
            df = clean_sessions(df)
            output = save_processed(df, f"sessions_{site}")
            print(f"  -> Saved to {output}")
        except FileNotFoundError:
            print(f"  -> No data found for {site}, skipping.")
