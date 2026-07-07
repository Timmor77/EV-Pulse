"""Feature engineering module for EV charging load prediction.

This module provides functions to transform raw datetime data into
model-ready features including calendar, cyclical, and weather-based features.
The `add_context_features` function is shared between training and API inference.
"""

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

LOAD_FILE = Path("data/processed/acn_timeseries_15min.parquet")
WEATHER_FILE = Path("data/processed/weather_data.parquet")
OUTPUT_FILE = Path("data/processed/model_context.parquet")

# --- EXPORTED CONSTANTS (To be imported by API and Training) ---
CAT_FEATURES = [
    "day_of_week",
    "month",
    "hour",
    "is_business_time",
    "is_holiday",
    "is_weekend",
    "is_active_hour",
]


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw DataFrame into a model-ready feature set.

    This function adds calendar features, cyclical encodings, and business
    logic indicators. It's designed to work for both batch training and
    real-time API inference.

    Args:
        df: DataFrame with a 'datetime' column containing timestamps.

    Returns:
        DataFrame with all original columns plus engineered features.

    Features added:
        - Calendar: hour, minute, day_of_week, month, year
        - Boolean: is_weekend, is_holiday, is_active_hour, is_business_time
        - Interactions: hour_x_weekend, hour_x_month
        - Cyclical: hour_sin/cos, day_sin/cos, month_sin/cos
    """
    # 1. Basic calendar
    # Ensure it's a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # Weekend & Holidays (dynamically computed based on data range)
    df["is_weekend"] = df["day_of_week"] >= 5

    # Get unique years present in the DataFrame to load correct holiday calendar
    unique_years = df["datetime"].dt.year.unique()
    ca_holidays = holidays.US(subdiv="CA", years=unique_years)

    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in ca_holidays)

    # Interactions & Business Logic
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]

    # Peak hours (7am-7pm)
    df["is_active_hour"] = df["hour"].between(7, 19).astype(int)

    # Business Time: Active hours on non-weekend, non-holiday days
    df["is_business_time"] = (
        (df["is_active_hour"] == 1)
        & (~df["is_weekend"])  # NOT weekend
        & (~df["is_holiday"])  # NOT holiday
    ).astype(int)

    # Hour x Month Interaction (captures seasonal patterns)
    df["hour_x_month"] = df["hour"] * df["month"]

    # 4. Cyclical Encoding
    time_float = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_float / 24.0)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def main():
    """Execute feature engineering pipeline for model training.

    Loads time series and weather data, merges them, applies feature
    engineering, and saves the result as a Parquet file.
    """
    if not LOAD_FILE.exists() or not WEATHER_FILE.exists():
        logger.error("Missing input files.")
        return

    logger.info("Loading & Merging...")
    df_load = pd.read_parquet(LOAD_FILE)
    df_weather = pd.read_parquet(WEATHER_FILE)

    # Inner Merge
    df = pd.merge(df_load, df_weather, on="datetime", how="inner")

    # Apply features (Call shared function)
    df = add_context_features(df)

    # Final cleanup
    cols_to_drop = ["active_chargers"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logger.info(f"Saving Context-Only dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\n--- ✅ 'Context-Only' Dataset Ready ---")


if __name__ == "__main__":
    main()
