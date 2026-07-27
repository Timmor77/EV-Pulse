import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

LOAD_FILE = Path("data/processed/acn_timeseries_15min.parquet")
WEATHER_FILE = Path("data/processed/weather_data.parquet")
OUTPUT_FILE = Path("data/processed/acn_ts_weather_holidays_data.parquet")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding calendar features (Cycles + Holidays)...")

    # 1. Base Time Features
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # 2. Boolean Features (Simple and Effective)
    # Weekend is often different
    df["is_weekend"] = df["day_of_week"] >= 5

    # 3. Holidays (US - California)
    # Key for avoiding major errors on Thanksgiving/Christmas
    ca_holidays = holidays.US(subdiv="CA", years=range(2018, 2022))
    # Create a boolean column: Is this a holiday?
    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in ca_holidays)

    # 4. Cyclic Encoding (For mathematical continuity)
    time_float = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_float / 24.0)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_robust_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'Safe' features for 24h ahead (Day-Ahead) forecasting.
    Only uses data available from the previous day.
    """
    logger.info("Adding robust lag features (Day-Ahead safe)...")
    target = "power_kw"

    # 1. What happened yesterday at the same time (24h Lag)
    # This is our main anchor.
    df["lag_24h"] = df[target].shift(96)  # 96 quarter-hours = 24h

    # 2. What happened a week ago (7-day Lag)
    # To capture weekly seasonality (Monday vs Sunday)
    df["lag_1week"] = df[target].shift(96 * 7)

    # 3. Average of YESTERDAY's consumption (not sliding 24h average)
    # Trick: Take the rolling average shifted by 24h
    # This represents "Average consumption from 24h to 48h ago"
    # This info is fully known when predicting for tomorrow.
    df["avg_energy_yesterday"] = df[target].shift(96).rolling(window=96).mean()

    return df


def main():
    if not LOAD_FILE.exists() or not WEATHER_FILE.exists():
        logger.error("Missing input files.")
        return

    logger.info("Loading & Merging...")
    df_load = pd.read_parquet(LOAD_FILE)
    df_weather = pd.read_parquet(WEATHER_FILE)

    df = pd.merge(df_load, df_weather, on="datetime", how="inner")

    df = add_calendar_features(df)
    df = add_robust_lag_features(df)

    # Clean NaN due to lags (7 days lost at beginning)
    df = df.dropna()

    # Select final columns
    # Keep active_chargers for analysis, but remove it from X in training
    logger.info(f"Saving robust dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\n--- Day-ahead features ready ---")
    print(df[["datetime", "is_holiday", "lag_24h", "avg_energy_yesterday"]].head())


if __name__ == "__main__":
    main()
