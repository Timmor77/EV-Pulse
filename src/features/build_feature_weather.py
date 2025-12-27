import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
LOAD_FILE = Path("data/processed/acn_timeseries_15min.parquet")
WEATHER_FILE = Path("data/processed/weather_data.parquet")
OUTPUT_FILE = Path("data/processed/acn_ts_weather_data.parquet")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical features to help the model understand
    temporal continuity (midnight close to 11pm, Sunday close to Monday).
    """
    logger.info("Adding temporal features...")

    # Basic extraction
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5

    # Cyclical Encoding (Mathematical transformation)
    # Hour (24-hour cycle)
    # Convert hour+minute to a fraction of the day (0 to 23.99)
    time_float = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_float / 24.0)

    # Day of week (7-day cycle)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Month (12-month cycle)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag variables.
    NOTE: We must not use future data to predict the future (Data Leakage).
    Here we do "One-Step Ahead" training.
    """
    logger.info("Adding lag features...")

    target = "power_kw"

    # 1. Immediate lag (t-1): What happened 15 min ago?
    # Often the strongest predictor (system inertia).
    df["lag_15m"] = df[target].shift(1)

    # 2. Hourly lag (t-4): 1 hour ago
    df["lag_1h"] = df[target].shift(4)

    # 3. Daily lag (t-96): Same time yesterday (4 * 24 = 96)
    # Critical for capturing day/night cycle
    df["lag_24h"] = df[target].shift(96)

    # 4. Weekly lag (t-672): Same time last week (4 * 24 * 7 = 672)
    # Critical for differentiating Monday vs Sunday
    df["lag_1week"] = df[target].shift(672)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling averages to capture recent trends.
    """
    logger.info("Adding rolling mean features...")

    # Average of last 4 hours (excluding current time t to avoid leakage when predicting t)
    # We shift(1) first to not include the value we want to predict in the average!
    df["rolling_mean_4h"] = df["power_kw"].shift(1).rolling(window=16).mean()

    # Average of last 24 hours
    df["rolling_mean_24h"] = df["power_kw"].shift(1).rolling(window=96).mean()

    return df


def main():
    if not LOAD_FILE.exists() or not WEATHER_FILE.exists():
        logger.error("Missing input files. Run data processing first.")
        return

    # 1. Load Data
    logger.info("Loading datasets...")
    df_load = pd.read_parquet(LOAD_FILE)
    df_weather = pd.read_parquet(WEATHER_FILE)

    # 2. Merge
    # On fait un INNER join. Cela va automatiquement filtrer sur la période commune
    # (Sept 2018 -> Mars 2020) puisque le fichier météo est déjà coupé.
    logger.info("Merging Load & Weather...")
    df = pd.merge(df_load, df_weather, on="datetime", how="inner")

    logger.info(f"Merged shape: {df.shape}")

    # 3. Feature Engineering
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # 4. Cleaning
    # Lags create NaN at the beginning of the file (we can't have a week's lag for the first day)
    # Remove these initial empty rows.
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows due to NaN in lags.")

    # 5. Save
    # Converting datetime to index is sometimes useful for LightGBM, but let's keep it as a column for splitting.
    logger.info(f"Saving features to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    # Preview final columns
    print("\n--- ✅ Final Dataset Ready for ML ---")
    print(f"Period: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head(3))


if __name__ == "__main__":
    main()
