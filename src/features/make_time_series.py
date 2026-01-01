"""Time series generation module for EV charging data.

This module transforms individual charging session records into a continuous
time series representation with 15-minute resolution, suitable for forecasting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/acn_data_cleaned.parquet")
OUTPUT_FILE = Path("data/processed/acn_timeseries_15min.parquet")


def create_timeseries_from_sessions(df: pd.DataFrame, interval_min: int = 15) -> pd.DataFrame:
    """Transform charging sessions into a continuous time series.

    Uses a uniform power distribution assumption (rectangular) over the charging
    duration. This simplification is appropriate for aggregated load prediction.

    Args:
        df: DataFrame with columns 'connectionTime', 'disconnectTime',
            'doneChargingTime', and 'kWhDelivered'.
        interval_min: Time resolution in minutes (default: 15).

    Returns:
        DataFrame with columns 'datetime', 'power_kw', and 'active_chargers'.
    """

    # 1. Define global time boundaries
    start_date = df["connectionTime"].min().floor("H")
    end_date = df["disconnectTime"].max().ceil("H")

    # Create complete time index (e.g., every 15 min from 2018 to 2021)
    # '15T' is the pandas alias for 15 minutes
    freq = f"{interval_min}T"
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq, tz="UTC")

    logger.info(f"Timeline created: {len(time_index)} points from {start_date} to {end_date}")

    # 2. Data structure preparation (Numpy for speed)
    # Create an array of zeros the size of the timeline
    load_curve = np.zeros(len(time_index))
    occupancy_curve = np.zeros(len(time_index))

    # Map dates to integer indices (0, 1, 2, ...)
    # This is the trick for speed: we no longer manipulate dates but array indices
    timestamps = time_index.to_numpy()

    # Iterate over sessions (fast here since we're doing simple math)
    # For a huge dataset, we could parallelize, but for <100k rows it's instant.
    logger.info("Projecting sessions onto timeline...")

    count = 0
    total = len(df)

    for row in df.itertuples():
        # Calculate active charging duration (in hours)
        # Use doneChargingTime because after that, the car is plugged in but not charging (0 kW)
        charge_start = row.connectionTime
        charge_end = row.doneChargingTime

        # Safety: if end is before start (data bug), skip
        if charge_end <= charge_start:
            continue

        duration_hours = (charge_end - charge_start).total_seconds() / 3600
        if duration_hours < (interval_min / 60):
            # Session too short, ignore or count as a spike
            continue

        avg_power_kw = row.kWhDelivered / duration_hours

        # Find indices in our large array
        # searchsorted is very fast for finding where a date fits
        idx_start = np.searchsorted(timestamps, charge_start)
        idx_end_charge = np.searchsorted(timestamps, charge_end)
        idx_end_conn = np.searchsorted(timestamps, row.disconnectTime)

        # Fill load array (Power)
        # Add average power over the entire charging duration
        if idx_end_charge > idx_start:
            load_curve[idx_start:idx_end_charge] += avg_power_kw

        # Fill occupancy array (Occupancy)
        # The car occupies the charger until disconnectTime, even if no longer charging
        if idx_end_conn > idx_start:
            occupancy_curve[idx_start:idx_end_conn] += 1

        count += 1
        if count % 10000 == 0:
            logger.info(f"Processed {count}/{total} sessions")

    # 3. Final assembly
    ts_df = pd.DataFrame(
        {
            "datetime": time_index,
            "power_kw": load_curve,
            "active_chargers": occupancy_curve,
        }
    )

    # Optimal typing
    ts_df["power_kw"] = ts_df["power_kw"].astype("float32")
    ts_df["active_chargers"] = ts_df["active_chargers"].astype("int32")

    return ts_df


def main():
    if not INPUT_FILE.exists():
        logger.error("Input file not found. Run process_data.py first.")
        return

    df = pd.read_parquet(INPUT_FILE)

    # Optional filtering: Can focus on Caltech to start if needed
    # df = df[df['source_site'] == 'caltech']

    logger.info("Generating Time Series (15 min intervals)...")
    ts_df = create_timeseries_from_sessions(df)

    logger.info(f"Saving Time Series to {OUTPUT_FILE}...")
    ts_df.to_parquet(OUTPUT_FILE, index=False)

    logger.info("Done! Data preview:")
    print(ts_df.head())
    print(ts_df.describe())


if __name__ == "__main__":
    main()
