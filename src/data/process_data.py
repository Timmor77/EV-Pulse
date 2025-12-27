import json
import logging
from pathlib import Path

import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FILES_TO_PROCESS = [
    "acn_jpl_FULL_2018-2024.json",
    "acn_caltech_FULL_2018-2024.json",
    "acn_office001_FULL_2018-2024.json",
]


def load_and_clean_json(filepath: Path) -> pd.DataFrame:
    """
    Loads a JSON file, cleans column names, parses dates, and handles missing values.
    """
    logger.info(f"Loading raw data from: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict) and "_items" in data:
        data = data["_items"]

    df = pd.DataFrame(data)

    # --- CRITICAL FIX ---
    # Force IDs to string to avoid PyArrow "int vs bytes" error
    id_cols = ["siteID", "clusterID", "stationID", "spaceID", "userID"]
    for col in id_cols:
        if col in df.columns:
            # Convert to string and replace 'nan' strings with actual NaN if needed,
            # but for Parquet, keeping everything as string is safer.
            df[col] = df[col].astype(str)
    # ------------------------

    # 1. Standardization: Drop MongoDB specific ID
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    # 2. Date Parsing
    date_cols = ["connectionTime", "disconnectTime", "doneChargingTime"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 3. Logic Fix: 'doneChargingTime'
    if "doneChargingTime" in df.columns and "disconnectTime" in df.columns:
        df["doneChargingTime"] = df["doneChargingTime"].fillna(df["disconnectTime"])

    # 4. Filter Invalid Rows
    initial_count = len(df)
    df = df.dropna(subset=["connectionTime"])

    if "kWhDelivered" in df.columns:
        df = df[df["kWhDelivered"] > 0]

    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        logger.warning(f"Dropped {dropped_count} rows due to invalid dates or zero energy.")

    # 5. Type Casting
    if "kWhDelivered" in df.columns:
        df["kWhDelivered"] = df["kWhDelivered"].astype("float32")

    # Add source identifier
    site_name = filepath.stem.replace("acn_", "").replace("_FULL_2018-2024", "")
    df["source_site"] = site_name

    logger.info(f"Successfully processed {len(df)} sessions from {site_name}.")
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []

    for filename in FILES_TO_PROCESS:
        file_path = RAW_DIR / filename
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        try:
            df = load_and_clean_json(file_path)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    if not all_dfs:
        logger.error("No data processed. Exiting.")
        return

    # Merge all sites into one master DataFrame
    logger.info("Concatenating all sites...")
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Sorting by time ensures efficient time-series operations later
    master_df = master_df.sort_values(by="connectionTime").reset_index(drop=True)

    # Save to Parquet
    output_path = PROCESSED_DIR / "acn_data_cleaned.parquet"
    logger.info(f"Saving merged dataset to {output_path}...")

    # Pyarrow engine is faster and supports more types
    master_df.to_parquet(output_path, engine="pyarrow", index=False)

    logger.info("Data processing pipeline completed successfully.")
    logger.info(f"Total shape: {master_df.shape}")
    logger.info(f"Columns: {list(master_df.columns)}")


if __name__ == "__main__":
    main()
