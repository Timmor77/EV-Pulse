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
OUTPUT_FILE = Path("data/processed/model_context.parquet")

# --- EXPORT DES CONSTANTES (Pour être importées par l'API et le Train) ---
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
    """
    Transforme un DataFrame brut (date + météo) en DataFrame prêt pour le modèle.
    Utilisable pour le training (batch) ET pour l'API (live).
    """
    # 1. Calendrier de base
    # On s'assure que c'est bien un datetime
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    # 2. Week-end & Jours Fériés (DYNAMIQUE)
    df["is_weekend"] = df["day_of_week"] >= 5

    # On récupère les années uniques présentes dans le DF pour charger les bons jours fériés
    unique_years = df["datetime"].dt.year.unique()
    ca_holidays = holidays.US(state="CA", years=unique_years)

    df["is_holiday"] = df["datetime"].dt.date.apply(lambda x: x in ca_holidays)

    # 3. Interactions & Business Logic
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]

    # Heures de pointe (7h-19h)
    df["is_active_hour"] = df["hour"].between(7, 19).astype(int)

    # Business Time
    df["is_business_time"] = (
        (df["is_active_hour"] == 1)
        & (df["is_weekend"] == False)
        & (df["is_holiday"] == False)
    ).astype(int)

    # Interaction Heure x Mois
    df["hour_x_month"] = df["hour"] * df["month"]

    # 4. Encodage Cyclique
    time_float = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_float / 24.0)

    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def main():
    """Pipeline d'exécution pour le Training uniquement"""
    if not LOAD_FILE.exists() or not WEATHER_FILE.exists():
        logger.error("Missing input files.")
        return

    logger.info("Loading & Merging...")
    df_load = pd.read_parquet(LOAD_FILE)
    df_weather = pd.read_parquet(WEATHER_FILE)

    # Merge Inner
    df = pd.merge(df_load, df_weather, on="datetime", how="inner")

    # Application des features (Appel de la fonction partagée)
    df = add_context_features(df)

    # Nettoyage final
    cols_to_drop = ["active_chargers"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    logger.info(f"Saving Context-Only dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("\n--- ✅ Dataset 'Context-Only' Ready ---")


if __name__ == "__main__":
    main()
