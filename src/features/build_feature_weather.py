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
    Crée des features cycliques pour aider le modèle à comprendre
    la continuité temporelle (minuit proche de 23h, Dimanche proche de Lundi).
    """
    logger.info("Adding temporal features...")

    # Extraction basique
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5

    # Encodage Cyclique (Magie Mathématique)
    # Heure (Cycle de 24h)
    # On convertit l'heure+minute en une fraction de la journée (0 à 23.99)
    time_float = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_float / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_float / 24.0)

    # Jour de la semaine (Cycle de 7 jours)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Mois (Cycle de 12 mois)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables de retard (Lags).
    ATTENTION : On ne doit pas utiliser le futur pour prédire le futur (Data Leakage).
    Mais ici on fait du "One-Step Ahead" training.
    """
    logger.info("Adding lag features...")

    target = "power_kw"

    # 1. Lag immédiat (t-1) : Que se passait-il il y a 15 min ?
    # C'est souvent le prédicteur le plus fort (inertie du système).
    df["lag_15m"] = df[target].shift(1)

    # 2. Lag Horaire (t-4) : Il y a 1 heure
    df["lag_1h"] = df[target].shift(4)

    # 3. Lag Journalier (t-96) : Hier à la même heure (4 * 24 = 96)
    # Crucial pour capturer le cycle jour/nuit
    df["lag_24h"] = df[target].shift(96)

    # 4. Lag Hebdomadaire (t-672) : La semaine dernière (4 * 24 * 7 = 672)
    # Crucial pour différencier Lundi vs Dimanche
    df["lag_1week"] = df[target].shift(672)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moyennes glissantes pour capturer la tendance récente.
    """
    logger.info("Adding rolling mean features...")

    # Moyenne des 4 dernières heures (excluant l'instant t actuel pour éviter le leakage si on prédit t)
    # On shift(1) d'abord pour ne pas inclure la valeur qu'on veut prédire dans la moyenne !
    df["rolling_mean_4h"] = df["power_kw"].shift(1).rolling(window=16).mean()

    # Moyenne des 24 dernières heures
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
    # Les Lags créent des NaN au début du fichier (on ne peut pas avoir le lag d'une semaine pour le 1er jour)
    # On supprime ces premières lignes vides.
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows due to NaN in lags.")

    # 5. Save
    # On convertit datetime en index pour LightGBM parfois c'est pratique, mais gardons-le en colonne pour le split.
    logger.info(f"Saving features to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)

    # Aperçu des colonnes finales
    print("\n--- ✅ Dataset Final Ready for ML ---")
    print(f"Période : {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Colonnes ({len(df.columns)}) : {list(df.columns)}")
    print(df.head(3))


if __name__ == "__main__":
    main()
