import pandas as pd
import numpy as np
import holidays
import logging
from pathlib import Path

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

LOAD_FILE = Path("data/processed/acn_timeseries_15min.parquet")
WEATHER_FILE = Path("data/processed/weather_data.parquet")
OUTPUT_FILE = Path("data/processed/acn_ts_weather_holidays_data.parquet")

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding calendar features (Cycles + Holidays)...")
    
    # 1. Base Time Features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # 2. Boolean Features (Simples et Efficaces)
    # Le week-end est souvent différent
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # 3. Holidays (US - California)
    # C'est la clé pour éviter les grosses erreurs sur Thanksgiving/Noël
    ca_holidays = holidays.US(state='CA', years=range(2018, 2022))
    # On crée une colonne booléenne : Est-ce un jour férié ?
    df['is_holiday'] = df['datetime'].dt.date.apply(lambda x: x in ca_holidays)
    
    # 4. Cyclic Encoding (Pour la continuité mathématique)
    time_float = df['hour'] + df['minute'] / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * time_float / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * time_float / 24.0)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    
    return df

def add_robust_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Création de features 'Safe' pour une prévision à 24h (Day-Ahead).
    On n'utilise QUE des données disponibles la veille.
    """
    logger.info("Adding robust lag features (Day-Ahead safe)...")
    target = 'power_kw'
    
    # 1. Ce qui s'est passé hier à la même heure (Lag 24h)
    # C'est notre ancre principale.
    df['lag_24h'] = df[target].shift(96) # 96 quarts d'heure = 24h
    
    # 2. Ce qui s'est passé il y a une semaine (Lag 7 jours)
    # Pour capturer la saisonnalité hebdo (Lundi vs Dimanche)
    df['lag_1week'] = df[target].shift(96 * 7)
    
    # 3. Moyenne de la journée d'HIER (et pas des dernières 24h glissantes)
    # Astuce : On prend la moyenne glissante décalée de 24h
    # Cela représente "La consommation moyenne d'il y a 24h à 48h"
    # C'est une info connue totalement au moment de prédire pour demain.
    df['avg_energy_yesterday'] = df[target].shift(96).rolling(window=96).mean()
    
    return df

def main():
    if not LOAD_FILE.exists() or not WEATHER_FILE.exists():
        logger.error("Missing input files.")
        return

    logger.info("Loading & Merging...")
    df_load = pd.read_parquet(LOAD_FILE)
    df_weather = pd.read_parquet(WEATHER_FILE)
    
    df = pd.merge(df_load, df_weather, on='datetime', how='inner')
    
    df = add_calendar_features(df)
    df = add_robust_lag_features(df)
    
    # Nettoyage des NaN dus aux lags (7 jours de perdu au début)
    df = df.dropna()
    
    # Sélection des colonnes finales
    # On garde active_chargers juste pour l'analyse, mais on l'enlèvera du X dans le train
    logger.info(f"Saving robust dataset to {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE, index=False)
    
    print("\n--- ✅ Features 'Day-Ahead' Ready ---")
    print(df[['datetime', 'is_holiday', 'lag_24h', 'avg_energy_yesterday']].head())

if __name__ == "__main__":
    main()