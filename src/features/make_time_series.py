import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/acn_data_cleaned.parquet")
OUTPUT_FILE = Path("data/processed/acn_timeseries_15min.parquet")

def create_timeseries_from_sessions(df: pd.DataFrame, interval_min: int = 15) -> pd.DataFrame:
    """
    Transforme une liste de sessions (Start, End, kWh) en une série temporelle continue.
    
    Hypothèse simplificatrice robuste : 
    La puissance est distribuée uniformément sur la durée de la charge (Rectangular assumption).
    C'est suffisant pour de la prédiction agrégée.
    """
    
    # 1. Définir les bornes temporelles globales
    start_date = df['connectionTime'].min().floor('H')
    end_date = df['disconnectTime'].max().ceil('H')
    
    # Création de l'index temporel complet (ex: toutes les 15 min de 2018 à 2021)
    # '15T' est l'alias pandas pour 15 minutes
    freq = f"{interval_min}T" 
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq, tz='UTC')
    
    logger.info(f"Timeline created: {len(time_index)} points from {start_date} to {end_date}")
    
    # 2. Préparation des structures de données (Numpy pour la vitesse)
    # On crée un tableau de zéros de la taille de la timeline
    load_curve = np.zeros(len(time_index))
    occupancy_curve = np.zeros(len(time_index))
    
    # Mapping des dates vers des indices entiers (0, 1, 2, ...)
    # C'est l'astuce pour aller vite : on ne manipule plus des dates mais des index de tableau
    timestamps = time_index.to_numpy()
    
    # On itère sur les sessions (c'est rapide ici car on fait juste des maths simples)
    # Pour un dataset géant, on pourrait paralléliser, mais pour <100k lignes c'est instantané.
    logger.info("Projecting sessions onto timeline...")
    
    count = 0
    total = len(df)
    
    for row in df.itertuples():
        # Calcul de la durée de charge active (en heures)
        # On utilise doneChargingTime car après, la voiture est branchée mais ne charge plus (0 kW)
        charge_start = row.connectionTime
        charge_end = row.doneChargingTime
        
        # Sécurité : si la fin est avant le début (bug data), on skip
        if charge_end <= charge_start:
            continue
            
        duration_hours = (charge_end - charge_start).total_seconds() / 3600
        if duration_hours < (interval_min / 60):
            # Session trop courte, on ignore ou on compte comme un pic
            continue
            
        avg_power_kw = row.kWhDelivered / duration_hours
        
        # Trouver les indices dans notre grand tableau
        # searchsorted est très rapide pour trouver où s'insère une date
        idx_start = np.searchsorted(timestamps, charge_start)
        idx_end_charge = np.searchsorted(timestamps, charge_end)
        idx_end_conn = np.searchsorted(timestamps, row.disconnectTime)
        
        # Remplissage du tableau de charge (Power)
        # On ajoute la puissance moyenne sur toute la durée de la charge
        if idx_end_charge > idx_start:
            load_curve[idx_start:idx_end_charge] += avg_power_kw
            
        # Remplissage du tableau d'occupation (Occupancy)
        # La voiture occupe la borne jusqu'au disconnectTime, même si elle ne charge plus
        if idx_end_conn > idx_start:
            occupancy_curve[idx_start:idx_end_conn] += 1
            
        count += 1
        if count % 10000 == 0:
            logger.info(f"Processed {count}/{total} sessions")

    # 3. Assemblage final
    ts_df = pd.DataFrame({
        'datetime': time_index,
        'power_kw': load_curve,
        'active_chargers': occupancy_curve
    })
    
    # Typage optimal
    ts_df['power_kw'] = ts_df['power_kw'].astype('float32')
    ts_df['active_chargers'] = ts_df['active_chargers'].astype('int32')
    
    return ts_df

def main():
    if not INPUT_FILE.exists():
        logger.error("Input file not found. Run process_data.py first.")
        return

    df = pd.read_parquet(INPUT_FILE)
    
    # Filtrage optionnel : On peut se concentrer sur Caltech pour commencer si on veut
    # df = df[df['source_site'] == 'caltech']
    
    logger.info("Generating Time Series (15 min intervals)...")
    ts_df = create_timeseries_from_sessions(df)
    
    logger.info(f"Saving Time Series to {OUTPUT_FILE}...")
    ts_df.to_parquet(OUTPUT_FILE, index=False)
    
    logger.info("Done! Aperçu des données :")
    print(ts_df.head())
    print(ts_df.describe())

if __name__ == "__main__":
    main()