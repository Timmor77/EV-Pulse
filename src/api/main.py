from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import SimulationRequest, SimulationResponse
from src.features.build_features_v3 import CAT_FEATURES, add_context_features

# --- CONFIGURATION ---
MODEL_PATH = Path("src/models/lgbm_context_model.pkl")
GRID_CAPACITY_LIMIT = 150.0
ml_models = {}

# --- CLIMATOLOGIE PASADENA (Normales de saison) ---
# Mois -> (Temp Moyenne °C, Ensoleillement Max W/m²)
# Source approximative : NOAA pour Pasadena, CA
CLIMATE_STATS = {
    1: (13.0, 450.0),  # Janvier
    2: (14.0, 500.0),  # Février
    3: (15.5, 600.0),  # Mars
    4: (17.5, 750.0),  # Avril
    5: (19.0, 850.0),  # Mai
    6: (21.0, 950.0),  # Juin
    7: (24.5, 950.0),  # Juillet
    8: (25.0, 900.0),  # Août
    9: (24.0, 800.0),  # Septembre
    10: (21.0, 650.0),  # Octobre
    11: (16.5, 500.0),  # Novembre
    12: (13.5, 400.0),  # Décembre
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        ml_models["lgbm"] = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    else:
        print("❌ ERREUR : Modèle introuvable.")
    yield
    ml_models.clear()


app = FastAPI(title="EV-Pulse API", lifespan=lifespan)


def get_climate_defaults(date_ts):
    """Retourne la température et le soleil moyens pour un mois donné."""
    month = date_ts.month
    return CLIMATE_STATS.get(month, (20.0, 600.0))  # Valeur refuge


def prepare_simulation_data(date_str: str, temp: float = None, sun: float = None):
    """Prépare les données avec fallback sur la climatologie."""

    # 1. Structure Temporelle
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:45:00"
    dates = pd.date_range(start=start, end=end, freq="15T")
    df = pd.DataFrame({"datetime": dates})

    # 2. Gestion Intelligente de la Météo
    # On calcule le mois pour chaque ligne (même si c'est le même jour)
    current_month = dates[0].month
    avg_temp, avg_sun = CLIMATE_STATS.get(current_month, (20.0, 600.0))

    # Logique : Si l'user donne une valeur, on prend. Sinon, on prend la normale de saison.
    sim_temp = temp if temp is not None else avg_temp
    sim_sun = sun if sun is not None else avg_sun

    df["temperature"] = sim_temp
    df["precipitation"] = 0.0

    # Simulation solaire (Cloche ajustée selon la saison)
    # En hiver, le soleil se lève plus tard (7h) et se couche plus tôt (17h)
    # En été, 6h -> 20h. On fait une approximation simple ici.
    df["hour"] = df["datetime"].dt.hour

    def simulate_sun_curve(h):
        # On simplifie : jour levé entre 6h et 18h
        if 6 < h < 18:
            return sim_sun * np.sin(np.pi * (h - 6) / 12)
        return 0

    df["solar_radiation"] = df["hour"].apply(simulate_sun_curve)

    # 3. Feature Engineering Partagé
    df = add_context_features(df)

    # 4. Conversion Types
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df, dates


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    if "lgbm" not in ml_models:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        df_features, dates = prepare_simulation_data(request.date, request.override_temp, request.override_sun)

        model_features = ml_models["lgbm"].feature_name_
        X = df_features[model_features]

        predictions = ml_models["lgbm"].predict(X)
        predictions = np.maximum(predictions, 0)

        points = []
        peak_val = 0.0
        peak_time = ""

        for dt, power in zip(dates, predictions):
            val = round(float(power), 2)
            if val > peak_val:
                peak_val = val
                peak_time = dt.strftime("%H:%M")

            points.append(
                {
                    "datetime": dt,
                    "predicted_power_kw": val,
                    "is_peak_warning": val > GRID_CAPACITY_LIMIT,
                }
            )

        return {
            "date": request.date,
            "total_energy_kwh": round(sum(predictions) / 4, 2),
            "peak_power_kw": round(float(peak_val), 2),
            "peak_hour": peak_time,
            "points": points,
            "message": f"Simulation basée sur {request.date} (Météo: {df_features['temperature'].iloc[0]}°C)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
