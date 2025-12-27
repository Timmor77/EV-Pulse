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

# --- PASADENA CLIMATOLOGY (Seasonal Averages) ---
# Month -> (Average Temp °C, Max Solar Radiation W/m²)
# Source: NOAA approximation for Pasadena, CA
CLIMATE_STATS = {
    1: (13.0, 450.0),  # January
    2: (14.0, 500.0),  # February
    3: (15.5, 600.0),  # March
    4: (17.5, 750.0),  # April
    5: (19.0, 850.0),  # May
    6: (21.0, 950.0),  # June
    7: (24.5, 950.0),  # July
    8: (25.0, 900.0),  # August
    9: (24.0, 800.0),  # September
    10: (21.0, 650.0),  # October
    11: (16.5, 500.0),  # November
    12: (13.5, 400.0),  # December
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        ml_models["lgbm"] = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print("❌ ERROR: Model not found.")
    yield
    ml_models.clear()


app = FastAPI(title="EV-Pulse API", lifespan=lifespan)


def get_climate_defaults(date_ts):
    """Return average temperature and solar radiation for a given month."""
    month = date_ts.month
    return CLIMATE_STATS.get(month, (20.0, 600.0))  # Fallback value


def prepare_simulation_data(date_str: str, temp: float = None, sun: float = None):
    """Prepare simulation data with fallback to climatology."""

    # 1. Time Structure
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:45:00"
    dates = pd.date_range(start=start, end=end, freq="15T")
    df = pd.DataFrame({"datetime": dates})

    # 2. Smart Weather Handling
    # Calculate month for each row (even if it's the same day)
    current_month = dates[0].month
    avg_temp, avg_sun = CLIMATE_STATS.get(current_month, (20.0, 600.0))

    # Logic: If user provides a value, use it. Otherwise, use seasonal average.
    sim_temp = temp if temp is not None else avg_temp
    sim_sun = sun if sun is not None else avg_sun

    df["temperature"] = sim_temp
    df["precipitation"] = 0.0

    # Solar simulation (Bell curve adjusted by season)
    # In winter, sun rises later (7am) and sets earlier (5pm)
    # In summer, 6am -> 8pm. Using a simplified approximation here.
    df["hour"] = df["datetime"].dt.hour

    def simulate_sun_curve(h):
        # Simplified: daylight between 6am and 6pm
        if 6 < h < 18:
            return sim_sun * np.sin(np.pi * (h - 6) / 12)
        return 0

    df["solar_radiation"] = df["hour"].apply(simulate_sun_curve)

    # 3. Shared Feature Engineering
    df = add_context_features(df)

    # 4. Type Conversion
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df, dates


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    if "lgbm" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
            "message": f"Simulation based on {request.date} (Weather: {df_features['temperature'].iloc[0]}°C)",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
