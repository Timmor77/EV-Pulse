"""EV-Pulse API - Smart Charging Simulation Backend.

This module provides the FastAPI backend for EV charging load prediction.
It exposes endpoints for simulating future power consumption based on
calendar context and weather conditions.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    HealthResponse,
    SimulationRequest,
    SimulationResponse,
)
from src.features.build_features_v3 import CAT_FEATURES, add_context_features

# --- CONFIGURATION ---
MODEL_PATH = Path("src/models/site_models.pkl")
GRID_CAPACITY_LIMIT = 150.0
API_VERSION = "1.0.0"
ml_models: dict[str, Any] = {}

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
        ml_models["bundle"] = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("ERROR: Model not found.")
    yield
    ml_models.clear()


app = FastAPI(
    title="EV-Pulse API",
    description="Day-ahead EV charging load simulator",
    version=API_VERSION,
    lifespan=lifespan,
)

# Enable CORS for dashboard integration
# Note: credentials are not allowed with a wildcard origin (CORS spec)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "EV-Pulse API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """Health check endpoint for monitoring and container orchestration."""
    model_loaded = "bundle" in ml_models
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_climate_defaults(date_ts: pd.Timestamp) -> tuple[float, float]:
    """Return average temperature and solar radiation for a given month."""
    month = date_ts.month
    return CLIMATE_STATS.get(month, (20.0, 600.0))  # Fallback value


def prepare_simulation_data(
    date_str: str,
    temp: float | None = None,
    sun: float | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Prepare simulation data with fallback to climatology.

    Creates a 15-minute resolution DataFrame for a full day with weather
    features and calendar context.

    Args:
        date_str: Target date in YYYY-MM-DD format.
        temp: Override temperature in Celsius. Uses seasonal average if None.
        sun: Override max solar radiation in W/m². Uses seasonal average if None.

    Returns:
        Tuple of (feature_dataframe, datetime_index).
    """
    # 1. Time Structure (96 intervals of 15 minutes per day)
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:45:00"
    dates = pd.date_range(start=start, end=end, freq="15min")
    df = pd.DataFrame({"datetime": dates})

    # 2. Smart Weather Handling with seasonal defaults
    avg_temp, avg_sun = get_climate_defaults(dates[0])

    # Use provided values or fall back to seasonal averages
    sim_temp = temp if temp is not None else avg_temp
    sim_sun = sun if sun is not None else avg_sun

    df["temperature"] = sim_temp
    df["precipitation"] = 0.0
    df["hour"] = df["datetime"].dt.hour

    def simulate_sun_curve(hour: int) -> float:
        """Generate realistic solar radiation curve (bell-shaped during daylight)."""
        if 6 < hour < 18:
            return sim_sun * np.sin(np.pi * (hour - 6) / 12)
        return 0.0

    df["solar_radiation"] = df["hour"].apply(simulate_sun_curve)

    # 3. Apply shared feature engineering pipeline
    df = add_context_features(df)

    # 4. Categorical type conversion for LightGBM
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, dates


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest) -> dict[str, Any]:
    """Run a power consumption simulation for a given date.

    Generates 15-minute resolution predictions for the entire day using
    the trained LightGBM model with calendar and weather context.

    Args:
        request: Simulation parameters including date and optional weather overrides.

    Returns:
        Complete simulation results with time series predictions and summary statistics.

    Raises:
        HTTPException: If model is not loaded or simulation fails.
    """
    if "bundle" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs.",
        )

    try:
        # Prepare features for prediction
        df_features, dates = prepare_simulation_data(
            request.date,
            request.override_temp,
            request.override_sun,
        )

        bundle = ml_models["bundle"]
        method = bundle["methods"][request.site]
        profile_index = pd.MultiIndex.from_frame(df_features[bundle["profile_keys"]])

        if method == "residual_recent":
            profile = bundle["profiles"][request.site]
            fallback = bundle["fallbacks"][request.site]
            recent_level = profile.reindex(profile_index).to_numpy(dtype=float)
            recent_level = np.nan_to_num(recent_level, nan=fallback)
            model = bundle["models"][request.site]
            residual = model.predict(df_features[bundle["features"]])
            predictions = np.maximum(recent_level + residual, 0)
        else:
            profile = bundle["calendar_profiles"][request.site]
            fallback = bundle["calendar_fallbacks"][request.site]
            predictions = profile.reindex(profile_index).to_numpy(dtype=float)
            predictions = np.maximum(np.nan_to_num(predictions, nan=fallback), 0)

        # Build response with predictions and statistics
        points = []
        peak_val = 0.0
        peak_time = ""
        warning_count = 0

        for dt, power in zip(dates, predictions):
            val = round(float(power), 2)
            is_warning = val > GRID_CAPACITY_LIMIT

            if val > peak_val:
                peak_val = val
                peak_time = dt.strftime("%H:%M")

            if is_warning:
                warning_count += 1

            points.append(
                {
                    "datetime": dt,
                    "predicted_power_kw": val,
                    "is_peak_warning": is_warning,
                }
            )

        # Calculate summary statistics
        total_energy = round(sum(predictions) / 4, 2)  # kWh (15min intervals)
        avg_power = round(float(np.mean(predictions)), 2)
        temp_used = df_features["temperature"].iloc[0]
        weather_source = "user-defined" if request.override_temp is not None else "seasonal-average"

        return {
            "date": request.date,
            "site": request.site,
            "method": method,
            "summary": {
                "total_energy_kwh": total_energy,
                "peak_power_kw": round(float(peak_val), 2),
                "average_power_kw": avg_power,
                "peak_hour": peak_time,
                "warning_count": warning_count,
                "grid_capacity_limit_kw": GRID_CAPACITY_LIMIT,
            },
            "weather": {
                "temperature_c": round(float(temp_used), 1),
                "source": weather_source,
            },
            "points": points,
            "message": f"Simulation completed for {request.date} ({weather_source}: {temp_used:.1f}°C)",
        }

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required feature: {e}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid input value: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}",
        )
