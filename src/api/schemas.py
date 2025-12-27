from datetime import datetime

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    date: str = Field(..., description="Target date in YYYY-MM-DD format", example="2025-07-14")
    override_temp: float | None = Field(None, description="Override average temperature (°C)", example=30.5)
    override_sun: float | None = Field(None, description="Override max solar radiation (W/m²)", example=800.0)


class PredictionPoint(BaseModel):
    datetime: datetime
    predicted_power_kw: float
    is_peak_warning: bool


class SimulationResponse(BaseModel):
    date: str
    total_energy_kwh: float
    peak_power_kw: float
    peak_hour: str
    points: list[PredictionPoint]
    message: str
