"""Pydantic schemas for EV-Pulse API request/response validation.

This module defines the data models used for API input validation
and response serialization.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Request model for power simulation endpoint."""

    date: str = Field(
        ...,
        description="Target date in YYYY-MM-DD format",
        json_schema_extra={"example": "2025-07-14"},
    )
    override_temp: float | None = Field(
        None,
        description="Override average temperature (°C). Uses seasonal average if not provided.",
        json_schema_extra={"example": 30.5},
        ge=-50,
        le=60,
    )
    override_sun: float | None = Field(
        None,
        description="Override max solar radiation (W/m²). Uses seasonal average if not provided.",
        json_schema_extra={"example": 800.0},
        ge=0,
        le=1500,
    )


class PredictionPoint(BaseModel):
    """Individual prediction point for a 15-minute interval."""

    datetime: datetime
    predicted_power_kw: float = Field(..., description="Predicted power consumption in kW")
    is_peak_warning: bool = Field(..., description="True if power exceeds grid capacity limit")


class SimulationSummary(BaseModel):
    """Summary statistics for the simulation."""

    total_energy_kwh: float = Field(..., description="Total energy consumption for the day in kWh")
    peak_power_kw: float = Field(..., description="Maximum power consumption in kW")
    average_power_kw: float = Field(..., description="Average power consumption in kW")
    peak_hour: str = Field(..., description="Time of peak power consumption (HH:MM)")
    warning_count: int = Field(..., description="Number of intervals exceeding grid capacity")
    grid_capacity_limit_kw: float = Field(..., description="Grid capacity threshold for warnings")


class WeatherInfo(BaseModel):
    """Weather information used in the simulation."""

    temperature_c: float = Field(..., description="Temperature used in simulation (°C)")
    source: str = Field(..., description="Weather data source: 'user-defined' or 'seasonal-average'")


class SimulationResponse(BaseModel):
    """Response model for power simulation endpoint."""

    date: str = Field(..., description="Simulation target date")
    summary: SimulationSummary = Field(..., description="Aggregated statistics")
    weather: WeatherInfo = Field(..., description="Weather conditions used")
    points: list[PredictionPoint] = Field(..., description="Time series predictions (96 points)")
    message: str = Field(..., description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2025-07-14",
                "summary": {
                    "total_energy_kwh": 1250.5,
                    "peak_power_kw": 145.2,
                    "average_power_kw": 52.1,
                    "peak_hour": "17:30",
                    "warning_count": 0,
                    "grid_capacity_limit_kw": 150.0,
                },
                "weather": {"temperature_c": 24.5, "source": "seasonal-average"},
                "points": [{"datetime": "2025-07-14T00:00:00", "predicted_power_kw": 5.2, "is_peak_warning": False}],
                "message": "Simulation completed for 2025-07-14 (seasonal-average: 24.5°C)",
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Service status: 'healthy' or 'degraded'")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Health check timestamp (ISO format)")
