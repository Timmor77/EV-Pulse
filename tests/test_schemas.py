"""Tests for Pydantic schemas validation."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    HealthResponse,
    PredictionPoint,
    SimulationRequest,
    SimulationSummary,
    WeatherInfo,
)


class TestSimulationRequest:
    """Test suite for SimulationRequest schema."""

    def test_valid_request_date_only(self):
        """Test valid request with only date."""
        request = SimulationRequest(date="2025-07-14")
        assert request.date == "2025-07-14"
        assert request.site == "caltech"
        assert request.override_temp is None
        assert request.override_sun is None

    def test_valid_request_with_overrides(self):
        """Test valid request with weather overrides."""
        request = SimulationRequest(
            date="2025-07-14",
            override_temp=30.5,
            override_sun=800.0,
        )
        assert request.date == "2025-07-14"
        assert request.override_temp == 30.5
        assert request.override_sun == 800.0

    def test_valid_site(self):
        request = SimulationRequest(date="2025-07-14", site="jpl")
        assert request.site == "jpl"

    def test_unknown_site_fails(self):
        with pytest.raises(ValidationError):
            SimulationRequest(date="2025-07-14", site="unknown")

    def test_missing_date_fails(self):
        """Test that missing date raises validation error."""
        with pytest.raises(ValidationError):
            SimulationRequest()

    def test_malformed_date_fails(self):
        """Test that a date not matching YYYY-MM-DD is rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(date="14/07/2025")

        with pytest.raises(ValidationError):
            SimulationRequest(date="tomorrow")

    def test_temperature_range_validation(self):
        """Test temperature validation bounds."""
        # Valid range
        request = SimulationRequest(date="2025-07-14", override_temp=-10.0)
        assert request.override_temp == -10.0

        request = SimulationRequest(date="2025-07-14", override_temp=50.0)
        assert request.override_temp == 50.0

        # Below minimum
        with pytest.raises(ValidationError):
            SimulationRequest(date="2025-07-14", override_temp=-60.0)

        # Above maximum
        with pytest.raises(ValidationError):
            SimulationRequest(date="2025-07-14", override_temp=70.0)

    def test_solar_radiation_range_validation(self):
        """Test solar radiation validation bounds."""
        # Valid range
        request = SimulationRequest(date="2025-07-14", override_sun=0.0)
        assert request.override_sun == 0.0

        request = SimulationRequest(date="2025-07-14", override_sun=1200.0)
        assert request.override_sun == 1200.0

        # Below minimum
        with pytest.raises(ValidationError):
            SimulationRequest(date="2025-07-14", override_sun=-100.0)

        # Above maximum
        with pytest.raises(ValidationError):
            SimulationRequest(date="2025-07-14", override_sun=2000.0)


class TestPredictionPoint:
    """Test suite for PredictionPoint schema."""

    def test_valid_prediction_point(self):
        """Test valid prediction point."""
        point = PredictionPoint(
            datetime="2025-07-14T10:00:00",
            predicted_power_kw=75.5,
            is_peak_warning=False,
        )
        assert point.predicted_power_kw == 75.5
        assert point.is_peak_warning is False

    def test_prediction_point_with_warning(self):
        """Test prediction point with peak warning."""
        point = PredictionPoint(
            datetime="2025-07-14T17:30:00",
            predicted_power_kw=165.0,
            is_peak_warning=True,
        )
        assert point.is_peak_warning is True


class TestSimulationSummary:
    """Test suite for SimulationSummary schema."""

    def test_valid_summary(self):
        """Test valid simulation summary."""
        summary = SimulationSummary(
            total_energy_kwh=1250.5,
            peak_power_kw=145.2,
            average_power_kw=52.1,
            peak_hour="17:30",
            warning_count=0,
            grid_capacity_limit_kw=150.0,
        )
        assert summary.total_energy_kwh == 1250.5
        assert summary.peak_power_kw == 145.2
        assert summary.warning_count == 0


class TestWeatherInfo:
    """Test suite for WeatherInfo schema."""

    def test_seasonal_weather(self):
        """Test seasonal average weather info."""
        weather = WeatherInfo(
            temperature_c=24.5,
            source="seasonal-average",
        )
        assert weather.source == "seasonal-average"

    def test_user_defined_weather(self):
        """Test user-defined weather info."""
        weather = WeatherInfo(
            temperature_c=35.0,
            source="user-defined",
        )
        assert weather.source == "user-defined"


class TestHealthResponse:
    """Test suite for HealthResponse schema."""

    def test_healthy_response(self):
        """Test healthy status response."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0",
            timestamp="2025-07-14T10:00:00",
        )
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_degraded_response(self):
        """Test degraded status response."""
        response = HealthResponse(
            status="degraded",
            model_loaded=False,
            version="1.0.0",
            timestamp="2025-07-14T10:00:00",
        )
        assert response.status == "degraded"
        assert response.model_loaded is False
