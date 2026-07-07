"""Tests for the EV-Pulse API endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class TestAPIEndpoints:
    """Test suite for API endpoints (real model loaded via lifespan)."""

    @pytest.fixture
    def client(self):
        """Create a test client. The lifespan loads the trained model."""
        with TestClient(app) as client:
            yield client

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert data["name"] == "EV-Pulse API"
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint_with_model(self, client):
        """Test health endpoint when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "version" in data
        assert "timestamp" in data

    def test_simulate_endpoint_success(self, client):
        """Test simulation endpoint with valid request."""
        payload = {"date": "2025-07-14"}

        response = client.post("/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()

        # Check response structure
        assert "date" in data
        assert data["date"] == "2025-07-14"
        assert "summary" in data
        assert "weather" in data
        assert "points" in data
        assert "message" in data

        # Check summary fields
        summary = data["summary"]
        assert "total_energy_kwh" in summary
        assert "peak_power_kw" in summary
        assert "average_power_kw" in summary
        assert "peak_hour" in summary
        assert "warning_count" in summary
        assert "grid_capacity_limit_kw" in summary

        # Check weather fields
        weather = data["weather"]
        assert "temperature_c" in weather
        assert "source" in weather
        assert weather["source"] == "seasonal-average"

        # Check points structure
        assert len(data["points"]) == 96  # 24h * 4 intervals per hour
        point = data["points"][0]
        assert "datetime" in point
        assert "predicted_power_kw" in point
        assert "is_peak_warning" in point

    def test_simulate_with_weather_override(self, client):
        """Test simulation with custom weather parameters."""
        payload = {
            "date": "2025-07-14",
            "override_temp": 35.0,
            "override_sun": 900.0,
        }

        response = client.post("/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["weather"]["source"] == "user-defined"
        assert data["weather"]["temperature_c"] == 35.0

    def test_simulate_with_zero_temperature(self, client):
        """Test that a 0°C override is still treated as user-defined."""
        payload = {"date": "2025-01-06", "override_temp": 0.0}

        response = client.post("/simulate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["weather"]["source"] == "user-defined"
        assert data["weather"]["temperature_c"] == 0.0

    def test_simulate_invalid_date_format(self, client):
        """Test simulation with invalid date format."""
        payload = {"date": "invalid-date"}

        response = client.post("/simulate", json=payload)
        assert response.status_code == 422  # Rejected by schema validation

    def test_simulate_impossible_date(self, client):
        """Test simulation with a well-formed but impossible date."""
        payload = {"date": "2025-02-30"}

        response = client.post("/simulate", json=payload)
        assert response.status_code == 422

    def test_simulate_missing_date(self, client):
        """Test simulation with missing required date field."""
        payload = {}

        response = client.post("/simulate", json=payload)
        assert response.status_code == 422  # Validation error


class TestAPIWithoutModel:
    """Test API behavior when model is not loaded."""

    @pytest.fixture
    def client_no_model(self):
        """Create a test client where the model file cannot be found."""
        with patch("src.api.main.MODEL_PATH", Path("nonexistent_model.pkl")):
            with TestClient(app) as client:
                yield client

    def test_health_degraded_without_model(self, client_no_model):
        """Test health endpoint shows degraded status without model."""
        response = client_no_model.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False

    def test_simulate_fails_without_model(self, client_no_model):
        """Test simulation fails gracefully without model."""
        payload = {"date": "2025-07-14"}

        response = client_no_model.post("/simulate", json=payload)
        assert response.status_code == 503

        data = response.json()
        assert "detail" in data
        assert "not loaded" in data["detail"].lower()
