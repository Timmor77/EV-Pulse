"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Sample data fixture for testing."""
    return {
        "timestamp": "2024-01-01 12:00:00",
        "station_id": "station_001",
        "power_kw": 50.0
    }
