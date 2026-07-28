"""Tests for the session-to-time-series conversion."""

import pandas as pd

from src.features.make_time_series import create_site_timeseries_from_sessions


def test_site_loads_are_not_aggregated_together():
    sessions = pd.DataFrame(
        {
            "source_site": ["caltech", "jpl"],
            "connectionTime": pd.to_datetime(
                ["2025-01-01 00:00:00", "2025-01-01 00:00:00"],
                utc=True,
            ),
            "doneChargingTime": pd.to_datetime(
                ["2025-01-01 01:00:00", "2025-01-01 01:00:00"],
                utc=True,
            ),
            "disconnectTime": pd.to_datetime(
                ["2025-01-01 01:00:00", "2025-01-01 01:00:00"],
                utc=True,
            ),
            "kWhDelivered": [10.0, 20.0],
        }
    )

    result = create_site_timeseries_from_sessions(sessions)
    first_slot = result[result["datetime"] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")]

    assert set(result["source_site"]) == {"caltech", "jpl"}
    assert first_slot.set_index("source_site")["power_kw"].to_dict() == {
        "caltech": 10.0,
        "jpl": 20.0,
    }
