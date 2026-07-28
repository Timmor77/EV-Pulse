"""Tests for the recent same-slot profile."""

import pandas as pd

from src.models.train_site_models import (
    RECENT_FEATURE,
    add_causal_recent_feature,
    apply_recent_profile,
)


def _weekly_frame(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-06 10:00:00", periods=len(values), freq="7D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "day_of_week": 0,
            "hour": 10,
            "minute": 0,
            "power_kw": values,
        }
    )


def test_causal_feature_never_uses_the_current_target():
    frame = _weekly_frame([10.0, 20.0, 999.0])

    result = add_causal_recent_feature(frame)

    assert result[RECENT_FEATURE].iloc[1] == 10.0
    assert result[RECENT_FEATURE].iloc[2] == 15.0


def test_test_targets_do_not_change_the_training_profile():
    train = _weekly_frame([10.0, 20.0, 30.0])
    test = _weekly_frame([1.0, 9999.0]).assign(datetime=pd.to_datetime(["2025-02-03 10:00:00", "2025-02-10 10:00:00"]))

    first = apply_recent_profile(train, test)
    second = apply_recent_profile(train, test.assign(power_kw=[500.0, 600.0]))

    assert first[RECENT_FEATURE].tolist() == [20.0, 20.0]
    assert second[RECENT_FEATURE].tolist() == [20.0, 20.0]
