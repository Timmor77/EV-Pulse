"""Small tests for the temporal evaluation helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.train_model_v2 import (
    calendar_baseline,
    metrics_by_period,
    split_final_holdout,
)


def test_final_holdout_is_the_last_calendar_days():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame({"datetime": dates, "power_kw": range(10)})

    development, holdout = split_final_holdout(frame, holdout_days=3)

    assert development["datetime"].max() == pd.Timestamp("2025-01-07")
    assert holdout["datetime"].min() == pd.Timestamp("2025-01-08")
    assert development["datetime"].max() < holdout["datetime"].min()
    assert len(holdout) == 3


def test_calendar_baseline_only_uses_training_values():
    train = pd.DataFrame(
        {
            "day_of_week": [0, 0, 1],
            "hour": [8, 8, 9],
            "minute": [0, 0, 0],
            "power_kw": [10.0, 30.0, 50.0],
        }
    )
    test = pd.DataFrame(
        {
            "day_of_week": [0, 6],
            "hour": [8, 23],
            "minute": [0, 45],
            "power_kw": [999.0, 999.0],
        }
    )

    predictions = calendar_baseline(train, test)

    assert predictions.tolist() == [20.0, 30.0]


def test_metrics_are_reported_for_each_month():
    dates = pd.Series(pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02"]))
    actual = pd.Series([5.0, 10.0, 20.0, 30.0])

    rows = metrics_by_period(
        dates,
        actual,
        model_pred=np.array([5.0, 10.0, 19.0, 31.0]),
        baseline_pred=np.array([2.0, 5.0, 10.0, 20.0]),
    )

    assert [row["period"] for row in rows] == ["2025-01", "2025-02"]
    assert [row["rows"] for row in rows] == [2, 2]
    assert rows[1]["model"]["mae_kw"] == 1.0


def test_original_report_matches_its_training_script():
    """The aggregate benchmark remains tied to its unchanged training code."""
    report = json.loads(Path("reports/model_evaluation.json").read_text(encoding="utf-8"))
    script_path = Path(report["source"]["training_script"])
    script_hash = hashlib.sha256(script_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()

    assert report["schema_version"] == 1
    assert not script_path.is_absolute()
    assert report["source"]["training_script_sha256"] == script_hash
    assert len(report["source"]["feature_script_sha256"]) == 64
    assert report["split"]["development"]["end"] < report["split"]["final_holdout"]["start"]
    assert report["final_holdout"]["model"]["mae_kw"] < report["final_holdout"]["baseline"]["mae_kw"]
    assert len(report["final_holdout"]["by_month"]) == 2
    assert report["split"]["incomplete_tail_excluded_from_evaluation"]["rows"] == 93
