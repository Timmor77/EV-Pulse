"""Train and evaluate the context-only EV charging forecast model.

The last 60 days are kept aside as a final temporal holdout. Cross-validation
and early stopping only use the older development period.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lgbm_power")

INPUT_FILE = Path("data/processed/model_context.parquet")
MODEL_PATH = Path("src/models/lgbm_context_model.pkl")
RESULTS_PATH = Path("reports/model_evaluation.json")

TARGET = "power_kw"
DROP_COLS = ["datetime", TARGET]
CAT_FEATURES = [
    "day_of_week",
    "month",
    "hour",
    "is_business_time",
    "is_holiday",
    "is_weekend",
    "is_active_hour",
]
BASELINE_KEYS = ["day_of_week", "hour", "minute"]

HOLDOUT_DAYS = 60
N_CV_SPLITS = 5
VALIDATION_FRACTION = 0.15
EARLY_STOPPING_ROUNDS = 150

LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 40,
    "random_state": 42,
    # One thread is slower but avoids platform-dependent OpenMP crashes and
    # makes the saved evaluation easier to reproduce on a laptop or in CI.
    "n_jobs": 1,
    "verbosity": -1,
}


def print_header(title: str) -> None:
    """Print a small readable section header."""
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


def regression_report(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the three regression metrics used throughout the report."""
    safe_pred = np.maximum(y_pred, 0)
    return {
        "mae_kw": float(mean_absolute_error(y_true, safe_pred)),
        "rmse_kw": float(np.sqrt(mean_squared_error(y_true, safe_pred))),
        "r2": float(r2_score(y_true, safe_pred)),
    }


def split_final_holdout(
    df: pd.DataFrame,
    holdout_days: int = HOLDOUT_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the last complete calendar days without shuffling."""
    if holdout_days < 1:
        raise ValueError("holdout_days must be at least 1")

    ordered = df.sort_values("datetime").reset_index(drop=True)
    intervals = ordered["datetime"].diff().dropna()
    interval = intervals[intervals > pd.Timedelta(0)].median()
    if pd.isna(interval) or interval > pd.Timedelta(days=1):
        raise ValueError("Cannot determine the dataset time interval")

    expected_rows = int(pd.Timedelta(days=1) / interval)
    day_values = ordered["datetime"].dt.normalize()
    last_complete_day = None
    for day in reversed(day_values.unique()):
        timestamps = ordered.loc[day_values == day, "datetime"]
        expected_end = day + pd.Timedelta(days=1) - interval
        if len(timestamps) == expected_rows and timestamps.min() == day and timestamps.max() == expected_end:
            last_complete_day = day
            break

    if last_complete_day is None:
        raise ValueError("No complete calendar day found")

    holdout_start = last_complete_day - pd.Timedelta(days=holdout_days - 1)
    holdout_end = last_complete_day + pd.Timedelta(days=1)
    development = ordered[ordered["datetime"] < holdout_start].copy()
    holdout = ordered[(ordered["datetime"] >= holdout_start) & (ordered["datetime"] < holdout_end)].copy()

    if development.empty or holdout.empty:
        raise ValueError("Not enough data for the requested temporal holdout")
    return development, holdout


def calendar_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Predict the historical mean for the same weekday and time slot.

    This deliberately simple baseline only learns from the training rows. If a
    calendar slot has never been seen, it falls back to the training mean.
    """
    missing = [column for column in [*BASELINE_KEYS, TARGET] if column not in train_df]
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")

    slot_means = train_df.groupby(BASELINE_KEYS, observed=True)[TARGET].mean()
    test_index = pd.MultiIndex.from_frame(test_df[BASELINE_KEYS])
    predictions = slot_means.reindex(test_index).to_numpy(dtype=float)
    return np.nan_to_num(predictions, nan=float(train_df[TARGET].mean()))


def metrics_by_period(
    datetimes: pd.Series,
    y_true: pd.Series,
    model_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> list[dict[str, object]]:
    """Calculate model and baseline metrics month by month."""
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(datetimes).to_numpy(),
            "actual": np.asarray(y_true),
            "model": np.maximum(model_pred, 0),
            "baseline": np.maximum(baseline_pred, 0),
        }
    )
    frame["period"] = frame["datetime"].dt.strftime("%Y-%m")

    rows: list[dict[str, object]] = []
    for period, group in frame.groupby("period", sort=True):
        month = pd.Period(period, freq="M")
        expected_start = month.start_time
        expected_end = month.end_time.floor("15min")
        period_start = group["datetime"].min()
        period_end = group["datetime"].max()
        if period_start.tzinfo is not None:
            expected_start = expected_start.tz_localize(period_start.tzinfo)
            expected_end = expected_end.tz_localize(period_start.tzinfo)
        rows.append(
            {
                "period": period,
                "rows": int(len(group)),
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
                "complete_month": bool(period_start == expected_start and period_end == expected_end),
                "model": regression_report(group["actual"], group["model"].to_numpy()),
                "baseline": regression_report(group["actual"], group["baseline"].to_numpy()),
            }
        )
    return rows


def fit_with_inner_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
) -> tuple[lgb.LGBMRegressor, int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Select tree count on the tail of training, then refit on all training."""
    validation_size = max(1, int(len(X_train) * VALIDATION_FRACTION))
    if validation_size >= len(X_train):
        raise ValueError("Training fold is too small for inner validation")

    split_at = len(X_train) - validation_size
    selector = lgb.LGBMRegressor(**LGBM_PARAMS, importance_type="gain")
    selector.fit(
        X_train.iloc[:split_at],
        y_train.iloc[:split_at],
        eval_set=[(X_train.iloc[split_at:], y_train.iloc[split_at:])],
        eval_metric="mae",
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = int(selector.best_iteration_ or LGBM_PARAMS["n_estimators"])

    model_params = {**LGBM_PARAMS, "n_estimators": best_iteration}
    model = lgb.LGBMRegressor(**model_params, importance_type="gain")
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    validation_range = (
        pd.Timestamp(X_train.index[split_at]),
        pd.Timestamp(X_train.index[-1]),
    )
    return model, best_iteration, validation_range


def _date_range(df: pd.DataFrame) -> dict[str, str]:
    return {
        "start": pd.Timestamp(df["datetime"].min()).isoformat(),
        "end": pd.Timestamp(df["datetime"].max()).isoformat(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash text sources independently from the platform's line endings."""
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _summary(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    metrics = ["mae_kw", "rmse_kw", "r2"]
    return {
        metric: {
            "mean": float(np.mean([row[key][metric] for row in rows])),
            "std": float(np.std([row[key][metric] for row in rows], ddof=1)),
        }
        for metric in metrics
    }


def train_and_evaluate() -> dict[str, object]:
    """Run temporal evaluation, save its JSON report, then train the served model."""
    print_header("LightGBM - context-only power forecast")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE).sort_values("datetime").reset_index(drop=True)
    development, holdout = split_final_holdout(df)

    existing_cat = [column for column in CAT_FEATURES if column in df.columns]
    for column in existing_cat:
        df[column] = df[column].astype("category")
        development[column] = development[column].astype("category")
        holdout[column] = holdout[column].astype("category")

    features = [column for column in df.columns if column not in DROP_COLS]
    development = development.set_index("datetime", drop=False)
    holdout = holdout.set_index("datetime", drop=False)
    X_dev, y_dev = development[features], development[TARGET]

    logger.info(
        "Rows: %d development + %d final holdout | holdout starts %s",
        len(development),
        len(holdout),
        holdout["datetime"].min(),
    )

    fold_rows: list[dict[str, object]] = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=N_CV_SPLITS).split(X_dev), start=1):
        train_df = development.iloc[train_idx]
        test_df = development.iloc[test_idx]
        model, best_iteration, validation_range = fit_with_inner_validation(
            train_df[features],
            train_df[TARGET],
            existing_cat,
        )
        model_pred = np.maximum(model.predict(test_df[features]), 0)
        baseline_pred = calendar_baseline(train_df, test_df)
        fold_rows.append(
            {
                "fold": fold,
                "train": _date_range(train_df),
                "inner_validation": {
                    "start": validation_range[0].isoformat(),
                    "end": validation_range[1].isoformat(),
                },
                "test": _date_range(test_df),
                "rows": int(len(test_df)),
                "best_iteration": best_iteration,
                "model": regression_report(test_df[TARGET], model_pred),
                "baseline": regression_report(test_df[TARGET], baseline_pred),
            }
        )

    print_header("Development cross-validation")
    cv_table = pd.DataFrame(
        [
            {
                "fold": row["fold"],
                "model_MAE": row["model"]["mae_kw"],
                "baseline_MAE": row["baseline"]["mae_kw"],
                "best_iter": row["best_iteration"],
            }
            for row in fold_rows
        ]
    )
    print(cv_table.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))

    evaluation_model, best_iteration, validation_range = fit_with_inner_validation(X_dev, y_dev, existing_cat)
    X_holdout = holdout[features]
    model_pred = np.maximum(evaluation_model.predict(X_holdout), 0)
    baseline_pred = calendar_baseline(development, holdout)
    model_metrics = regression_report(holdout[TARGET], model_pred)
    baseline_metrics = regression_report(holdout[TARGET], baseline_pred)
    improvement = 100 * (baseline_metrics["mae_kw"] - model_metrics["mae_kw"]) / baseline_metrics["mae_kw"]

    gain_importance = sorted(
        zip(features, evaluation_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    results: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce": "uv run python -m src.models.train_model_v2",
        "source": {
            "training_script": Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "training_script_sha256": _source_sha256(Path(__file__)),
            "feature_script": Path("src/features/build_features_v3.py").as_posix(),
            "feature_script_sha256": _source_sha256(Path("src/features/build_features_v3.py")),
            "uv_lock_sha256": _sha256(Path("uv.lock")),
            "versions": {
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "dataset": {
            "path": INPUT_FILE.as_posix(),
            "sha256": _sha256(INPUT_FILE),
            "rows": int(len(df)),
            **_date_range(df),
        },
        "method": {
            "target": TARGET,
            "features": features,
            "final_holdout_days": HOLDOUT_DAYS,
            "cv_splits": N_CV_SPLITS,
            "inner_validation_fraction": VALIDATION_FRACTION,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "lightgbm_parameters": LGBM_PARAMS,
            "categorical_features": existing_cat,
            "baseline": "mean power by weekday, hour and minute; global train mean fallback",
        },
        "split": {
            "development": {"rows": int(len(development)), **_date_range(development)},
            "final_holdout": {"rows": int(len(holdout)), **_date_range(holdout)},
            "incomplete_tail_excluded_from_evaluation": {
                "rows": int((df["datetime"] > holdout["datetime"].max()).sum()),
                "reason": "The final source day does not contain all 15-minute intervals.",
            },
        },
        "cross_validation": {
            "folds": fold_rows,
            "model_summary": _summary(fold_rows, "model"),
            "baseline_summary": _summary(fold_rows, "baseline"),
        },
        "final_holdout": {
            "selected_iteration": best_iteration,
            "inner_validation": {
                "start": validation_range[0].isoformat(),
                "end": validation_range[1].isoformat(),
            },
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_vs_baseline_pct": float(improvement),
            "by_month": metrics_by_period(
                holdout["datetime"],
                holdout[TARGET],
                model_pred,
                baseline_pred,
            ),
        },
        "feature_importance_gain": [
            {"feature": feature, "importance": int(importance)} for feature, importance in gain_importance
        ],
        "served_model": {
            "path": MODEL_PATH.as_posix(),
            "trained_on_rows": int(len(df)),
            "n_estimators": best_iteration,
            "note": "Refit on all rows after the final holdout evaluation.",
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    final_params = {**LGBM_PARAMS, "n_estimators": best_iteration}
    final_model = lgb.LGBMRegressor(**final_params, importance_type="gain")
    final_model.fit(df[features], df[TARGET], categorical_feature=existing_cat)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print_header("Final untouched holdout")
    print(f"Model MAE:    {model_metrics['mae_kw']:.3f} kW")
    print(f"Baseline MAE: {baseline_metrics['mae_kw']:.3f} kW")
    print(f"Improvement:  {improvement:.1f}%")
    logger.info("Results saved to %s", RESULTS_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return results


if __name__ == "__main__":
    train_and_evaluate()
