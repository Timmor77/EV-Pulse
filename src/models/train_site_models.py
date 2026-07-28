"""Train one EV charging forecast model per site.

Each site model uses calendar/weather context plus a simple recent-level
feature: the mean of the last eight observed weeks for the same weekday and
15-minute slot. Evaluation profiles are always fitted on the training side of
the temporal split.
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
from sklearn.model_selection import TimeSeriesSplit

from src.features.build_features_v3 import CAT_FEATURES
from src.models.train_model_v2 import (
    EARLY_STOPPING_ROUNDS,
    LGBM_PARAMS,
    VALIDATION_FRACTION,
    calendar_baseline,
    regression_report,
    split_final_holdout,
)

logger = logging.getLogger("site_models")

INPUT_FILE = Path("data/processed/model_site_context.parquet")
MODEL_PATH = Path("src/models/site_models.pkl")
RESULTS_PATH = Path("reports/site_model_evaluation.json")

TARGET = "power_kw"
RECENT_FEATURE = "recent_slot_mean_kw"
RECENT_WEEKS = 8
PROFILE_KEYS = ["day_of_week", "hour", "minute"]
DROP_COLS = ["datetime", "source_site", TARGET]
N_CV_SPLITS = 3
RESIDUAL_PARAMS = {
    **LGBM_PARAMS,
    "n_estimators": 1000,
    "num_leaves": 20,
    "min_child_samples": 50,
    "reg_lambda": 1.0,
    "force_col_wise": True,
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


def _date_range(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": pd.Timestamp(frame["datetime"].min()).isoformat(),
        "end": pd.Timestamp(frame["datetime"].max()).isoformat(),
    }


def add_causal_recent_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a rolling same-slot mean using earlier target values only."""
    result = frame.sort_index().copy() if frame.index.name == "datetime" else frame.sort_values("datetime").copy()
    recent = result.groupby(PROFILE_KEYS, observed=True)[TARGET].transform(
        lambda values: values.shift(1).rolling(RECENT_WEEKS, min_periods=1).mean()
    )
    global_past_mean = result[TARGET].shift(1).expanding().mean()
    result[RECENT_FEATURE] = recent.fillna(global_past_mean).fillna(0.0)
    return result


def make_recent_profile(frame: pd.DataFrame) -> tuple[pd.Series, float]:
    """Build the latest eight-week profile available at a training cutoff."""
    ordered = frame.sort_index() if frame.index.name == "datetime" else frame.sort_values("datetime")
    recent_rows = ordered.groupby(PROFILE_KEYS, observed=True, group_keys=False).tail(RECENT_WEEKS)
    profile = recent_rows.groupby(PROFILE_KEYS, observed=True)[TARGET].mean()
    return profile, float(ordered[TARGET].mean())


def apply_recent_profile(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    """Apply a profile learned only from train_frame to later rows."""
    result = test_frame.copy()
    profile, fallback = make_recent_profile(train_frame)
    index = pd.MultiIndex.from_frame(result[PROFILE_KEYS])
    values = profile.reindex(index).to_numpy(dtype=float)
    result[RECENT_FEATURE] = np.nan_to_num(values, nan=fallback)
    return result


def fit_residual_model(
    train_frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> tuple[lgb.LGBMRegressor, pd.Series, float, int, dict[str, str]]:
    """Fit a model that corrects the fixed recent-profile prediction."""
    validation_size = max(1, int(len(train_frame) * VALIDATION_FRACTION))
    split_at = len(train_frame) - validation_size
    fit_frame = train_frame.iloc[:split_at]
    validation_frame = train_frame.iloc[split_at:]

    fit_augmented = add_causal_recent_feature(fit_frame)
    validation_augmented = apply_recent_profile(fit_frame, validation_frame)

    fit_residual = fit_augmented[TARGET] - fit_augmented[RECENT_FEATURE]
    validation_residual = validation_augmented[TARGET] - validation_augmented[RECENT_FEATURE]

    selector = lgb.LGBMRegressor(**RESIDUAL_PARAMS, importance_type="gain")
    selector.fit(
        fit_augmented[features],
        fit_residual,
        eval_set=[(validation_augmented[features], validation_residual)],
        eval_metric="mae",
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = int(selector.best_iteration_ or RESIDUAL_PARAMS["n_estimators"])

    full_augmented = add_causal_recent_feature(train_frame)
    full_residual = full_augmented[TARGET] - full_augmented[RECENT_FEATURE]
    params = {**RESIDUAL_PARAMS, "n_estimators": best_iteration}
    model = lgb.LGBMRegressor(**params, importance_type="gain")
    model.fit(
        full_augmented[features],
        full_residual,
        categorical_feature=categorical_features,
    )
    profile, fallback = make_recent_profile(train_frame)
    validation_range = _date_range(validation_frame)
    return model, profile, fallback, best_iteration, validation_range


def _predict_residual(
    model: lgb.LGBMRegressor,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    features: list[str],
) -> np.ndarray:
    augmented = apply_recent_profile(train_frame, test_frame)
    return np.maximum(augmented[RECENT_FEATURE].to_numpy() + model.predict(augmented[features]), 0)


def _monthly_metrics(
    frame: pd.DataFrame,
    residual_pred: np.ndarray,
    selected_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> list[dict[str, object]]:
    values = pd.DataFrame(
        {
            "datetime": frame["datetime"].to_numpy(),
            "actual": frame[TARGET].to_numpy(),
            "residual": residual_pred,
            "selected": selected_pred,
            "baseline": baseline_pred,
        }
    )
    values["period"] = pd.to_datetime(values["datetime"]).dt.strftime("%Y-%m")
    rows = []
    for period, group in values.groupby("period", sort=True):
        rows.append(
            {
                "period": period,
                "rows": int(len(group)),
                "residual_model": regression_report(group["actual"], group["residual"].to_numpy()),
                "selected_method": regression_report(group["actual"], group["selected"].to_numpy()),
                "baseline": regression_report(group["actual"], group["baseline"].to_numpy()),
            }
        )
    return rows


def _fit_final_model(
    full_frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
    n_estimators: int,
) -> tuple[lgb.LGBMRegressor, pd.Series, float]:
    augmented = add_causal_recent_feature(full_frame)
    residual = augmented[TARGET] - augmented[RECENT_FEATURE]
    params = {**RESIDUAL_PARAMS, "n_estimators": n_estimators}
    model = lgb.LGBMRegressor(**params, importance_type="gain")
    model.fit(augmented[features], residual, categorical_feature=categorical_features)
    profile, fallback = make_recent_profile(full_frame)
    return model, profile, fallback


def train_and_evaluate_sites() -> dict[str, object]:
    """Evaluate site models and save the bundle used by the API."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    data = pd.read_parquet(INPUT_FILE).sort_values(["source_site", "datetime"]).reset_index(drop=True)
    sites = sorted(data["source_site"].unique())
    logger.info("Training independent models for: %s", ", ".join(sites))

    base_features = [column for column in data.columns if column not in DROP_COLS]
    categorical_features = [column for column in CAT_FEATURES if column in base_features]

    site_results: dict[str, object] = {}
    served_models = {}
    served_profiles = {}
    served_fallbacks = {}
    served_calendar_profiles = {}
    served_calendar_fallbacks = {}
    served_methods = {}
    aggregate_actual = []
    aggregate_residual = []
    aggregate_selected = []
    aggregate_baseline = []

    for site in sites:
        site_frame = data[data["source_site"] == site].copy()
        for column in categorical_features:
            site_frame[column] = site_frame[column].astype("category")
        development, holdout = split_final_holdout(site_frame)
        development = development.set_index("datetime", drop=False)
        holdout = holdout.set_index("datetime", drop=False)

        folds = []
        splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        for fold, (train_index, test_index) in enumerate(splitter.split(development), start=1):
            train_frame = development.iloc[train_index]
            test_frame = development.iloc[test_index]

            residual_model, _, _, residual_iteration, _ = fit_residual_model(
                train_frame,
                base_features,
                categorical_features,
            )

            residual_pred = _predict_residual(residual_model, train_frame, test_frame, base_features)
            baseline_pred = calendar_baseline(train_frame, test_frame)
            folds.append(
                {
                    "fold": fold,
                    "train": _date_range(train_frame),
                    "test": _date_range(test_frame),
                    "rows": int(len(test_frame)),
                    "residual_iteration": residual_iteration,
                    "residual_model": regression_report(test_frame[TARGET], residual_pred),
                    "baseline": regression_report(test_frame[TARGET], baseline_pred),
                }
            )

        residual_model, _, _, residual_iteration, residual_validation = fit_residual_model(
            development,
            base_features,
            categorical_features,
        )

        residual_pred = _predict_residual(residual_model, development, holdout, base_features)
        baseline_pred = calendar_baseline(development, holdout)

        residual_metrics = regression_report(holdout[TARGET], residual_pred)
        baseline_metrics = regression_report(holdout[TARGET], baseline_pred)
        mean_cv_residual = float(np.mean([fold["residual_model"]["mae_kw"] for fold in folds]))
        mean_cv_baseline = float(np.mean([fold["baseline"]["mae_kw"] for fold in folds]))
        selected_method = "residual_recent" if mean_cv_residual < mean_cv_baseline else "calendar_baseline"
        selected_pred = residual_pred if selected_method == "residual_recent" else baseline_pred
        selected_metrics = regression_report(holdout[TARGET], selected_pred)

        full_frame = site_frame.set_index("datetime", drop=False)
        final_model, final_profile, final_fallback = _fit_final_model(
            full_frame,
            base_features,
            categorical_features,
            residual_iteration,
        )
        served_models[site] = final_model
        served_profiles[site] = final_profile
        served_fallbacks[site] = final_fallback
        calendar_profile = full_frame.groupby(PROFILE_KEYS, observed=True)[TARGET].mean()
        served_calendar_profiles[site] = calendar_profile
        served_calendar_fallbacks[site] = float(full_frame[TARGET].mean())
        served_methods[site] = selected_method

        site_results[site] = {
            "rows": int(len(site_frame)),
            "development": {"rows": int(len(development)), **_date_range(development)},
            "final_holdout": {
                "rows": int(len(holdout)),
                **_date_range(holdout),
                "residual_iteration": residual_iteration,
                "residual_inner_validation": residual_validation,
                "residual_model": residual_metrics,
                "baseline": baseline_metrics,
                "selection": {
                    "method": selected_method,
                    "development_mean_mae_kw": {
                        "residual_model": mean_cv_residual,
                        "baseline": mean_cv_baseline,
                    },
                },
                "selected_method_metrics": selected_metrics,
                "selected_mae_improvement_vs_baseline_pct": float(
                    100 * (baseline_metrics["mae_kw"] - selected_metrics["mae_kw"]) / baseline_metrics["mae_kw"]
                ),
                "by_month": _monthly_metrics(
                    holdout,
                    residual_pred,
                    selected_pred,
                    baseline_pred,
                ),
            },
            "cross_validation": folds,
        }

        aggregate_actual.append(holdout[TARGET].to_numpy())
        aggregate_residual.append(residual_pred)
        aggregate_selected.append(selected_pred)
        aggregate_baseline.append(baseline_pred)

        logger.info(
            "%s holdout MAE | residual %.3f | selected %.3f | baseline %.3f",
            site,
            residual_metrics["mae_kw"],
            selected_metrics["mae_kw"],
            baseline_metrics["mae_kw"],
        )

    actual = np.concatenate(aggregate_actual)
    residual = np.concatenate(aggregate_residual)
    selected = np.concatenate(aggregate_selected)
    baseline = np.concatenate(aggregate_baseline)
    aggregate_metrics = {
        "rows": int(len(actual)),
        "residual_model": regression_report(pd.Series(actual), residual),
        "selected_method": regression_report(pd.Series(actual), selected),
        "baseline": regression_report(pd.Series(actual), baseline),
    }
    aggregate_metrics["selected_mae_improvement_vs_baseline_pct"] = float(
        100
        * (aggregate_metrics["baseline"]["mae_kw"] - aggregate_metrics["selected_method"]["mae_kw"])
        / aggregate_metrics["baseline"]["mae_kw"]
    )

    bundle = {
        "schema_version": 2,
        "sites": sites,
        "features": base_features,
        "categorical_features": categorical_features,
        "recent_feature": RECENT_FEATURE,
        "profile_keys": PROFILE_KEYS,
        "recent_weeks": RECENT_WEEKS,
        "models": served_models,
        "profiles": served_profiles,
        "fallbacks": served_fallbacks,
        "calendar_profiles": served_calendar_profiles,
        "calendar_fallbacks": served_calendar_fallbacks,
        "methods": served_methods,
    }
    joblib.dump(bundle, MODEL_PATH)

    results: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce": [
            "uv run python -m src.features.make_time_series",
            "uv run python -m src.features.build_features_v3",
            "uv run python -m src.models.train_site_models",
        ],
        "source": {
            "training_script": Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            "training_script_sha256": _source_sha256(Path(__file__)),
            "feature_script_sha256": _source_sha256(Path("src/features/build_features_v3.py")),
            "timeseries_script_sha256": _source_sha256(Path("src/features/make_time_series.py")),
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
            "rows": int(len(data)),
            "sites": sites,
        },
        "method": {
            "final_holdout_days": 60,
            "cv_splits_per_site": N_CV_SPLITS,
            "recent_profile": (
                "Mean of the last eight training observations for the same weekday, hour and minute; "
                "fixed at each evaluation cutoff."
            ),
            "lightgbm_parameters": RESIDUAL_PARAMS,
        },
        "aggregate_holdout": aggregate_metrics,
        "sites": site_results,
        "served_bundle": {
            "path": MODEL_PATH.as_posix(),
            "model_type": "development-selected residual model or calendar baseline per site",
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print("\nSite holdout MAE")
    for site, result in site_results.items():
        holdout = result["final_holdout"]
        print(
            f"{site:10s} residual={holdout['residual_model']['mae_kw']:.3f} "
            f"selected={holdout['selected_method_metrics']['mae_kw']:.3f} "
            f"baseline={holdout['baseline']['mae_kw']:.3f}"
        )
    print(
        f"aggregate  residual={aggregate_metrics['residual_model']['mae_kw']:.3f} "
        f"selected={aggregate_metrics['selected_method']['mae_kw']:.3f} "
        f"baseline={aggregate_metrics['baseline']['mae_kw']:.3f}"
    )
    return results


if __name__ == "__main__":
    train_and_evaluate_sites()
