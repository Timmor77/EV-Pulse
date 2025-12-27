import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# =========================
# Config
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("lgbm_power")

INPUT_FILE = Path("data/processed/model_context.parquet")
MODEL_DIR = Path("src/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lgbm_context_model.pkl"

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

LGBM_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=40,
    random_state=42,
    n_jobs=-1,
)

EARLY_STOPPING_ROUNDS = 150


# =========================
# Utils
# =========================
def print_header(title: str):
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}\n")


def regression_report(y_true, y_pred) -> dict:
    y_pred = np.maximum(y_pred, 0)  # sécurité physique
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# =========================
# Main training
# =========================
def train_and_evaluate():
    print_header("LightGBM — Power Forecast (Context Only)")

    if not INPUT_FILE.exists():
        logger.error("Dataset introuvable: %s", INPUT_FILE)
        return

    df = pd.read_parquet(INPUT_FILE).sort_values("datetime").reset_index(drop=True)

    # Categorical casting on full df for stable categories across splits
    existing_cat = [c for c in CAT_FEATURES if c in df.columns]
    for c in existing_cat:
        df[c] = df[c].astype("category")

    features = [c for c in df.columns if c not in DROP_COLS]
    X = df[features]
    y = df[TARGET]

    logger.info(
        "Samples: %d | Features: %d | Categorical: %s",
        len(df),
        len(features),
        existing_cat,
    )

    # -------------------------
    # 1) CV metrics (TimeSeriesSplit)
    # -------------------------
    tscv = TimeSeriesSplit(n_splits=10)
    fold_rows = []

    last_fold_artifacts = {}  # store last fold for nice plots + importances

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS, importance_type="gain")

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            categorical_feature=existing_cat,
            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        preds = model.predict(X_test)
        rep = regression_report(y_test, preds)
        fold_rows.append({"fold": fold, **rep, "best_iter": getattr(model, "best_iteration_", None)})

        if fold == tscv.get_n_splits():
            last_fold_artifacts = dict(model=model, X_test=X_test, y_test=y_test, preds=preds)

    cv_df = pd.DataFrame(fold_rows)
    print_header("Cross-validation (TimeSeriesSplit)")
    print(cv_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    mean_row = cv_df[["MAE", "RMSE", "R2"]].mean()
    std_row = cv_df[["MAE", "RMSE", "R2"]].std()

    print("\nSummary:")
    print(f"  MAE : {mean_row['MAE']:.4f} ± {std_row['MAE']:.4f} kW")
    print(f"  RMSE: {mean_row['RMSE']:.4f} ± {std_row['RMSE']:.4f} kW")
    print(f"  R2  : {mean_row['R2']:.4%} ± {std_row['R2']:.4%}")

    # -------------------------
    # 2) “Vraies” importances = permutation importance on last fold (holdout)
    # -------------------------
    print_header("Feature importances (Permutation on last fold = holdout)")

    # Permutation importance uses the estimator + holdout data.
    # Use MAE: lower is better, so we use scoring='neg_mean_absolute_error'.
    pi = permutation_importance(
        last_fold_artifacts["model"],
        last_fold_artifacts["X_test"],
        last_fold_artifacts["y_test"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature": features,
            "importance": pi.importances_mean,
            "std": pi.importances_std,
        }
    ).sort_values("importance", ascending=False)

    # Note: importance here is in "delta MAE" scale (via neg MAE), higher = more important
    print(perm_df.head(30).to_string(index=False, float_format=lambda x: f"{x:,.6f}"))

    # For comparison only (not “vrai”): native gain importances
    print_header("Feature importances (LightGBM gain — indicative)")
    gain_df = pd.DataFrame(
        {
            "feature": features,
            "importance": last_fold_artifacts["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print(gain_df.head(30).to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # -------------------------
    # 3) Train final model on full data + save
    # -------------------------
    print_header("Training final model on full data + save")

    final_model = lgb.LGBMRegressor(**LGBM_PARAMS, importance_type="gain")
    final_model.fit(X, y, categorical_feature=existing_cat)

    joblib.dump(final_model, MODEL_PATH)
    logger.info("Model saved: %s", MODEL_PATH)

    print("\n✅ Done.")


if __name__ == "__main__":
    train_and_evaluate()
