import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = Path("data/processed/acn_ts_weather_holidays_data.parquet")
MODEL_DIR = Path("src/models")
MODEL_PATH = MODEL_DIR / "lgbm_model_v1.pkl"


def train_and_evaluate():
    # 1. Chargement
    if not INPUT_FILE.exists():
        logger.error("Dataset introuvable. Lance build_features.py d'abord.")
        return

    df = pd.read_parquet(INPUT_FILE)

    # Tri temporel CRITIQUE (au cas où)
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. Séparation Features (X) / Target (y)
    target = "power_kw"
    drop_cols = [
        "datetime",
        "power_kw",
        "active_chargers",
    ]  # On enlève la cible et les infos "futures" ou métadonnées

    # Vérification que toutes les colonnes sont numériques
    features_to_remove = [
        "lag_15m",
        "lag_1h",
        "rolling_mean_4h",
        "lag_24h",
        "lag_1week",
        "avg_energy_yesterday",
    ]
    features = [c for c in df.columns if c not in drop_cols and c not in features_to_remove]

    X = df[features]
    y = df[target]

    logger.info(f"Training on {len(X)} samples with {len(features)} features.")
    logger.info(f"Features: {features}")

    # 3. Time Series Cross-Validation
    # 5 splits = On va tester le modèle sur 5 périodes différentes du futur
    tscv = TimeSeriesSplit(n_splits=5)

    fold_metrics = []

    logger.info("-" * 30)
    logger.info("🚀 Démarrage de la Cross-Validation (TimeSeriesSplit)")
    logger.info("-" * 30)

    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Modèle LightGBM (Configuration Robuste)
        model = lgb.LGBMRegressor(
            n_estimators=1000,  # Nombre max d'arbres
            learning_rate=0.05,  # Apprentissage doux pour éviter l'overfitting
            num_leaves=31,  # Complexité de l'arbre
            random_state=42,
            n_jobs=-1,  # Utiliser tous les coeurs CPU
            importance_type="gain",  # Pour l'explicabilité plus tard
        )

        # Entraînement avec Early Stopping
        # Si le modèle ne s'améliore pas pendant 50 itérations sur le set de test, on arrête.
        # Note: callbacks est la nouvelle façon propre de faire du early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Silence les logs
        ]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="mae",
            callbacks=callbacks,
        )

        # Prédictions
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        logger.info(f"Fold {fold + 1}: MAE={mae:.2f} kW | RMSE={rmse:.2f} kW | R2={r2:.2%}")
        fold_metrics.append({"mae": mae, "rmse": rmse, "r2": r2})

    # 4. Bilan Global
    avg_mae = np.mean([m["mae"] for m in fold_metrics])
    avg_r2 = np.mean([m["r2"] for m in fold_metrics])

    logger.info("-" * 30)
    logger.info(f"✅ CV Terminée. Moyenne MAE: {avg_mae:.2f} kW | Moyenne R2: {avg_r2:.2%}")
    logger.info("-" * 30)

    # 5. Entraînement Final (Production)
    # On réentraîne sur TOUT le dataset pour avoir le modèle le plus "savant" possible
    logger.info("Entraînement du modèle final sur l'ensemble des données...")

    final_model = lgb.LGBMRegressor(
        n_estimators=1000,  # On garde les mêmes hyperparams (ou ceux du meilleur fold idéalement)
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    # 6. Sauvegarde
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    logger.info(f"Modèle sauvegardé sous : {MODEL_PATH}")

    # 7. Feature Importance (Bonus Expert)
    # Pour vérifier que le modèle n'a pas appris n'importe quoi
    importance = pd.DataFrame({"feature": features, "importance": final_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )

    print("\n--- 🏆 Top 5 Features les plus importantes ---")
    print(importance.head(5))


if __name__ == "__main__":
    train_and_evaluate()
