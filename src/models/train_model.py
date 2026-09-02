"""Train, compare, and select a champion churn model.

Trains a Logistic Regression baseline plus Random Forest, XGBoost, and
LightGBM candidates, evaluates all of them with the same metrics, and
promotes the best one (by PR-AUC, see src/models/model_registry.py) to
models/champion_model.joblib.

Run with: python -m src.models.train_model
"""

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.clean_data import clean_telco_data
from src.data.load_data import load_raw_data
from src.evaluation.metrics import classification_metrics, top_k_metrics
from src.features.build_features import build_features
from src.models.model_registry import (
    load_registry,
    register_run,
    save_registry,
    select_champion,
)
from src.utils.config import (
    CATEGORICAL_FEATURES,
    CHAMPION_MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Shared preprocessing for all candidates: scale numeric features and
    one-hot encode categoricals. Tree models do not strictly need scaling,
    but sharing one preprocessor keeps every candidate's input contract
    identical, which simplifies both this script and src/models/predict.py.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_candidate_models(scale_pos_weight: float) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


def load_model_ready_data() -> pd.DataFrame:
    raw_df = load_raw_data()
    clean_df = clean_telco_data(raw_df)
    return build_features(clean_df)


def train_and_evaluate(
    df: pd.DataFrame | None = None,
) -> tuple[dict[str, Pipeline], dict[str, dict], pd.DataFrame, pd.Series]:
    """
    Train every candidate model and return:
    - fitted_pipelines: dict of model_name -> fitted sklearn Pipeline
    - all_metrics: dict of model_name -> {classification metrics, top_k metrics}
    - X_test, y_test: held-out split, useful for downstream SHAP/explainability
    """
    if df is None:
        df = load_model_ready_data()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    fitted_pipelines: dict[str, Pipeline] = {}
    all_metrics: dict[str, dict] = {}

    for name, model in get_candidate_models(scale_pos_weight).items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", model)]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = classification_metrics(y_test, y_pred, y_proba)
        metrics["top_10pct"] = top_k_metrics(y_test, y_proba, k_pct=0.1)

        fitted_pipelines[name] = pipeline
        all_metrics[name] = metrics

    return fitted_pipelines, all_metrics, X_test, y_test


def run_training_job() -> dict:
    """Train all candidates, update the registry, and persist the champion."""
    fitted_pipelines, all_metrics, _, _ = train_and_evaluate()

    registry = load_registry()
    for name, metrics in all_metrics.items():
        flat_metrics = {k: v for k, v in metrics.items() if k != "top_10pct"}
        flat_metrics.update(
            {f"top10pct_{k}": v for k, v in metrics["top_10pct"].items()}
        )
        register_run(registry, name, flat_metrics)

    champion = select_champion(registry)
    save_registry(registry)

    if champion is not None:
        champion_pipeline = fitted_pipelines[champion["model_name"]]
        CHAMPION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_pipeline, CHAMPION_MODEL_PATH)

    return {
        "metrics": all_metrics,
        "champion": champion,
    }


if __name__ == "__main__":
    result = run_training_job()

    print("Model comparison:")
    for name, metrics in result["metrics"].items():
        print(
            f"  {name}: roc_auc={metrics['roc_auc']:.3f} "
            f"pr_auc={metrics['average_precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"top10%_capture={metrics['top_10pct']['capture_rate']:.3f}"
        )

    champion = result["champion"]
    if champion:
        print(f"\nChampion model: {champion['model_name']}")
        print(f"Selected on: {champion['selection_metric']}")
        print(f"Saved to: {CHAMPION_MODEL_PATH}")
