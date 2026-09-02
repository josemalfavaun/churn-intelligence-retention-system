from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import classification_metrics, gain_table, top_k_metrics
from src.models import model_registry
from src.models.train_model import train_and_evaluate
from src.utils.config import ALL_MODEL_FEATURES, TARGET_COLUMN


@pytest.fixture(scope="module")
def synthetic_features_df() -> pd.DataFrame:
    """
    A small synthetic dataset shaped like the real feature-engineered
    dataframe (same columns train_and_evaluate expects), so the full
    training path (preprocessing, 4 candidate models, evaluation) can be
    exercised in tests without needing the real Telco CSV or paying its
    training cost.
    """
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame(
        {
            "tenure": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(18, 120, n),
            "total_charges": rng.uniform(18, 8000, n),
            "senior_citizen": rng.integers(0, 2, n),
            "service_count": rng.integers(0, 9, n),
            "contract_risk_score": rng.integers(1, 4, n),
            "payment_risk_score": rng.integers(1, 4, n),
            "high_value_customer": rng.integers(0, 2, n),
            "estimated_customer_value": rng.uniform(200, 1500, n),
            "gender": rng.choice(["Male", "Female"], n),
            "partner": rng.choice(["Yes", "No"], n),
            "dependents": rng.choice(["Yes", "No"], n),
            "phone_service": rng.choice(["Yes", "No"], n),
            "multiple_lines": rng.choice(["Yes", "No", "No phone service"], n),
            "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "online_security": rng.choice(["Yes", "No", "No internet service"], n),
            "online_backup": rng.choice(["Yes", "No", "No internet service"], n),
            "device_protection": rng.choice(["Yes", "No", "No internet service"], n),
            "tech_support": rng.choice(["Yes", "No", "No internet service"], n),
            "streaming_tv": rng.choice(["Yes", "No", "No internet service"], n),
            "streaming_movies": rng.choice(["Yes", "No", "No internet service"], n),
            "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "paperless_billing": rng.choice(["Yes", "No"], n),
            "payment_method": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n,
            ),
            "tenure_bucket": rng.choice(["0-6m", "7-12m", "13-24m", "25-48m", "49m+"], n),
            "monthly_spend_band": rng.choice(["low", "medium", "high"], n),
            TARGET_COLUMN: rng.integers(0, 2, n),
        }
    )
    assert set(ALL_MODEL_FEATURES) <= set(df.columns)
    return df


def test_train_and_evaluate_returns_fitted_pipelines_for_every_candidate(
    synthetic_features_df: pd.DataFrame,
) -> None:
    pipelines, metrics, X_test, y_test = train_and_evaluate(synthetic_features_df)

    assert set(pipelines.keys()) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    }
    for name, pipeline in pipelines.items():
        preds = pipeline.predict(X_test)
        assert len(preds) == len(y_test)


def test_train_and_evaluate_produces_expected_metric_keys(
    synthetic_features_df: pd.DataFrame,
) -> None:
    _, metrics, _, _ = train_and_evaluate(synthetic_features_df)

    for name, model_metrics in metrics.items():
        for key in ("roc_auc", "average_precision", "precision", "recall", "f1"):
            assert key in model_metrics
            assert 0.0 <= model_metrics[key] <= 1.0
        assert "top_10pct" in model_metrics
        assert "capture_rate" in model_metrics["top_10pct"]


def test_classification_metrics_are_bounded() -> None:
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.1, 0.8, 0.4, 0.2, 0.9]

    metrics = classification_metrics(y_true, y_pred, y_proba)
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_top_k_metrics_flags_perfect_ranking_with_lift_above_one() -> None:
    y_true = [0] * 8 + [1] * 2
    y_proba = [0.1] * 8 + [0.9, 0.95]

    result = top_k_metrics(y_true, y_proba, k_pct=0.2)
    assert result["precision_at_k"] == 1.0
    assert result["capture_rate"] == 1.0
    assert result["lift"] > 1.0


def test_gain_table_cumulative_capture_reaches_one_at_last_decile() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.uniform(0, 1, 100)

    gains = gain_table(y_true, y_proba, n_bins=10)
    assert len(gains) == 10
    assert gains["cumulative_capture_rate"].iloc[-1] == pytest.approx(1.0)


def test_model_registry_selects_champion_by_highest_average_precision(tmp_path: Path) -> None:
    registry = model_registry.load_registry(tmp_path / "does_not_exist.json")
    model_registry.register_run(registry, "model_a", {"average_precision": 0.5})
    model_registry.register_run(registry, "model_b", {"average_precision": 0.8})

    champion = model_registry.select_champion(registry)

    assert champion["model_name"] == "model_b"
    assert model_registry.get_champion(registry)["model_name"] == "model_b"


def test_model_registry_round_trips_through_disk(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry = model_registry.load_registry(registry_path)
    model_registry.register_run(registry, "model_a", {"average_precision": 0.6})
    model_registry.select_champion(registry)
    model_registry.save_registry(registry, registry_path)

    reloaded = model_registry.load_registry(registry_path)
    assert reloaded["champion"]["model_name"] == "model_a"
