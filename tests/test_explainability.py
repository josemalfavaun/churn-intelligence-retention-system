import pandas as pd
import pytest

from src.explainability.reason_codes import (
    describe_feature,
    dominant_reason_category,
    generate_reason_codes,
)
from src.explainability.shap_explainer import (
    compute_shap_values,
    explain_customer,
    global_feature_importance,
)
from src.models.train_model import train_and_evaluate
from src.utils.config import TARGET_COLUMN


@pytest.fixture(scope="module")
def trained_pipeline_and_test_set():
    import numpy as np

    rng = np.random.default_rng(1)
    n = 150
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
    pipelines, _, X_test, _ = train_and_evaluate(df)
    return pipelines["random_forest"], X_test


def test_compute_shap_values_shape_matches_input_rows(trained_pipeline_and_test_set) -> None:
    pipeline, X_test = trained_pipeline_and_test_set
    sample = X_test.head(10)

    shap_values, feature_names = compute_shap_values(pipeline, sample)

    assert shap_values.shape[0] == len(sample)
    assert shap_values.shape[1] == len(feature_names)


def test_global_feature_importance_is_sorted_descending(trained_pipeline_and_test_set) -> None:
    pipeline, X_test = trained_pipeline_and_test_set
    sample = X_test.head(20)
    shap_values, feature_names = compute_shap_values(pipeline, sample)

    importance = global_feature_importance(shap_values, feature_names, top_n=5)

    assert len(importance) == 5
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_explain_customer_returns_requested_top_n(trained_pipeline_and_test_set) -> None:
    pipeline, X_test = trained_pipeline_and_test_set
    sample = X_test.head(5)
    shap_values, feature_names = compute_shap_values(pipeline, sample)

    top_features = explain_customer(shap_values, 0, feature_names, top_n=3)

    assert len(top_features) == 3
    magnitudes = [abs(value) for _, value in top_features]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_describe_feature_reports_numeric_value_and_direction() -> None:
    row = pd.Series({"tenure": 3})
    sentence = describe_feature("numeric__tenure", shap_value=0.4, raw_row=row)
    assert "3 months" in sentence
    assert "increases churn risk" in sentence


def test_describe_feature_reports_categorical_value_and_direction() -> None:
    row = pd.Series({"contract_type": "Month-to-month"})
    sentence = describe_feature(
        "categorical__contract_type_Month-to-month", shap_value=0.3, raw_row=row
    )
    assert "Month-to-month" in sentence
    assert "increases churn risk" in sentence


def test_generate_reason_codes_preserves_order() -> None:
    row = pd.Series({"tenure": 2, "monthly_charges": 90.0})
    top_features = [
        ("numeric__tenure", 0.5),
        ("numeric__monthly_charges", -0.2),
    ]
    reasons = generate_reason_codes(top_features, row)
    assert len(reasons) == 2
    assert "2 months" in reasons[0]
    assert "decreases churn risk" in reasons[1]


def test_dominant_reason_category_picks_strongest_risk_increasing_feature() -> None:
    top_features = [
        ("numeric__tenure", -0.5),
        ("categorical__contract_type_Month-to-month", 0.6),
        ("categorical__online_security_No", 0.2),
    ]
    assert dominant_reason_category(top_features) == "contract"


def test_dominant_reason_category_returns_none_when_nothing_increases_risk() -> None:
    top_features = [("numeric__tenure", -0.5), ("numeric__monthly_charges", -0.1)]
    assert dominant_reason_category(top_features) is None
