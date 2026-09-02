import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_contract_and_payment_risk,
    add_customer_value_features,
    add_monthly_spend_band,
    add_risk_value_segment,
    add_service_count,
    add_tenure_bucket,
    add_value_at_risk,
    build_features,
)


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "tenure": [1, 15, 30, 60],
            "monthly_charges": [20.0, 50.0, 80.0, 110.0],
            "contract_type": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "payment_method": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "phone_service": ["Yes", "Yes", "No", "Yes"],
            "multiple_lines": ["No", "Yes", "No phone service", "Yes"],
            "internet_service": ["DSL", "Fiber optic", "No", "Fiber optic"],
            "online_security": ["No", "Yes", "No internet service", "Yes"],
            "online_backup": ["No", "Yes", "No internet service", "No"],
            "device_protection": ["No", "No", "No internet service", "Yes"],
            "tech_support": ["No", "Yes", "No internet service", "No"],
            "streaming_tv": ["No", "Yes", "No internet service", "Yes"],
            "streaming_movies": ["No", "No", "No internet service", "Yes"],
        }
    )


def test_add_tenure_bucket_assigns_expected_labels(clean_sample: pd.DataFrame) -> None:
    result = add_tenure_bucket(clean_sample)
    assert result["tenure_bucket"].tolist() == ["0-6m", "13-24m", "25-48m", "49m+"]


def test_add_monthly_spend_band_creates_three_bands(clean_sample: pd.DataFrame) -> None:
    result = add_monthly_spend_band(clean_sample)
    assert set(result["monthly_spend_band"].cat.categories) <= {"low", "medium", "high"}
    assert result["monthly_spend_band"].notna().all()


def test_add_service_count_counts_only_active_services(clean_sample: pd.DataFrame) -> None:
    result = add_service_count(clean_sample)
    # C3 has no phone and no internet service, so every service column reads
    # as inactive ("No", "No phone service" or "No internet service").
    c3_count = result.loc[result["customer_id"] == "C3", "service_count"].iloc[0]
    assert c3_count == 0

    # C2 has phone + fiber + security/backup/support/tv, i.e. 6 of 9 active
    # (multiple_lines=Yes, internet=Fiber, security/backup/support/tv=Yes,
    # streaming_movies=No, phone_service=Yes).
    c2_count = result.loc[result["customer_id"] == "C2", "service_count"].iloc[0]
    assert c2_count == 7


def test_add_contract_and_payment_risk_maps_known_values(clean_sample: pd.DataFrame) -> None:
    result = add_contract_and_payment_risk(clean_sample)
    assert result["contract_risk_score"].tolist() == [3, 2, 1, 3]
    assert result["payment_risk_score"].tolist() == [3, 2, 1, 1]


def test_add_customer_value_features_flags_high_value_customers(clean_sample: pd.DataFrame) -> None:
    result = add_customer_value_features(clean_sample)
    assert (result["estimated_customer_value"] == result["monthly_charges"] * 12).all()
    # The highest monthly charge customer should be flagged high value.
    top_customer = result.loc[result["monthly_charges"].idxmax()]
    assert top_customer["high_value_customer"] == 1


def test_add_value_at_risk_uses_proxy_when_no_probability_column(clean_sample: pd.DataFrame) -> None:
    df = add_contract_and_payment_risk(clean_sample)
    df = add_customer_value_features(df)
    result = add_value_at_risk(df)

    assert "customer_value_at_risk" in result.columns
    assert (result["customer_value_at_risk"] >= 0).all()
    assert (result["customer_value_at_risk"] <= result["estimated_customer_value"]).all()


def test_add_value_at_risk_uses_model_probability_when_available(clean_sample: pd.DataFrame) -> None:
    df = add_contract_and_payment_risk(clean_sample)
    df = add_customer_value_features(df)
    df["churn_probability"] = [0.9, 0.1, 0.5, 0.0]

    result = add_value_at_risk(df, churn_probability_col="churn_probability")

    expected = df["estimated_customer_value"] * df["churn_probability"]
    np.testing.assert_allclose(result["customer_value_at_risk"], expected)


def test_add_risk_value_segment_produces_one_of_four_labels(clean_sample: pd.DataFrame) -> None:
    df = add_contract_and_payment_risk(clean_sample)
    df = add_customer_value_features(df)
    df = add_value_at_risk(df)
    result = add_risk_value_segment(df)

    valid_segments = {
        "high_value_high_risk",
        "high_value_low_risk",
        "low_value_high_risk",
        "low_value_low_risk",
    }
    assert set(result["risk_value_segment"].unique()) <= valid_segments


def test_build_features_end_to_end_adds_all_engineered_columns(clean_sample: pd.DataFrame) -> None:
    result = build_features(clean_sample)

    expected_new_columns = {
        "tenure_bucket",
        "monthly_spend_band",
        "service_count",
        "contract_risk_score",
        "payment_risk_score",
        "estimated_customer_value",
        "high_value_customer",
        "customer_value_at_risk",
        "risk_value_segment",
    }
    assert expected_new_columns <= set(result.columns)
    assert len(result) == len(clean_sample)
