"""Feature engineering for the churn intelligence system.

Turns the cleaned Telco dataset into business-ready modeling features, as
documented in docs/data_dictionary.md under "Engineered Features".
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import clean_telco_data
from src.data.load_data import load_raw_data
from src.utils.config import INTERIM_DATA_PATH

SERVICE_COLUMNS = [
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Contract risk: month-to-month customers can leave with no penalty, so they
# carry the highest risk score. Longer commitments carry lower risk.
CONTRACT_RISK_MAP = {
    "Month-to-month": 3,
    "One year": 2,
    "Two year": 1,
}

# Payment risk: electronic check has historically shown the highest observed
# churn rate in the Telco dataset EDA (docs/retention_strategy.md), followed
# by mailed check, then the two automatic/bank-linked methods.
PAYMENT_RISK_MAP = {
    "Electronic check": 3,
    "Mailed check": 2,
    "Bank transfer (automatic)": 1,
    "Credit card (automatic)": 1,
}

TENURE_BUCKET_BINS = [-np.inf, 6, 12, 24, 48, np.inf]
TENURE_BUCKET_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def _service_active(value: str) -> bool:
    """A service column counts as 'active' unless it explicitly says No."""
    return value not in ("No", "No internet service", "No phone service")


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=TENURE_BUCKET_BINS, labels=TENURE_BUCKET_LABELS
    )
    return df


def add_monthly_spend_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["monthly_spend_band"] = pd.qcut(
        df["monthly_charges"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_count"] = df[SERVICE_COLUMNS].apply(
        lambda row: sum(_service_active(v) for v in row), axis=1
    )
    return df


def add_contract_and_payment_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["contract_risk_score"] = df["contract_type"].map(CONTRACT_RISK_MAP)
    df["payment_risk_score"] = df["payment_method"].map(PAYMENT_RISK_MAP)
    return df


def add_customer_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a simple forward-looking customer value.

    estimated_customer_value: a 12-month forward projection of billing based
    on the customer's current monthly charge. This is a simplification (it
    ignores discounts, upsell, and downgrade) documented as a limitation in
    docs/modeling_report.md.
    """
    df = df.copy()
    df["estimated_customer_value"] = df["monthly_charges"] * 12

    high_value_threshold = df["estimated_customer_value"].quantile(0.75)
    df["high_value_customer"] = (
        df["estimated_customer_value"] >= high_value_threshold
    ).astype(int)

    return df


def add_value_at_risk(
    df: pd.DataFrame, churn_probability_col: str | None = None
) -> pd.DataFrame:
    """
    Compute customer_value_at_risk.

    If a churn probability column is available (post-modeling), value at risk
    is estimated_customer_value * churn_probability. Before modeling, a
    tenure/contract-based proxy risk is used instead so the feature exists
    end-to-end and the dashboard can fall back to it if needed.
    """
    df = df.copy()

    if churn_probability_col is not None and churn_probability_col in df.columns:
        risk = df[churn_probability_col]
    else:
        tenure_risk = 1 - (df["tenure"] / df["tenure"].max()).clip(0, 1)
        contract_risk = df["contract_risk_score"] / df["contract_risk_score"].max()
        risk = (0.5 * tenure_risk + 0.5 * contract_risk).clip(0, 1)

    df["customer_value_at_risk"] = df["estimated_customer_value"] * risk
    return df


def add_risk_value_segment(
    df: pd.DataFrame, risk_col: str = "customer_value_at_risk"
) -> pd.DataFrame:
    """
    Combine value and risk into an operational 2x2 segment used by the
    retention recommendation engine.
    """
    df = df.copy()

    value_median = df["estimated_customer_value"].median()
    risk_median = df[risk_col].median()

    def _segment(row: pd.Series) -> str:
        high_value = row["estimated_customer_value"] >= value_median
        high_risk = row[risk_col] >= risk_median
        if high_value and high_risk:
            return "high_value_high_risk"
        if high_value and not high_risk:
            return "high_value_low_risk"
        if not high_value and high_risk:
            return "low_value_high_risk"
        return "low_value_low_risk"

    df["risk_value_segment"] = df.apply(_segment, axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a cleaned dataframe."""
    df = add_tenure_bucket(df)
    df = add_monthly_spend_band(df)
    df = add_service_count(df)
    df = add_contract_and_payment_risk(df)
    df = add_customer_value_features(df)
    df = add_value_at_risk(df)
    df = add_risk_value_segment(df)
    return df


def save_features(df: pd.DataFrame, output_path: str | Path = INTERIM_DATA_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_telco_data(raw_df)
    features_df = build_features(clean_df)
    save_features(features_df)

    print(f"Feature dataset saved to: {INTERIM_DATA_PATH}")
    print(f"Feature data shape: {features_df.shape}")
    print(features_df["risk_value_segment"].value_counts())
