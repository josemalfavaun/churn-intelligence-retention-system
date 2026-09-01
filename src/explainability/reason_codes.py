"""Turn SHAP feature contributions into human-readable reason codes.

A reason code is a short, business-readable sentence a retention agent or a
dashboard user can act on, e.g. "Month-to-month contract (increases risk)"
instead of a raw SHAP value on a one-hot encoded column name.
"""

import pandas as pd

from src.utils.config import CATEGORICAL_FEATURES

NUMERIC_LABELS: dict[str, str] = {
    "tenure": "Tenure of {value:.0f} months",
    "monthly_charges": "Monthly charges of ${value:.2f}",
    "total_charges": "Total charges to date of ${value:,.2f}",
    "senior_citizen": "Senior citizen status",
    "service_count": "{value:.0f} active services",
    "contract_risk_score": "Contract commitment level",
    "payment_risk_score": "Payment method risk profile",
    "high_value_customer": "High-value customer flag",
    "estimated_customer_value": "Estimated annual value of ${value:,.2f}",
}

CATEGORICAL_LABELS: dict[str, str] = {
    "gender": "Gender",
    "partner": "Partner status",
    "dependents": "Dependents",
    "phone_service": "Phone service",
    "multiple_lines": "Multiple phone lines",
    "internet_service": "Internet service",
    "online_security": "Online security add-on",
    "online_backup": "Online backup add-on",
    "device_protection": "Device protection add-on",
    "tech_support": "Tech support add-on",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming movies",
    "contract_type": "Contract type",
    "paperless_billing": "Paperless billing",
    "payment_method": "Payment method",
    "tenure_bucket": "Tenure bucket",
    "monthly_spend_band": "Monthly spend band",
}

# Sorted longest-first so prefix matching picks the most specific column name
# (e.g. "streaming_movies" before "streaming").
_CATEGORICAL_COLUMNS_BY_LENGTH = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)


def _parse_transformed_name(name: str) -> tuple[str, str | None, str | None]:
    """
    Parse a ColumnTransformer output name like "numeric__tenure" or
    "categorical__contract_type_Month-to-month" into
    (kind, column_name, category_value).
    """
    if name.startswith("numeric__"):
        column = name.removeprefix("numeric__")
        return "numeric", column, None

    if name.startswith("categorical__"):
        raw = name.removeprefix("categorical__")
        for column in _CATEGORICAL_COLUMNS_BY_LENGTH:
            prefix = f"{column}_"
            if raw.startswith(prefix):
                return "categorical", column, raw[len(prefix):]
        return "categorical", raw, None

    return "unknown", name, None


def describe_feature(
    transformed_name: str, shap_value: float, raw_row: pd.Series
) -> str:
    """Build one human-readable reason code sentence for a single feature."""
    kind, column, category = _parse_transformed_name(transformed_name)
    direction = "increases churn risk" if shap_value > 0 else "decreases churn risk"

    if kind == "numeric" and column in NUMERIC_LABELS:
        raw_value = raw_row.get(column)
        try:
            label = NUMERIC_LABELS[column].format(value=raw_value)
        except (TypeError, ValueError):
            label = NUMERIC_LABELS[column]
        return f"{label} ({direction})"

    if kind == "categorical" and column in CATEGORICAL_LABELS and category:
        label = CATEGORICAL_LABELS[column]
        return f"{label}: {category} ({direction})"

    # Fallback for anything not explicitly labeled, so the function never
    # crashes on an unexpected feature name.
    return f"{transformed_name} ({direction})"


def generate_reason_codes(
    top_features: list[tuple[str, float]], raw_row: pd.Series
) -> list[str]:
    """
    Convert a list of (transformed_feature_name, shap_value) tuples, as
    returned by src.explainability.shap_explainer.explain_customer, into
    ordered, human-readable reason code strings (most influential first).
    """
    return [
        describe_feature(name, value, raw_row) for name, value in top_features
    ]


def dominant_reason_category(top_features: list[tuple[str, float]]) -> str | None:
    """
    Identify which business category (contract, payment, service, value,
    tenure) the single strongest risk-increasing feature belongs to. Used by
    the retention engine to pick a targeted action.
    """
    risk_increasing = [f for f in top_features if f[1] > 0]
    if not risk_increasing:
        return None

    top_name, _ = max(risk_increasing, key=lambda f: f[1])
    _, column, _ = _parse_transformed_name(top_name)

    if column in ("contract_type", "contract_risk_score"):
        return "contract"
    if column in ("payment_method", "payment_risk_score"):
        return "payment"
    if column in (
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "internet_service",
        "multiple_lines",
        "phone_service",
        "service_count",
    ):
        return "service"
    if column in ("tenure", "tenure_bucket"):
        return "tenure"
    if column in ("monthly_charges", "monthly_spend_band", "estimated_customer_value"):
        return "value"

    return "other"
