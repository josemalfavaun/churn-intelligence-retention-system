"""Central configuration for paths and shared constants."""

from pathlib import Path

# Project root (this file lives at src/utils/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
INTERIM_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "telco_churn_features.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"

# Reproducibility
RANDOM_STATE = 42

# Modeling
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
TEST_SIZE = 0.2

# Business assumptions (documented in docs/modeling_report.md)
# Average estimated cost of a successful retention offer per customer, used to
# turn model output into a business-oriented cost/benefit read in the dashboard.
RETENTION_OFFER_COST = 50.0
# Assumed average tenure (in months) recovered when a high-value, high-risk
# customer is successfully retained. Used only for illustrative ROI framing.
ASSUMED_RETAINED_MONTHS = 12
# Assumed probability that a contacted at-risk customer accepts the retention
# offer and is actually saved. Illustrative only, see docs/modeling_report.md.
ASSUMED_SAVE_RATE = 0.30

# Modeling feature contract.
# Numeric features passed to the preprocessing pipeline as-is (scaled inside
# the pipeline). These are all deterministic functions of raw account data,
# not of the model's own predictions, so there is no target leakage.
NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "service_count",
    "contract_risk_score",
    "payment_risk_score",
    "high_value_customer",
    "estimated_customer_value",
]

# Categorical features, one-hot encoded inside the pipeline.
CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method",
    "tenure_bucket",
    "monthly_spend_band",
]

# customer_value_at_risk and risk_value_segment are intentionally excluded
# from the model's input features: at the feature-engineering stage they are
# built from a tenure/contract proxy for risk (see
# src/features/build_features.py), not from the trained model. Using the
# champion model's own predicted probability to recompute them afterwards
# (done in src/models/predict.py) avoids feeding a proxy of the target into
# the model that predicts the target.
ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
