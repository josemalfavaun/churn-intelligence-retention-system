"""Score customers with the champion model and finalize business features.

This module is the bridge between the trained model and the business-facing
layer (dashboard, recommendation engine): it loads the champion pipeline,
scores customers, and recomputes customer_value_at_risk /
risk_value_segment using the model's real churn_probability instead of the
tenure/contract proxy used during feature engineering (see
src/features/build_features.py and src/utils/config.py for why that
proxy-vs-model-probability distinction matters).
"""

import joblib
import pandas as pd

from src.data.clean_data import clean_telco_data
from src.data.load_data import load_raw_data
from src.features.build_features import (
    add_risk_value_segment,
    add_value_at_risk,
    build_features,
)
from src.utils.config import (
    ALL_MODEL_FEATURES,
    CHAMPION_MODEL_PATH,
    ID_COLUMN,
)


def load_champion_model(path=CHAMPION_MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.models.train_model` first."
        )
    return joblib.load(path)


def score_customers(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Add churn_probability and churn_prediction columns to a feature-engineered
    dataframe, then recompute value-at-risk features using the real
    predicted probability.
    """
    if model is None:
        model = load_champion_model()

    df = df.copy()
    X = df[ALL_MODEL_FEATURES]

    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["churn_prediction"] = model.predict(X)

    df = add_value_at_risk(df, churn_probability_col="churn_probability")
    df = add_risk_value_segment(df)

    return df


def score_new_customers(raw_df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Full pipeline entry point: raw Telco-schema rows in, scored rows out."""
    clean_df = clean_telco_data(raw_df)
    features_df = build_features(clean_df)
    return score_customers(features_df, model=model)


def get_customer_score(scored_df: pd.DataFrame, customer_id: str) -> pd.Series:
    match = scored_df[scored_df[ID_COLUMN] == customer_id]
    if match.empty:
        raise KeyError(f"Customer {customer_id!r} not found.")
    return match.iloc[0]


if __name__ == "__main__":
    raw_df = load_raw_data()
    scored = score_new_customers(raw_df)
    print(f"Scored {len(scored)} customers.")
    print(
        scored[["customer_id", "churn_probability", "customer_value_at_risk", "risk_value_segment"]]
        .sort_values("customer_value_at_risk", ascending=False)
        .head(10)
    )
