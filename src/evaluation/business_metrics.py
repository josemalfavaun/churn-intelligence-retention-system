"""Business-oriented metrics that translate model output into $ terms.

These are illustrative, transparent business framings (documented as such in
docs/modeling_report.md) intended to make the model's output usable by a
non-technical retention/growth stakeholder, not a certified financial model.
"""

import pandas as pd

from src.utils.config import ASSUMED_SAVE_RATE, RETENTION_OFFER_COST


def total_value_at_risk(df: pd.DataFrame, value_at_risk_col: str = "customer_value_at_risk") -> float:
    """Sum of estimated value at risk across all customers."""
    return float(df[value_at_risk_col].sum())


def value_captured_at_k(
    df: pd.DataFrame,
    proba_col: str = "churn_probability",
    value_at_risk_col: str = "customer_value_at_risk",
    k_pct: float = 0.1,
) -> dict[str, float]:
    """
    Of the total value at risk, how much is captured by contacting only the
    top k% of customers ranked by churn probability.
    """
    n = len(df)
    k = max(1, int(round(n * k_pct)))
    ranked = df.sort_values(proba_col, ascending=False)
    top = ranked.head(k)

    total_risk = total_value_at_risk(df, value_at_risk_col)
    captured = float(top[value_at_risk_col].sum())

    return {
        "k_pct": k_pct,
        "k_customers": k,
        "value_captured": captured,
        "total_value_at_risk": total_risk,
        "share_of_value_captured": (captured / total_risk) if total_risk > 0 else 0.0,
    }


def retention_campaign_roi(
    df: pd.DataFrame,
    proba_col: str = "churn_probability",
    value_at_risk_col: str = "customer_value_at_risk",
    k_pct: float = 0.1,
    offer_cost: float = RETENTION_OFFER_COST,
    save_rate: float = ASSUMED_SAVE_RATE,
) -> dict[str, float]:
    """
    Illustrative ROI of running a retention campaign on the top k% highest
    risk customers.

    campaign_cost: offer_cost * customers contacted.
    expected_value_saved: value at risk in that slice * assumed save_rate.
    roi: (expected_value_saved - campaign_cost) / campaign_cost.

    This is intentionally simple. It does not model incremental lift (some of
    those customers might have stayed anyway), so it should be read as a
    directional business case, not a guaranteed return. See docs/modeling_report.md
    "Limitations".
    """
    slice_metrics = value_captured_at_k(df, proba_col, value_at_risk_col, k_pct)
    campaign_cost = offer_cost * slice_metrics["k_customers"]
    expected_value_saved = slice_metrics["value_captured"] * save_rate
    roi = (
        (expected_value_saved - campaign_cost) / campaign_cost
        if campaign_cost > 0
        else 0.0
    )

    return {
        **slice_metrics,
        "offer_cost_per_customer": offer_cost,
        "assumed_save_rate": save_rate,
        "campaign_cost": campaign_cost,
        "expected_value_saved": expected_value_saved,
        "roi": roi,
    }


def prioritized_customer_list(
    df: pd.DataFrame,
    proba_col: str = "churn_probability",
    value_at_risk_col: str = "customer_value_at_risk",
    top_n: int = 100,
) -> pd.DataFrame:
    """Return the top_n customers ranked by value at risk, for outreach lists."""
    return (
        df.sort_values(value_at_risk_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
