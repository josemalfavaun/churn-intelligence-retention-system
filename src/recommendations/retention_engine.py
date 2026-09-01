"""Rule-based retention recommendation engine.

Combines churn risk, customer value, and the dominant SHAP-derived reason
category into a specific, actionable recommendation. Deliberately
rule-based (not another model) so the recommendation logic stays
transparent and easy for a business stakeholder to audit and adjust.
"""

# Priority tiers, used for sorting outreach lists in the dashboard.
PRIORITY_TIERS = {
    "urgent": 1,
    "high": 2,
    "medium": 3,
    "low": 4,
    "monitor": 5,
}

# Category-specific plays, chosen when we know *why* a customer is at risk
# (from src.explainability.reason_codes.dominant_reason_category).
_CATEGORY_PLAYS = {
    "contract": "Offer a contract upgrade (annual/two-year) with a loyalty discount to lock in commitment.",
    "payment": "Nudge toward automatic payment (bank transfer or credit card) with a small one-time incentive.",
    "service": "Offer a free trial of tech support / online security to increase product stickiness.",
    "tenure": "Enroll in a structured first-90-day onboarding and check-in journey.",
    "value": "Review pricing/bundle fit; consider a tailored plan adjustment for a high-spend customer.",
    "other": "Flag for manual review; no single dominant risk driver identified.",
}


def recommend_action(
    risk_value_segment: str,
    churn_probability: float,
    dominant_category: str | None = None,
) -> dict[str, str]:
    """
    Return a recommendation dict with keys: priority, action, rationale.

    Rules (in order of business intent, not code order):
    - high_value_high_risk: always urgent/high priority, human-reviewed save
      play, tailored by dominant_category when available.
    - high_value_low_risk: proactive relationship touch, no urgent offer.
    - low_value_high_risk: low-cost automated nudge, do not over-invest.
    - low_value_low_risk: passive monitoring only.
    """
    category_play = _CATEGORY_PLAYS.get(dominant_category, _CATEGORY_PLAYS["other"])

    if risk_value_segment == "high_value_high_risk":
        priority = "urgent" if churn_probability >= 0.6 else "high"
        action = f"Priority save outreach (human review). {category_play}"
        rationale = (
            "High estimated value and high churn risk: the cost of a human-reviewed "
            "retention offer is justified by the value at stake."
        )
        return {"priority": priority, "action": action, "rationale": rationale}

    if risk_value_segment == "high_value_low_risk":
        return {
            "priority": "medium",
            "action": "Proactive loyalty touch (no urgent offer) — recognize the relationship, watch for change.",
            "rationale": "High value but currently low risk: protect the relationship without discounting unnecessarily.",
        }

    if risk_value_segment == "low_value_high_risk":
        return {
            "priority": "low",
            "action": f"Low-cost automated nudge (email/SMS). {category_play}",
            "rationale": "High risk but lower value: an automated, low-cost touch is more cost-effective than a human save call.",
        }

    # low_value_low_risk
    return {
        "priority": "monitor",
        "action": "No action needed now — passive monitoring.",
        "rationale": "Low value and low risk: intervention cost would likely exceed the value protected.",
    }


def rank_by_priority(recommendations: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort a list of recommendation dicts by priority tier (most urgent first)."""
    return sorted(
        recommendations, key=lambda r: PRIORITY_TIERS.get(r["priority"], 99)
    )
