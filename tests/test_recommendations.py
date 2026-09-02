from src.recommendations.retention_engine import rank_by_priority, recommend_action


def test_high_value_high_risk_with_high_probability_is_urgent() -> None:
    result = recommend_action("high_value_high_risk", churn_probability=0.75)
    assert result["priority"] == "urgent"
    assert "save" in result["action"].lower()


def test_high_value_high_risk_with_moderate_probability_is_high_not_urgent() -> None:
    result = recommend_action("high_value_high_risk", churn_probability=0.55)
    assert result["priority"] == "high"


def test_high_value_low_risk_recommends_proactive_touch_not_a_discount() -> None:
    result = recommend_action("high_value_low_risk", churn_probability=0.1)
    assert result["priority"] == "medium"
    assert "discount" not in result["action"].lower()


def test_low_value_high_risk_recommends_low_cost_automated_action() -> None:
    result = recommend_action("low_value_high_risk", churn_probability=0.8)
    assert result["priority"] == "low"
    assert "automated" in result["action"].lower()


def test_low_value_low_risk_recommends_monitoring_only() -> None:
    result = recommend_action("low_value_low_risk", churn_probability=0.05)
    assert result["priority"] == "monitor"
    assert "no action" in result["action"].lower()


def test_dominant_category_tailors_the_high_risk_play() -> None:
    contract_driven = recommend_action(
        "high_value_high_risk", churn_probability=0.7, dominant_category="contract"
    )
    service_driven = recommend_action(
        "high_value_high_risk", churn_probability=0.7, dominant_category="service"
    )
    assert contract_driven["action"] != service_driven["action"]
    assert "contract" in contract_driven["action"].lower()
    assert "tech support" in service_driven["action"].lower()


def test_unknown_segment_falls_back_to_monitoring() -> None:
    result = recommend_action("unexpected_segment", churn_probability=0.5)
    assert result["priority"] == "monitor"


def test_rank_by_priority_orders_urgent_before_monitor() -> None:
    recommendations = [
        {"priority": "monitor", "action": "a", "rationale": "r"},
        {"priority": "urgent", "action": "b", "rationale": "r"},
        {"priority": "low", "action": "c", "rationale": "r"},
    ]
    ranked = rank_by_priority(recommendations)
    assert [r["priority"] for r in ranked] == ["urgent", "low", "monitor"]
