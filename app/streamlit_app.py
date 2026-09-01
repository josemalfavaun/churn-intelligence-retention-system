"""Churn Intelligence & Retention dashboard.

Run with: streamlit run app/streamlit_app.py
Requires a trained champion model: python -m src.models.train_model
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow running via `streamlit run app/streamlit_app.py` from the repo root
# without installing the package.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.business_metrics import (  # noqa: E402
    retention_campaign_roi,
    total_value_at_risk,
)
from src.evaluation.metrics import gain_table  # noqa: E402
from src.explainability.reason_codes import (  # noqa: E402
    dominant_reason_category,
    generate_reason_codes,
)
from src.explainability.shap_explainer import (  # noqa: E402
    build_explainer,
    compute_shap_values,
    explain_customer,
)
from src.models.model_registry import load_registry  # noqa: E402
from src.models.predict import load_champion_model, score_customers  # noqa: E402
from src.models.train_model import load_model_ready_data  # noqa: E402
from src.recommendations.retention_engine import recommend_action  # noqa: E402
from src.utils.config import CHAMPION_MODEL_PATH  # noqa: E402

st.set_page_config(
    page_title="Churn Intelligence & Retention",
    page_icon="📉",
    layout="wide",
)


@st.cache_data(show_spinner="Loading and scoring customers...")
def get_scored_customers() -> pd.DataFrame:
    features_df = load_model_ready_data()
    model = load_champion_model()
    return score_customers(features_df, model=model)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_champion_model()


@st.cache_resource(show_spinner="Building SHAP explainer...")
def get_explainer_and_values(_model, sample_key: int):
    """
    Build a SHAP explainer and precompute SHAP values for a bounded sample of
    customers (SHAP on the full 7k-row dataset is unnecessarily slow for an
    interactive dashboard; the sample is large enough to be representative).
    """
    df = load_model_ready_data()
    sample = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    explainer = build_explainer(_model, sample)
    shap_values, feature_names = compute_shap_values(_model, sample, explainer=explainer)
    return sample, shap_values, feature_names


def render_missing_model_message() -> None:
    st.error(
        "No trained model found. Train one first from the project root:\n\n"
        "```\npython -m src.models.train_model\n```"
    )


def render_overview_tab(scored: pd.DataFrame) -> None:
    st.subheader("Portfolio Overview")

    total_customers = len(scored)
    churn_rate = scored["churn"].mean()
    predicted_at_risk = (scored["churn_probability"] >= 0.5).sum()
    value_at_risk = total_value_at_risk(scored)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{total_customers:,}")
    col2.metric("Historical Churn Rate", f"{churn_rate:.1%}")
    col3.metric("Predicted High-Risk Customers", f"{predicted_at_risk:,}")
    col4.metric("Total Value at Risk", f"${value_at_risk:,.0f}")

    st.markdown("#### Customers by Risk / Value Segment")
    segment_counts = (
        scored["risk_value_segment"].value_counts().rename_axis("segment").reset_index(name="customers")
    )
    fig = px.bar(
        segment_counts,
        x="segment",
        y="customers",
        text="customers",
        color="segment",
        title="Customer Count by Risk/Value Segment",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Illustrative Retention Campaign ROI")
    st.caption(
        "Directional business case only — see docs/modeling_report.md for assumptions "
        "(offer cost, assumed save rate) and limitations."
    )
    k_pct = st.slider("Contact top X% of customers by churn probability", 5, 50, 10, step=5) / 100
    roi = retention_campaign_roi(scored, k_pct=k_pct)

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    rcol1.metric("Customers Contacted", f"{roi['k_customers']:,}")
    rcol2.metric("Campaign Cost", f"${roi['campaign_cost']:,.0f}")
    rcol3.metric("Expected Value Saved", f"${roi['expected_value_saved']:,.0f}")
    rcol4.metric("Expected ROI", f"{roi['roi']:.1%}")


def render_customer_explorer_tab(scored: pd.DataFrame) -> None:
    st.subheader("Customer Explorer")

    col1, col2, col3 = st.columns(3)
    segment_filter = col1.multiselect(
        "Risk / value segment",
        options=sorted(scored["risk_value_segment"].unique()),
        default=None,
    )
    min_prob = col2.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
    contract_filter = col3.multiselect(
        "Contract type", options=sorted(scored["contract_type"].unique()), default=None
    )

    filtered = scored.copy()
    if segment_filter:
        filtered = filtered[filtered["risk_value_segment"].isin(segment_filter)]
    if contract_filter:
        filtered = filtered[filtered["contract_type"].isin(contract_filter)]
    filtered = filtered[filtered["churn_probability"] >= min_prob]

    display_cols = [
        "customer_id",
        "contract_type",
        "tenure",
        "monthly_charges",
        "churn_probability",
        "customer_value_at_risk",
        "risk_value_segment",
    ]
    st.dataframe(
        filtered[display_cols]
        .sort_values("customer_value_at_risk", ascending=False)
        .style.format(
            {
                "churn_probability": "{:.1%}",
                "monthly_charges": "${:.2f}",
                "customer_value_at_risk": "${:,.0f}",
            }
        ),
        width="stretch",
        height=500,
    )
    st.caption(f"{len(filtered):,} customers match the current filters.")


def render_customer_detail_tab(scored: pd.DataFrame, model) -> None:
    st.subheader("Customer Detail: Risk, Reasons & Recommended Action")

    customer_id = st.selectbox("Select a customer", options=scored["customer_id"].tolist())
    row = scored[scored["customer_id"] == customer_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{row['churn_probability']:.1%}")
    col2.metric("Estimated Value at Risk", f"${row['customer_value_at_risk']:,.0f}")
    col3.metric("Segment", row["risk_value_segment"].replace("_", " ").title())

    sample, shap_values, feature_names = get_explainer_and_values(model, 42)

    if customer_id in sample["customer_id"].values:
        row_position = sample.index[sample["customer_id"] == customer_id][0]
        top_features = explain_customer(shap_values, row_position, feature_names, top_n=5)
        raw_row = sample.iloc[row_position]
        reasons = generate_reason_codes(top_features, raw_row)
        category = dominant_reason_category(top_features)
    else:
        reasons = []
        category = None
        st.info(
            "This customer is outside the cached explainability sample. "
            "Reason codes are computed on a representative 1,000-customer sample "
            "for dashboard responsiveness; the risk score and recommendation above are exact."
        )

    if reasons:
        st.markdown("#### Why this customer is flagged (top SHAP drivers)")
        for reason in reasons:
            st.markdown(f"- {reason}")

    recommendation = recommend_action(
        row["risk_value_segment"], row["churn_probability"], category
    )
    st.markdown("#### Recommended Action")
    st.info(f"**Priority: {recommendation['priority'].upper()}**\n\n{recommendation['action']}")
    st.caption(recommendation["rationale"])


def render_model_performance_tab(scored: pd.DataFrame) -> None:
    st.subheader("Model Performance")

    registry = load_registry()
    runs = registry.get("runs", [])
    champion = registry.get("champion")

    if not runs:
        st.warning("No training runs found in the model registry.")
        return

    metrics_df = pd.DataFrame(
        [
            {
                "model": run["model_name"],
                "roc_auc": run["metrics"].get("roc_auc"),
                "pr_auc": run["metrics"].get("average_precision"),
                "recall": run["metrics"].get("recall"),
                "precision": run["metrics"].get("precision"),
                "top10%_capture": run["metrics"].get("top10pct_capture_rate"),
                "top10%_lift": run["metrics"].get("top10pct_lift"),
            }
            for run in runs
        ]
    )
    if champion:
        st.success(
            f"Champion model: **{champion['model_name']}** "
            f"(selected on {champion['selection_metric']})"
        )
    st.dataframe(metrics_df.set_index("model"), width="stretch")

    st.markdown("#### Gain Chart (Champion Model, Test Set Proxy)")
    st.caption(
        "Decile ranking of the full scored population by predicted churn probability. "
        "Decile 1 is the highest-risk 10% of customers."
    )
    gains = gain_table(scored["churn"], scored["churn_probability"], n_bins=10)
    fig = px.line(
        gains,
        x="decile",
        y="cumulative_capture_rate",
        markers=True,
        title="Cumulative Churn Capture by Decile",
        labels={"decile": "Decile (1 = highest risk)", "cumulative_capture_rate": "Cumulative capture rate"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("📉 Churn Intelligence & Retention Decision System")
    st.caption(
        "Predicts churn risk, explains it, estimates value at risk, and recommends a retention action."
    )

    if not CHAMPION_MODEL_PATH.exists():
        render_missing_model_message()
        return

    scored = get_scored_customers()
    model = get_model()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Customer Explorer", "Customer Detail", "Model Performance"]
    )
    with tab1:
        render_overview_tab(scored)
    with tab2:
        render_customer_explorer_tab(scored)
    with tab3:
        render_customer_detail_tab(scored, model)
    with tab4:
        render_model_performance_tab(scored)


if __name__ == "__main__":
    main()
