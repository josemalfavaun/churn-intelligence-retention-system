# Modeling Report

## Objective

Predict which Telco customers are likely to churn, rank them so a retention
team can prioritize outreach under limited capacity, and support that
ranking with per-customer explanations and a recommended action.

## Data

7,032 customers after cleaning (see `docs/data_dictionary.md`), overall
churn rate 26.6%. Split 80/20 (train/test), stratified on `churn`,
`random_state=42` for reproducibility.

## Feature Set

`src/utils/config.py` defines the exact modeling feature contract:

- **Numeric (9):** `tenure`, `monthly_charges`, `total_charges`,
  `senior_citizen`, `service_count`, `contract_risk_score`,
  `payment_risk_score`, `high_value_customer`, `estimated_customer_value`.
- **Categorical (17, one-hot encoded):** demographics, service subscriptions,
  contract, billing, and payment fields, plus the engineered
  `tenure_bucket` and `monthly_spend_band` buckets.

`customer_value_at_risk` and `risk_value_segment` are **excluded** from the
model's inputs. At the feature-engineering stage they are built from a
tenure/contract-based proxy for risk (see `src/features/build_features.py`),
not from a trained model. Feeding a proxy of the target into the model that
predicts the target would be a subtle form of leakage. Instead,
`src/models/predict.py` recomputes both features *after* scoring, using the
model's real `churn_probability` in place of the proxy. This is the
intended design, not an oversight — see the code comments in
`src/utils/config.py` and `src/features/build_features.py`.

## Models Compared

Four candidates were trained on identical preprocessing (`StandardScaler`
for numeric features, `OneHotEncoder(handle_unknown="ignore")` for
categorical features, both inside one `sklearn.Pipeline` so the exact same
input contract is used for training, evaluation, and inference):

| Model | ROC-AUC | PR-AUC (avg. precision) | Recall | Precision | Top-10% Capture | Top-10% Lift |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.836 | 0.638 | 0.797 | 0.496 | 0.270 | 2.69 |
| Random Forest | 0.834 | 0.638 | 0.765 | 0.522 | 0.275 | 2.75 |
| **XGBoost (champion)** | 0.834 | **0.647** | 0.781 | 0.512 | **0.281** | **2.80** |
| LightGBM | 0.823 | 0.631 | 0.727 | 0.520 | 0.273 | 2.72 |

Class imbalance (~27% positive rate) is handled with `class_weight="balanced"`
for Logistic Regression, Random Forest, and LightGBM, and `scale_pos_weight`
(ratio of negative to positive training examples) for XGBoost — rather than
resampling (e.g. SMOTE). This keeps the training data's real distribution
intact and avoids synthetic examples the SHAP explanations would then have
to explain.

## Model Selection

Champion is selected on **PR-AUC (average precision)**, not ROC-AUC. With a
27% positive rate, ROC-AUC can look deceptively strong while still ranking
the minority (churn) class poorly; PR-AUC is more sensitive to exactly the
thing this system is for — ranking the customers who will actually churn
above the ones who won't, so a capacity-limited retention team spends effort
on the right people. See `src.models.model_registry.CHAMPION_METRIC`.

**Champion: XGBoost.** It wins on PR-AUC and on both top-10% capture rate
and lift, meaning it's the best of the four candidates at surfacing true
churners into the highest-priority slice — the population the retention
engine and dashboard actually act on.

## Business-Oriented Read

At the champion's 10%-of-portfolio operating point (contacting the 141
highest-risk test customers): precision@10% ≈ 74%, i.e. roughly 3 in 4
customers contacted in that slice are genuinely at risk, and that slice
captures ~28% of all churners in the test set — a 2.8x lift over contacting
a random 10% of the base.

`src/evaluation/business_metrics.py` turns this into an illustrative ROI
figure (`retention_campaign_roi`) using two explicit, documented
assumptions in `src/utils/config.py`:

- `RETENTION_OFFER_COST = 50.0` — assumed cost per customer contacted.
- `ASSUMED_SAVE_RATE = 0.30` — assumed share of contacted at-risk customers
  who are actually retained as a result of the offer.

These are placeholders for illustration, not measured constants. **This
figure ignores incremental lift** (some contacted customers might have
stayed regardless of the offer), so it should be read as a directional
business case, not a promised return. A real deployment would want a
holdout/control group to measure true incrementality before sizing a
campaign budget on this number.

## Explainability

`src/explainability/shap_explainer.py` computes SHAP values for the
champion pipeline (`shap.Explainer`'s tree path for XGBoost). Global
importance and per-customer top drivers both operate on the *transformed*
feature space (post one-hot encoding), and
`src/explainability/reason_codes.py` maps those back into
business-readable sentences (e.g. "Contract type: Month-to-month (increases
churn risk)") by parsing the `ColumnTransformer`'s output feature names
against the known numeric/categorical feature lists.

Consistently with the original EDA (`docs/retention_strategy.md`), the top
global drivers are contract commitment, tenure, and the absence of add-on
services (online security, tech support) — no surprises relative to the
business hypotheses formed during EDA, which is itself a useful sanity
check on the model.

## Limitations

- The dataset is a single snapshot (no time dimension), so the model cannot
  capture trends, seasonality, or a customer's trajectory over time.
- `estimated_customer_value` is a simple 12-month projection of the current
  monthly charge; it ignores discounts, upsell/downgrade, and non-billing
  value (referrals, brand effects).
- The retention campaign ROI is illustrative, not measured — see above.
- The recommendation engine (`src/recommendations/retention_engine.py`) is
  intentionally rule-based, not learned, so it stays transparent and
  auditable, but it also means it will not automatically improve as more
  outcome data becomes available; it should be revisited periodically
  against real campaign results.
- No hyperparameter search was run; each candidate uses one reasonable,
  fixed configuration. A tuned XGBoost would likely widen its lead further,
  but the relative model ranking is unlikely to flip.
