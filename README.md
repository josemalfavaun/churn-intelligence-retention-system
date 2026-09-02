# Churn Intelligence & Retention Decision System

An end-to-end data science and machine learning system that predicts customer churn, explains customer-level risk drivers, estimates business value at risk, and recommends retention actions.

## Project Goal

This project goes beyond a traditional churn prediction notebook. The goal is to build a decision-support system for retention, growth, product, or customer success teams.

The system is designed to answer:

- Which customers are most likely to churn?
- Why are they at risk?
- Which customer segments are most vulnerable?
- How much business value is at risk?
- Which customers should be contacted first?
- What retention action should be recommended?

## Core Capabilities

- Customer churn prediction
- Business-oriented model evaluation
- Top-k prioritization metrics
- SHAP interpretability
- Customer-level reason codes
- Retention recommendation engine
- Streamlit dashboard
- Professional documentation and testing

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- XGBoost / LightGBM
- SHAP
- Streamlit
- pytest

## Dataset

The initial dataset will be the IBM Telco Customer Churn dataset.

The dataset includes customer-level information such as tenure, contract type, payment method, monthly charges, total charges, services subscribed, and churn label.

## Project Status

Complete. All phases in `docs/project_plan.md` (data acquisition through
portfolio packaging) are implemented and tested.

## Quickstart

```bash
make install                       # install dependencies
python -m src.features.build_features   # optional: inspect the feature table
python -m src.models.train_model        # trains 4 candidates, saves the champion
python -m src.models.predict            # scores all customers with the champion
make test                          # 36 tests across data, features, models, explainability, recommendations
make run-app                       # launch the Streamlit dashboard
```

The dashboard needs a trained model first (`python -m src.models.train_model`);
it will tell you if one isn't found yet.

## Results Summary

Champion model: **XGBoost**, selected on PR-AUC (0.647) because churn is
imbalanced (~27% positive rate) and PR-AUC better reflects ranking quality
for the minority class than ROC-AUC does. At a 10%-of-portfolio contact
rate, the champion captures ~28% of churners with ~74% precision (2.8x lift
over random). Full comparison and business framing in
`docs/modeling_report.md`.

## Project Structure

```
src/
  data/            raw loading + cleaning
  features/        business feature engineering (tenure buckets, risk scores, value at risk, ...)
  models/          training, comparison, registry, champion persistence, scoring
  evaluation/      classification + business/ROI metrics
  explainability/  SHAP values -> human-readable reason codes
  recommendations/ rule-based retention action engine
app/
  streamlit_app.py dashboard: overview, customer explorer, customer detail, model performance
docs/              data dictionary, EDA-driven retention strategy, modeling report, project plan
tests/             36 tests covering data, features, models, explainability, recommendations
```