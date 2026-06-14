# Project Plan

## Project Name

Churn Intelligence & Retention Decision System

## Objective

Build an end-to-end machine learning system that predicts customer churn, explains risk drivers, estimates business value at risk, and recommends retention actions.

## Business Problem

Companies lose revenue when customers churn. Many businesses have customer, payment, contract, and usage data, but lack a structured way to convert that data into retention decisions.

This project aims to help retention, growth, product, and customer success teams prioritize customers at risk and recommend actions based on churn probability and business value.

## Dataset

Initial dataset: IBM Telco Customer Churn.

The dataset includes customer-level information such as tenure, contract type, payment method, monthly charges, total charges, services subscribed, and churn label.

## MVP Scope

The MVP includes:

- Data ingestion
- Data cleaning
- Feature engineering
- Exploratory Data Analysis
- Baseline churn model
- Advanced model comparison
- Business-oriented metrics
- SHAP interpretability
- Customer-level reason codes
- Retention recommendation engine
- Streamlit dashboard
- Basic tests
- Professional documentation

## Out of Scope for Initial MVP

The following items are optional extensions:

- FastAPI
- PostgreSQL
- MLflow
- Docker
- GitHub Actions
- LLM-generated executive explanations

## Project Phases

### Phase 0 — Project definition and repo setup

Create the repository structure, documentation base, dependencies, and GitHub repository.

### Phase 1 — Data acquisition and data dictionary

Download the dataset, inspect columns, document variables, and create the first data loading utilities.

### Phase 2 — EDA and business framing

Analyze churn rate, vulnerable customer segments, contract risk, payment risk, and customer value patterns.

### Phase 3 — Feature engineering

Create business and modeling features such as tenure buckets, monthly spend bands, contract risk score, payment risk score, service count, high value flag, and value at risk.

### Phase 4 — Baseline model

Train a Logistic Regression baseline and evaluate with classification and business metrics.

### Phase 5 — Advanced models and evaluation

Compare Random Forest and XGBoost/LightGBM. Evaluate ROC-AUC, PR-AUC, recall, F1, lift, gain, and top-k metrics.

### Phase 6 — SHAP interpretability and reason codes

Generate global and customer-level explanations using SHAP and convert explanations into reason codes.

### Phase 7 — Retention recommendation engine

Create a rule-based recommendation system that combines churn risk, customer value, and reason codes into retention actions.

### Phase 8 — Streamlit dashboard

Build an interactive dashboard for business users.

### Phase 9 — Tests and engineering polish

Add tests, optional Docker, optional GitHub Actions, and repo cleanup.

### Phase 10 — Portfolio packaging

Create final README, executive summary, LinkedIn post, and CV bullets.