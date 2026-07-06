# Retention Strategy

## Business Problem

The objective of this project is to identify which customers are most likely to churn and which segments should be prioritized for retention intervention. The business goal is not just prediction accuracy; it is to support better retention decisions by combining churn risk, customer value, and actionable customer attributes.

The cleaned Telco dataset contains 7,032 customers with an overall churn rate of 26.6%. This creates a meaningful retention problem with enough signal to support both modeling and operational decision rules.

## Initial EDA-Based Insights

The exploratory analysis points to a small number of clear patterns:

- Churn is heavily concentrated in month-to-month contracts.
- Early-tenure customers are materially more vulnerable than long-tenured customers.
- Payment method appears to separate higher-risk and lower-risk customers.
- Higher monthly charges are associated with stronger churn sensitivity in several segments.
- Support and security services appear linked to lower churn, suggesting that service depth and product stickiness matter.

These patterns are business-relevant because they identify retention levers that can be acted on before a customer leaves.

## Customer Segments of Interest

The first segments to monitor closely are:

- Month-to-month customers, especially those in the first 6 to 12 months.
- Customers with high monthly charges and weak service attachment.
- Customers using payment methods that appear associated with higher churn.
- Customers without technical support, online security, online backup, or device protection.
- Customers in high-risk contract and spend combinations where the value at risk is likely higher.

These segments are candidates for targeted offers, proactive support, or human review depending on their estimated business value.

## Initial Retention Hypotheses

The EDA supports the following working hypotheses:

- Month-to-month customers may respond to contract upgrade or renewal incentives.
- Early-tenure customers may need stronger onboarding, education, and first-90-day engagement.
- High-spend high-risk customers may deserve priority treatment because they are more valuable to save.
- Customers without support or security services may benefit from proactive service activation or bundle recommendations.
- Retention should be prioritized using both churn risk and customer value, not risk alone.

These hypotheses are intentionally practical and should be tested further in feature engineering, modeling, and recommendation logic.

## How These Insights Will Inform Feature Engineering

The EDA suggests several feature directions for the next phase:

- Tenure-based buckets to capture early-life, mid-life, and mature customers.
- Spend bands to represent price sensitivity and commercial importance.
- Contract and payment indicators to encode commitment and billing behavior.
- Service-availability indicators to capture product depth and stickiness.
- Segment-level risk combinations that support prioritization logic.

These features will help the modeling phase move beyond raw fields and toward business-ready retention signals.

## Limitations

- The EDA is observational, so it identifies associations rather than causal drivers.
- The estimated value at risk in the notebook is only a proxy, not a full financial model.
- The dataset is a single snapshot, so seasonality and time-based effects are not captured.
- Customer behavior may change over time, which means these patterns should be revalidated during model monitoring and dashboard use.
- Churn risk should always be interpreted together with customer value and operational cost.

This document will be refined as feature engineering and modeling produce more evidence.
