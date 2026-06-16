# Data Dictionary

## Dataset

IBM Telco Customer Churn.

This dataset contains customer-level information from a fictional telecommunications company. It includes customer demographics, account information, subscribed services, charges, and whether the customer churned.

## Target Variable

| Raw Column | Clean Column | Description | Type |
|---|---|---|---|
| Churn | churn | Whether the customer churned. Converted from Yes/No to 1/0. | Binary |

## Raw Columns

| Raw Column | Clean Column | Description | Type |
|---|---|---|---|
| customerID | customer_id | Unique customer identifier. | String |
| gender | gender | Customer gender. | Categorical |
| SeniorCitizen | senior_citizen | Whether the customer is a senior citizen. | Binary |
| Partner | partner | Whether the customer has a partner. | Categorical |
| Dependents | dependents | Whether the customer has dependents. | Categorical |
| tenure | tenure | Number of months the customer has stayed with the company. | Numeric |
| PhoneService | phone_service | Whether the customer has phone service. | Categorical |
| MultipleLines | multiple_lines | Whether the customer has multiple phone lines. | Categorical |
| InternetService | internet_service | Customer internet service type. | Categorical |
| OnlineSecurity | online_security | Whether the customer has online security. | Categorical |
| OnlineBackup | online_backup | Whether the customer has online backup. | Categorical |
| DeviceProtection | device_protection | Whether the customer has device protection. | Categorical |
| TechSupport | tech_support | Whether the customer has technical support. | Categorical |
| StreamingTV | streaming_tv | Whether the customer has streaming TV. | Categorical |
| StreamingMovies | streaming_movies | Whether the customer has streaming movies. | Categorical |
| Contract | contract_type | Customer contract type. | Categorical |
| PaperlessBilling | paperless_billing | Whether the customer uses paperless billing. | Categorical |
| PaymentMethod | payment_method | Customer payment method. | Categorical |
| MonthlyCharges | monthly_charges | Monthly amount charged to the customer. | Numeric |
| TotalCharges | total_charges | Total amount charged to the customer. Converted to numeric during cleaning. | Numeric |
| Churn | churn | Whether the customer churned. | Binary |

## Cleaning Rules

| Step | Description |
|---|---|
| Column renaming | Raw column names are standardized to snake_case. |
| total_charges conversion | `total_charges` is converted from object/string to numeric. Invalid values are coerced to missing. |
| Missing total_charges | Rows with missing `total_charges` are removed. |
| churn encoding | `Yes` is mapped to `1`, and `No` is mapped to `0`. |
| customer ID | `customerID` is renamed to `customer_id`. |

## Engineered Features

These features will be created during the feature engineering phase.

| Feature | Description | Source |
|---|---|---|
| tenure_bucket | Customer tenure grouped into business-friendly buckets. | Derived |
| monthly_spend_band | Monthly charges grouped into low, medium, and high spend bands. | Derived |
| estimated_customer_value | Estimated customer value based on tenure and monthly charges. | Derived |
| customer_value_at_risk | Estimated value exposed to churn risk. | Derived |
| high_value_customer | Whether the customer belongs to a high-value segment. | Derived |
| contract_risk_score | Risk score based on contract type. | Derived |
| payment_risk_score | Risk score based on payment method. | Derived |
| service_count | Number of active services. | Derived |
| risk_value_segment | Segment combining churn risk and customer value. | Derived |

## Synthetic Features

No synthetic features are included in the initial MVP.

If synthetic features are added later, they must be clearly documented in this section.