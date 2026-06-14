# Data Dictionary

## Dataset

IBM Telco Customer Churn.

## Target Variable

| Column | Description | Type |
|---|---|---|
| Churn | Whether the customer churned or not | Binary |

## Raw Columns

| Column | Description | Type |
|---|---|---|
| customerID | Unique customer identifier | String |
| gender | Customer gender | Categorical |
| SeniorCitizen | Whether the customer is a senior citizen | Binary |
| Partner | Whether the customer has a partner | Categorical |
| Dependents | Whether the customer has dependents | Categorical |
| tenure | Number of months the customer has stayed with the company | Numeric |
| PhoneService | Whether the customer has phone service | Categorical |
| MultipleLines | Whether the customer has multiple lines | Categorical |
| InternetService | Customer internet service type | Categorical |
| OnlineSecurity | Whether the customer has online security | Categorical |
| OnlineBackup | Whether the customer has online backup | Categorical |
| DeviceProtection | Whether the customer has device protection | Categorical |
| TechSupport | Whether the customer has tech support | Categorical |
| StreamingTV | Whether the customer has streaming TV | Categorical |
| StreamingMovies | Whether the customer has streaming movies | Categorical |
| Contract | Customer contract type | Categorical |
| PaperlessBilling | Whether the customer uses paperless billing | Categorical |
| PaymentMethod | Customer payment method | Categorical |
| MonthlyCharges | Monthly amount charged to the customer | Numeric |
| TotalCharges | Total amount charged to the customer | Numeric |
| Churn | Whether the customer churned | Binary |

## Engineered Features

These features will be created during the feature engineering phase.

| Feature | Description | Source |
|---|---|---|
| tenure_bucket | Customer tenure grouped into business-friendly buckets | Derived |
| monthly_spend_band | Monthly charges grouped into low, medium, and high spend bands | Derived |
| estimated_customer_value | Estimated value of the customer based on charges and tenure | Derived |
| customer_value_at_risk | Estimated value exposed to churn risk | Derived |
| high_value_customer | Whether the customer belongs to a high-value segment | Derived |
| contract_risk_score | Risk score based on contract type | Derived |
| payment_risk_score | Risk score based on payment method | Derived |
| service_count | Number of active services | Derived |
| risk_value_segment | Segment combining churn risk and customer value | Derived |

## Synthetic Features

No synthetic features are included in the initial MVP.

If synthetic features are added later, they must be clearly documented in this section.