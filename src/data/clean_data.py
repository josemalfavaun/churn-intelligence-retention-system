from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_data


PROCESSED_DATA_PATH = Path("data/processed/telco_churn_clean.csv")


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the IBM Telco Customer Churn dataset.

    Cleaning steps:
    - Standardize column names.
    - Convert TotalCharges to numeric.
    - Convert Churn to binary target.
    - Remove rows with missing TotalCharges.
    - Rename customerID to customer_id.
    """
    df = df.copy()

    df = df.rename(
        columns={
            "customerID": "customer_id",
            "SeniorCitizen": "senior_citizen",
            "PhoneService": "phone_service",
            "MultipleLines": "multiple_lines",
            "InternetService": "internet_service",
            "OnlineSecurity": "online_security",
            "OnlineBackup": "online_backup",
            "DeviceProtection": "device_protection",
            "TechSupport": "tech_support",
            "StreamingTV": "streaming_tv",
            "StreamingMovies": "streaming_movies",
            "PaperlessBilling": "paperless_billing",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charges",
            "TotalCharges": "total_charges",
            "Contract": "contract_type",
            "Churn": "churn",
        }
    )

    df.columns = [col.lower() for col in df.columns]

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df = df.dropna(subset=["total_charges"]).reset_index(drop=True)

    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    return df


def save_clean_data(
    df: pd.DataFrame,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> None:
    """
    Save cleaned dataset to processed data folder.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_telco_data(raw_df)
    save_clean_data(clean_df)

    print(f"Clean dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Clean data shape: {clean_df.shape}")
    print(clean_df.head())
    print(clean_df.dtypes)