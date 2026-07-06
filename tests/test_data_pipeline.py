from pathlib import Path

import pandas as pd
import pytest

from src.data.clean_data import clean_telco_data, save_clean_data
from src.data.load_data import load_raw_data


def test_load_raw_data_raises_file_not_found_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)


def test_load_raw_data_reads_small_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    expected = pd.DataFrame(
        {
            "customerID": ["C0001", "C0002"],
            "Churn": ["Yes", "No"],
        }
    )
    expected.to_csv(input_path, index=False)

    result = load_raw_data(input_path)

    pd.testing.assert_frame_equal(result, expected)


def test_clean_telco_data_standardizes_columns_converts_total_charges_and_maps_churn() -> None:
    raw_df = pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003"],
            "SeniorCitizen": [0, 1, 0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "MonthlyCharges": [29.85, 56.95, 42.30],
            "TotalCharges": ["29.85", "", "1889.50"],
            "Churn": ["Yes", "No", "Yes"],
        }
    )

    cleaned = clean_telco_data(raw_df)

    assert list(cleaned.columns) == [
        "customer_id",
        "senior_citizen",
        "contract_type",
        "monthly_charges",
        "total_charges",
        "churn",
    ]
    assert cleaned.shape == (2, 6)
    assert cleaned["total_charges"].tolist() == [29.85, 1889.50]
    assert cleaned["churn"].tolist() == [1, 1]
    assert cleaned["customer_id"].tolist() == ["0001", "0003"]
    assert pd.api.types.is_numeric_dtype(cleaned["total_charges"])


def test_save_clean_data_writes_csv_to_tmp_path(tmp_path: Path) -> None:
    output_path = tmp_path / "processed" / "telco_churn_clean.csv"
    df = pd.DataFrame(
        {
            "customer_id": ["C0001"],
            "senior_citizen": [0],
            "contract_type": ["Month-to-month"],
            "monthly_charges": [29.85],
            "total_charges": [29.85],
            "churn": [1],
        }
    )

    save_clean_data(df, output_path)

    assert output_path.exists()
    saved = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(saved, df)
