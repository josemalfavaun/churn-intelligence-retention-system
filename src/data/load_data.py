from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw IBM Telco Customer Churn dataset.

    Parameters
    ----------
    path:
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw churn dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download the IBM Telco Customer Churn dataset and place it in data/raw/."
        )

    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded raw data with shape: {df.shape}")
    print(df.head())