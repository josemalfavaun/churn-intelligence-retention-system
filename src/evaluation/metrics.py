"""Classification and ranking metrics for the churn model.

Includes standard classification metrics plus top-k / gain metrics, which
matter more for this project than raw accuracy: retention teams can only
act on a limited number of customers, so how well the model ranks the
highest-risk customers into the top slice is the operationally relevant
question.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute standard classification metrics for a binary churn model."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def top_k_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    k_pct: float = 0.1,
) -> dict[str, float]:
    """
    Metrics for the top k% of customers ranked by predicted churn probability.

    - precision_at_k: of the customers flagged, what share actually churned.
    - capture_rate (a.k.a. recall_at_k): of all churners, what share fall in
      the top k%.
    - lift: precision_at_k relative to the overall base churn rate. A lift of
      2.0 means the top slice churns twice as often as a random sample.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_proba = pd.Series(y_proba).reset_index(drop=True)

    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))

    ranked_idx = y_proba.sort_values(ascending=False).index[:k]
    top_true = y_true.loc[ranked_idx]

    base_rate = y_true.mean()
    precision_at_k = top_true.mean() if k > 0 else 0.0
    total_churners = y_true.sum()
    capture_rate = (top_true.sum() / total_churners) if total_churners > 0 else 0.0
    lift = (precision_at_k / base_rate) if base_rate > 0 else 0.0

    return {
        "k_pct": k_pct,
        "k_customers": k,
        "precision_at_k": float(precision_at_k),
        "capture_rate": float(capture_rate),
        "lift": float(lift),
        "base_rate": float(base_rate),
    }


def gain_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Decile gain table: customers ranked by predicted probability and split
    into n_bins equal-sized groups (decile 1 = highest risk).

    Returns a DataFrame with, per decile: customer count, churners captured,
    churn rate within the decile, and cumulative capture rate.
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba)}
    ).sort_values("y_proba", ascending=False).reset_index(drop=True)

    df["decile"] = (np.floor(df.index * n_bins / len(df)) + 1).astype(int)
    df["decile"] = df["decile"].clip(upper=n_bins)

    total_churners = df["y_true"].sum()

    summary = (
        df.groupby("decile")
        .agg(customers=("y_true", "size"), churners=("y_true", "sum"))
        .reset_index()
    )
    summary["churn_rate"] = summary["churners"] / summary["customers"]
    summary["cumulative_churners"] = summary["churners"].cumsum()
    summary["cumulative_capture_rate"] = (
        summary["cumulative_churners"] / total_churners if total_churners > 0 else 0.0
    )

    return summary
