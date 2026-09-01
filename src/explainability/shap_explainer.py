"""SHAP-based explainability for the champion model.

Provides both global feature importance (what drives churn overall) and
per-customer explanations (why this specific customer is flagged), which
src/explainability/reason_codes.py turns into human-readable text.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.utils.config import ALL_MODEL_FEATURES


def _get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after the ColumnTransformer (numeric + one-hot columns)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def build_explainer(pipeline: Pipeline, background_X: pd.DataFrame) -> shap.Explainer:
    """
    Build a SHAP explainer for the trained model inside the pipeline.

    Uses shap.Explainer's generic dispatch, which selects TreeExplainer for
    tree-based models (RandomForest/XGBoost/LightGBM) and an appropriate
    fallback for linear models. The explainer operates on the *transformed*
    feature space, since that is what the underlying model actually sees.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    background_transformed = preprocessor.transform(background_X[ALL_MODEL_FEATURES])

    # Dense array: SHAP's tree explainers expect dense input, and the
    # background sample here is small enough that this is cheap.
    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    return shap.Explainer(model, background_transformed)


def compute_shap_values(
    pipeline: Pipeline, X: pd.DataFrame, explainer: shap.Explainer | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for every row in X.

    Returns (shap_values_array, feature_names) where shap_values_array has
    shape (n_rows, n_transformed_features).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X[ALL_MODEL_FEATURES])
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    if explainer is None:
        explainer = build_explainer(pipeline, X)

    shap_result = explainer(X_transformed)
    values = shap_result.values

    # Binary classifiers via shap.Explainer sometimes return shape
    # (n_rows, n_features, n_classes); keep the positive class (churn=1).
    if values.ndim == 3:
        values = values[:, :, 1]

    feature_names = _get_transformed_feature_names(pipeline)
    return values, feature_names


def global_feature_importance(
    shap_values: np.ndarray, feature_names: list[str], top_n: int = 15
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, the standard global importance view."""
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def explain_customer(
    shap_values: np.ndarray,
    row_index: int,
    feature_names: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Top contributing features for a single customer (by row position, not
    customer_id), sorted by absolute SHAP contribution.

    Returns a list of (feature_name, shap_value) tuples. A positive
    shap_value pushes the prediction toward churn; negative pushes toward
    staying.
    """
    row_values = shap_values[row_index]
    order = np.argsort(-np.abs(row_values))[:top_n]
    return [(feature_names[i], float(row_values[i])) for i in order]
