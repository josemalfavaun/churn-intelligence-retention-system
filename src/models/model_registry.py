"""Lightweight JSON-backed registry for tracking trained model runs.

The registry does not store the model objects themselves (those are saved
separately with joblib). It stores each run's metrics so different model
candidates can be compared and a champion selected transparently.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import MODEL_REGISTRY_PATH

# Metric used to rank candidate models and pick the champion. Average
# precision (PR-AUC) is preferred over ROC-AUC here because churn is
# imbalanced (~27% positive rate) and PR-AUC is more sensitive to how well
# the model ranks the minority (churn) class, which is what the business
# actually acts on. See docs/modeling_report.md.
CHAMPION_METRIC = "average_precision"


def load_registry(path: str | Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"runs": [], "champion": None}
    with open(path) as f:
        return json.load(f)


def save_registry(registry: dict[str, Any], path: str | Path = MODEL_REGISTRY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_run(
    registry: dict[str, Any],
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a training run to the registry (in memory) and return the run entry."""
    run = {
        "model_name": model_name,
        "metrics": metrics,
        "params": params or {},
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    registry.setdefault("runs", []).append(run)
    return run


def select_champion(
    registry: dict[str, Any], metric: str = CHAMPION_METRIC
) -> dict[str, Any] | None:
    """Pick the run with the highest value of `metric` and mark it as champion."""
    runs = registry.get("runs", [])
    if not runs:
        registry["champion"] = None
        return None

    champion = max(runs, key=lambda run: run["metrics"].get(metric, float("-inf")))
    registry["champion"] = {
        "model_name": champion["model_name"],
        "metrics": champion["metrics"],
        "selection_metric": metric,
    }
    return registry["champion"]


def get_champion(registry: dict[str, Any]) -> dict[str, Any] | None:
    return registry.get("champion")
