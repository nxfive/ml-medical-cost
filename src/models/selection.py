from pathlib import Path
from typing import Any, Tuple

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils.loading import load_metrics, load_model

MODELS_MAPPING: dict[str, tuple[type[BaseEstimator], str]] = {
    "RandomForestRegressor": (RandomForestRegressor, "rf"),
    "LinearRegression": (LinearRegression, "linear"),
    "DecisionTreeRegressor": (DecisionTreeRegressor, "tree"),
    "KNeighborsRegressor": (KNeighborsRegressor, "knn"),
}


def find_best_run(results_dir: str) -> Tuple[Path, dict[str, Any]]:
    """
    Find the run directory with the highest test R2.
    """
    results_dir = Path(results_dir)
    best_run = None
    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}

    for metrics_file in results_dir.glob("*/metrics.yaml"):
        data = load_metrics(metrics_file)
        r2 = data.get("metrics", {}).get("test_r2", float("-inf"))
        if r2 > best_score:
            best_score = r2
            best_run = metrics_file.parent
            best_metrics = data

    if best_run is None:
        raise ValueError(f"No valid metrics found in {results_dir}")

    return best_run, best_metrics


def pick_best(results_dir: str) -> Tuple[BaseEstimator, dict[str, Any]]:
    """
    Load the best trained pipeline and its metrics from a directory of runs.
    """
    best_run, metrics = find_best_run(results_dir)
    pipeline = load_model(best_run / "pipeline.pkl")
    return pipeline, metrics


def get_model_class_and_short(name: str) -> tuple[type[BaseEstimator], str | None]:
    """
    Returns the scikit-learn model class and its short alias based on the model name.
    """
    for full_name, (model_class, short_alias) in MODELS_MAPPING.items():
        if name == full_name:
            return model_class, short_alias
        elif name == short_alias:
            return model_class, None
    else:
        raise ValueError(f"Model '{name}' not found in MODELS_MAPPING")
