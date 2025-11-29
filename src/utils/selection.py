from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


def pick_best(
    results_dir: str,
) -> tuple[Pipeline | TransformedTargetRegressor, dict[str, Any]]:
    """
    Loads the best trained pipeline and its metrics from a directory of runs.
    """
    results_dir = Path(results_dir)
    best_run = None
    best_score = float("-inf")

    for run_dir in results_dir.iterdir():
        metrics_file = run_dir / "metrics.yaml"
        if metrics_file.exists():
            data = yaml.safe_load(metrics_file.read_text())
            r2 = data["metrics"].get("test_r2", float("-inf"))
            if r2 > best_score:
                best_score = r2
                best_run = run_dir

    if best_run is None:
        raise ValueError(f"No valid metrics found in {results_dir}")

    pipeline_file = best_run / "pipeline.pkl"
    pipeline = joblib.load(pipeline_file)

    metrics = yaml.safe_load((best_run / "metrics.yaml").read_text())

    return pipeline, metrics


def get_model_class_and_short(name: str) -> tuple[type[BaseEstimator], str | None]:
    """
    Returns the scikit-learn model class and its short alias based on the model name.
    """
    models_mapping = {
        "RandomForestRegressor": (RandomForestRegressor, "rf"),
        "LinearRegression": (LinearRegression, "linear"),
        "DecisionTreeRegressor": (DecisionTreeRegressor, "tree"),
        "KNeighborsRegressor": (KNeighborsRegressor, "knn"),
    }

    for full_name, (model_class, short_alias) in models_mapping.items():
        if name == full_name:
            return model_class, short_alias
        elif name == short_alias:
            return model_class, None
    else:
        raise ValueError(f"Model '{name}' not found in models_mapping")
