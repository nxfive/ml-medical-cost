from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import yaml
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


def build_metadata(model_name: str, cfg: DictConfig, params: dict, metrics: dict) -> dict:
    return {
        "model_name": model_name,
        "version": "1.0",
        "date_trained": datetime.today().strftime("%Y-%m-%d"),
        "features_processed": {
            "cat_features": list(cfg.features.categorical),
            "num_features": list(cfg.features.numeric),
            "bin_features": list(cfg.features.binary),
        },
        "params": params or {},
        "metrics": metrics,
    }


def save_metrics(metrics: dict, path: Path):
    """
    Save metrics dictionary to a YAML file on disk.
    """
    with open(path, "w") as f:
        yaml.safe_dump(metrics, f)


def save_model(model: BaseEstimator, path: Path):
    """
    Save any scikit-learn estimator or pipeline to disk.
    """
    joblib.dump(model, path)


def save_model_with_metadata(
    model: Any,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, float | int],
    cfg: DictConfig,
):
    """
    Save model and corresponding metadata to disk.
    """
    file_name = model_name.lower()
    models_path = Path(cfg.models.output_dir)
    metadata_path = models_path / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    metadata = build_metadata(model_name, cfg, params, metrics)
    save_model(model, models_path / f"{file_name}.pkl")
    save_metrics(metadata, metadata_path / f"{file_name}.yml")


def save_run(
    results: dict, pipeline: Pipeline | TransformedTargetRegressor, cfg: DictConfig
):
    """
    Saves the training run results and the trained pipeline to disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_path = Path(cfg.training.output_dir) / timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    save_metrics(results, results_path / "metrics.yaml")
    save_model(pipeline, results_path / "pipeline.pkl")
