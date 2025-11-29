from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import yaml
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


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

    joblib.dump(model, models_path / f"{file_name}.pkl")

    metadata = {
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

    with open(metadata_path / f"{file_name}.yml", "w") as f:
        yaml.safe_dump(metadata, f)


def save_run(
    results: dict, pipeline: Pipeline | TransformedTargetRegressor, cfg: DictConfig
):
    """
    Saves the training run results and the trained pipeline to disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_path = Path(cfg.training.output_dir) / timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_path = results_path / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.safe_dump(results, f)

    pipeline_path = results_path / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
