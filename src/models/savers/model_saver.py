from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator

from src.data.core import DataSaver
from src.io.file_ops import PathManager


class ModelSaver:
    def __init__(self, models_dir: Path, data_saver: DataSaver):
        self.models_dir = models_dir
        self.data_saver = data_saver

    def save_model_with_metadata(
        self, model: BaseEstimator, metadata: dict[str, Any]
    ) -> None:
        """
        Save model and metadata to disk using `model_name` as the base filename.
        """
        file_name = metadata.get("model_name").lower()
        metadata_dir = self.models_dir / "metadata"
        PathManager.ensure_dir(metadata_dir)

        self.data_saver.save_model(model, self.models_dir / f"{file_name}.pkl")
        self.data_saver.save_metrics(metadata, metadata_dir / f"{file_name}.yml")
