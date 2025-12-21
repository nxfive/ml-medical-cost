from pathlib import Path

from src.conf.schema import FeaturesConfig
from src.containers.results import StageResult
from src.data.core import DataSaver
from src.io.file_ops import PathManager
from src.serializers.model_metadata import ModelMetadataSerializer


class ModelSaver:
    def __init__(self, models_dir: Path, data_saver: DataSaver):
        self.models_dir = models_dir
        self.data_saver = data_saver

    def save_model_with_metadata(
        self, result: StageResult, features: FeaturesConfig
    ) -> None:
        """
        Save model and metadata to disk using `model_name` as the base filename.
        """
        file_name = result.model_name.lower()
        metadata_dir = self.models_dir / "metadata"
        PathManager.ensure_dir(metadata_dir)

        metadata = ModelMetadataSerializer.from_stage(result, features=features)
        metadata_dict = ModelMetadataSerializer.to_dict(metadata)

        model_path = self.models_dir / f"{file_name}.pkl"
        metadata_path = metadata_dir / f"{file_name}.yml"

        self.data_saver.save_model(result.estimator, model_path)
        self.data_saver.save_metrics(metadata_dict, metadata_path)
