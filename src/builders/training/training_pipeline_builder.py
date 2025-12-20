from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.containers.builder import TrainingBuildResult
from src.data.core import DataLoader, DataSaver
from src.factories.io_factory import IOFactory
from src.factories.model_factory import ModelFactory
from src.io.file_ops import PathManager
from src.models.savers.run_saver import RunSaver
from src.models.spec import ModelSpec
from src.training.train import TrainModel

from .training_builder import TrainingBuilder


class TrainingPipelineBuilder:
    def __init__(self, cfg: TrainingStageConfig):
        self.cfg = cfg

    def _build_data_io(self) -> tuple[DataLoader, DataSaver]:
        """
        Builds the DataLoader and DataSaver using IOFactory readers/writers.
        """
        readers = IOFactory.create_readers()
        writers = IOFactory.create_writers()
        return DataLoader(readers), DataSaver(writers)

    def _build_run_saver(self, data_saver: DataSaver) -> RunSaver:
        """
        Builds a RunSaver to save training runs.
        """
        return RunSaver(trainig_dir=self.cfg.training_dir, data_saver=data_saver)

    def _build_model_spec(self) -> ModelSpec:
        """
        Builds the model specification from the ModelFactory.
        """
        return ModelFactory.get_spec(self.cfg.model.name)

    def _build_training(self, model_class: type[BaseEstimator]) -> TrainModel:
        """
        Builds a TrainModel object for the given model class and config.
        """
        return TrainingBuilder.build(model_class, self.cfg)

    def build(self) -> TrainingBuildResult:
        """
        Builds all components for the training pipeline:
        - ensures the training directory exists
        - DataLoader and DataSaver
        - RunSaver for saving training runs
        - ModelSpec from ModelFactory
        - TrainModel for the given model class and config
        """
        PathManager.ensure_dir(self.cfg.training_dir.output_dir)
        data_loader, data_saver = self._build_data_io()
        run_saver = self._build_run_saver(data_saver)
        model_spec = self._build_model_spec()
        training = self._build_training(model_spec.model_class)

        return TrainingBuildResult(
            model_spec=model_spec,
            run_saver=run_saver,
            loader=data_loader,
            training=training,
        )
