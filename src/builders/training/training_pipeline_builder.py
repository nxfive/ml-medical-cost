from sklearn.base import BaseEstimator

from src.builders.base.base_pipeline_builder import BasePipelineBuilder
from src.conf.schema import TrainingStageConfig
from src.containers.builder import TrainingBuildResult
from src.data.core import DataSaver
from src.io.file_ops import PathManager
from src.models.savers.run_saver import RunSaver
from src.training.train import TrainModel

from .training_builder import TrainingBuilder


class TrainingPipelineBuilder(
    BasePipelineBuilder[TrainingStageConfig, RunSaver, TrainingBuildResult]
):
    def _build_saver(self, data_saver: DataSaver) -> RunSaver:
        """
        Builds a RunSaver to save training runs.
        """
        return RunSaver(training_dir=self.cfg.training_dir, data_saver=data_saver)

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
        run_saver = self._build_saver(data_saver)
        model_spec = self._build_model_spec()
        training = self._build_training(model_spec.model_class)

        return TrainingBuildResult(
            model_spec=model_spec,
            run_saver=run_saver,
            loader=data_loader,
            training=training,
        )
