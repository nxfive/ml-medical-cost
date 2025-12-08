from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.data.core import DataLoader, DataSaver
from src.data.types import SplitData
from src.evaluation.metrics import flatten_metrics, get_metrics
from src.factories.io_factory import IOFactory
from src.factories.training_factory import TrainingFactory
from src.io.file_ops import PathManager
from src.mlflow.logging import log_model, setup_mlflow
from src.models.registry import get_model_class_and_short
from src.models.savers.run_saver import RunSaver
from src.models.types import ModelLog, ModelResult, ModelRun
from src.patterns.base_pipeline import BasePipeline
from src.patterns.types import BuildResult

from .train import TrainModel
from .types import TrainResult
from .validation import ModelDiagnostics


@dataclass
class TrainingBuildResult(BuildResult):
    model_class: type[BaseEstimator]
    training: TrainModel
    model_name: str = field(init=False)

    def __post_init__(self):
        self.model_name = self.model_class.__name__


class TrainingPipeline(BasePipeline[TrainingBuildResult, None]):
    def __init__(self, cfg: TrainingStageConfig):
        self.cfg = cfg

    def build(self) -> TrainingBuildResult:
        """
        Constructs and returns the components required for training pipeline.
        """
        PathManager.ensure_dir(self.cfg.training_dir.output_dir)
        model_class, _ = get_model_class_and_short(self.cfg.model.name)
        writers = IOFactory.create_writers()
        readers = IOFactory.create_readers()
        data_loader = DataLoader(readers)
        data_saver = DataSaver(writers)
        training = TrainingFactory.create(model_class, self.cfg)
        return TrainingBuildResult(
            model_class=model_class,
            saver=data_saver,
            loader=data_loader,
            training=training,
        )

    def load_data(self, data_loader: DataLoader) -> SplitData:
        """
        Loads preprocessed training and test data using the provided readers.
        """
        return data_loader.load_splitted_data(
            processed_dir=self.cfg.data_dir.processed_dir
        )

    @staticmethod
    def _save_run(
        train_result: TrainResult,
        run_saver: RunSaver,
        flatten_metrics: dict[str, float],
        model_name: str,
    ) -> None:
        """
        Saves the training results and pipeline to persistent storage.
        """
        run_saver.save(
            ModelRun(
                result=ModelResult(
                    model_name=model_name,
                    param_grid=train_result.param_grid,
                    folds_scores_mean=float(
                        train_result.runner_result.folds_scores_mean
                    ),
                    metrics=flatten_metrics,
                    transformation=train_result.transformation,
                ),
                estimator=train_result.runner_result.trained,
            )
        )

    @staticmethod
    def _log_model(
        train_result: TrainResult,
        flatten_metrics: dict[str, float],
        X_train: pd.DataFrame,
        model_class: type[BaseEstimator],
    ) -> None:
        """
        Logs the training results and metrics to MLflow.
        """
        log_model(
            ModelLog(
                model_class=model_class,
                estimator=train_result.runner_result.trained,
                X_train=X_train,
                param_grid=train_result.param_grid,
                transformation=train_result.transformation,
                metrics=flatten_metrics,
                folds_scores=train_result.runner_result.folds_scores,
                folds_scores_mean=train_result.runner_result.folds_scores_mean,
            )
        )

    def run(self) -> None:
        """
        Trains and evaluates all defined models using cross-validation, logs results to MLflow
        and saves pipeline and training results to disk.
        """
        setup_mlflow()
        builder = self.build()
        split_data = self.load_data(builder.loader)

        run_saver = RunSaver(
            trainig_dir=self.cfg.training_dir, data_saver=builder.saver
        )

        for train_result in builder.training.run(
            X_train=split_data.X_train,
            X_test=split_data.X_test,
            y_train=split_data.y_train,
        ):

            metrics = get_metrics(
                y_train=split_data.y_train,
                y_test=split_data.y_test,
                train_predictions=train_result.runner_result.train_predictions,
                test_predictions=train_result.runner_result.test_predictions,
            )
            diagnostics = ModelDiagnostics(
                folds_scores=train_result.runner_result.folds_scores
            )
            diagnostics.report(
                model_name=builder.model_name,
                train_r2=metrics.train.r2,
                test_r2=metrics.test.r2,
            )
            fm = flatten_metrics(metrics)
            self._log_model(
                train_result,
                flatten_metrics=fm,
                X_train=split_data.X_train,
                model_class=builder.model_class,
            )
            self._save_run(
                train_result,
                run_saver,
                flatten_metrics=fm,
                model_name=builder.model_name,
            )
