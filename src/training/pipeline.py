from src.builders.training.training_pipeline_builder import \
    TrainingPipelineBuilder
from src.conf.schema import TrainingStageConfig
from src.containers.builder import TrainingBuildResult
from src.containers.results import StageResult
from src.evaluation.metrics import flatten_metrics
from src.mlflow.logger import logger
from src.models.savers.run_saver import RunSaver
from src.patterns.base_pipeline import BasePipeline
from src.serializers.prediction_set import PredictionSetSerializer
from src.serializers.stage_result import StageResultSerializer

from .validation import ModelDiagnostics


class TrainingPipeline(BasePipeline[TrainingBuildResult, None]):
    def __init__(self, cfg: TrainingStageConfig):
        self.cfg = cfg

    def build(self) -> TrainingBuildResult:
        """
        Constructs and returns the components required for training pipeline.
        """
        builder = TrainingPipelineBuilder(self.cfg)
        return builder.build()

    @staticmethod
    def _save_run(run_saver: RunSaver, stage_result: StageResult) -> None:
        """
        Saves the training results and pipeline to persistent storage.
        """
        run_saver.save(stage_result)

    def run(self) -> None:
        """
        Trains and evaluates all defined models using cross-validation, logs results
        to MLflow and saves pipeline and training results to disk.
        """
        builder = self.build()
        split_data = self.load_data(builder.loader)
        model_name = builder.model_spec.model_class.__name__

        for train_result in builder.training.run(
            X_train=split_data.X_train,
            X_test=split_data.X_test,
            y_train=split_data.y_train,
        ):
            pred_set = PredictionSetSerializer.from_stage_pipeline(
                result=train_result.runner_result, split_data=split_data
            )
            metrics = self._compute_metrics(pred_set)

            diagnostics = ModelDiagnostics(
                folds_scores=train_result.runner_result.folds_scores
            )
            diagnostics.report(
                model_name=model_name,
                train_r2=metrics.train.r2,
                test_r2=metrics.test.r2,
            )
            stage_result = StageResultSerializer.from_stage(
                result=train_result,
                metrics=flatten_metrics(metrics),
                model_name=model_name,
            )
            self._log_model(
                logger,
                stage_result,
                X_train=split_data.X_train,
            )
            self._save_run(
                run_saver=builder.run_saver,
                stage_result=stage_result,
            )
