from src.builders.optuna.optuna_pipeline_builder import OptunaPipelineBuilder
from src.conf.schema import FeaturesConfig, OptunaStageConfig
from src.containers.builder import OptunaBuildResult
from src.containers.results import StageResult
from src.evaluation.metrics import flatten_metrics
from src.factories.model_factory import ModelFactory
from src.mlflow.logging import setup_mlflow
from src.models.savers.model_saver import ModelSaver
from src.patterns.base_pipeline import BasePipeline
from src.serializers.experiment import ExperimentSerializer
from src.serializers.prediction_set import PredictionSetSerializer
from src.serializers.stage_result import StageResultSerializer

from .manager import OptunaExperimentManager


class OptunaPipeline(BasePipeline[OptunaBuildResult, None]):
    def __init__(self, cfg: OptunaStageConfig):
        self.cfg = cfg

    def build(self) -> OptunaBuildResult:
        """
        Constructs and returns the components required for optuna pipeline.
        """
        builder = OptunaPipelineBuilder(self.cfg)
        return builder.build()

    def _save_model(
        self, model_saver: ModelSaver, result: StageResult, features: FeaturesConfig
    ) -> None:
        """
        Saves the trained model along with its metadata.
        """
        model_saver.save_model_with_metadata(result, features)

    def run(self) -> None:
        """
        Optimizes hyperparameters for the best-performing model using Optuna,
        retrains it, and logs the final version to MLflow.
        """
        if not self.cfg.model.params:
            print("No params to optimize, skipping optimization...")
            return

        setup_mlflow()

        model_spec = ModelFactory.get_spec(self.cfg.model.name)
        builder = self.build()
        split_data = self.load_data(builder.data_loader)

        context = ExperimentSerializer.from_optuna_stage(
            cfg=self.cfg,
            split_data=split_data,
        )

        run_result = OptunaExperimentManager(
            context=context,
            optimizer=builder.optimizer,
            cross_runner=builder.cross_runner,
            search_runner=builder.search_runner,
        ).manage()

        pred_set = PredictionSetSerializer.from_stage_pipeline(
            run_result.runner_result, split_data
        )
        metrics = self._compute_metrics(pred_set)

        stage_result = StageResultSerializer.from_stage(
            result=run_result,
            metrics=flatten_metrics(metrics),
            model_name=model_spec.model_class.__name__,
        )

        self._log_model(stage_result, X_train=split_data.X_train, register=True)
        self._save_model(
            model_saver=builder.model_saver,
            result=stage_result,
            features=self.cfg.features,
        )
