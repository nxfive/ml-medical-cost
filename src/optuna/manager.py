from sklearn.base import BaseEstimator

import optuna
from src.builders.pipeline_builder import PipelineBuilder
from src.optuna.runners import DirectOptunaRunner, WrapperOptunaRunner
from src.tuning.runners import CrossValidationRunner, OptunaSearchRunner
from src.tuning.types import RunnerResult

from .tuning import OptunaOptimize
from .types import ExperimentContext


class OptunaExperimentManager:
    def __init__(
        self,
        context: ExperimentContext,
        optimizer: OptunaOptimize,
        cross_runner: CrossValidationRunner,
        search_runner: OptunaSearchRunner,
    ):
        self.context = context
        self.optimizer = optimizer
        self.cross_runner = cross_runner
        self.search_runner = search_runner

    @property
    def has_transformation(self) -> bool:
        return self.context.model_cfg.target_transformations

    def _build_estimator_from_study(self, study: optuna.Study) -> BaseEstimator:
        """
        Builds a pipeline estimator configured with the best parameters from an
        Optuna study.
        """
        estimator = PipelineBuilder.build(
            model_cfg=self.context.model_cfg,
            features_cfg=self.context.features_cfg,
            transformation=study.best_params["transformation"],
        )
        param_grid = {
            k: v for k, v in study.best_params.items() if k != "transformation"
        }
        return estimator.set_params(**param_grid)

    def manage(self) -> RunnerResult:
        if self.has_transformation:
            runner = WrapperOptunaRunner(
                optimizer=self.optimizer,
                runner=self.cross_runner,
            )
            study = runner.run(self.context)
            estimator = self._build_estimator_from_study(study)

            return self.cross_runner.run(
                estimator=estimator,
                X_train=self.context.X_train,
                X_test=self.context.X_test,
                y_train=self.context.y_train,
            )

        else:
            runner = DirectOptunaRunner(runner=self.search_runner)
            return runner.run(self.context)
