from typing import Any

from sklearn.base import BaseEstimator

import optuna
from src.builders.pipeline.pipeline_builder import PipelineBuilder
from src.containers.experiment import ExperimentContext
from src.containers.results import RunResult
from src.optuna.runners import DirectOptunaRunner, WrapperOptunaRunner
from src.tuning.runners import CrossValidationRunner, OptunaSearchRunner

from .tuning import OptunaOptimize


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

    def _build_estimator_from_study(
        self, study: optuna.Study
    ) -> tuple[BaseEstimator, dict[str, Any]]:
        """
        Builds a pipeline estimator configured with the best parameters from an
        Optuna study.
        """
        estimator = PipelineBuilder.build(
            model_cfg=self.context.model_cfg,
            features_cfg=self.context.features_cfg,
            transformation=study.best_params["transformation"],
        )
        best_params = {
            k: v for k, v in study.best_params.items() if k != "transformation"
        }
        return estimator.set_params(**best_params), best_params

    def manage(self) -> RunResult:
        if self.has_transformation:
            runner = WrapperOptunaRunner(
                optimizer=self.optimizer,
                runner=self.cross_runner,
            )
            study = runner.run(self.context)
            estimator, best_params = self._build_estimator_from_study(study)
            runner_res = self.cross_runner.run(
                estimator=estimator,
                X_train=self.context.X_train,
                X_test=self.context.X_test,
                y_train=self.context.y_train,
            )
            return RunResult(
                runner_result=runner_res,
                param_grid=best_params,
                transformation=study.best_params["transformation"],
            )

        else:
            runner = DirectOptunaRunner(runner=self.search_runner)
            return runner.run(self.context)
