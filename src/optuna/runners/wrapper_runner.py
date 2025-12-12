import numpy as np

import optuna
from src.factories.optuna_runner_factory import OptunaRunnerFactory
from src.optuna.tuning import OptunaOptimize
from src.optuna.types import ExperimentContext

from .base import BaseExperimentRunner


class WrapperOptunaRunner(BaseExperimentRunner[optuna.Study]):
    def __init__(self, optimizer: OptunaOptimize):
        self.optimizer = optimizer

    def objective(self, trial: optuna.Trial, context: ExperimentContext) -> np.float64:
        """
        Objective function to evaluate a single trial.

        Builds the experiment setup, runs cross-validation using the wrapper runner,
        and returns the mean score across folds.
        """
        exp_setup = self.build(
            exp_config=context.to_experiment_config(),
            trial=trial,
        )
        search_runner = OptunaRunnerFactory.create_wrapper_runner(
            cv_cfg=context.cv_cfg, study=self.optimizer.study
        )

        result = search_runner.run(
            estimator=exp_setup.pipeline,
            param_grid=exp_setup.params,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
        )
        return result.folds_scores_mean

    def run(self, context: ExperimentContext) -> optuna.Study:
        """
        Runs the Optuna optimization using the objective function.
        """
        return self.optimizer.optimize(lambda trial: self.objective(trial, context))
