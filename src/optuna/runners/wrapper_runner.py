import numpy as np

import optuna
from src.optuna.tuning import OptunaOptimize
from src.optuna.types import ExperimentContext
from src.tuning.runners import CrossValidationRunner

from .base import BaseExperimentRunner


class WrapperOptunaRunner(BaseExperimentRunner[optuna.Study]):
    def __init__(self, optimizer: OptunaOptimize, runner: CrossValidationRunner):
        self.optimizer = optimizer
        self.runner = runner

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
        exp_setup.pipeline.set_params(**exp_setup.params)

        result = self.runner.run(
            estimator=exp_setup.pipeline,
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
