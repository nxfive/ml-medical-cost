from typing import Callable

import numpy as np

import optuna
from optuna.pruners import BasePruner
from src.conf.schema import OptunaConfig


class OptunaOptimize:
    def __init__(
        self,
        optuna_cfg: OptunaConfig,
        pruner: BasePruner,
        study_name: str = "MedicalRegressor",
    ):
        self.optuna_cfg = optuna_cfg
        self.study = optuna.create_study(
            study_name=study_name, pruner=pruner, direction="maximize"
        )

    def optimize(
        self, objective_fn: Callable[[optuna.Trial], np.float64]
    ) -> optuna.Study:
        """
        Runs the Optuna study using the provided objective function.
        """
        self.study.optimize(
            func=objective_fn,
            n_trials=self.optuna_cfg.trials,
            timeout=self.optuna_cfg.timeout,
        )
        return self.study
