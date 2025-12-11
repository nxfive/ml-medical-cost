from typing import Any

from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

import optuna
from optuna.integration import OptunaSearchCV
from src.conf.schema import OptunaConfig

from .search_runner import SearchRunner


class OptunaSearchRunner(SearchRunner[OptunaSearchCV]):
    def __init__(
        self,
        optuna_cfg: OptunaConfig,
        study: optuna.Study,
        cv: KFold,
        scoring: str = "r2",
    ):
        self.cfg = optuna_cfg
        self.study = study
        self.cv = cv
        self.scoring = scoring

    def perform_search(
        self, estimator: BaseEstimator, param_grid: dict[str, Any]
    ) -> OptunaSearchCV:
        """
        Creates a OptunaSearchCV object for the given estimator and parameter grid.
        """
        return OptunaSearchCV(
            estimator,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_trials=self.cfg.trials,
            timeout=self.cfg.timeout,
            study=self.study,
            return_train_score=True,
        )
