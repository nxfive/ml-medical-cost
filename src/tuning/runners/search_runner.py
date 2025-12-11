from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.tuning.types import RunnerResult, RunnerType

from .base_runner import BaseRunner


class SearchRunner(BaseRunner, Generic[RunnerType], ABC):
    @abstractmethod
    def perform_search(
        self, estimator: BaseEstimator, param_grid: dict[str, Any]
    ) -> RunnerType: ...

    @staticmethod
    def get_folds_scores(runner: RunnerType) -> list[np.float64]:
        """
        Extracts test scores for each fold from a fitted runner.
        """
        cv_results = runner.cv_results_
        n_splits = runner.cv.get_n_splits()
        return [
            cv_results[f"split{i}_test_score"][runner.best_index_]
            for i in range(n_splits)
        ]

    def run(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> RunnerResult:
        """
        Runs the search/fitting process for the given estimator, fits the estimator,
        and generates predictions on the training and test sets.
        """
        grid_search = self.perform_search(estimator, param_grid)
        trained = self.fit_estimator(grid_search, X_train, y_train, return_best=True)
        folds_scores = self.get_folds_scores(grid_search)

        return self._collect_results(trained, folds_scores, X_train, X_test)
