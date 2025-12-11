import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold

from src.tuning.types import RunnerResult

from .base_runner import BaseRunner


class GridSearchRunner(BaseRunner):
    def __init__(self, cv: KFold):
        self.cv = cv

    def perform_grid_search(
        self, estimator: BaseEstimator, param_grid: dict[str, list]
    ) -> GridSearchCV:
        """
        Creates a GridSearchCV object for the given estimator and parameter grid.
        """
        return GridSearchCV(
            estimator,
            param_grid,
            cv=self.cv,
            scoring="r2",
            return_train_score=True,
        )

    @staticmethod
    def fit_grid_search(
        grid: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseEstimator:
        """
        Fits GridSearchCV on the training data and returns the best estimator.
        """
        grid.fit(X_train, y_train)
        return grid.best_estimator_

    @staticmethod
    def get_grid_folds_scores(grid: GridSearchCV) -> list[np.float64]:
        """
        Extracts test scores for each fold from a fitted GridSearchCV.
        """
        cv_results = grid.cv_results_
        n_splits = grid.cv.get_n_splits()
        return [
            cv_results[f"split{i}_test_score"][grid.best_index_]
            for i in range(n_splits)
        ]

    def run(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> RunnerResult:
        """
        Performs a GridSearchCV on the given estimator, fits the best estimator and
        generates predictions on training and test sets.
        """
        grid = self.perform_grid_search(estimator, param_grid)
        trained = self.fit_grid_search(grid, X_train, y_train)
        folds_scores = self.get_grid_folds_scores(grid)

        return self._collect_results(trained, folds_scores, X_train, X_test)
