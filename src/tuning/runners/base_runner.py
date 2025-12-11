from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.evaluation.metrics import compute_scores_mean
from src.training.types import RunnerResult


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> RunnerResult: ...

    @staticmethod
    def make_predictions(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained estimator.
        """
        return estimator.predict(X)
    
    @staticmethod
    def fit_estimator(
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        return_best: bool = False
    ) -> BaseEstimator:
        """
        Fits the estimator. For search objects like GridSearchCV or OptunaSearchCV,
        `return_best=True` returns the best_estimator_.
        """
        estimator.fit(X_train, y_train)
        if return_best and hasattr(estimator, "best_estimator_"):
            return estimator.best_estimator_
        return estimator

    def _collect_results(
        self,
        trained: BaseEstimator,
        folds_scores: list[np.float64],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> RunnerResult:
        """
        Creates RunnerResult from trained estimator and CV scores.
        """
        folds_scores_mean = compute_scores_mean(folds_scores)
        train_predictions = self.make_predictions(trained, X_train)
        test_predictions = self.make_predictions(trained, X_test)
        return RunnerResult(
            trained=trained,
            folds_scores=folds_scores,
            folds_scores_mean=folds_scores_mean,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
        )
