from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.evaluation.metrics import compute_scores_mean
from src.training.types import RunnerResult


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs) -> RunnerResult:
        """
        Runs the training/evaluation. Must be implemented in subclasses.
        """

    @staticmethod
    def make_predictions(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained estimator.
        """
        return estimator.predict(X)

    @staticmethod
    def _collect_results(
        trained: BaseEstimator,
        folds_scores: list[float],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> RunnerResult:
        """
        Creates RunnerResult from trained estimator and CV scores.
        """
        folds_scores_mean = compute_scores_mean(folds_scores)
        train_predictions = BaseRunner.make_predictions(trained, X_train)
        test_predictions = BaseRunner.make_predictions(trained, X_test)
        return RunnerResult(
            trained=trained,
            folds_scores=folds_scores,
            folds_scores_mean=folds_scores_mean,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
        )
