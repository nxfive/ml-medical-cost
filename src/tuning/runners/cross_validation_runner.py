import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score

from src.tuning.types import RunnerResult

from .base_runner import BaseRunner


class CrossValidationRunner(BaseRunner):
    def __init__(self, cv: KFold):
        self.cv = cv

    def perform_cross_validation(
        self, estimator: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series
    ) -> list[np.float64]:
        """
        Performs cross-validation on the given estimator using the configuration provided.
        """
        return list(
            cross_val_score(estimator, X_train, y_train, cv=self.cv, scoring="r2")
        )

    @staticmethod
    def fit_estimator(
        estimator: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseEstimator:
        """
        Fits the given estimator on the training data.
        """
        estimator.fit(X_train, y_train)
        return estimator

    def run(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> RunnerResult:
        """
        Performs cross-validation, fits the estimator and generates predictions for train and
        test sets.
        """
        folds_scores = self.perform_cross_validation(estimator, X_train, y_train)
        trained = self.fit_estimator(estimator, X_train, y_train)

        return self._collect_results(trained, folds_scores, X_train, X_test)
