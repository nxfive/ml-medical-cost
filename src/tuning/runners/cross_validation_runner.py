import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_val_score

from src.tuning.types import RunnerResult

from .base_runner import BaseRunner


class CrossValidationRunner(BaseRunner):
    def __init__(self, cv: KFold, scoring: str = "r2"):
        self.cv = cv
        self.scoring = scoring

    def _perform_cross_validation(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> list[np.float64]:
        """
        Performs cross-validation on the given estimator using the configuration provided.
        """
        return list(
            cross_val_score(
                estimator, X_train, y_train, cv=self.cv, scoring=self.scoring
            )
        )

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
        folds_scores = self._perform_cross_validation(estimator, X_train, y_train)
        trained = self.fit_estimator(estimator, X_train, y_train)

        return self._collect_results(trained, folds_scores, X_train, X_test)
