from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.containers.results import RunnerResult
from src.evaluation.metrics import compute_scores_mean


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
    ) -> BaseEstimator:
        """
        Fits the estimator. For search objects like GridSearchCV or OptunaSearchCV
        returns the best_estimator_.
        """
        estimator.fit(X_train, y_train)
        if hasattr(estimator, "best_estimator_"):
            return estimator.best_estimator_
        return estimator

    @staticmethod
    def _unwrap_estimator(
        estimator: BaseEstimator,
    ) -> tuple[BaseEstimator, BaseEstimator | None]:
        """
        Unwraps an estimator to get the underlying model and optional transformer.
        """
        if hasattr(estimator, "best_estimator_"):
            estimator = estimator.best_estimator_

        transformer = None
        while isinstance(estimator, TransformedTargetRegressor):  # for nested ttr
            transformer = estimator.transformer
            estimator = estimator.regressor

        if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
            estimator = estimator.named_steps["model"]

        return estimator, transformer

    def _get_params(self, estimator: BaseEstimator) -> dict[str, Any]:
        """
        Returns parameters from the model and optional transformer with prefixes.
        """
        estimator, transformer = self._unwrap_estimator(estimator)
        params = {}

        model_params = estimator.get_params(deep=False)
        params.update({f"model__{k}": v for k, v in model_params.items()})

        if transformer is not None:
            params.update(
                {
                    f"transformer__{k}": v
                    for k, v in transformer.get_params(deep=False).items()
                }
            )

        return params

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
        params = self._get_params(estimator=trained)
        return RunnerResult(
            trained=trained,
            folds_scores=folds_scores,
            folds_scores_mean=folds_scores_mean,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            params=params,
        )
