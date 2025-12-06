from abc import ABC, abstractmethod
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.conf.schema import TransformersConfig
from src.evaluation.metrics import compute_scores_mean
from src.params.grid import ParamGrid

from .types import EvaluationResult, RunnerResult


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


class CrossValidationRunner(BaseRunner):
    def __init__(self, cv: KFold):
        self.cv = cv

    def perform_cross_validation(
        self, estimator: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series
    ) -> list[np.float64]:
        """
        Performs cross-validation on the given estimator using the configuration provided.
        """
        return cross_val_score(estimator, X_train, y_train, self.cv, scoring="r2")

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


class TargetTransformer:
    TRANSFORMERS: dict[str, BaseEstimator | None] = {
        "log": FunctionTransformer(np.log, inverse_func=np.exp),
        "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=100),
        "none": None,
    }

    def __init__(self, cfg_transform: TransformersConfig):
        self.cfg_transform = cfg_transform

    @classmethod
    def get(cls, name: str) -> BaseEstimator | None:
        """
        Maps a string name to a scikit-learn transformer.
        """
        return cls.TRANSFORMERS.get(name)

    @staticmethod
    def build_wrapper_pipeline(
        pipeline: Pipeline, transformer: BaseEstimator | None
    ) -> BaseEstimator:
        """
        Wraps the pipeline in TransformedTargetRegressor if transformer is provided.
        """
        if transformer is None:
            return pipeline
        return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)

    @staticmethod
    def prepare_param_grid(
        param_grid: dict[str, list], params: dict[str, list]
    ) -> dict[str, list]:
        """
        Merges base param grid with transformer parameters.
        """
        grid_copy = param_grid.copy() if param_grid else {}
        grid_copy = ParamGrid.prefix(grid_copy, "regressor")
        if params:
            grid_copy.update(ParamGrid.prefix(params, "transformer"))
        return grid_copy

    def evaluate(
        self, pipeline: Pipeline, param_grid: dict[str, list]
    ) -> Generator[EvaluationResult, None, None]:
        """
        Generates pipelines with target transformations and updated param grid.
        """
        for transformation, value in self.cfg_transform.to_dict().items():
            transformer = TargetTransformer.get(transformation)
            estimator = self.build_wrapper_pipeline(pipeline, transformer)

            if transformer is not None:
                params = value["params"]
                local_param_grid = self.prepare_param_grid(param_grid, params)
            else:
                local_param_grid = param_grid

            yield EvaluationResult(
                estimator=estimator,
                param_grid=local_param_grid,
                transformation=transformation,
            )
