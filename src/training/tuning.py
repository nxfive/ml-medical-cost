from abc import ABC, abstractmethod
from typing import Generator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.utils.grid import update_param_grid

from .core import compute_scores_mean


class BaseRunner(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Runs the training/evaluation. Must be implemented in subclasses.
        """

    @staticmethod
    def make_predictions(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the trained estimator.
        """
        return estimator.predict(X)


class CrossValidationRunner(BaseRunner):
    def __init__(self, cv: KFold, pipeline: Pipeline):
        self.cv = cv
        self.pipeline = pipeline

    def perform_cross_validation(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> list[np.float64]:
        """
        Performs cross-validation on the given pipeline using the configuration provided.
        """
        return cross_val_score(self.pipeline, X_train, y_train, self.cv, scoring="r2")

    def train_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Fits the given pipeline on the training data.
        """
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def run(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
    ) -> tuple[Pipeline, Sequence[float], float, np.ndarray, np.ndarray]:
        """
        Performs cross-validation, fits the pipeline and generates predictions for train and
        test sets.
        """
        folds_scores = self.perform_cross_validation(self.pipeline, X_train, y_train)
        folds_scores_mean = compute_scores_mean(folds_scores)
        trained = self.train_pipeline(self.pipeline, X_train, y_train)
        train_predictions = self.make_predictions(trained, X_train)
        test_predictions = self.make_predictions(trained, X_test)

        return (
            trained,
            folds_scores,
            folds_scores_mean,
            train_predictions,
            test_predictions,
        )


class GridSearchRunner(BaseRunner):
    def __init__(self, cv: KFold, pipeline: Pipeline, param_grid: dict[str, list]):
        self.cv = cv
        self.pipeline = pipeline
        self.param_grid = param_grid

    def perform_grid_search(self) -> GridSearchCV:
        """
        Creates a GridSearchCV object for the given pipeline and parameter grid.
        """
        return GridSearchCV(
            self.pipeline,
            self.param_grid,
            self.cv,
            scoring="r2",
            return_train_score=True,
        )

    @staticmethod
    def train_grid_search(
        grid: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series
    ) -> BaseEstimator:
        """
        Fits GridSearchCV on the training data and returns the best estimator pipeline.
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
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
    ) -> tuple[GridSearchCV, Sequence[float], float, np.ndarray, np.ndarray]:
        """
        Performs a GridSearchCV on the given pipeline, fits the best estimator and 
        generates predictions on training and test sets.
        """
        grid = self.perform_grid_search()
        trained = self.train_grid_search(grid, X_train, y_train)
        train_predictions = self.make_predictions(trained, X_train)
        test_predictions = self.make_predictions(trained, X_test)
        folds_scores = self.get_grid_folds_scores(grid)
        folds_scores_mean = compute_scores_mean(folds_scores)

        return (
            trained,
            folds_scores,
            folds_scores_mean,
            train_predictions,
            test_predictions,
        )


class TargetTransformer:
    TRANSFORMERS: dict[str, BaseEstimator | None] = {
        "log": FunctionTransformer(np.log, inverse_func=np.exp),
        "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=100),
        "none": None,
    }

    def __init__(
        self, pipeline: Pipeline, param_grid: dict[str, list], cfg_transform: DictConfig
    ):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cfg_transform = cfg_transform

    @classmethod
    def get(cls, name) -> BaseEstimator | None:
        """
        Maps a string name to a scikit-learn transformer.
        """
        return cls.TRANSFORMERS.get(name)

    def build_wrapper_pipeline(
        self, transformer: BaseEstimator | None
    ) -> BaseEstimator:
        """
        Wraps the pipeline in TransformedTargetRegressor if transformer is provided.
        """
        if transformer is None:
            return self.pipeline
        return TransformedTargetRegressor(
            regressor=self.pipeline, transformer=transformer
        )

    def prepare_param_grid(self, params: dict[str, list]) -> dict[str, list]:
        """
        Merges base param grid with transformer parameters.
        """
        grid_copy = self.param_grid.copy() if self.param_grid else {}
        grid_copy = update_param_grid(grid_copy, "regressor")
        if params:
            grid_copy.update(update_param_grid(params, "transformer"))
        return grid_copy

    def evaluate(
        self,
    ) -> Generator[tuple[BaseEstimator, dict[str, list], str], None, None]:
        """
        Generates pipelines with target transformations and updated param grid.
        """
        for transformation_name, value in self.cfg_transform.items():
            transformer = TargetTransformer.get(transformation_name)
            estimator = self.build_wrapper_pipeline(transformer)

            if transformer is not None:
                params = {k: list(v) for k, v in value["params"].items()}
                local_param_grid = self.prepare_param_grid(params)
            else:
                local_param_grid = self.param_grid

            yield estimator, local_param_grid, transformation_name
