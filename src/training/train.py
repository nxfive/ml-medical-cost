from typing import Generator

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.conf.schema import (CVConfig, FeaturesConfig, ModelConfig,
                             TransformersConfig)
from src.tuning.runners import CrossValidationRunner, GridSearchRunner
from src.tuning.transformers import TargetTransformer
from src.tuning.types import EvaluationResult, RunnerResult

from .types import TrainResult


class TrainModel:
    def __init__(
        self,
        model: type[BaseEstimator],
        cfg_model: ModelConfig,
        cfg_features: FeaturesConfig,
        cfg_cv: CVConfig,
        cfg_transform: TransformersConfig,
        param_grid: dict[str, list],
        pipeline: Pipeline,
        grid_runner: GridSearchRunner,
        cross_runner: CrossValidationRunner,
        target_transformer: TargetTransformer,
    ):
        self.model_class = model
        self.cfg_cv = cfg_cv
        self.cfg_features = cfg_features
        self.cfg_model = cfg_model
        self.cfg_transform = cfg_transform
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.grid_runner = grid_runner
        self.cross_runner = cross_runner
        self.target_transformer = target_transformer

    def fit_estimator(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> RunnerResult:
        """
        Fits an estimator using either GridSearchCV or simple cross-validation.
        """
        if self.cfg_model.params:
            return self.grid_runner.run(
                estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
            )
        else:
            return self.cross_runner.run(
                estimator=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
            )

    def transform_estimator(self) -> Generator[EvaluationResult, None, None]:
        """
        Generates estimators with optional target transformations and updated parameter grids.
        """
        if self.cfg_model.target_transformations:
            yield from self.target_transformer.evaluate(self.pipeline, self.param_grid)
        else:
            yield EvaluationResult(
                estimator=self.pipeline,
                param_grid=self.param_grid,
            )

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> Generator[TrainResult, None, None]:
        """
        Runs training over all estimators (with optional target transformations)
        and performs either grid search or cross-validation.
        """
        for evaluation in self.transform_estimator():
            results = self.fit_estimator(
                evaluation.estimator, evaluation.param_grid, X_train, X_test, y_train
            )
            yield TrainResult(
                runner_result=results,
                param_grid=evaluation.param_grid,
                transformation=evaluation.transformation,
            )
