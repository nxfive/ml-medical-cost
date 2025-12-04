from typing import Generator

import pandas as pd
from sklearn.base import BaseEstimator

from src.conf.schema import (CVConfig, FeaturesConfig, ModelConfig,
                             TransformersConfig)
from src.models.pipeline import create_model_pipeline
from src.utils.grid import ParamGrid

from .tuning import CrossValidationRunner, GridSearchRunner, TargetTransformer
from .types import EvaluationResult, RunnerResult, TrainResult


class TrainModel:
    def __init__(
        self,
        model: type[BaseEstimator],
        cfg_model: ModelConfig,
        cfg_features: FeaturesConfig,
        cfg_cv: CVConfig,
        cfg_transform: TransformersConfig,
        grid_runner: GridSearchRunner,
        cross_runner: CrossValidationRunner,
        target_transformer: TargetTransformer,
    ):
        self.model_class = model
        self.cfg_cv = cfg_cv
        self.cfg_features = cfg_features
        self.cfg_model = cfg_model
        self.cfg_transform = cfg_transform
        self.pipeline = create_model_pipeline(
            self.cfg_model.preprocess_num_features, self.cfg_features, model=self.model_class
        )
        self.param_grid = ParamGrid.create(self.cfg_model.params)
        self.grid_runner = grid_runner
        self.cross_runner = cross_runner
        self.target_transformer = target_transformer

    def fit_estimator(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> RunnerResult:
        """
        Fits an estimator using either GridSearchCV or simple cross-validation.
        """
        if self.cfg_model.params:
            return self.grid_runner.run(
                pipeline=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
            )
        else:
            return self.cross_runner.run(
                pipeline=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
            )

    def transform_estimator(self) -> Generator[EvaluationResult, None, None]:
        """
        Generates estimators with optional target transformations and updated parameter grids.
        """
        if self.cfg_model.target_transformations:
            for (
                estimator,
                local_param_grid,
                transformer_name,
            ) in self.target_transformer.evaluate():
                yield EvaluationResult(
                    estimator=estimator,
                    param_grid=local_param_grid,
                    transformation_name=transformer_name,
                )
        else:
            yield EvaluationResult(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                transformation_name=None,
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
        for (
            estimator,
            local_param_grid,
            transformer_name,
        ) in self.transform_estimator():
            results = self.fit_estimator(estimator, X_train, X_test, y_train)
            yield TrainResult(
                runner_result=results,
                param_grid=local_param_grid,
                transformation_name=transformer_name,
            )
