from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from src.models.pipeline import create_model_pipeline
from src.utils.grid import prepare_grid

PipelineResult = tuple[
    BaseEstimator,
    Sequence[float],
    float,
    np.ndarray,
    np.ndarray,
]


class TrainModel:
    def __init__(
        self,
        model: type[BaseEstimator],
        cfg_model: DictConfig,
        cfg_features: DictConfig,
        cfg_cv: DictConfig,
        cfg_transform: DictConfig,
        grid_runner,
        cross_runner,
        target_transformer,
    ):
        self.model_class = model
        self.cfg_cv = cfg_cv
        self.cfg_features = cfg_features
        self.cfg_model = cfg_model
        self.cfg_transform = cfg_transform
        self.pipeline = create_model_pipeline(
            self.cfg_model.preprocess_num_features, self.cfg_features
        )
        self.param_grid = prepare_grid(self.cfg_model.params)

        self.grid_runner = grid_runner
        self.cross_runner = cross_runner
        self.target_runner = target_transformer

    def fit_estimator(
        self,
        estimator: BaseEstimator,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> PipelineResult:
        """
        Fits an estimator using either GridSearchCV or simple cross-validation.
        """
        if self.model_params:
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

    def transform_estimator(
        self,
    ) -> Iterator[tuple[BaseEstimator, dict[str, list], str | None]]:
        """
        Generates estimators with optional target transformations and updated parameter grids.
        """
        if self.model_target_transformations:
            for (
                estimator,
                local_param_grid,
                transformer_name,
            ) in self.target_transformer.evaluate():
                yield estimator, local_param_grid, transformer_name
        else:
            yield self.pipeline, self.param_grid, None

    def run(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> Iterator[tuple[PipelineResult, dict[str, list], str | None]]:
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
            yield results, local_param_grid, transformer_name
