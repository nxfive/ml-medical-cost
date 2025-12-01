from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.models.models import create_model_pipeline
from src.training.tuning import (evaluate_target_transformers,
                                 perform_cross_validation, perform_grid_search)
from src.utils.grid import prepare_grid

PipelineResult = tuple[
    BaseEstimator,
    Sequence[float],
    float,
    np.ndarray,
    np.ndarray,
]


def fit_estimator(
    estimator: BaseEstimator,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
) -> PipelineResult:
    """
    Fit an estimator using either GridSearchCV or simple cross-validation.
    """
    if cfg.model.params:
        return perform_grid_search(
            estimator,
            param_grid,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            cfg=cfg,
        )
    else:
        return perform_cross_validation(
            estimator, X_train=X_train, X_test=X_test, y_train=y_train, cfg=cfg
        )


def generate_estimators_with_transformers(
    pipeline: Pipeline, param_grid: dict[str, list], cfg: DictConfig
) -> Iterator[tuple[BaseEstimator, dict[str, list], str | None]]:
    """
    Generate estimators with optional target transformations and updated parameter grids.
    """
    if cfg.model.target_transformations:
        for (
            estimator,
            local_param_grid,
            transformer_name,
        ) in evaluate_target_transformers(pipeline, param_grid, cfg):
            yield estimator, local_param_grid, transformer_name
    else:
        yield pipeline, param_grid, None


def run_training(
    model: type,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
) -> Iterator[tuple[PipelineResult, dict[str, list], str | None]]:
    """
    Run training over all estimators (with optional target transformations)
    and perform either grid search or cross-validation.
    """
    pipeline = create_model_pipeline(cfg, model)
    param_grid = prepare_grid(cfg)

    for (
        estimator,
        local_param_grid,
        transformer_name,
    ) in generate_estimators_with_transformers(pipeline, param_grid, cfg):
        results = fit_estimator(
            estimator, local_param_grid, X_train, X_test, y_train, cfg
        )
        yield results, local_param_grid, transformer_name
