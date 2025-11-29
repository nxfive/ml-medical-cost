from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.models.models import create_model_pipeline
from src.training.tuning import (evaluate_target_transformers,
                                 perform_cross_validation, perform_grid_search)
from src.utils.grid import prepare_grid

PipelineResult = tuple[
    Pipeline | TransformedTargetRegressor,
    Sequence[float],
    float,
    np.ndarray,
    np.ndarray,
]


def run_training(
    model: type, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> Iterator[tuple[PipelineResult, dict[str, list], str | None]]:
    """
    Runs the model pipeline with optional target transformations and hyperparameter tuning
    (using cross-validation or grid search with cross-validation).
    """
    inner_pipeline = create_model_pipeline(cfg, model)
    param_grid = prepare_grid(cfg)

    def fit_pipeline(
        pipeline: Pipeline | TransformedTargetRegressor, param_grid: dict[str, list], cfg: DictConfig
    ) -> PipelineResult:
        if cfg.model.params:
            return perform_grid_search(
                pipeline, param_grid, X_train=X_train, X_test=X_test, y_train=y_train, cfg=cfg
            )
        else:
            return perform_cross_validation(
                pipeline, X_train=X_train, X_test=X_test, y_train=y_train, cfg=cfg
            )

    if cfg.model.target_transformations:
        for (
            pipeline,
            local_param_grid,
            transformer_name,
        ) in evaluate_target_transformers(inner_pipeline, param_grid, cfg):
            results = fit_pipeline(pipeline, local_param_grid, cfg)
            yield results, local_param_grid, transformer_name
    else:
        results = fit_pipeline(inner_pipeline, param_grid, cfg)
        yield results, param_grid, None
