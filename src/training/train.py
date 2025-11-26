import numpy as np
import pandas as pd
from typing import Iterator, Sequence

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.models.models import create_model_pipeline
from src.models.settings import pipeline_config
from src.models.tuning import (
    evaluate_target_transformers,
    perform_cross_validation,
    perform_grid_search,
)
from src.models.utils import prepare_grid


PipelineResult = tuple[
    Pipeline | TransformedTargetRegressor,
    Sequence[float],
    float,
    np.ndarray,
    np.ndarray,
]


def run_pipeline(
    model: type, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> Iterator[tuple[PipelineResult, dict[str, list], str | None]]:
    """
    Runs the model pipeline with optional target transformations and hyperparameter tuning
    (using cross-validation or grid search with cross-validation).
    """
    inner_pipeline = create_model_pipeline(model)
    param_grid = prepare_grid(model)

    def fit_pipeline(
        pipeline: Pipeline | TransformedTargetRegressor, param_grid: dict[str, list]
    ) -> PipelineResult:
        if pipeline_config.models[model.__name__].params:
            return perform_grid_search(
                pipeline, param_grid, X_train=X_train, X_test=X_test, y_train=y_train
            )
        else:
            return perform_cross_validation(
                pipeline, X_train=X_train, X_test=X_test, y_train=y_train
            )

    if pipeline_config.models[model.__name__].target_transformations:
        for (
            pipeline,
            local_param_grid,
            transformer_name,
        ) in evaluate_target_transformers(inner_pipeline, param_grid):
            results = fit_pipeline(pipeline, local_param_grid)
            yield results, local_param_grid, transformer_name
    else:
        results = fit_pipeline(inner_pipeline, param_grid)
        yield results, param_grid, None
