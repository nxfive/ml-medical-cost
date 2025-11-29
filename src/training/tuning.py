from typing import Generator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.utils.cv import get_cv
from src.utils.grid import update_param_grid


def perform_cross_validation(
    pipeline: Pipeline,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
) -> tuple[Pipeline, Sequence[float], float, np.ndarray, np.ndarray]:
    """
    Performs cross-validation on the given pipeline.
    """
    folds_scores = cross_val_score(
        pipeline, X_train, y_train, cv=get_cv(cfg), scoring="r2"
    )
    folds_scores_mean = np.mean(folds_scores)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    return pipeline, folds_scores, folds_scores_mean, y_train_pred, y_test_pred


def perform_grid_search(
    pipeline: Pipeline,
    param_grid: dict,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
) -> tuple[GridSearchCV, Sequence[float], float, np.ndarray, np.ndarray]:
    """
    Performs grid search with cross-validation a on the given pipeline.
    """
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=get_cv(cfg),
        scoring="r2",
        return_train_score=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)

    y_test_pred = gs.best_estimator_.predict(X_test)
    y_train_pred = gs.best_estimator_.predict(X_train)
    cv_results = gs.cv_results_
    n_splits = gs.cv.get_n_splits()
    folds_scores = [
        cv_results[f"split{i}_test_score"][gs.best_index_] for i in range(n_splits)
    ]
    folds_scores_mean = np.mean(folds_scores)

    return (
        gs.best_estimator_,
        folds_scores,
        folds_scores_mean,
        y_train_pred,
        y_test_pred,
    )


def evaluate_target_transformers(
    inner_pipeline: Pipeline, param_grid: dict[str, list], cfg: DictConfig
) -> Generator[
    tuple[Pipeline | TransformedTargetRegressor, dict[str, list], str], None, None
]:
    """
    Generates pipelines with different target transformations and updated parameter grids.
    """
    transformer_map = {
        "log": FunctionTransformer(np.log, inverse_func=np.exp),
        "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=100),
        "none": None,
    }

    for transformation, value in cfg.transform.items():
        local_param_grid = param_grid.copy()

        transformer = transformer_map[transformation]
        params = {k: list(v) for k, v in value["params"].items()}

        if transformer is not None:
            ttr = TransformedTargetRegressor(
                regressor=inner_pipeline, transformer=transformer
            )
            local_param_grid = (
                update_param_grid(local_param_grid, "regressor")
                if local_param_grid
                else {}
            )
        else:
            ttr = inner_pipeline

        if params:
            local_param_grid.update(update_param_grid(params, "transformer"))

        yield ttr, local_param_grid, transformation
