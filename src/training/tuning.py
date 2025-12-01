from typing import Generator, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import QuantileTransformer

from src.utils.cv import get_cv
from src.utils.grid import update_param_grid


def run_cross_validation(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> list[np.float64]:
    """
    Perform cross-validation on the given pipeline using the configuration provided.
    """
    cv = get_cv(cfg)
    return cross_val_score(pipeline, X_train, y_train, cv, scoring="r2")


def compute_scores_mean(fold_scores: list[np.float64]) -> np.float64:
    """
    Compute the mean score from a list of fold scores.
    """
    return np.mean(fold_scores)


def train_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """
    Fit the given pipeline on the training data.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def make_predictions(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions using the trained estimator.
    """
    return estimator.predict(X)


def perform_cross_validation(
    pipeline: Pipeline,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cfg: DictConfig,
) -> tuple[Pipeline, Sequence[float], float, np.ndarray, np.ndarray]:
    """
    Performs cross-validation, fits the pipeline,
    and generates predictions for train and test sets.
    """
    folds_scores = run_cross_validation(pipeline, X_train, y_train, cfg)
    folds_scores_mean = compute_scores_mean(folds_scores)
    trained = train_pipeline(pipeline, X_train, y_train)
    train_predictions = make_predictions(trained, X_train)
    test_predictions = make_predictions(trained, X_test)

    return trained, folds_scores, folds_scores_mean, train_predictions, test_predictions


def run_grid_search(
    pipeline: Pipeline, param_grid: dict[str, list], cfg: DictConfig
) -> GridSearchCV:
    """
    Create a GridSearchCV object for the given pipeline and parameter grid.
    """
    cv = get_cv(cfg)
    return GridSearchCV(pipeline, param_grid, cv, scoring="r2", return_train_score=True)


def train_grid_search(
    grid: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series
) -> BaseEstimator:
    """
    Fit GridSearchCV on the training data and return the best estimator pipeline.
    """
    grid.fit(X_train, y_train)
    return grid.best_estimator_


def get_grid_folds_scores(grid: GridSearchCV) -> list[np.float64]:
    """
    Extract test scores for each fold from a fitted GridSearchCV.
    """
    cv_results = grid.cv_results_
    n_splits = grid.cv.get_n_splits()
    return [
        cv_results[f"split{i}_test_score"][grid.best_index_] for i in range(n_splits)
    ]


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
    Perform a GridSearchCV on the given pipeline, fit the best estimator,
    and generate predictions on training and test sets.
    """
    grid = run_grid_search(pipeline, param_grid, cfg)
    trained = train_grid_search(grid, X_train, y_train)
    train_predictions = make_predictions(trained, X_train)
    test_predictions = make_predictions(trained, X_test)
    folds_scores = get_grid_folds_scores(grid)
    folds_scores_mean = compute_scores_mean(folds_scores)

    return (
        trained,
        folds_scores,
        folds_scores_mean,
        train_predictions,
        test_predictions,
    )


def get_transformer(name: str) -> BaseEstimator | None:
    """
    Map a string name to a scikit-learn transformer.
    """
    transformer_map = {
        "log": FunctionTransformer(np.log, inverse_func=np.exp),
        "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=100),
        "none": None,
    }
    return transformer_map[name]


def build_pipeline_with_transformer(
    pipeline: Pipeline, transformer: BaseEstimator | None
) -> BaseEstimator:
    """
    Wrap the pipeline in TransformedTargetRegressor if transformer is provided.
    """
    if transformer is None:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


def prepare_transformer_param_grid(
    base_grid: dict[str, list], params: dict[str, list]
) -> dict[str, list]:
    """
    Merge base param grid with transformer parameters.
    """
    grid_copy = base_grid.copy() if base_grid else {}
    grid_copy = update_param_grid(grid_copy, "regressor")
    if params:
        grid_copy.update(update_param_grid(params, "transformer"))
    return grid_copy


def evaluate_target_transformers(
    inner_pipeline: Pipeline, param_grid: dict[str, list], cfg: DictConfig
) -> Generator[tuple[BaseEstimator, dict[str, list], str], None, None]:
    """
    Generate pipelines with target transformations and updated param grid.
    """
    for transformation_name, value in cfg.transform.items():
        transformer = get_transformer(transformation_name)
        estimator = build_pipeline_with_transformer(inner_pipeline, transformer)
        params = {k: list(v) for k, v in value["params"].items()}

        if transformer is not None:
            local_param_grid = prepare_transformer_param_grid(param_grid, params)
        else:
            local_param_grid = param_grid

        yield estimator, local_param_grid, transformation_name
