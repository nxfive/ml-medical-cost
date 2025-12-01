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


def make_predictions(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """
    Generate predictions using the trained pipeline.
    """
    return pipeline.predict(X)


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
