import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold

from src.models.settings import pipeline_config


def update_param_grid(param_grid: dict, step_name: str) -> dict:
    """
    Prefixes all keys in param_grid with 'step_name__' for pipeline compatibility.
    """
    param_grid = param_grid.copy()
    return {f"{step_name}__{k}": v for k, v in param_grid.items()}


def prepare_grid(model: type) -> dict:
    """
    Prepares param_grid for GridSearchCV with 'model' prefixes.
    """
    param_grid = {}
    model_params = pipeline_config.models[model.__name__].params

    if model_params:
        param_grid = update_param_grid(model_params, "model")

    return param_grid


def get_cv() -> KFold:
    """
    Returns a KFold cross-validator configured based on the values from pipeline_config.
    """
    return KFold(
        n_splits=pipeline_config.cv["n_splits"], 
        shuffle=pipeline_config.cv["shuffle"], 
        random_state=pipeline_config.cv["random_state"]
    )


def get_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict[str, float]:
    """
    Computes train and test metrics (R², MAE, RMSE) for model predictions.
    """
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_rmse": root_mean_squared_error(y_train, y_train_pred),
        "test_rmse": root_mean_squared_error(y_test, y_test_pred),
    }

    return metrics


def check_fold_stability(folds_scores: Sequence[float], threshold: float = 0.1) -> bool:
    """
    Checks the stability of a model based on cross-validation fold scores.
    """
    max_score = max(folds_scores)
    min_score = min(folds_scores)
    difference = max_score - min_score
    
    print(f"Fold scores: {folds_scores}, max-min difference: {difference:.3f}")
    
    return difference <= threshold


def check_overfitting(train_r2: float, test_r2: float, threshold: float = 0.2) -> bool:
    """
    Checks whether a model is likely overfitting based on the difference between 
    training and test R² scores.
    """
    difference = train_r2 - test_r2

    print(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, Difference: {difference:.3f}")

    return difference > threshold


def check_model_results(
    model: type,
    metrics: dict,
    folds_scores: Sequence[float],
    fold_threshold=0.1,
    overfit_threshold=0.2,
):
    """
    Checks model stability and potential overfitting.
    """
    stable = True
    if folds_scores is not None and len(folds_scores) > 0:
        stable = check_fold_stability(folds_scores, threshold=fold_threshold)
    if not stable:
        print(f"{model.__name__}: fold results indicate potential instability")

    overfit = check_overfitting(
        metrics["train_r2"], metrics["test_r2"], threshold=overfit_threshold
    )
    if overfit:
        print(
            f"{model.__name__} may be overfitting. Consider adjusting hyperparameters."
        )
