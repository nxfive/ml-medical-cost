import os
from datetime import datetime
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import KFold

from src.models.settings import pipeline_config
from src.utils.paths import MODELS_DIR


def update_param_grid(param_grid: dict, step_name: str) -> dict:
    """
    Prefixes all keys in param_grid with 'step_name__' for pipeline compatibility.
    """
    param_grid = param_grid.copy()
    step_name = step_name.strip().strip("_")

    if step_name:
        return {f"{step_name}__{k}": v for k, v in param_grid.items()}
    return param_grid


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


def update_params_with_optuna(
    model_params: dict[str, Any], optuna_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Updates model parameters with optimized values from Optuna results.
    """
    updated_params = {}
    for key, values in model_params.items():
        for param_name, best_value in optuna_params.items():
            if key.endswith(param_name):
                updated_params[key] = best_value
                break
        else:
            if isinstance(values, list):
                print(
                    f"Parameter '{key}' not found in Optuna results. "
                    f"Using first value from config: {values[0]!r}"
                )
                updated_params[key] = values[0]
            else:
                print(
                    f"Parameter '{key}' not found in Optuna results. "
                    f"Using config value: {values!r}"
                )
                updated_params[key] = values

    return updated_params


def save_model_with_metadata(
    model: Any,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, float | int],
):
    """
    Save model and corresponding metadata to disk.
    """
    file_name = model_name.lower()
    model_path = os.path.join(MODELS_DIR, f"{file_name}.pkl")
    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "version": "1.0",
        "date_trained": datetime.today().strftime("%Y-%m-%d"),
        "features_processed": {
            "cat_features": pipeline_config.features.categorical,
            "num_features": pipeline_config.features.numeric,
            "bin_features": pipeline_config.features.binary,
        },
        "params": params or {},
        "metrics": metrics
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata", f"{file_name}.yml")
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)
