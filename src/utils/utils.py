from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def update_param_grid(param_grid: dict, step_name: str) -> dict:
    """
    Prefixes all keys in param_grid with 'step_name__' for pipeline compatibility.
    """
    param_grid = param_grid.copy()
    step_name = step_name.strip().strip("_")

    if step_name:
        return {f"{step_name}__{k}": v for k, v in param_grid.items()}
    return param_grid


def prepare_grid(cfg: DictConfig) -> dict:
    """
    Prepares param_grid for GridSearchCV with 'model' prefixes.
    """
    param_grid: dict[str, list] = {}
    model_params = {k: list(v) for k, v in cfg.model.params.items()}

    if model_params:
        param_grid = update_param_grid(model_params, "model")

    return param_grid


def get_cv(cfg: DictConfig) -> KFold:
    """
    Returns a KFold cross-validator configured based on the values from config.
    """
    return KFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
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

    print(
        f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, Difference: {difference:.3f}"
    )

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
    cfg: DictConfig,
):
    """
    Save model and corresponding metadata to disk.
    """
    file_name = model_name.lower()
    models_path = Path(cfg.models.output_dir)
    metadata_path = models_path / "metadata"

    metadata_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_path / f"{file_name}.pkl")

    metadata = {
        "model_name": model_name,
        "version": "1.0",
        "date_trained": datetime.today().strftime("%Y-%m-%d"),
        "features_processed": {
            "cat_features": list(cfg.features.categorical),
            "num_features": list(cfg.features.numeric),
            "bin_features": list(cfg.features.binary),
        },
        "params": params or {},
        "metrics": metrics,
    }

    with open(metadata_path / f"{file_name}.yml", "w") as f:
        yaml.safe_dump(metadata, f)


def load_splitted_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load splitted data from data/processed directory.
    """
    X_train = pd.read_parquet(Path(cfg.data.processed_dir) / "X_train.parquet")
    X_test = pd.read_parquet(Path(cfg.data.processed_dir) / "X_test.parquet")
    y_train = pd.read_parquet(Path(cfg.data.processed_dir) / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(Path(cfg.data.processed_dir) / "y_test.parquet").squeeze()

    return X_train, X_test, y_train, y_test


def save_run(
    results: dict, pipeline: Pipeline | TransformedTargetRegressor, cfg: DictConfig
):
    """
    Saves the training run results and the trained pipeline to disk.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_path = Path(cfg.training.output_dir) / timestamp
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_path = results_path / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.safe_dump(results, f)

    pipeline_path = results_path / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)


def pick_best(
    results_dir: str,
) -> tuple[Pipeline | TransformedTargetRegressor, dict[str, Any]]:
    """
    Loads the best trained pipeline and its metrics from a directory of runs.
    """
    results_dir = Path(results_dir)
    best_run = None
    best_score = float("-inf")

    for run_dir in results_dir.iterdir():
        metrics_file = run_dir / "metrics.yaml"
        if metrics_file.exists():
            data = yaml.safe_load(metrics_file.read_text())
            r2 = data["metrics"].get("test_r2", float("-inf"))
            if r2 > best_score:
                best_score = r2
                best_run = run_dir

    if best_run is None:
        raise ValueError(f"No valid metrics found in {results_dir}")

    pipeline_file = best_run / "pipeline.pkl"
    pipeline = joblib.load(pipeline_file)

    metrics = yaml.safe_load((best_run / "metrics.yaml").read_text())

    return pipeline, metrics
