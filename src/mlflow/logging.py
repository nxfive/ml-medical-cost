import os
from datetime import datetime
from typing import Any, Sequence

import mlflow
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
import uuid


def setup_mlflow(create_experiment: bool = True):
    """
    Configures MLflow tracking for CI/CD or local runs.
    """
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_uri or "http://localhost:5000")
    
    commit_hash = os.getenv("GITHUB_SHA", "")[:7]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    
    experiment_prefix = "ci-build" if remote_uri else "local-test"
    experiment_name = f"{experiment_prefix}-{commit_hash or timestamp}"
    
    if create_experiment:
        mlflow.set_experiment(experiment_name)


def _log_params(param_grid: dict[str, Any] | None, transformer_name: str | None):
    """
    Log model hyperparameters and optional transformer name to MLflow.
    """
    if param_grid:
        for param, value in param_grid.items():
            mlflow.log_param(param, value)
    if transformer_name:
        mlflow.log_param("transformer", transformer_name)


def _log_metrics(metrics: dict[str, float], folds_scores: Sequence[float] | None, folds_scores_mean: float | None):
    """
    Log evaluation metrics and cross-validation fold scores to MLflow.
    """
    i = 1
    for name, value in metrics.items():
        mlflow.log_metric(name, value, step=i)
        i += 1

    if folds_scores is not None and folds_scores_mean is not None:
        for s, score in enumerate(folds_scores):
            mlflow.log_metric(f"fold_{s+1}_r2", score, step=i)
            i += 1
        mlflow.log_metric("folds_r2_mean", folds_scores_mean, step=i)


def _log_model_artifacts(estimator: BaseEstimator, model: type, X_train: pd.DataFrame):
    """
    Log the trained model to MLflow with input example and signature.
    """
    example_input = X_train.iloc[:5]
    example_output = estimator.predict(example_input)
    signature = mlflow.models.infer_signature(example_input, example_output)
    mlflow.sklearn.log_model(
        estimator,
        name=model.__name__,
        signature=signature,
        input_example=example_input,
    )


def _register_model_if_study(study: optuna.Study | None, model_name: str):
    """
    Register the trained model to MLflow Model Registry if an Optuna study is provided.
    """
    if study:
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            name="MedicalRegressor",
        )


def log_model(
    estimator: BaseEstimator,
    param_grid: dict[str, Any] | None,
    X_train: pd.DataFrame,
    model: type,
    metrics: dict[str, float],
    folds_scores: Sequence[float] | None = None,
    folds_scores_mean: float | None = None,
    study: optuna.Study | None = None,
    transformer_name: str | None = None,
):
    """
    Orchestrates logging a model, parameters, metrics, and artifacts to MLflow.
    """
    uuid_id = uuid.uuid4().hex[:6]
    run_name = f"{model.__name__}-{uuid_id}"

    with mlflow.start_run(run_name=run_name):
        _log_params(param_grid, transformer_name)
        _log_metrics(metrics, folds_scores, folds_scores_mean)
        _log_model_artifacts(estimator, model, X_train)
        _register_model_if_study(study, model.__name__)
