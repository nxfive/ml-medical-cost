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


def log_model(
    estimator: BaseEstimator,
    param_grid: dict[str, Any] | None,
    X_train: pd.DataFrame,
    model: type,
    metrics: dict[str, float],
    folds_scores: Sequence[float] | None,
    folds_scores_mean: float | None,
    study: optuna.Study | None,
    transformer_name: str | None,
):
    """
    Logs model parameters, metrics, and artifacts to MLflow.
    If an Optuna study is provided, registers the model to the MLflow Model Registry.
    """
    uuid_id = uuid.uuid4().hex[:6]
    run_name = f"{model.__name__}-{uuid_id}"
    with mlflow.start_run(run_name=run_name) as run:
        if param_grid:
            for param, value in param_grid.items():
                mlflow.log_param(param, value)

        if transformer_name:
            mlflow.log_param("transformer", transformer_name)

        i=1
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=i)
            i+=1

        if folds_scores is not None and folds_scores_mean is not None:
            for s, score in enumerate(folds_scores):
                mlflow.log_metric(f"fold_{s+1}_r2", score, step=i)
                i+=1

            mlflow.log_metric("folds_r2_mean", folds_scores_mean, step=i)

        example_input = X_train.iloc[:5]
        example_output = estimator.predict(example_input)

        signature = mlflow.models.infer_signature(example_input, example_output)

        mlflow.sklearn.log_model(
                estimator,
                name=model.__name__,
                signature=signature,
                input_example=example_input,
            )

        if study:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{model.__name__}",
                name="MedicalRegressor",
            )
