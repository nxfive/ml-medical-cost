import os
from datetime import datetime
from typing import Any, Sequence

import mlflow
import optuna
import pandas as pd
from sklearn.base import BaseEstimator


def setup_mlflow():
    """
    Configures MLflow tracking for CI/CD or local runs.
    """
    remote_uri = os.getenv("MLFLOW_TRACKING_URI")

    commit_hash = os.getenv("GITHUB_SHA", "")[:7]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")

    if remote_uri:
        mlflow.set_tracking_uri(remote_uri)
        experiment_name = f"ci-build-{commit_hash or timestamp}"
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment_name = f"local-test-{timestamp}"

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
    with mlflow.start_run(run_name=model.__name__, nested=True):

        if param_grid:
            for param, value in param_grid.items():
                mlflow.log_param(param, value)

        if transformer_name:
            mlflow.log_param("transformer", transformer_name)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        if folds_scores is not None and folds_scores_mean is not None:
            for i, score in enumerate(folds_scores):
                mlflow.log_metric(f"fold_{i+1}_r2", score)

            mlflow.log_metric("folds_r2_mean", folds_scores_mean)

        example_input = X_train.iloc[:5]
        example_output = estimator.predict(example_input)

        signature = mlflow.models.infer_signature(example_input, example_output)

        mlflow.sklearn.log_model(
            estimator,
            name=f"{model.__name__}",
            signature=signature,
            input_example=example_input,
        )

        current_file = os.path.abspath(__file__)
        mlflow.log_artifact(current_file)

        if study:
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{model.__name__}",
                "MedicalRegressor",
            )
