from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

import mlflow
from mlflow.tracking import MlflowClient
from src.settings import Settings

MLFLOW_REGISTER_NAME = "MedicalRegressor"


class MLflowService:
    def __init__(self):
        self.mlflow = mlflow
        self.client = MlflowClient()

    def setup(self, create_experiment: bool = True) -> None:
        """
        Sets MLflow tracking URI and optionally creates/sets an experiment.
        """
        tracking_uri = Settings.tracking_uri()
        self.mlflow.set_tracking_uri(tracking_uri)

        if create_experiment:
            experiment_name = Settings.experiment_name()
            self.mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """
        Starts a new MLflow run.
        """
        return self.mlflow.start_run(run_name=run_name)

    def get_latest_model_version(self, model_name: str) -> int:
        """
        Gets the latest version number of a registered MLflow model.
        """
        all_versions = self.client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            raise ValueError(f"No versions found for model {model_name}")
        return max(int(v.version) for v in all_versions)

    def load_model(self, model_name: str, version: int) -> BaseEstimator:
        """
        Loads a specific version of a registered MLflow model.
        """
        return self.mlflow.sklearn.load_model(f"models:/{model_name}/{version}")

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Logs model hyperparameters to MLflow.
        """
        for param, value in params.items():
            self.mlflow.log_param(param, value)

    def log_metrics(
        self,
        metrics: dict[str, float],
        folds_scores: list[float],
        folds_scores_mean: float,
    ) -> None:
        """
        Logs evaluation metrics and cross-validation fold scores to MLflow.
        """
        step = 1
        for name, value in metrics.items():
            self.mlflow.log_metric(name, value, step)
            step += 1

        for idx, score in enumerate(folds_scores, start=1):
            self.mlflow.log_metric(f"fold_{idx}_r2", score, step)
            step += 1
        self.mlflow.log_metric("folds_r2_mean", folds_scores_mean, step)

    def log_artifacts(
        self, estimator: BaseEstimator, model_name: str, X_train: pd.DataFrame
    ) -> None:
        """
        Logs the trained model to MLflow with input example and signature.
        """
        example_input = X_train.iloc[:5]
        example_output = estimator.predict(example_input)
        signature = self.mlflow.models.infer_signature(example_input, example_output)
        self.mlflow.sklearn.log_model(
            estimator,
            name=model_name,
            signature=signature,
            input_example=example_input,
        )

    def register_model(self, model_name: str) -> None:
        """
        Registers the trained model to MLflow Model Registry.
        """
        self.mlflow.register_model(
            f"runs:/{self.mlflow.active_run().info.run_id}/{model_name}",
            name=MLFLOW_REGISTER_NAME,
        )
