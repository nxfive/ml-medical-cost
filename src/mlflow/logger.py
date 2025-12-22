import uuid

import pandas as pd

from src.containers.results import StageResult

from .service import MLflowService


class MLflowLogger:
    def __init__(self, service: MLflowService):
        self.service = service
        self.service.setup()

    def log_model(
        self, result: StageResult, X_train: pd.DataFrame, register: bool = False
    ) -> None:
        """
        Orchestrates logging a model, parameters, metrics, and artifacts to MLflow.
        """
        run_name = f"{result.model_name}-{uuid.uuid4().hex[:6]}"
        with self.service.start_run(run_name):
            self.service.log_params(result.params)
            self.service.log_metrics(
                result.metrics, result.folds_scores, result.folds_scores_mean
            )
            self.service.log_artifacts(result.estimator, result.model_name, X_train)
            if register:
                self.service.register_model(result.model_name)


logger = MLflowLogger(service=MLflowService())
