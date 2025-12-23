from abc import ABC, abstractmethod
from typing import Generic

import pandas as pd

from src.containers.data import SplitData
from src.containers.results import PredictionSet, StageResult
from src.containers.types import BuildResultType, RunResultType
from src.data.core import DataLoader
from src.dto.metrics import AllMetrics
from src.evaluation.metrics import get_metrics
from src.mlflow.logger import MLflowLogger


class BasePipeline(ABC, Generic[BuildResultType, RunResultType]):
    @abstractmethod
    def build(self) -> BuildResultType: ...

    @abstractmethod
    def run(self) -> RunResultType: ...

    def load_data(self, data_loader: DataLoader) -> SplitData:
        """
        Loads preprocessed training and test data using the provided readers.
        """
        return data_loader.load_splitted_data(
            processed_dir=self.cfg.data_dir.processed_dir
        )

    @staticmethod
    def _compute_metrics(pred_set: PredictionSet) -> AllMetrics:
        """
        Computes training and test metrics from a PredictionSet.
        """
        return get_metrics(
            y_train=pred_set.y_train,
            y_test=pred_set.y_test,
            train_predictions=pred_set.train_predictions,
            test_predictions=pred_set.test_predictions,
        )

    @staticmethod
    def _log_model(
        logger: MLflowLogger,
        stage_result: StageResult,
        X_train: pd.DataFrame,
        register: bool = False,
    ) -> None:
        """
        Logs the training results and metrics to MLflow.
        """
        logger.log_model(result=stage_result, X_train=X_train, register=register)
