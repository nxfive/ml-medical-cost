from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.evaluation.metrics import get_metrics
from src.evaluation.types import AllMetrics
from src.patterns.types import PredictionSet

B = TypeVar("B")
T = TypeVar("T")


class BasePipeline(ABC, Generic[B, T]):
    @abstractmethod
    def build(self) -> B: ...

    @abstractmethod
    def run(self) -> T: ...

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
