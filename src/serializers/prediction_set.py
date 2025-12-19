from src.containers.data import SplitData
from src.containers.results import PredictionSet, RunnerResult


class PredictionSetSerializer:
    @staticmethod
    def from_stage_pipeline(
        result: RunnerResult, split_data: SplitData
    ) -> PredictionSet:
        return PredictionSet(
            y_train=split_data.y_train,
            y_test=split_data.y_test,
            train_predictions=result.train_predictions,
            test_predictions=result.test_predictions,
        )
