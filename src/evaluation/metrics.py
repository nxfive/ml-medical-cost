from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)

from .types import AllMetrics, SplitMetrics, YType


def compute_split_metrics(y: YType, y_pred: YType) -> SplitMetrics:
    return {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": root_mean_squared_error(y, y_pred),
    }


def get_metrics(
    y_train: YType, y_test: YType, y_train_pred: YType, y_test_pred: YType
) -> AllMetrics:
    return {
        "train": compute_split_metrics(y_train, y_train_pred),
        "test": compute_split_metrics(y_test, y_test_pred),
    }
