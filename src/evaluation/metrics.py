from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)

from .types import AllMetrics, SplitMetrics, YType


def compute_split_metrics(y: YType, y_pred: YType) -> SplitMetrics:
    """
    Computes regression metrics for a single data split.
    """
    return SplitMetrics(
        r2=r2_score(y, y_pred),
        mae=mean_absolute_error(y, y_pred),
        rmse=root_mean_squared_error(y, y_pred),
    )


def get_metrics(
    y_train: YType, y_test: YType, train_predictions: YType, test_predictions: YType
) -> AllMetrics:
    """
    Computes train and test metrics and returns them as structured results.
    """
    return AllMetrics(
        train=compute_split_metrics(y_train, train_predictions),
        test=compute_split_metrics(y_test, test_predictions),
    )
