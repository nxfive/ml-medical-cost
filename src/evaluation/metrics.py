from dataclasses import asdict

import numpy as np
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)

from .types import AllMetrics, SplitMetrics, YType


def compute_scores_mean(fold_scores: list[np.float64]) -> np.float64:
    """
    Computes the mean score from a list of fold scores.
    """
    return np.mean(fold_scores)


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


def flatten_dict(prefix: str, d: dict) -> dict[str, float]:
    """
    Recursively flattens a nested dictionary into a single-level dict,
    prefixing nested keys with their parent keys.
    """
    flat = {}
    for key, value in d.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(name, value))
        else:
            flat[name] = value
    return flat


def flatten_metrics(metrics: AllMetrics) -> dict[str, float]:
    """
    Converts an AllMetrics dataclass into a flat dictionary.
    """
    return flatten_dict("", asdict(metrics))
