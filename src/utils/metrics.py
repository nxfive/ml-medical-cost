import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)


def get_metrics(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> dict[str, float]:
    """
    Computes train and test metrics (R², MAE, RMSE) for model predictions.
    """
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "train_rmse": root_mean_squared_error(y_train, y_train_pred),
        "test_rmse": root_mean_squared_error(y_test, y_test_pred),
    }

    return metrics