from unittest import mock

import numpy as np
import pandas as pd

from src.utils.metrics import get_metrics
                             

def test_get_metrics():
    y_train = pd.Series([1, 2, 3])
    y_test = pd.Series([4, 5, 6])

    y_train_pred = np.array([1.1, 2.1, 3.2])
    y_test_pred = np.array([4.8, 5.5, 5.4])

    with (
        mock.patch("src.utils.utils.mean_absolute_error", return_value=0.8) as mock_mae,
        mock.patch("src.utils.utils.r2_score", return_value=0.5) as mock_r2,
        mock.patch(
            "src.utils.utils.root_mean_squared_error", return_value=0.3
        ) as mock_rmse,
    ):

        metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)

        assert mock_mae.call_count == 2
        assert mock_r2.call_count == 2
        assert mock_rmse.call_count == 2

        expected_keys = {
            "train_r2",
            "test_r2",
            "train_mae",
            "test_mae",
            "train_rmse",
            "test_rmse",
        }

        assert expected_keys == set(metrics.keys())

        mock_mae.assert_called()
        mock_r2.assert_called()
        mock_rmse.assert_called()
