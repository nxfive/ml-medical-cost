from pathlib import Path
from unittest import mock

import pandas as pd

from src.utils.loading import load_metrics, load_model, load_splitted_data


def test_load_splitted_data_unit(cfg_loading):
    cfg = cfg_loading
    fake_X_train, fake_X_test = pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})
    fake_y_train, fake_y_test = pd.Series([1, 2]), pd.Series([2, 3])

    with mock.patch(
        "pandas.read_parquet",
        side_effect=[
            fake_X_train,
            fake_X_test,
            fake_y_train.to_frame(),
            fake_y_test.to_frame(),
        ],
    ) as mock_read:
        X_train, X_test, y_train, y_test = load_splitted_data(cfg)

    assert X_train.equals(fake_X_train)
    assert X_test.equals(fake_X_test)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert y_train.equals(fake_y_train)
    assert y_test.equals(fake_y_test)

    assert mock_read.call_count == 4
    expected_calls = [
        mock.call(Path(cfg.data.processed_dir) / "X_train.parquet"),
        mock.call(Path(cfg.data.processed_dir) / "X_test.parquet"),
        mock.call(Path(cfg.data.processed_dir) / "y_train.parquet"),
        mock.call(Path(cfg.data.processed_dir) / "y_test.parquet"),
    ]
    mock_read.assert_has_calls(expected_calls, any_order=False)


def test_load_metrics():
    metrics_path = Path("fake/path/metrics.yml")
    metrics_data = "r2: 0.9"

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=metrics_data)) as mock_open,
        mock.patch("yaml.safe_load", return_value={"r2": 0.9}) as mock_safe_load,
    ):
        results = load_metrics(metrics_path)
    mock_open.assert_called_once_with(metrics_path)

    handle = mock_open()
    mock_safe_load.assert_called_once_with(handle)
    assert results == {"r2": 0.9}


def test_load_model():
    model_path = Path("fake/path/pipeline.pkl")

    with mock.patch("joblib.load", return_value=mock.Mock()) as mock_joblib_load:
        results = load_model(model_path)

    mock_joblib_load.assert_called_once_with(model_path)
    assert results == mock_joblib_load.return_value
