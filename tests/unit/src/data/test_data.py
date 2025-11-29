from unittest import mock

import pandas as pd

from src.data.data import fetch_data, split_data


def test_fetch_data_reads_existing_parquet(sample_df, cfg_data):
    with (
        mock.patch("pathlib.Path.mkdir"),
        mock.patch("pathlib.Path.exists", side_effect=[True]),  # parquet exists
        mock.patch("pandas.read_parquet", return_value=sample_df) as mock_read,
    ):
        df = fetch_data(cfg_data)

    mock_read.assert_called_once()
    pd.testing.assert_frame_equal(df, sample_df)


def test_fetch_data_reads_csv_and_converts(sample_df, cfg_data):
    with (
        mock.patch("pathlib.Path.mkdir"),
        mock.patch("pathlib.Path.exists", side_effect=[False, True]),
        mock.patch("pandas.read_csv", return_value=sample_df) as mock_read_csv,
        mock.patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
        mock.patch("pathlib.Path.unlink") as mock_unlink,
    ):
        df = fetch_data(cfg_data)

    mock_read_csv.assert_called_once()
    mock_to_parquet.assert_called_once()
    mock_unlink.assert_called_once()
    pd.testing.assert_frame_equal(df, sample_df)


def test_split_data(cfg_data):
    df = pd.DataFrame(
        {"age": [20, 30], "bmi": [18.5, 22.8], "charges": [2345.3, 7891.1]}
    )

    mock_X = df[["age", "bmi"]]
    mock_y = df["charges"]

    mock_split_return = (mock_X, mock_X, mock_y, mock_y)

    with (
        mock.patch("pathlib.Path.mkdir") as mock_mkdir,
        mock.patch("pathlib.Path.exists"),
        mock.patch(
            "src.data.data.train_test_split", return_value=mock_split_return
        ) as mock_split,
        mock.patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
    ):
        split_data(df, cfg_data)

    mock_split.assert_called_once()
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert mock_to_parquet.call_count == 4
