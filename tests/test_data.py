from unittest import mock

import pandas as pd
import pytest

from src.data.data import fetch_data, split_data
from src.data.pipeline import data_pipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame({"age": [20, 30], "bmi": [18.5, 22.8]})


def test_fetch_data_reads_exisiting_parquet(tmp_path, sample_df):
    parquet_path = tmp_path / "insurance.parquet"
    sample_df.to_parquet(parquet_path)

    with (
        mock.patch("src.data.data.RAW_DATA_DIR", tmp_path),
        mock.patch("src.data.data.PARQUET_FILE_NAME", "insurance.parquet"),
        mock.patch("pandas.read_parquet", return_value=sample_df) as mock_read,
    ):

        df = fetch_data()
        mock_read.assert_called_once_with(str(parquet_path))
        pd.testing.assert_frame_equal(df, sample_df)


def test_fetch_data_download_if_not_existing(tmp_path, sample_df):
    csv_path = tmp_path / "insurance.csv"
    sample_df.to_csv(csv_path, index=False)

    with (
        mock.patch("src.data.data.RAW_DATA_DIR", tmp_path),
        mock.patch("src.data.data.CSV_FILE_NAME", "insurance.csv"),
        mock.patch("src.data.data.PARQUET_FILE_NAME", "insurance.parquet"),
        mock.patch(
            "src.data.data.kagglehub.dataset_download", return_value="fake_kaggle_path"
        ),
        mock.patch("pandas.read_csv", return_value=sample_df) as mock_read_csv,
        mock.patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
        mock.patch("os.remove") as mock_remove,
    ):

        df = fetch_data()
        mock_read_csv.assert_called_once_with(str(csv_path))
        mock_to_parquet.assert_called_once()
        mock_remove.assert_called_once_with(str(csv_path))
        pd.testing.assert_frame_equal(df, sample_df)


def test_split_data():
    sample_df = pd.DataFrame(
        {"age": [20, 30], "bmi": [18.5, 22.8], "charges": [2345.3, 7891.1]}
    )

    with (
        mock.patch("os.makedirs") as mock_makedirs,
        mock.patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
        mock.patch(
            "src.data.data.train_test_split",
            return_value=(
                sample_df.drop("charges", axis=1),
                sample_df.drop("charges", axis=1),
                sample_df["charges"],
                sample_df["charges"],
            ),
        ) as mock_split,
    ):

        X_train, X_test, y_train, y_test = split_data(sample_df)

        mock_makedirs.assert_called_once()
        assert all(not data.empty for data in (X_train, X_test, y_train, y_test))
        mock_to_parquet.assert_called()
        mock_split.assert_called_once()


def test_data_pipeline(sample_df):
    with (
        mock.patch(
            "src.data.pipeline.fetch_data", return_value=sample_df
        ) as mock_fetch_data,
        mock.patch(
            "src.data.pipeline.convert_features_type", return_value=sample_df
        ) as mock_convert_features,
        mock.patch(
            "src.data.pipeline.split_data",
            return_value=("X_train, X_test, y_train, y_test"),
        ) as mock_split_data,
    ):

        result = data_pipeline()

        mock_fetch_data.assert_called_once()
        mock_convert_features.assert_called_once_with(sample_df)
        mock_split_data.assert_called_once_with(sample_df)

        assert result == ("X_train, X_test, y_train, y_test")
