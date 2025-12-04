from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from src.io.file_ops import PathManager
from src.io.readers import JoblibReader, ParquetReader, YamlReader
from src.io.writers import JoblibWriter, ParquetWriter, YamlWriter

from .converters import CSVToParquetConverter
from .download import DatasetDownloader
from .types import SplitData


class DataLoader:
    def __init__(
        self,
        processed_dir: Path,
        parquet_reader: ParquetReader,
        yaml_reader: YamlReader,
        joblib_reader: JoblibReader,
    ):
        self.processed_dir = processed_dir
        self.parquet_reader = parquet_reader
        self.yaml_reader = yaml_reader
        self.joblib_reader = joblib_reader

    def load_splitted_data(self) -> SplitData:
        """
        Loads train/test splits from processed directory into a SplitData object.
        """
        X_train = self.parquet_reader.read(self.processed_dir / "X_train.parquet")
        X_test = self.parquet_reader.read(self.processed_dir / "X_test.parquet")
        y_train = self.parquet_reader.read(self.processed_dir / "y_train.parquet")
        y_test = self.parquet_reader.read(self.processed_dir / "y_test.parquet")

        return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def load_metrics(self, metrics_path: Path) -> dict[str, Any]:
        """
        Loads metrics from a YAML file and return as a dictionary.
        """
        return self.yaml_reader.read(metrics_path)

    def load_model(self, model_path: Path) -> BaseEstimator:
        """
        Loads model from Pickle file and return estimator.
        """
        return self.joblib_reader.read(model_path)


class DataSaver:
    def __init__(
        self,
        processed_dir: Path,
        parquet_writer: ParquetWriter,
        joblib_writer: JoblibWriter,
        yaml_writer: YamlWriter,
    ):
        self.processed_dir = processed_dir
        self.parquet_writer = parquet_writer
        self.joblib_writer = joblib_writer
        self.yaml_writer = yaml_writer

    def save_splitted_data(self, splits: SplitData) -> None:
        """
        Saves each split from a SplitData object as a Parquet file in the processed
        directory.
        """
        for name, value in splits.to_dict().items():
            self.parquet_writer.write(value, self.processed_dir / f"{name}.parquet")

    def save_metrics(self, metrics: dict[str, Any], metrics_path: Path) -> None:
        """
        Saves evaluation metrics as a YAML file.
        """
        self.yaml_writer.write(metrics, metrics_path)

    def save_model(self, model: BaseEstimator, model_path: Path) -> None:
        """
        Saves a scikit-learn model using Joblib.
        """
        self.joblib_writer.write(model, model_path)


class DataFetcher:
    def __init__(
        self,
        raw_dir: Path,
        downloader: DatasetDownloader,
        converter: CSVToParquetConverter,
        parquet_reader: ParquetReader,
    ):
        self.raw_dir = raw_dir
        self.downloader = downloader
        self.converter = converter
        self.parquet_reader = parquet_reader

    def fetch(self, filename: str = "insurance.parquet") -> pd.DataFrame:
        """
        Fetches dataset as a Pandas DataFrame. Downloads CSV and converts to
        Parquet if needed.
        """
        parquet_path = self.raw_dir / filename

        if PathManager.exists(parquet_path):
            return self.parquet_reader.read(parquet_path)

        csv_path = self.downloader.download()
        parquet_path = self.converter.convert(csv_path, parquet_path)
        PathManager.remove_file(csv_path)
        return self.parquet_reader.read(parquet_path)
