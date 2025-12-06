from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from src.io.file_ops import PathManager
from src.io.types import Readers, Writers

from .converters import CSVToParquetConverter
from .download import DatasetDownloader
from .types import SplitData, SplitDataDict


class DataLoader:
    def __init__(self, readers: Readers):
        self.readers = readers

    def load_splitted_data(self, processed_dir: Path) -> SplitData:
        """
        Loads train/test splits from processed directory into a SplitData object.
        """
        files = ["X_train", "X_test", "y_train", "y_test"]
        data = {
            file: self.readers.parquet.read(processed_dir / f"{file}.parquet")
            for file in files
        }
        return SplitData.from_dict(data)

    def load_metrics(self, metrics_path: Path) -> dict[str, Any]:
        """
        Loads metrics from a YAML file and return as a dictionary.
        """
        return self.readers.yaml.read(metrics_path)

    def load_model(self, model_path: Path) -> BaseEstimator:
        """
        Loads model from Pickle file and return estimator.
        """
        return self.readers.joblib.read(model_path)


class DataSaver:
    def __init__(self, writers: Writers):
        self.writers = writers

    def save_splitted_data(self, splits: SplitDataDict, processed_dir: Path) -> None:
        """
        Saves each split from a SplitData object as a Parquet file in the processed
        directory.
        """
        for name, value in splits.items():
            self.writers.parquet.write(value, processed_dir / f"{name}.parquet")

    def save_metrics(self, metrics: dict[str, Any], metrics_path: Path) -> None:
        """
        Saves evaluation metrics as a YAML file.
        """
        self.writers.yaml.write(metrics, metrics_path)

    def save_model(self, model: BaseEstimator, model_path: Path) -> None:
        """
        Saves a scikit-learn model using Joblib.
        """
        self.writers.joblib.write(model, model_path)


class DataFetcher:
    def __init__(
        self,
        raw_dir: Path,
        downloader: DatasetDownloader,
        converter: CSVToParquetConverter,
        readers: Readers,
    ):
        self.raw_dir = raw_dir
        self.downloader = downloader
        self.converter = converter
        self.readers = readers

    def fetch(self, filename: str = "insurance.parquet") -> pd.DataFrame:
        """
        Fetches dataset as a Pandas DataFrame. Downloads CSV and converts to
        Parquet if needed.
        """
        parquet_path = self.raw_dir / filename

        if PathManager.exists(parquet_path):
            return self.readers.parquet.read(parquet_path)

        csv_path = self.downloader.download()
        parquet_path = self.converter.convert(csv_path, parquet_path)
        PathManager.remove_file(csv_path)
        return self.readers.parquet.read(parquet_path)
