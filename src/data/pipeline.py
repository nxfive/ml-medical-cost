from pathlib import Path

import pandas as pd

from src.features.core import convert_features_type

from .core import split_features_target, split_train_test
from .io import PARQUET_FILE_NAME


class DataPipeline:
    def __init__(self, downloader, io_rw, raw_dir: Path, processed_dir: Path):
        self.downloader = downloader
        self.io_rw = io_rw
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches the dataset from Kaggle if not already downloaded locally
        and return it as a pandas DataFrame.
        """
        parquet_path = Path(self.raw_dir) / PARQUET_FILE_NAME

        if parquet_path.exists():
            return self.io_rw.read_parquet(parquet_path)

        csv_path = self.downloader.download(self.raw_dir)
        df = self.io_rw.read_csv(csv_path)
        self.io_rw.write_parquet(df, parquet_path)
        csv_path.unlink(missing_ok=True)
        return df

    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Saves train/test splits as Parquet files.
        """
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        self.io_rw.write_parquet(X_train, self.processed_dir / "X_train.parquet")
        self.io_rw.write_parquet(X_test, self.processed_dir / "X_test.parquet")
        self.io_rw.write_parquet(
            y_train.to_frame(), self.processed_dir / "y_train.parquet"
        )
        self.io_rw.write_parquet(
            y_test.to_frame(), self.processed_dir / "y_test.parquet"
        )

    def split_data(self, df: pd.DataFrame, target_col: str = "charges") -> None:
        """
        Splits a DataFrame into features (X) and target (y), then into training and test sets,
        and save the resulting datasets as Parquet files.
        """
        X, y = split_features_target(df, target_col)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        self.save_splits(X_train, X_test, y_train, y_test)

    def run(self) -> None:
        """
        Loads and preprocesses the dataset, then splits it into training and test sets.
        """
        df = self.fetch_data()
        df = convert_features_type(df)
        self.split_data(df)
