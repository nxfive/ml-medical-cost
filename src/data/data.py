from pathlib import Path

import pandas as pd

from src.data.split import split_features_target, split_train_test

from .core import DataFetcher, DataSaver


class Data:
    def __init__(
        self, data_saver: DataSaver, data_fetcher: DataFetcher, processed_dir: Path
    ):
        self.data_saver = data_saver
        self.data_fetcher = data_fetcher
        self.processed_dir = processed_dir

    def fetch(self) -> pd.DataFrame:
        """
        Delegates dataset fetching to DataFetcher, which handles downloading,
        conversion from CSV to Parquet, and cleanup of temporary files.
        """
        return self.data_fetcher.fetch()

    def split(self, df: pd.DataFrame, target_col: str = "charges") -> None:
        """
        Splits a DataFrame into features (X) and target (y), then into training and test sets,
        and save the resulting datasets as Parquet files.
        """
        X, y = split_features_target(df, target_col)
        split_data = split_train_test(X, y)
        self.data_saver.save_splitted_data(
            splits=split_data, processed_dir=self.processed_dir
        )
