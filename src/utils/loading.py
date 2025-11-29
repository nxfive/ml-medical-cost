from pathlib import Path

import pandas as pd


def load_splitted_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load splitted data from data/processed directory.
    """
    X_train = pd.read_parquet(Path(cfg.data.processed_dir) / "X_train.parquet")
    X_test = pd.read_parquet(Path(cfg.data.processed_dir) / "X_test.parquet")
    y_train = pd.read_parquet(Path(cfg.data.processed_dir) / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(Path(cfg.data.processed_dir) / "y_test.parquet").squeeze()

    return X_train, X_test, y_train, y_test
