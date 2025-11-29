from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.base import BaseEstimator


def load_splitted_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load splitted data from data/processed directory.
    """
    X_train = pd.read_parquet(Path(cfg.data.processed_dir) / "X_train.parquet")
    X_test = pd.read_parquet(Path(cfg.data.processed_dir) / "X_test.parquet")
    y_train = pd.read_parquet(Path(cfg.data.processed_dir) / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(Path(cfg.data.processed_dir) / "y_test.parquet").squeeze()

    return X_train, X_test, y_train, y_test


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """
    Load metrics from a YAML file and return as a dictionary.
    """
    with open(metrics_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: Path) -> BaseEstimator:
    """
    Load a scikit-learn model or pipeline from a file.
    """
    return joblib.load(model_path)
