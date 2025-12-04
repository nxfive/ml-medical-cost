from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import joblib
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

T = TypeVar("T")


class BaseReader(ABC, Generic[T]):
    @abstractmethod
    def read(self, path: Path) -> T: ...


class CSVReader(BaseReader[pd.DataFrame]):
    def read(self, path: Path) -> pd.DataFrame:
        """
        Reads a CSV file from the given path into a pandas DataFrame.
        """
        return pd.read_csv(path)


class YamlReader(BaseReader[dict]):
    def read(self, path: Path) -> dict:
        """
        Reads a YAML file from the given path into a Python dictionary.
        """
        with open(path) as f:
            return yaml.safe_load(f)


class ParquetReader(BaseReader[pd.DataFrame]):
    def read(self, path: Path) -> pd.DataFrame:
        """
        Reads a Parquet file from the given path into a pandas DataFrame.
        """
        return pd.read_parquet(path)


class JoblibReader(BaseReader[BaseEstimator]):
    def read(self, path: Path) -> BaseEstimator:
        """
        Reads a Joblib file from the given path into a scikit-learn model.
        """
        return joblib.load(path)
