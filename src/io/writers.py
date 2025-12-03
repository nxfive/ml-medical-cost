from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml
import joblib
from sklearn.base import BaseEstimator

from typing import Generic, TypeVar

T = TypeVar("T")


class BaseWriter(ABC, Generic[T]):
    @abstractmethod
    def write(self, data: T, path: Path) -> None: ...


class ParquetWriter(BaseWriter[pd.DataFrame]):
    def write(self, data: pd.DataFrame, path: Path) -> None:
        """
        Writes a pandas DataFrame to a Parquet file at the given path.
        """
        data.to_parquet(path, index=False)


class YamlWriter(BaseWriter[dict]):
    def write(self, data: dict, path: Path) -> None:
        """
        Writes a Python dictionary to a YAML file at the given path.
        """
        with open(path, "w") as f:
            yaml.safe_dump(data, f)


class JoblibWriter(BaseWriter[BaseEstimator]):
    def write(self, data: BaseEstimator, path: Path) -> None:
        """
        Writes a Python object to a joblib file at the given path.
        """
        joblib.dump(data, path)
