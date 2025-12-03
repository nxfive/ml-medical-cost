from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


class BaseReader(ABC):
    @abstractmethod
    def read(self, path: Path) -> Any: ...


class CSVReader(BaseReader):
    def read(self, path: Path) -> pd.DataFrame:
        """
        Reads a CSV file from the given path into a pandas DataFrame.
        """
        return pd.read_csv(path)


class YamlReader(BaseReader):
    def read(self, path: Path) -> dict:
        """
        Reads a YAML file from the given path into a Python dictionary.
        """
        with open(path) as f:
            return yaml.safe_load(f)


class ParquetReader(BaseReader):
    def read(self, path: Path) -> pd.DataFrame:
        """
        Reads a Parquet file from the given path into a pandas DataFrame.
        """
        return pd.read_parquet(path)


class JoblibReader(BaseReader):
    def read(self, path: Path) -> Any:
        """
        Reads a Python object from a joblib file at the given path.
        """
        return joblib.load(path)
