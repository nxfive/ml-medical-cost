from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseWriter(ABC):
    @abstractmethod
    def write(self, df: pd.DataFrame, path: Path): ...


class ParquetWriter(BaseWriter):
    def write(self, df: pd.DataFrame, path: Path, index: bool = False):
        """
        Writes a pandas DataFrame to a Parquet file at the given path.
        """
        df.to_parquet(path, index=index)
