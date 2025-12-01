import shutil
from pathlib import Path

import kagglehub
import pandas as pd

CSV_FILE_NAME = "insurance.csv"
PARQUET_FILE_NAME = "insurance.parquet"


class DatasetDownloader:
    def download(self, raw_dir: Path) -> Path:
        """
        Downloads datasets from Kaggle if not already present and return path to CSV.
        """
        if not (raw_dir / CSV_FILE_NAME).exists():
            temp_dir = kagglehub.dataset_download("mirichoi0218/insurance")
            shutil.copytree(temp_dir, raw_dir, dirs_exist_ok=True)
        return raw_dir / CSV_FILE_NAME


class DataReaderWriter:
    def read_csv(self, path: Path) -> pd.DataFrame:
        """
        Reads a CSV file from the given path into a pandas DataFrame.
        """
        return pd.read_csv(path)

    def read_parquet(self, path: Path) -> pd.DataFrame:
        """
        Reads a Parquet file from the given path into a pandas DataFrame.
        """
        return pd.read_parquet(path)

    def write_parquet(self, df, path: Path):
        """
        Writes a pandas DataFrame to a Parquet file at the given path.
        """
        df.to_parquet(path, index=False)
