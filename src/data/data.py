import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

CSV_FILE_NAME = "insurance.csv"
PARQUET_FILE_NAME = "insurance.parquet"


def download_dataset_if_missing(cache_dir: Path) -> Path:
    """
    Download dataset from Kaggle if not present and return path to CSV.
    """
    if not (cache_dir / CSV_FILE_NAME).exists():
        temp_path = kagglehub.dataset_download("mirichoi0218/insurance")
        shutil.copytree(temp_path, cache_dir, dirs_exist_ok=True)
    return cache_dir / CSV_FILE_NAME


def save_parquet(df: pd.DataFrame, parquet_path: Path):
    """
    Save DataFrame as parquet file.
    """
    df.to_parquet(parquet_path, index=False)


def fetch_data(cfg: DictConfig) -> pd.DataFrame:
    """
    Fetches the dataset from Kaggle if not already downloaded locally,
    and returns it as a pandas DataFrame.
    """
    raw_data_path = Path(cfg.data.raw_dir)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_data_path / PARQUET_FILE_NAME

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_path = download_dataset_if_missing(raw_data_path)

    df = pd.read_csv(csv_path)
    save_parquet(df, parquet_path)
    csv_path.unlink(missing_ok=True)

    return df


def split_data(df: pd.DataFrame, cfg: DictConfig):
    """
    Splits the input DataFrame into training and test sets (features and target), converts
    integer columns to float, and saves the resulting datasets as Parquet files.
    """
    df = df.copy()

    X = df.drop(["charges"], axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    processed_data_path = Path(cfg.data.processed_dir)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(processed_data_path / "X_train.parquet", index=False)
    X_test.to_parquet(processed_data_path / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(processed_data_path / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(processed_data_path / "y_test.parquet", index=False)
