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


def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Splits DataFrame into features X and target y.
    """
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):
    """
    Splits features and target into train/test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_splits(X_train, X_test, y_train, y_test, processed_data_path: Path):
    """
    Saves train/test splits as Parquet files.
    """
    processed_data_path.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(processed_data_path / "X_train.parquet", index=False)
    X_test.to_parquet(processed_data_path / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(processed_data_path / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(processed_data_path / "y_test.parquet", index=False)


def split_data(df: pd.DataFrame, cfg: DictConfig, target_col="charges"):
    """
    Split a DataFrame into features (X) and target (y), then into training and test sets,
    and save the resulting datasets as Parquet files.
    """
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    save_splits(X_train, X_test, y_train, y_test, Path(cfg.data.processed_dir))
