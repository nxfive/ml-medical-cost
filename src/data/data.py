import os
import shutil
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split

from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR

CSV_FILE_NAME = "insurance.csv"
PARQUET_FILE_NAME = "insurance.parquet"


def fetch_data() -> pd.DataFrame:
    """
    Fetches the dataset from Kaggle if not already downloaded locally, and returns 
    it as a pandas DataFrame.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    parquet_path = os.path.join(RAW_DATA_DIR, PARQUET_FILE_NAME)
    csv_path = os.path.join(RAW_DATA_DIR, CSV_FILE_NAME)

    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    if not os.path.exists(csv_path):
        cache_path = kagglehub.dataset_download("mirichoi0218/insurance")
        shutil.copytree(cache_path, RAW_DATA_DIR, dirs_exist_ok=True)

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)
    os.remove(csv_path)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the input DataFrame into training and test sets (features and target), converts 
    integer columns to float, and saves the resulting datasets as Parquet files.
    """
    df = df.copy()
    df["age"] = df["age"].astype(float)
    df["children"] = df["children"].astype(float)

    X = df.drop(["charges"], axis=1)
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.2, random_state=42
    )

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    X_train.to_parquet(os.path.join(PROCESSED_DATA_DIR, "X_train.parquet"), index=False)
    X_test.to_parquet(os.path.join(PROCESSED_DATA_DIR, "X_test.parquet"), index=False)
    y_train.to_frame().to_parquet(os.path.join(PROCESSED_DATA_DIR, "y_train.parquet"), index=False)
    y_test.to_frame().to_parquet(os.path.join(PROCESSED_DATA_DIR, "y_test.parquet"), index=False)

    return X_train, X_test, y_train, y_test
