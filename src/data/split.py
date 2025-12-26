from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.containers.data import SplitData
from src.containers.types import SplitDataDict
from src.io.file_ops import PathManager
from src.serializers.split_data import SplitDataSerializer

from .constants import SPLIT_FILES


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits DataFrame into features X and target y.
    """
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42
) -> SplitDataDict:
    """
    Splits features and target into train/test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    split_data = SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return SplitDataSerializer.to_dict(split_data)


def get_missing_split_files(processed_dir: Path) -> list[str]:
    """
    Check which expected split files are missing in the processed directory.
    """
    return [
        file
        for file in SPLIT_FILES
        if not PathManager.exists(processed_dir / f"{file}.parquet")
    ]
