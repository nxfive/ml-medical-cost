import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.types import SplitData, SplitDataDict


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
    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    ).to_dict()
