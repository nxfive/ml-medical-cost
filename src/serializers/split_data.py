import pandas as pd

from src.containers.data import SplitData
from src.containers.types import SplitDataDict


class SplitDataSerializer:
    def to_dict(self) -> SplitDataDict:
        def to_df(s: pd.Series) -> pd.DataFrame:
            return s.to_frame() if isinstance(s, pd.Series) else s

        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": to_df(self.y_train),
            "y_test": to_df(self.y_test),
        }

    @staticmethod
    def from_dict(data: SplitDataDict) -> SplitData:
        def to_series(df: pd.DataFrame) -> pd.Series:
            return df.iloc[:, 0] if df.shape[1] == 1 else df

        return SplitData(
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=to_series(data["y_train"]),
            y_test=to_series(data["y_test"]),
        )
