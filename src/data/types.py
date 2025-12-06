from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd


class SplitDataDict(TypedDict):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | pd.DataFrame
    y_test: pd.Series | pd.DataFrame

    def to_dict(self) -> SplitDataDict:
        def to_df(s: pd.Series) -> pd.DataFrame:
            return s.to_frame() if isinstance(s, pd.Series) else s

        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": to_df(self.y_train),
            "y_test": to_df(self.y_test),
        }

    @classmethod
    def from_dict(cls, data: SplitDataDict) -> SplitData:
        def to_series(df: pd.DataFrame) -> pd.Series:
            return df.iloc[:, 0] if df.shape[1] == 1 else df

        return cls(
            X_train=data["X_train"],
            X_test=data["X_test"],
            y_train=to_series(data["y_train"]),
            y_test=to_series(data["y_test"]),
        )
