from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    def to_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }
