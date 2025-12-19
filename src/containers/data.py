from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series | pd.DataFrame
    y_test: pd.Series | pd.DataFrame
