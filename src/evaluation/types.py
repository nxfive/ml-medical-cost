from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

YType = pd.Series | npt.NDArray[np.float_]


class SplitMetrics(TypedDict):
    r2: float
    mae: float
    rmse: float


class AllMetrics(TypedDict):
    train: SplitMetrics
    test: SplitMetrics
