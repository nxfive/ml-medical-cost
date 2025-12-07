from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

YType = pd.Series | npt.NDArray[np.float64]


@dataclass
class SplitMetrics:
    r2: float
    mae: float
    rmse: float


@dataclass
class AllMetrics:
    train: SplitMetrics
    test: SplitMetrics
