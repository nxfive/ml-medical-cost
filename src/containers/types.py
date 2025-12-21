from typing import TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator

"""Target / prediction vector"""
YType = pd.Series | npt.NDArray[np.float64]

"""Type returned by build() method of the base pipeline"""
BuildResultType = TypeVar("BuildResultType")

"""Type returned by run() method of the base pipeline"""
RunResultType = TypeVar("RunResultType")

"""Type returned by perform_search() method of the search runner"""
RunnerType = TypeVar("RunnerType", bound=BaseEstimator)

"""Type returned by run() method of the base runner"""
OptunaRunnerResult = TypeVar("OptunaRunnerResult")


class SplitDataDict(TypedDict):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
