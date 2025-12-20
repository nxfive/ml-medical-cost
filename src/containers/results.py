from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .types import YType


@dataclass
class EvaluationResult:
    estimator: BaseEstimator
    param_grid: dict[str, list]
    transformation: str = "none"


@dataclass
class PredictionResult:
    trained: BaseEstimator
    train_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass
class RunnerResult(PredictionResult):
    folds_scores: list[np.float64]
    folds_scores_mean: np.float64
    params: dict[str, Any]


@dataclass
class RunResult:
    runner_result: RunnerResult
    param_grid: dict[str, list]
    transformation: str = "none"


@dataclass
class StageResult:
    model_name: str
    estimator: BaseEstimator
    params: dict[str, Any]
    param_grid: dict[str, list]
    folds_scores: list[float]
    folds_scores_mean: float
    metrics: dict[str, float]
    transformation: str | None = None


@dataclass
class LoadedModelResults:
    runs: dict[str, StageResult]


@dataclass
class PredictionSet:
    y_train: pd.Series
    y_test: pd.Series
    train_predictions: YType
    test_predictions: YType
