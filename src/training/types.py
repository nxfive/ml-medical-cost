from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class PredictionResult:
    trained: BaseEstimator
    train_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass
class RunnerResult(PredictionResult):
    folds_scores: list[np.float64]
    folds_scores_mean: np.float64


@dataclass
class EvaluationResult:
    estimator: BaseEstimator
    param_grid: dict[str, list]
    transformation: str = "none"


@dataclass
class TrainResult:
    runner_result: RunnerResult
    param_grid: dict[str, list]
    transformation: str | None


class TransformersDict(TypedDict):
    log: BaseEstimator
    quantile: BaseEstimator
    none: None
