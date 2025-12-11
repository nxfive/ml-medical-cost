from dataclasses import dataclass
from typing import TypedDict, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import FunctionTransformer

RunnerType = TypeVar("RunnerType", bound=BaseEstimator)


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


class TransformersDict(TypedDict):
    log: FunctionTransformer
    sqrt: FunctionTransformer
    square: FunctionTransformer
    none: None
