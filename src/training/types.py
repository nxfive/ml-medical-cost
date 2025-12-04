from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class RunnerResult:
    trained: BaseEstimator
    folds_scores: list[np.float64]
    folds_scores_mean: np.float64
    train_predictions: np.ndarray
    test_predictions: np.ndarray


@dataclass
class EvaluationResult:
    estimator: BaseEstimator
    param_grid: dict[str, list]
    transformation_name: str
