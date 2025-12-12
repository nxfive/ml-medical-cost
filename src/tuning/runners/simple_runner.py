from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.tuning.types import RunnerResult

from .base_runner import BaseRunner


class SimpleRunner(BaseRunner):
    def run(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        folds_scores: list[np.float64] = None,
    ) -> RunnerResult:
        """
        Runs a single estimator with the given parameters grid and collects results.
        """
        estimator = estimator.set_params(**param_grid)
        trained = self.fit_estimator(estimator, X_train, y_train)
        return self._collect_results(trained, folds_scores, X_train, X_test)
