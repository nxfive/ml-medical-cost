import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.models.utils import get_cv


def perform_cross_validation(
    pipeline: Pipeline,
    *, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series
) -> tuple[Pipeline, Sequence[float], float, np.ndarray, np.ndarray]:
    """
    Performs cross-validation on the given pipeline.
    """
    folds_scores = cross_val_score(
        pipeline, X_train, y_train, cv=get_cv(), scoring="r2"
    )
    folds_scores_mean = np.mean(folds_scores)
    print("Fold scores:", folds_scores)
    print("Mean R²:", folds_scores_mean)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    return pipeline, folds_scores, folds_scores_mean, y_train_pred, y_test_pred


def perform_grid_search(
    pipeline: Pipeline,
    param_grid: dict,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[GridSearchCV, Sequence[float], float, np.ndarray, np.ndarray]:
    """
    Performs grid search with cross-validation a on the given pipeline.
    """
    gs = GridSearchCV(
        pipeline, param_grid, cv=get_cv(), scoring="r2", return_train_score=True
    )
    gs.fit(X_train, y_train)

    y_test_pred = gs.best_estimator_.predict(X_test)
    y_train_pred = gs.best_estimator_.predict(X_train)
    cv_results = gs.cv_results_
    n_splits = gs.cv.get_n_splits()
    folds_scores = [
        cv_results[f"split{i}_test_score"][gs.best_index_] for i in range(n_splits)
    ]
    folds_scores_mean = np.mean(folds_scores)

    return gs, folds_scores, folds_scores_mean, y_train_pred, y_test_pred
