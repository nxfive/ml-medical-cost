import numpy as np
from sklearn.model_selection import cross_val_score

from src.models.utils import get_cv


def perform_cross_validation(pipeline, *, X_train, X_test, y_train):
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
