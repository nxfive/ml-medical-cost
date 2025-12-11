from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold

from .search_runner import SearchRunner


class GridSearchRunner(SearchRunner[GridSearchCV]):
    def __init__(self, cv: KFold, scoring: str = "r2"):
        self.cv = cv
        self.scoring = scoring

    def perform_search(
        self, estimator: BaseEstimator, param_grid: dict[str, list]
    ) -> GridSearchCV:
        """
        Creates a GridSearchCV object for the given estimator and parameter grid.
        """
        return GridSearchCV(
            estimator,
            param_grid,
            cv=self.cv,
            scoring=self.scoring,
            return_train_score=True,
        )
