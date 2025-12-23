from typing import Sequence

from src.logger.setup import logger


class ModelDiagnostics:
    def __init__(
        self,
        folds_scores: Sequence[float] = None,
        fold_threshold: float = 0.1,
        overfit_threshold: float = 0.2,
    ):
        self.folds_scores = folds_scores
        self.fold_threshold = fold_threshold
        self.overfit_threshold = overfit_threshold

    def is_stable(self) -> bool:
        """
        Checks the stability of a model based on cross-validation fold scores.
        """
        if not self.folds_scores:
            return True
        difference = max(self.folds_scores) - min(self.folds_scores)
        return difference <= self.fold_threshold

    def is_overfitting(self, train_r2: float, test_r2: float) -> bool:
        """
        Checks whether a model is likely overfitting based on the difference between
        training and test R² scores.
        """
        return (train_r2 - test_r2) > self.overfit_threshold

    def report(self, model_name: str, train_r2: float, test_r2: float):
        """
        Logs warnings about potential model instability or overfitting.
        """
        if self.folds_scores and not self.is_stable():
            logger.warning(f"{model_name}: fold results indicate potential instability")
        
        if self.is_overfitting(train_r2, test_r2):
            logger.warning(
                f"{model_name} may be overfitting. Consider adjusting hyperparameters"
            )
