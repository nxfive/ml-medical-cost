from typing import Sequence


def check_fold_stability(folds_scores: Sequence[float], threshold: float = 0.1) -> bool:
    """
    Checks the stability of a model based on cross-validation fold scores.
    """
    max_score = max(folds_scores)
    min_score = min(folds_scores)
    difference = max_score - min_score

    print(f"Fold scores: {folds_scores}, max-min difference: {difference:.3f}")

    return difference <= threshold


def check_overfitting(train_r2: float, test_r2: float, threshold: float = 0.2) -> bool:
    """
    Checks whether a model is likely overfitting based on the difference between
    training and test R² scores.
    """
    difference = train_r2 - test_r2

    print(
        f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}, Difference: {difference:.3f}"
    )

    return difference > threshold


def check_model_results(
    model: type,
    metrics: dict,
    folds_scores: Sequence[float],
    fold_threshold=0.1,
    overfit_threshold=0.2,
):
    """
    Checks model stability and potential overfitting.
    """
    stable = True
    if folds_scores is not None and len(folds_scores) > 0:
        stable = check_fold_stability(folds_scores, threshold=fold_threshold)
    if not stable:
        print(f"{model.__name__}: fold results indicate potential instability")

    overfit = check_overfitting(
        metrics["train_r2"], metrics["test_r2"], threshold=overfit_threshold
    )
    if overfit:
        print(
            f"{model.__name__} may be overfitting. Consider adjusting hyperparameters."
        )
