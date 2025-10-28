import pandas as pd
from typing import Any

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.train import run_pipeline
from src.models.utils import check_model_results, get_metrics, update_params_with_optuna
from src.models.mlflow_logging import log_model
from src.models.optuna_tuning import optimize_model


MODELS = [
    LinearRegression,
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
]


def models_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """
    Trains and evaluates all defined models using cross-validation and logs results to MLflow.
    """
    models_results = []

    for model in MODELS:
        for pipeline_results, param_grid, transformer_name in run_pipeline(model, X_train, X_test, y_train):
            estimator, fold_scores, folds_scores_mean, y_train_pred, y_test_pred = pipeline_results

            metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)
            check_model_results(model, metrics, fold_scores)

            log_model(
                estimator=estimator,
                param_grid=param_grid,
                X_train=X_train,
                model=model,
                metrics=metrics,
                folds_scores=fold_scores,
                folds_scores_mean=folds_scores_mean,
                study=None,
                transformer_name=transformer_name
            )

            models_results.append({
                "model": model,
                "estimator": estimator,
                "param_grid": param_grid,
                "transformer": transformer_name,
                "folds_scores_mean": folds_scores_mean,
                "metrics": metrics
            })

    return max(models_results, key=lambda x: x["metrics"]["test_r2"])


def optuna_pipeline(best_model_info: dict[str, Any], X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Optimizes hyperparameters for the best-performing model using Optuna, retrains it,
    and logs the final version to MLflow.
    """
    if (optuna_results := optimize_model(best_model_info["model"], X_train, y_train)) is not None:
        study, best_params = optuna_results

        updated_params = update_params_with_optuna(best_model_info["param_grid"], optuna_params=best_params)

        best_estimator = best_model_info["estimator"]
        best_estimator.set_params(**updated_params)
        best_estimator.fit(X_train, y_train)
        y_test_pred = best_estimator.predict(X_test)
        y_train_pred = best_estimator.predict(X_train)

        metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)

        log_model(
                estimator=best_estimator,
                param_grid=updated_params,
                X_train=X_train,
                model=best_model_info["model"],
                metrics=metrics,
                folds_scores=None,
                folds_scores_mean=None,
                study=study,
                transformer_name=best_model_info["transformer"]
            )
