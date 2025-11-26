import pandas as pd
from typing import Any

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.utils import (
    get_metrics,
    update_params_with_optuna,
    save_model_with_metadata,
)
from src.models.mlflow_logging import log_model
from src.models.optuna_tuning import optimize_model


MODELS = [
    LinearRegression,
    KNeighborsRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
]


def optuna_pipeline(
    best_model_info: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Optimizes hyperparameters for the best-performing model using Optuna, retrains it,
    and logs the final version to MLflow.
    """
    if (
        optuna_results := optimize_model(best_model_info["model"], X_train, y_train)
    ) is not None:
        study, best_params = optuna_results

        updated_params = update_params_with_optuna(
            best_model_info["param_grid"], optuna_params=best_params
        )

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
            transformer_name=best_model_info["transformer"],
        )

        save_model_with_metadata(
            best_estimator, best_model_info["model"].__name__, metrics, updated_params
        )
