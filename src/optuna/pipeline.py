from omegaconf import DictConfig

from src.mlflow.logging import log_model, setup_mlflow
from src.optuna.tuning import optimize_model
from src.utils.grid import update_params_with_optuna
from src.utils.loading import load_splitted_data
from src.utils.metrics import get_metrics
from src.utils.saving import save_model_with_metadata
from src.utils.selection import get_model_class_and_short, pick_best


def run(cfg: DictConfig) -> None:
    """
    Optimizes hyperparameters for the best-performing model using Optuna, retrains it,
    and logs the final version to MLflow.
    """
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_splitted_data(cfg)

    estimator, data = pick_best(cfg.training.output_dir)

    model_class, short = get_model_class_and_short(data["model"])
    cfg.model.name = short

    if (optuna_results := optimize_model(model_class, X_train, y_train, cfg)) is not None:
        study, best_params = optuna_results

        updated_params = update_params_with_optuna(
            data["param_grid"], optuna_params=best_params
        )

        best_estimator = estimator
        best_estimator.set_params(**updated_params)
        best_estimator.fit(X_train, y_train)
        y_test_pred = best_estimator.predict(X_test)
        y_train_pred = best_estimator.predict(X_train)

        metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)

        log_model(
            estimator=best_estimator,
            param_grid=updated_params,
            X_train=X_train,
            model=model_class,
            metrics=metrics,
            folds_scores=None,
            folds_scores_mean=None,
            study=study,
            transformer_name=data["transformer"],
        )

        save_model_with_metadata(
            best_estimator, model_class.__name__, metrics, updated_params, cfg
        )
