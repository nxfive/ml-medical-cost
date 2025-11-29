from omegaconf import DictConfig

from src.mlflow.logging import log_model, setup_mlflow
from src.training.train import run_training
from src.utils.loading import load_splitted_data
from src.utils.metrics import get_metrics
from src.utils.saving import save_run
from src.utils.selection import get_model_class_and_short
from src.utils.validation import check_model_results


def run(cfg: DictConfig) -> None:
    """
    Trains and evaluates all defined models using cross-validation, logs results to MLflow
    and saves pipeline and training results to disk.
    """
    setup_mlflow()

    X_train, X_test, y_train, y_test = load_splitted_data(cfg)

    model_class, _ = get_model_class_and_short(cfg.model.name)

    for pipeline_results, param_grid, transformer_name in run_training(
        model_class, X_train, X_test, y_train, cfg
    ):
        estimator, fold_scores, folds_scores_mean, y_train_pred, y_test_pred = (
            pipeline_results
        )

        metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)
        check_model_results(model_class, metrics, fold_scores)

        log_model(
            estimator=estimator,
            param_grid=param_grid,
            X_train=X_train,
            model=model_class,
            metrics=metrics,
            folds_scores=fold_scores,
            folds_scores_mean=folds_scores_mean,
            study=None,
            transformer_name=transformer_name,
        )
        save_run(
            {
                "model": model_class.__name__,
                "param_grid": param_grid,
                "transformer": transformer_name,
                "folds_scores_mean": float(folds_scores_mean),
                "metrics": metrics,
            },
            estimator,
            cfg,
        )
