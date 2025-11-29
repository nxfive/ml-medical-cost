from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

import optuna
from sklearn.base import BaseEstimator
from src.models.models import create_model_pipeline
from src.utils.cv import get_cv


def _build_trial_params(trial: optuna.Trial, optuna_params: dict) -> dict[str, Any]:
    """
    Convert Optuna parameter definitions into trial parameters.
    """
    trial_params: dict[str, Any] = {}

    for param_name, param_config in optuna_params.items():
        if isinstance(param_config, list):
            trial_params[param_name] = trial.suggest_categorical(param_name, param_config)
            continue

        if isinstance(param_config, dict):
            if "choices" in param_config:
                trial_params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif all(k in param_config for k in ("min", "max")):
                step = param_config.get("step")
                if isinstance(param_config["min"], int) and isinstance(param_config["max"], int):
                    trial_params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        step=step or 1,
                    )
                else:
                    trial_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["min"],
                        param_config["max"],
                        step=step or 1.0,
                    )
            else:
                raise ValueError(f"Invalid param format for '{param_name}': {param_config}")
        else:
            raise ValueError(f"Unexpected param type for '{param_name}': {type(param_config)}")

    return trial_params


def _instantiate_model(model_class: type, trial_params: dict[str, Any]) -> BaseEstimator:
    """
    Instantiate the model with trial parameters, using a fixed random_state 
    if the model supports a `random_state` parameter.
    """
    if "random_state" in model_class.__init__.__code__.co_varnames:
        return model_class(**trial_params, random_state=42)
    return model_class(**trial_params)


def _evaluate_pipeline(
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    trial: optuna.Trial
) -> float:
    """
    Fit the pipeline on CV folds, report intermediate metrics to Optuna, and return mean R2.
    """
    r2_scores = []

    for step, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_train_cv, y_train_cv)
        r2_score = pipeline.score(X_val_cv, y_val_cv)
        r2_scores.append(r2_score)
        trial.report(r2_score, step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(r2_scores)


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: type,
    optuna_params: dict,
    cfg: DictConfig
) -> float:
    """
    Generic Optuna objective function for scikit-learn models.
    """
    trial_params = _build_trial_params(trial, optuna_params)
    model_instance = _instantiate_model(model, trial_params)
    pipeline = create_model_pipeline(cfg, model_instance=model_instance)
    cv = get_cv(cfg)
    return _evaluate_pipeline(pipeline, X_train, y_train, cv, trial)


def create_study(study_name: str = "MedicalRegressor") -> optuna.Study:
    """
    Creates and returns an Optuna study with a MedianPruner and maximize direction.
    """
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )


def run_study(
    study: optuna.Study, objective_fn, n_trials: int) -> dict[str, Any]:
    """
    Run the Optuna study with the given objective function and number of trials.
    """
    study.optimize(objective_fn, n_trials=n_trials)
    return study.best_params


def optimize_model(
    model: type, X_train: pd.DataFrame, y_train: pd.Series, cfg: DictConfig
) -> tuple[optuna.Study, dict] | None:
    """
    Performs hyperparameter optimization for a given model using Optuna.
    """
    params_node = cfg.optuna.get(cfg.model.name, {})

    if not params_node:
        print(f"No Optuna parameters defined for {model.__name__}, skipping optimization.")
        return None

    optuna_params = {k: dict(v) for k, v in params_node["params"].items()}

    study = create_study()
    best_params = run_study(
        study,
        lambda trial: objective(trial, X_train, y_train, model, optuna_params, cfg),
        n_trials=cfg.optuna.trials
    )

    best_params = study.best_params
    return study, best_params
