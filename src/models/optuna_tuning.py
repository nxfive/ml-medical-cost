import numpy as np
import pandas as pd
import optuna
from typing import Any
from sklearn.model_selection import KFold

from src.models.models import create_model_pipeline
from src.models.settings import OptunaConfig
from src.models.utils import get_cv
from src.utils.paths import OPTUNA_CONFIG


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: type,
    optuna_params: dict[str, Any],
    cv: KFold,
) -> float:
    """
    Generic Optuna objective for any scikit-learn model.
    Supports int, float, and categorical parameters in YAML.
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

    if "random_state" in model.__init__.__code__.co_varnames:
        model_instance = model(**trial_params, random_state=42)
    else:
        model_instance = model(**trial_params)

    pipeline = create_model_pipeline(model_instance=model_instance)

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


def optimize_model(
    model: type, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[optuna.Study, dict] | None:
    """
    Performs hyperparameter optimization for a given model using Optuna.
    """
    optuna_config = OptunaConfig(OPTUNA_CONFIG, model)

    if not optuna_config.params:
        print(f"No Optuna parameters defined for {model.__name__}, skipping optimization.")
        return None

    cv = get_cv()

    study = optuna.create_study(
        study_name="MedicalRegressor",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, model, optuna_config.params, cv
        ),
        n_trials=optuna_config.trials,
    )

    best_params = study.best_params
    return study, best_params
