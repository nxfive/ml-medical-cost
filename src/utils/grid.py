from typing import Any

from omegaconf import DictConfig


def update_param_grid(param_grid: dict, step_name: str) -> dict:
    """
    Prefixes all keys in param_grid with 'step_name__' for pipeline compatibility.
    """
    param_grid = param_grid.copy()
    step_name = step_name.strip().strip("_")

    if step_name:
        return {f"{step_name}__{k}": v for k, v in param_grid.items()}
    return param_grid


def prepare_grid(cfg: DictConfig) -> dict:
    """
    Prepares param_grid for GridSearchCV with 'model' prefixes.
    """
    param_grid: dict[str, list] = {}
    model_params: dict[str, list] = {}

    if cfg.model.params:
        model_params = {k: list(v) for k, v in cfg.model.params.items()}

    if model_params:
        param_grid = update_param_grid(model_params, "model")

    return param_grid


def update_params_with_optuna(
    model_params: dict[str, Any], optuna_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Updates model parameters with optimized values from Optuna results.
    """
    updated_params = {}
    for key, values in model_params.items():
        for param_name, best_value in optuna_params.items():
            if key.endswith(param_name):
                updated_params[key] = best_value
                break
        else:
            if isinstance(values, list):
                print(
                    f"Parameter '{key}' not found in Optuna results. "
                    f"Using first value from config: {values[0]!r}"
                )
                updated_params[key] = values[0]
            else:
                print(
                    f"Parameter '{key}' not found in Optuna results. "
                    f"Using config value: {values!r}"
                )
                updated_params[key] = values

    return updated_params
