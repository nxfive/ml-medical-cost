from sklearn.model_selection import KFold
from src.models.settings import pipeline_config


def update_param_grid(param_grid: dict, step_name: str) -> dict:
    """
    Prefixes all keys in param_grid with 'step_name__' for pipeline compatibility.
    """
    param_grid = param_grid.copy()
    return {f"{step_name}__{k}": v for k, v in param_grid.items()}


def prepare_grid(model: type) -> dict:
    """
    Prepares param_grid for GridSearchCV with 'model' prefixes.
    """
    param_grid = {}
    model_params = pipeline_config.models[model.__name__].params

    if model_params:
        param_grid = update_param_grid(model_params, "model")

    return param_grid


def get_cv() -> KFold:
    """
    Returns a KFold cross-validator configured based on the values from pipeline_config.
    """
    return KFold(
        n_splits=pipeline_config.cv["n_splits"], 
        shuffle=pipeline_config.cv["shuffle"], 
        random_state=pipeline_config.cv["random_state"]
    )
