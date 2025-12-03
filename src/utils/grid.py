from typing import Any


class ParamGrid:
    @staticmethod
    def create(params: dict) -> dict:
        """
        Creates param gird for model pipeline.
        """
        if not params:
            return {}
        return ParamGrid.prefix(params, "model")

    @staticmethod
    def prefix(params: dict, step_name: str) -> dict:
        """
        Adds prefix to params for pipeline compatibility.
        """
        step_name = step_name.strip().strip("_")
        return {f"{step_name}__{k}": v for k, v in params.items()}


class OptunaParamUpdater:
    @staticmethod
    def update(
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
                updated_params[key] = values[0] if isinstance(values, list) else values
        return updated_params
