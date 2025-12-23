from typing import Any


class ParamValidator:
    @staticmethod
    def validate_optuna(params_config: dict[str, Any]) -> None:
        """
        Validates Optuna parameter configuration:

        - Numeric parameters: must have 'min' and 'max', optional 'step'
        - Categorical parameters: must have 'choices' list or can be plain list
        """
        for name, param in params_config.items():
            if isinstance(param, list):
                if not param:
                    raise ValueError(f"{name}: list cannot be empty")
                continue

            if "choices" in param:
                values = param["choices"]
                if not isinstance(values, list) or not values:
                    raise ValueError(f"{name}: 'choices' must be a non-empty list")

                if "min" in param or "max" in param:
                    raise ValueError(
                        f"{name}: cannot have both 'choices' and 'min/max'"
                    )
            else:
                if "min" not in param or "max" not in param:
                    raise ValueError(f"{name}: if using min/max, both must be defined")

                step = param.get("step")
                if step is not None and not isinstance(step, (int, float)):
                    raise TypeError(f"{name}: 'step' must be a number")

    @staticmethod
    def validate_grid(param_grid: dict[str, list]) -> None:
        """
        Validates parameter grid for GridSearch:

        - Each parameter must be a non-empty list of values.
        """
        for name, values in param_grid.items():
            if not isinstance(values, list):
                raise TypeError(f"{name}: values must be a list")
            if not values:
                raise ValueError(f"{name}: values list cannot be empty")
