from typing import Any

LARGE_RANGE_THRESHOLD = 100
DEFAULT_LARGE_RANGE_STEP = 5
DEFAULT_SMALL_RANGE_STEP = 1


class OptunaParamUpdater:
    @staticmethod
    def update(
        model_params: dict[str, Any],
        optuna_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Updates model parameters with optimized values from Optuna results.

        - For missing numeric parameters: creates min/max/step
        - For missing categorical parameters: creates choices
        """
        optuna_params = optuna_params or {}
        missing_params = {}

        for key, values in model_params.items():
            if key in optuna_params:
                continue

            if not isinstance(values, list) or not values:
                raise ValueError(f"Invalid parameter grid for '{key}': {values}")

            first_value = values[0]

            if isinstance(first_value, (bool, str)):
                missing_params[key] = {"choices": values}

            elif isinstance(first_value, (int, float)):
                min_val, max_val = min(values), max(values)
                step = (
                    DEFAULT_LARGE_RANGE_STEP
                    if (max_val - min_val) > LARGE_RANGE_THRESHOLD
                    else DEFAULT_SMALL_RANGE_STEP
                )
                missing_params[key] = {
                    "min": min_val,
                    "max": max_val,
                    "step": step,
                }

            else:
                raise ValueError(
                    f"Unsupported parameter type for '{key}': {type(first_value)}"
                )

        return {**optuna_params, **missing_params}
