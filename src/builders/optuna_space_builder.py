from typing import Any

from src.params.optuna_grid import OptunaGrid
from src.params.optuna_updater import OptunaParamUpdater
from src.params.validator import ParamValidator


class OptunaSpaceBuilder:
    @staticmethod
    def build(
        optuna_params: dict[str, Any], model_params: dict[str, list]
    ) -> dict[str, Any]:
        """
        Builds an Optuna search space by validating inputs, filling missing parameter values, 
        converting them to Optuna distributions, and applying pipeline namespaces.
        """
        ParamValidator.validate_optuna(optuna_params)
        ParamValidator.validate_grid(model_params)
        params = OptunaParamUpdater.update(
            model_params=model_params, optuna_params=optuna_params
        )

        return OptunaGrid(params).create_optuna_space()
