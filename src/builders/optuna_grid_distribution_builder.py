from typing import Any

from optuna.distributions import BaseDistribution
from src.params.grid import ParamGridPrefixer

from .optuna_space_builder import OptunaSpaceBuilder


class OptunaGridDistributionBuilder:
    @staticmethod
    def build(
        optuna_params: dict[str, Any], model_params: dict[str, list]
    ) -> dict[str, BaseDistribution]:
        """
        Builds an Optuna parameter space by creating distributions and
        applying pipeline namespaces.
        """
        space = OptunaSpaceBuilder.build(optuna_params, model_params)
        return ParamGridPrefixer().prepare_pipeline_grid(space)
