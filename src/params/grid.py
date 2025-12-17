from typing import Any

from .types import Prefixes


class ParamGridPrefixer:
    @staticmethod
    def prefix(params: dict[str, Any], step: Prefixes) -> dict[str, Any]:
        """
        Adds the given step prefix to all parameter keys.
        """
        return {f"{step.value}{k}": v for k, v in params.items()}

    def prepare_pipeline_grid(self, model_params: dict[str, Any]) -> dict[str, Any]:
        """
        Prefixes model parameters so they can be used in a pipeline step.
        """
        return self.prefix(model_params, Prefixes.PIPELINE_MODEL)

    def prepare_wrapper_grid(
        self,
        model_params: dict[str, Any] | None,
        transformer_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Merges model and transformer parameters with appropriate prefixes for a wrapper.
        """
        grid = {}
        if model_params:
            grid |= self.prefix(model_params, Prefixes.WRAPPER_REGRESSOR)
        if transformer_params:
            grid |= self.prefix(transformer_params, Prefixes.WRAPPER_TRANSFORMER)
        return grid
