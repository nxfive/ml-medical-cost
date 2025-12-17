from src.params.grid import ParamGridPrefixer
from src.params.validator import ParamValidator


class WrapperGridBuilder:
    @staticmethod
    def build(
        param_grid: dict[str, list], transformer_params: dict[str, list] | None
    ) -> dict[str, list]:
        """
        Builds a validated parameter grid for wrapper estimators, applying
        namespaces for model and transformer parameters.
        """
        if transformer_params:
            ParamValidator.validate_grid(transformer_params)

        return ParamGridPrefixer.prepare_wrapper_grid(
            param_grid=param_grid, transformer_params=transformer_params
        )
