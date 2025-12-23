from src.params.prefixer import ParamGridPrefixer
from src.params.validator import ParamValidator


class WrapperGridBuilder:
    @staticmethod
    def build(
        param_grid: dict[str, list], transformer_params: dict[str, list] | None
    ) -> dict[str, list]:
        """
        Builds a validated parameter grid for wrapper estimators, applying
        namespaces for model and transformer parameters.

        Transformer parameters are those that can influence the model's
        performance.
        """
        if transformer_params:
            ParamValidator.validate_grid(transformer_params)

        return ParamGridPrefixer().prepare_wrapper_grid(
            model_params=param_grid, transformer_params=transformer_params
        )
