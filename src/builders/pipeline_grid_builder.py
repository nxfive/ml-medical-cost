from src.params.grid import ParamGridPrefixer
from src.params.validator import ParamValidator


class PipelineGridBuilder:
    @staticmethod
    def build(model_params):
        """
        Builds a validated parameter grid for pipeline models, applying necessary prefixes.
        """
        ParamValidator.validate_grid(model_params)
        return ParamGridPrefixer().prepare_pipeline_grid(model_params)
