from typing import Generator

from sklearn.pipeline import Pipeline

from src.builders import TransformerBuilder, TransformerPipelineBuilder
from src.conf.schema import TransformersConfig
from src.params.grid import ParamGrid
from src.tuning.types import EvaluationResult


class TargetTransformer:
    def __init__(self, cfg_transform: TransformersConfig):
        self.cfg_transform = cfg_transform

    @staticmethod
    def prepare_param_grid(
        param_grid: dict[str, list], params: dict[str, list]
    ) -> dict[str, list]:
        """
        Merges base param grid with transformer parameters.
        """
        grid_copy = param_grid.copy() if param_grid else {}
        grid_copy = ParamGrid.prefix(grid_copy, "regressor")
        if params:
            grid_copy.update(ParamGrid.prefix(params, "transformer"))
        return grid_copy

    def evaluate(
        self, pipeline: Pipeline, param_grid: dict[str, list]
    ) -> Generator[EvaluationResult, None, None]:
        """
        Generates pipelines with target transformations and updated param grid.
        """
        for transformation, value in self.cfg_transform.to_dict().items():

            transformer = TransformerBuilder.build(name=transformation)
            estimator = TransformerPipelineBuilder.build(pipeline, transformer)

            if transformer is not None:
                params = value.params
                local_param_grid = self.prepare_param_grid(param_grid, params)
            else:
                local_param_grid = param_grid

            yield EvaluationResult(
                estimator=estimator,
                param_grid=local_param_grid,
                transformation=transformation,
            )
