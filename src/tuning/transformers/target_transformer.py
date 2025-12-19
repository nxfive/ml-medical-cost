from typing import Generator

from sklearn.pipeline import Pipeline

from src.builders.transformer.transformer_wrapper_builder import \
    TransformerWrapperBuilder
from src.builders.transformer.wrapper_grid_builder import WrapperGridBuilder
from src.conf.schema import TransformersConfig
from src.containers.results import EvaluationResult
from src.factories.transformer_factory import TargetTransformerFactory

from .registry import TRANSFORMERS


class TargetTransformer:
    def __init__(self, cfg_transform: TransformersConfig):
        self.cfg_transform = cfg_transform

    def evaluate(
        self, pipeline: Pipeline, param_grid: dict[str, list]
    ) -> Generator[EvaluationResult, None, None]:
        """
        Generates pipelines with target transformations and updated param grid.
        """
        for transformation in self.cfg_transform.to_dict():

            if TRANSFORMERS[transformation].is_identity:
                yield EvaluationResult(estimator=pipeline, param_grid=param_grid)

            else:
                transformer = TargetTransformerFactory.create(
                    transformation=transformation
                )
                wrapper = TransformerWrapperBuilder.build(
                    pipeline=pipeline, transformer=transformer
                )
                wrapper_grid = WrapperGridBuilder.build(
                    param_grid=param_grid, transformer_params=None
                )

                yield EvaluationResult(
                    estimator=wrapper,
                    param_grid=wrapper_grid,
                    transformation=transformation,
                )
