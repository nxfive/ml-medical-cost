from typing import Any

from sklearn.base import BaseEstimator

from src.conf.schema import StageConfig
from src.models.registry import get_model_class_and_short

from .preprocessor_builder import PreprocessorBuilder
from .model_pipeline_builder import ModelPipelineBuilder
from .transformer_builder import TransformerBuilder
from .transformer_pipeline_builder import TransformerPipelineBuilder


class PipelineBuilder:
    @staticmethod
    def build(
        stage_cfg: StageConfig,
        model_params: dict[str, Any] | None = None,
        transformer_params: dict[str, Any] | None = None,
        transformation: str = "none",
    ) -> BaseEstimator:
        """
        Builds a complete pipeline with optional target transformation for a stage.
        """
        model_class, _ = get_model_class_and_short(name=stage_cfg.model.name)

        preprocessor = PreprocessorBuilder.build(
            preprocess_num_features=stage_cfg.model.preprocess_num_features,
            cfg=stage_cfg.features,
        )

        pipeline = ModelPipelineBuilder.build(
            preprocessor=preprocessor,
            model=model_class,
            params=model_params,
        )

        if not stage_cfg.model.target_transformations or transformation == "none":
            return pipeline

        transformer = TransformerBuilder.build(
            name=transformation, params=transformer_params
        )

        return TransformerPipelineBuilder.build(
            pipeline=pipeline, transformer=transformer
        )
