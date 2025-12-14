from sklearn.base import BaseEstimator

from src.conf.schema import FeaturesConfig, ModelConfig
from src.factories.model_factory import ModelFactory
from src.factories.transformer_factory import TargetTransformerFactory

from .model_pipeline_builder import ModelPipelineBuilder
from .preprocessor_builder import PreprocessorBuilder
from .transformer_wrapper_builder import TransformerWrapperBuilder


class PipelineBuilder:
    @staticmethod
    def build(
        model_cfg: ModelConfig,
        features_cfg: FeaturesConfig,
        transformation: str = "none",
    ) -> BaseEstimator:
        """
        Builds a complete pipeline with optional target transformation for a stage.
        """
        model_spec = ModelFactory.get_spec(name_or_alias=model_cfg.name)

        preprocessor = PreprocessorBuilder.build(
            preprocess_num_features=model_cfg.preprocess_num_features,
            cfg=features_cfg,
        )

        pipeline = ModelPipelineBuilder.build(
            preprocessor=preprocessor,
            model=model_spec.model_class,
        )

        if not model_cfg.target_transformations:
            return pipeline

        transformer = TargetTransformerFactory.create(
            transformation=transformation,
        )

        return TransformerWrapperBuilder.build(
            pipeline=pipeline, transformer=transformer
        )
