import inspect
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.conf.schema import FeaturesConfig, StageConfig
from src.models.registry import get_model_class_and_short
from src.models.transformers import get_transformer


class PreprocessorBuilder:
    @staticmethod
    def build(preprocess_num_features: bool, cfg: FeaturesConfig) -> ColumnTransformer:
        """
        Builds a ColumnTransformer for feature preprocessing.
        """
        if preprocess_num_features:
            return ColumnTransformer(
                [
                    ("num", StandardScaler(), cfg.numeric),
                    ("bin", "passthrough", cfg.binary),
                    (
                        "cat",
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
                        cfg.categorical,
                    ),
                ]
            )
        else:
            return ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        cfg.categorical,
                    ),
                    (
                        "rest",
                        "passthrough",
                        cfg.numeric + cfg.binary,
                    ),
                ]
            )


class ModelPipelineBuilder:
    @staticmethod
    def build(
        preprocessor: ColumnTransformer,
        model: type[BaseEstimator],
        params: dict[str, Any] | None = None,
    ) -> Pipeline:
        """
        Builds a pipeline consisting of a preprocessor and a model.
        """
        params = params or {}

        signature = inspect.signature(model.__init__)
        if "random_state" in signature.parameters:
            params.setdefault("random_state", 42)

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model(**params)),
            ]
        )


class TransformerBuilder:
    @staticmethod
    def build(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
        """
        Builds a target transformer instance.
        """
        transformer = get_transformer(name=name)

        if params:
            transformer.set_params(**params)

        return transformer


class TransformerPipelineBuilder:
    @staticmethod
    def build(
        pipeline: Pipeline, transformer: BaseEstimator
    ) -> TransformedTargetRegressor:
        """
        Wraps a pipeline in a TransformedTargetRegressor using the provided transformer.
        """
        return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


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
