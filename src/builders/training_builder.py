from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.params.grid import ParamGrid
from src.training.cv import get_cv
from src.training.train import TrainModel
from src.training.tuning import (CrossValidationRunner, GridSearchRunner,
                                 TargetTransformer)

from .model_pipeline_builder import ModelPipelineBuilder
from .preprocessor_builder import PreprocessorBuilder


class TrainingBuilder:
    @staticmethod
    def build(model_class: type[BaseEstimator], cfg: TrainingStageConfig) -> TrainModel:
        """
        Builds and configures a TrainModel instance with the given model class and config.
        """
        cv = get_cv(cfg.cv)

        preprocessor = PreprocessorBuilder.build(
            preprocess_num_features=cfg.model.preprocess_num_features, cfg=cfg.features
        )

        pipeline = ModelPipelineBuilder.build(
            preprocessor=preprocessor,
            model=model_class,
        )

        param_grid = ParamGrid.create(cfg.model.params)

        grid_runner = GridSearchRunner(cv=cv)
        cross_runner = CrossValidationRunner(cv=cv)

        target_transformer = TargetTransformer(
            cfg_transform=cfg.transformers,
        )

        return TrainModel(
            model=model_class,
            cfg_model=cfg.model,
            cfg_features=cfg.features,
            cfg_cv=cfg.cv,
            cfg_transform=cfg.transformers,
            param_grid=param_grid,
            pipeline=pipeline,
            grid_runner=grid_runner,
            cross_runner=cross_runner,
            target_transformer=target_transformer,
        )
