from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.training.cv import get_cv
from src.training.train import TrainModel
from src.tuning.runners import CrossValidationRunner, GridSearchRunner
from src.tuning.transformers import TargetTransformer

from .pipeline_builder import PipelineBuilder
from .pipeline_grid_builder import PipelineGridBuilder


class TrainingBuilder:
    @staticmethod
    def build(model_class: type[BaseEstimator], cfg: TrainingStageConfig) -> TrainModel:
        """
        Builds and configures a TrainModel instance with the given model class and config.
        """
        cv = get_cv(cfg.cv)

        pipeline = PipelineBuilder.build(
            model_cfg=cfg.model,
            features_cfg=cfg.features,
        )
        param_grid = PipelineGridBuilder.build(model_params=cfg.model.params)

        grid_runner = GridSearchRunner(cv=cv)
        cross_runner = CrossValidationRunner(cv=cv)

        target_transformer = TargetTransformer(
            cfg_transform=cfg.transformers,
        )

        return TrainModel(
            model=model_class,
            cfg_model=cfg.model,
            param_grid=param_grid,
            pipeline=pipeline,
            grid_runner=grid_runner,
            cross_runner=cross_runner,
            target_transformer=target_transformer,
        )
