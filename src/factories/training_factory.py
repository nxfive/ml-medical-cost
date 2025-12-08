from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.models.models import create_model_pipeline
from src.params.grid import ParamGrid
from src.training.cv import get_cv
from src.training.train import TrainModel
from src.training.tuning import (CrossValidationRunner, GridSearchRunner,
                                 TargetTransformer)


class TrainingFactory:
    @staticmethod
    def create(
        model_class: type[BaseEstimator], cfg: TrainingStageConfig
    ) -> TrainModel:
        """
        Creates and configures a TrainModel instance with the given model class and config.
        """
        cv = get_cv(cfg.cv)

        pipeline = create_model_pipeline(
            preprocess_num_features=cfg.model.preprocess_num_features,
            features_cfg=cfg.features,
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
