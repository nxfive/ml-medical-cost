from sklearn.base import BaseEstimator

from src.conf.schema import TrainingStageConfig
from src.models.pipeline import create_model_pipeline
from src.training.train import TrainModel
from src.training.tuning import (CrossValidationRunner, GridSearchRunner,
                                 TargetTransformer)
from src.utils.cv import get_cv
from src.utils.grid import ParamGrid


class TrainingFactory:
    @staticmethod
    def create(model_class: type[BaseEstimator], training_cfg: TrainingStageConfig):
        cv = get_cv(training_cfg.cv)

        pipeline = create_model_pipeline(
            training_cfg.model.preprocess_num_features,
            training_cfg.features,
            model=model_class,
        )

        param_grid = ParamGrid.create(training_cfg.model.params)

        grid_runner = GridSearchRunner(cv=cv, pipeline=pipeline, param_grid=param_grid)
        cross_runner = CrossValidationRunner(cv=cv, pipeline=pipeline)

        target_transformer = TargetTransformer(
            pipeline=pipeline,
            param_grid=param_grid,
            training_cfg_transform=training_cfg.transformers,
        )

        return TrainModel(
            model=model_class,
            cfg_model=training_cfg.model,
            cfg_features=training_cfg.features,
            cfg_cv=training_cfg.cv,
            cfg_transform=training_cfg.transformers,
            grid_runner=grid_runner,
            cross_runner=cross_runner,
            target_transformer=target_transformer,
        )
