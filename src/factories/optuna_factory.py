from omegaconf import DictConfig
from sklearn.base import BaseEstimator

from src.conf.schema import (CVConfig, DataDir, FeaturesConfig, ModelConfig,
                             ModelsDir, OptunaConfig, OptunaModelConfig,
                             OptunaStageConfig, TrainingDir,
                             TransformersConfig)
from src.optuna.types import DynamicConfig


class OptunaConfigFactory:
    """
    Factory that constructs dataclasses while dynamically overriding parts of the configuration.
    As a result, it depends on an OmegaConf/Hydra DictConfig.

    This is an exception to the standard flow, used specifically for Optuna stage where the
    configuration must be updated at runtime based on the selected best model.
    """

    @staticmethod
    def create(
        cfg: DictConfig, dynamic_cfg: DynamicConfig, model_class: type[BaseEstimator]
    ) -> OptunaStageConfig:
        cv_cfg = CVConfig.from_omegaconf(cfg.cv)
        features_cfg = FeaturesConfig.from_omegaconf(cfg.features)
        optuna_cfg = OptunaConfig.from_omegaconf(cfg.optuna.study)
        transform_cfg = TransformersConfig.from_omegaconf(cfg.transform)

        model_cfg = ModelConfig.from_omegaconf(dynamic_cfg.model)
        optuna_model_cfg = OptunaModelConfig.from_omegaconf(
            dynamic_cfg.optuna_model.model
        )

        model_cfg.model_class = model_class

        return OptunaStageConfig(
            data_dir=DataDir(**cfg.data),
            training_dir=TrainingDir(**cfg.training),
            models_dir=ModelsDir(**cfg.models),
            model=model_cfg,
            optuna_config=optuna_cfg,
            optuna_model_config=optuna_model_cfg,
            features=features_cfg,
            cv=cv_cfg,
            transformers=transform_cfg,
        )
