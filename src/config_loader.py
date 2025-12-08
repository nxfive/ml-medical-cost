from omegaconf import DictConfig

from src.conf.schema import (CVConfig, DataDir, DataStageConfig,
                             FeaturesConfig, KaggleConfig, ModelConfig,
                             TrainingDir, TrainingStageConfig,
                             TransformersConfig)


def load_stage_configs(cfg: DictConfig) -> tuple[DataStageConfig, TrainingStageConfig]:
    """
    Converts a DictConfig into typed configurations for data and training stages.
    """    
    cv_cfg = CVConfig.from_omegaconf(cfg.cv)
    features_cfg = FeaturesConfig.from_omegaconf(cfg.features)
    model_cfg = ModelConfig.from_omegaconf(cfg.model)
    transform_cfg = TransformersConfig.from_omegaconf(cfg.transform)

    data_dir = DataDir(**cfg.data)
    training_dir = TrainingDir(**cfg.training)
    kaggle_cfg = KaggleConfig(**cfg.kaggle)

    training_stage_cfg = TrainingStageConfig(
        data_dir=data_dir,
        training_dir=training_dir,
        cv=cv_cfg,
        features=features_cfg,
        model=model_cfg,
        transformers=transform_cfg,
    )

    data_stage_cfg = DataStageConfig(data_dir=data_dir, kaggle=kaggle_cfg)

    return data_stage_cfg, training_stage_cfg
