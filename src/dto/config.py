from dataclasses import dataclass
from typing import TypedDict

from omegaconf import DictConfig

from src.conf.schema import (DataStageConfig, FeaturesConfig, ModelConfig,
                             OptunaModelConfig, TrainingStageConfig,
                             TransformersConfig)


@dataclass
class DynamicConfig:
    model: DictConfig
    optuna_model: DictConfig
    patient: DictConfig | None


@dataclass
class OptunaExperimentConfig:
    model: ModelConfig
    features: FeaturesConfig
    transformers: TransformersConfig
    optuna_model_config: OptunaModelConfig


class StageConfigMap(TypedDict):
    data: DataStageConfig
    training: TrainingStageConfig
    optuna: DictConfig
