from typing import TypedDict
from omegaconf import DictConfig

from src.conf.schema import DataStageConfig, TrainingStageConfig


class StageConfigMap(TypedDict):
    data: DataStageConfig
    training: TrainingStageConfig
    optuna: DictConfig
