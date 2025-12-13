from dataclasses import dataclass
from typing import Any

import pandas as pd
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

import optuna
from optuna.pruners import BasePruner
from src.conf.schema import (CVConfig, FeaturesConfig, ModelConfig,
                             OptunaConfig, OptunaModelConfig,
                             TransformersConfig)
from src.models.types import ModelRun
from src.patterns.types import BuildResult


@dataclass
class LoadedModelResults:
    runs: dict[str, ModelRun]


@dataclass
class OptunaResult:
    model_class: type[BaseEstimator]
    best_estimator: BaseEstimator
    best_params: dict[str, list]
    transformation: str
    metrics: dict[str, float]
    study: optuna.Study


@dataclass
class DynamicConfig:
    model: DictConfig
    optuna_model: DictConfig
    patient: DictConfig | None


@dataclass
class OptunaBuildResult(BuildResult):
    pruner: BasePruner


@dataclass
class OptunaExperimentConfig:
    model: ModelConfig
    features: FeaturesConfig
    transformers: TransformersConfig
    optuna_model_config: OptunaModelConfig


@dataclass
class ExperimentContext:
    cv_cfg: CVConfig
    model_cfg: ModelConfig
    features_cfg: FeaturesConfig
    optuna_cfg: OptunaConfig
    optuna_model_cfg: OptunaModelConfig
    transformers_cfg: TransformersConfig
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series

    def to_experiment_config(self) -> OptunaExperimentConfig:
        return OptunaExperimentConfig(
            model=self.model_cfg,
            features=self.features_cfg,
            transformers=self.transformers_cfg,
            optuna_model_config=self.optuna_model_cfg,
        )


@dataclass
class ExperimentSetup:
    pipeline: BaseEstimator
    params: dict[str, Any]
