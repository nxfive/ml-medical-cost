from dataclasses import dataclass

from omegaconf import DictConfig
from sklearn.base import BaseEstimator

import optuna
from src.models.types import ModelRun


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
