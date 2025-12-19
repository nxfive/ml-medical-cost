from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from src.conf.schema import (FeaturesConfig, ModelConfig, OptunaModelConfig,
                             TransformersConfig)


@dataclass
class ExperimentSetup:
    pipeline: BaseEstimator
    params: dict[str, Any]


@dataclass
class ExperimentContext:
    model_cfg: ModelConfig
    features_cfg: FeaturesConfig
    optuna_model_cfg: OptunaModelConfig
    transformers_cfg: TransformersConfig
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
