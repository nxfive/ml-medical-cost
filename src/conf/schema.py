from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator

ConfigClass = TypeVar("ConfigClass", bound="ConvertConfig")


@dataclass
class ConvertConfig:
    @classmethod
    def from_omegaconf(cls: type[ConfigClass], cfg: DictConfig) -> ConfigClass:
        data = OmegaConf.to_container(cfg, resolve=True)
        return cls(**data)


@dataclass
class CVConfig(ConvertConfig):
    n_splits: int
    shuffle: bool
    random_state: int | None = None


@dataclass
class FeaturesConfig(ConvertConfig):
    categorical: list[str]
    numeric: list[str]
    binary: list[str]


@dataclass
class ModelConfig(ConvertConfig):
    name: str
    preprocess_num_features: bool
    target_transformations: bool
    params: dict[str, list]
    model_class: type[BaseEstimator] | None = None


@dataclass
class OptunaConfig(ConvertConfig):
    trials: int
    timeout: float | None = None


@dataclass
class OptunaModelConfig(ConvertConfig):
    name: str
    params: dict[str, Any]


@dataclass
class SingleTransformerConfig(ConvertConfig):
    params: dict[str, list]


@dataclass
class TransformersConfig(ConvertConfig):
    log: SingleTransformerConfig
    none: SingleTransformerConfig
    quantile: SingleTransformerConfig

    def to_dict(self) -> dict[str, SingleTransformerConfig]:
        return {
            "log": self.log,
            "none": self.none,
            "quantile": self.quantile,
        }

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> TransformersConfig:
        return cls(
            log=SingleTransformerConfig.from_omegaconf(cfg.log),
            none=SingleTransformerConfig.from_omegaconf(cfg.none),
            quantile=SingleTransformerConfig.from_omegaconf(cfg.quantile),
        )


@dataclass
class DataDir:
    root_dir: Path
    raw_dir: Path
    processed_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.raw_dir = Path(self.raw_dir)
        self.processed_dir = Path(self.processed_dir)


@dataclass
class TrainingDir:
    output_dir: Path
    model_file: str
    metrics_file: str

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class ModelsDir:
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class KaggleConfig:
    handle: str
    filename: str


@dataclass
class DataStageConfig:
    data_dir: DataDir
    kaggle: KaggleConfig


@dataclass
class TrainingStageConfig:
    data_dir: DataDir
    training_dir: TrainingDir
    cv: CVConfig
    features: FeaturesConfig
    model: ModelConfig
    transformers: TransformersConfig


@dataclass
class OptunaStageConfig:
    data_dir: DataDir
    training_dir: TrainingDir
    models_dir: ModelsDir
    model: ModelConfig
    optuna_config: OptunaConfig
    optuna_model_config: OptunaModelConfig
    features: FeaturesConfig
    cv: CVConfig
