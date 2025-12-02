from dataclasses import dataclass, field
from typing import Any


@dataclass
class CVConfig:
    folds: int
    shuffle: bool = True
    random_state: int | None = None


@dataclass
class FeaturesConfig:
    features: dict[str, list]


@dataclass
class ModelConfig:
    name: str
    preprocess_num_features: bool
    target_transformations: bool
    params: dict[str, Any]


@dataclass
class OptunaConfig:
    trials: int
    timeout: float | None = None


@dataclass
class SingleOptunaModelConfig:
    params: dict[str, Any]


@dataclass
class OptunaModelsConfig:
    knn: SingleOptunaModelConfig = field(default_factory=SingleOptunaModelConfig)
    linear: SingleOptunaModelConfig = field(default_factory=SingleOptunaModelConfig)
    rf: SingleOptunaModelConfig = field(default_factory=SingleOptunaModelConfig)
    tree: SingleOptunaModelConfig = field(default_factory=SingleOptunaModelConfig)


@dataclass
class SingleTransformerConfig:
    params: dict[str, list]


@dataclass
class TransformersConfig:
    log: SingleTransformerConfig = field(default_factory=SingleTransformerConfig)
    none: SingleTransformerConfig = field(default_factory=SingleTransformerConfig)
    quantile: SingleTransformerConfig = field(default_factory=SingleTransformerConfig)
