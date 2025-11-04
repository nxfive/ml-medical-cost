from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import os
import yaml

from src.models.config_defaults import (MODEL_DEFAULT_CONFIG,
                                        OPTUNA_DEFAULT_CONFIG)
from src.utils.paths import MODEL_CONFIG_FILE


class BaseConfig(ABC):

    def _merge_with_defaults(self, base: dict, extra: dict) -> dict:
        merged = {}
        for key, base_value in base.items():
            if key in extra:
                extra_value = extra[key]
                if isinstance(base_value, dict) and isinstance(extra_value, dict):
                    merged[key] = self._merge_with_defaults(base_value, extra_value)
                else:
                    merged[key] = extra_value
            else:
                merged[key] = base_value
        return merged

    def _load_yaml(self, yaml_path: str) -> dict:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (FileNotFoundError, yaml.YAMLError, OSError):
            return {}


class PipelineConfig(BaseConfig):

    @dataclass(frozen=True)
    class _FeaturesConfig:
        numeric: list[str] = field(default_factory=list)
        categorical: list[str] = field(default_factory=list)
        binary: list[str] = field(default_factory=list)

        @classmethod
        def from_dict(cls, cfg: dict[str, Any]):
            return cls(
                numeric=cfg.get("num_features", []),
                categorical=cfg.get("cat_features", []),
                binary=cfg.get("bin_features", []),
            )

    @dataclass(frozen=True)
    class _ModelConfig:
        preprocess_num_features: bool = True
        target_transformations: bool = False
        params: dict[str, Any] = field(default_factory=dict)

        @classmethod
        def from_dict(cls, cfg: dict[str, Any]):
            return cls(**cfg)

    def __init__(self, yaml_path: str):
        default_cfg = MODEL_DEFAULT_CONFIG
        loaded_cfg = self._load_yaml(yaml_path)
        cfg = (
            self._merge_with_defaults(default_cfg, loaded_cfg)
            if loaded_cfg
            else default_cfg
        )

        self.features = self._FeaturesConfig.from_dict(cfg)
        self.models = {
            name: self._ModelConfig.from_dict(cfg["models"][name])
            for name in cfg["models"]
        }
        self.cv = cfg["cv"]
        self.transformations = cfg["transformations"]


class OptunaConfig(BaseConfig):
    def __init__(self, yaml_path: str, model: type):
        default_cfg = OPTUNA_DEFAULT_CONFIG
        loaded_cfg = self._load_yaml(yaml_path)
        cfg = (
            self._merge_with_defaults(default_cfg, loaded_cfg)
            if loaded_cfg
            else default_cfg
        )

        self.model = model
        self.params = cfg["models"].get(model.__name__, {})
        self.trials = cfg["optuna"]["trials"]


pipeline_config = PipelineConfig(MODEL_CONFIG_FILE)
