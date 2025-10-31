import os
import yaml
from abc import ABC
from typing import Any

from src.utils.paths import MODEL_CONFIG_FILE
from src.models.config_defaults import MODEL_DEFAULT_CONFIG, OPTUNA_DEFAULT_CONFIG


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
        if not os.path.exists(yaml_path):
            return {}
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except (yaml.YAMLError, OSError):
            return {}


class FeaturesConfig:
    def __init__(self, config: dict[str, Any]):
        self.numeric = config.get("num_features", [])
        self.categorical = config.get("cat_features", [])
        self.binary = config.get("bin_features", [])


class ModelConfig:
    def __init__(self, config: dict[str, Any]):
        self.preprocess_num_features = config.get("preprocess_num_features", True)
        self.target_transformation = config.get("target_transformations", False)
        self.params = config.get("params", {})


class PipelineConfig(BaseConfig):
    def __init__(self, yaml_path: str):
        default_cfg = MODEL_DEFAULT_CONFIG
        loaded_cfg = self._load_yaml(yaml_path)
        cfg = self._merge_with_defaults(default_cfg, loaded_cfg) if loaded_cfg else default_cfg

        self.features = FeaturesConfig(cfg)
        self.models = {name: ModelConfig(cfg["models"][name]) for name in cfg["models"]}
        self.cv = cfg["cv"]
        self.transformations = cfg["transformations"]


class OptunaConfig(BaseConfig):
    def __init__(self, yaml_path: str, model: type):
        default_cfg = OPTUNA_DEFAULT_CONFIG
        loaded_cfg = self._load_yaml(yaml_path)
        cfg = self._merge_with_defaults(default_cfg, loaded_cfg) if loaded_cfg else default_cfg

        self.model = model
        self.params = cfg["models"][model.__name__]
        self.trials = cfg["optuna"]["trials"]


pipeline_config = PipelineConfig(MODEL_CONFIG_FILE)
