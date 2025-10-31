import os
import yaml
from typing import Any

from src.utils.paths import MODEL_CONFIG_FILE
from src.models.config_defaults import MODEL_DEFAULT_CONFIG


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


class PipelineConfig:
    def __init__(self, yaml_path: str):
        default_cfg = MODEL_DEFAULT_CONFIG
        loaded_cfg = {}
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    loaded_cfg = yaml.safe_load(f) or {}
            except (yaml.YAMLError, OSError):
                pass

        cfg = self._merge_with_defaults(default_cfg, loaded_cfg)

        self.features = FeaturesConfig(cfg)
        self.models = {name: ModelConfig(cfg["models"][name]) for name in cfg["models"]}
        self.cv = cfg.get("cv", default_cfg["cv"])
        self.transformations = cfg.get(
            "transformations", default_cfg["transformations"]
        )

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


class OptunaConfig:
    def __init__(self, yaml_path: str, model: type):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self.model = model
        self.params = cfg.get(model.__name__, None)
        self.trials = cfg["optuna"].get("trials", 50)


pipeline_config = PipelineConfig(MODEL_CONFIG_FILE)
