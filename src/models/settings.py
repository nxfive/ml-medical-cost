import yaml

from src.utils.paths import MODEL_CONFIG


class FeaturesConfig:
    def __init__(self, config):
        self.numeric = config.get("num_features", [])
        self.categorical = config.get("cat_features", [])
        self.binary = config.get("bin_features", [])


class ModelConfig:
    def __init__(self, config):
        self.preprocess_num_features = config.get("preprocess_num_features", True)
        self.target_transformation = config.get("target_transformations", False)
        self.params = config.get("params", {})


class PipelineConfig:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        self.features = FeaturesConfig(cfg)
        self.models = {
            name: ModelConfig(cfg[name])
            for name in [
                "LinearRegression",
                "KNeighborsRegressor",
                "DecisionTreeRegressor",
                "RandomForestRegressor",
            ]
        }
        self.cv = cfg.get("cv", {})
        self.scoring = cfg.get("scoring", [])
        self.transformations = cfg.get("transformations", {})


pipeline_config = PipelineConfig(MODEL_CONFIG)
