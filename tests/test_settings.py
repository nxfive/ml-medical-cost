import yaml

from src.models.settings import MODEL_DEFAULT_CONFIG, PipelineConfig


def test_pipeline_config_file_not_found(tmp_path):
    yaml_path = tmp_path / "fake_yaml.yml"
    config = PipelineConfig(yaml_path)

    assert config.cv == MODEL_DEFAULT_CONFIG["cv"]
    assert config.features.numeric == MODEL_DEFAULT_CONFIG["num_features"]


def test_pipeline_config_invalid_yaml(tmp_path):
    yaml_path = tmp_path / "invalid.yml"
    yaml_path.write_text("key: value\n - arg")

    config = PipelineConfig(yaml_path)

    assert config.features.binary == MODEL_DEFAULT_CONFIG["bin_features"]
    assert len(config.models) == 4


def test_pipeline_config_partial_yaml(tmp_path):
    data = {
        "cat_features": ["region"],
        "num_features": ["age"],
        "bin_features": ["sex"],
        "cv": {"n_splits": 3, "shuffle": False, "random_state": 11},
        "models": {"LinearRegression": {"params": {"fit_intercept": [True, False]}}},
    }

    yaml_path = tmp_path / "partial.yml"
    yaml_path.write_text(yaml.safe_dump(data))

    config = PipelineConfig(yaml_path)

    assert len(config.features.binary) == 1
    assert len(config.features.categorical) == 1
    assert len(config.features.numeric) == 1
    assert config.models["LinearRegression"].preprocess_num_features == True
    assert "RandomForestRegressor" in config.models
