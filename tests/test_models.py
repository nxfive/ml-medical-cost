from unittest import mock

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.models import create_model_pipeline, get_preprocessor


def test_get_preprocessor_num_true(monkeypatch):

    class FakeFeaturesConfig:
        numeric = ["age", "bmi"]
        binary = ["sex"]
        categorical = ["region"]

    class FakeModelConfig:
        preprocess_num_features = True

    class FakePipelineConfig:
        features = FakeFeaturesConfig()
        models = {"LinearRegression": FakeModelConfig()}

    monkeypatch.setattr("src.models.models.pipeline_config", FakePipelineConfig())

    preprocessor = get_preprocessor(LinearRegression)
    assert isinstance(preprocessor, ColumnTransformer)

    transformers = {
        name: transformer for name, transformer, _ in preprocessor.transformers
    }

    assert list(transformers.keys()) == ["num", "bin", "cat"]
    assert isinstance(transformers["num"], StandardScaler)
    assert isinstance(transformers["cat"], OneHotEncoder)


def test_get_preprocessor_num_false(monkeypatch):
    class FakeFeaturesConfig:
        numeric = ["age", "bmi"]
        binary = ["sex"]
        categorical = ["region"]

    class FakeModelConfig:
        preprocess_num_features = False

    class FakePipelineConfig:
        features = FakeFeaturesConfig()
        models = {"RandomForestRegressor": FakeModelConfig()}

    monkeypatch.setattr("src.models.models.pipeline_config", FakePipelineConfig())
    preprocessor = get_preprocessor(RandomForestRegressor)

    transformers = {
        transformer: features for _, transformer, features in preprocessor.transformers
    }

    assert "passthrough" in list(transformers.keys())
    assert sorted(FakeFeaturesConfig.numeric + FakeFeaturesConfig.binary) == sorted(
        transformers["passthrough"]
    )


def test_create_model_pipeline_with_class(monkeypatch):
    mock_preprocessor = mock.Mock()
    monkeypatch.setattr(
        "src.models.models.get_preprocessor", lambda model: mock_preprocessor
    )

    pipeline = create_model_pipeline(model=LinearRegression)
    assert isinstance(pipeline, Pipeline)
    
    named_steps = dict(pipeline.named_steps)
    assert named_steps["preprocessor"] == mock_preprocessor
    assert isinstance(named_steps["model"], LinearRegression)


def test_create_model_pipeline_with_instance(monkeypatch):
    mock_preprocessor = mock.Mock()

    monkeypatch.setattr(
        "src.models.models.get_preprocessor", lambda model: mock_preprocessor
    )
    model_instance = LinearRegression()

    pipeline = create_model_pipeline(model_instance=model_instance)
    named_steps = dict(pipeline.named_steps)

    assert isinstance(pipeline, Pipeline)
    assert named_steps["model"] == model_instance
    assert named_steps["preprocessor"] == mock_preprocessor


def test_create_model_pipeline_no_args():
    with pytest.raises(
        ValueError, match="Provide either `model` class or `model_instance`."
    ):
        create_model_pipeline()
