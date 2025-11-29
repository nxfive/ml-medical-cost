from unittest import mock

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.models import create_model_pipeline, get_preprocessor


def test_get_preprocessor_num_true(cfg_models_process_num):
    preprocessor = get_preprocessor(LinearRegression, cfg_models_process_num)
    assert isinstance(preprocessor, ColumnTransformer)

    transformers = {
        name: transformer for name, transformer, _ in preprocessor.transformers
    }

    assert list(transformers.keys()) == ["num", "bin", "cat"]
    assert isinstance(transformers["num"], StandardScaler)
    assert isinstance(transformers["cat"], OneHotEncoder)


def test_get_preprocessor_num_false(cfg_models_not_process_num):
    cfg = cfg_models_not_process_num
    preprocessor = get_preprocessor(RandomForestRegressor, cfg)

    transformers = {
        transformer: features for _, transformer, features in preprocessor.transformers
    }

    assert "passthrough" in list(transformers.keys())
    assert sorted(cfg.features.numeric + cfg.features.binary) == sorted(
        transformers["passthrough"]
    )


def test_create_model_pipeline_with_class(monkeypatch, cfg_models_process_num):
    cfg = cfg_models_process_num
    mock_preprocessor = mock.Mock()
    monkeypatch.setattr(
        "src.models.models.get_preprocessor", lambda model, cfg: mock_preprocessor
    )

    pipeline = create_model_pipeline(cfg=cfg, model=LinearRegression)
    assert isinstance(pipeline, Pipeline)

    named_steps = dict(pipeline.named_steps)
    assert named_steps["preprocessor"] == mock_preprocessor
    assert isinstance(named_steps["model"], LinearRegression)


def test_create_model_pipeline_with_instance(monkeypatch, cfg_models_process_num):
    cfg = cfg_models_process_num
    mock_preprocessor = mock.Mock()

    monkeypatch.setattr(
        "src.models.models.get_preprocessor", lambda model, cfg: mock_preprocessor
    )
    model_instance = LinearRegression()

    pipeline = create_model_pipeline(cfg=cfg, model_instance=model_instance)
    named_steps = dict(pipeline.named_steps)

    assert isinstance(pipeline, Pipeline)
    assert named_steps["model"] == model_instance
    assert named_steps["preprocessor"] == mock_preprocessor


def test_create_model_pipeline_no_args(cfg_models_process_num):
    cfg = cfg_models_process_num
    with pytest.raises(
        ValueError, match="Provide either `model` class or `model_instance`."
    ):
        create_model_pipeline(cfg)
