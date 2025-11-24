from unittest import mock

import pytest

from services.backend.bento.register import get_model_version, register_bento_model


def test_get_model_version_loads_latest(monkeypatch):
    mock_client = mock.Mock()
    mock_client.search_model_versions.return_value = [
        mock.Mock(version="1"),
        mock.Mock(version="3"),
        mock.Mock(version="2"),
    ]

    monkeypatch.setattr("services.backend.bento.register.MlflowClient", lambda: mock_client)

    mock_mlflow = mock.Mock()
    mock_mlflow.sklearn.load_model.return_value = "fake_model"
    monkeypatch.setattr("services.backend.bento.register.mlflow", mock_mlflow)

    result = get_model_version("MyModel")

    mock_mlflow.sklearn.load_model.assert_called_once_with("models:/MyModel/3")
    assert result == "fake_model"


def test_get_model_version_no_versions(monkeypatch):
    mock_client = mock.Mock()
    mock_client.search_model_versions.return_value = []
    monkeypatch.setattr("services.backend.bento.register.MlflowClient", lambda: mock_client)

    with pytest.raises(ValueError, match="No versions found for model MyModel"):
        get_model_version("MyModel")


def test_register_bento_model(capsys):
    with (
        mock.patch("services.backend.bento.register.datetime") as mock_datetime,
        mock.patch(
            "services.backend.bento.register.get_model_version", return_value="fake_model"
        ) as mock_get_model_version,
        mock.patch("services.backend.bento.register.bentoml") as mock_bentoml,
    ):

        mock_now = mock.Mock()
        mock_now.strftime.return_value = "20251105133030"
        mock_datetime.now.return_value = mock_now

        mock_bentoml.sklearn.save_model.return_value.tag = "bento:20251105133030"

        register_bento_model("MyModel", "BentoMedical")

    captured = capsys.readouterr()

    mock_get_model_version.assert_called_once_with("MyModel")

    mock_bentoml.sklearn.save_model.assert_called_once_with(
        name="BentoMedical:20251105133030",
        model="fake_model",
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata={
            "description": "Model MyModel for medical cost prediction",
            "source": "MLflow Registry",
            "model_name": "MyModel",
        },
    )

    assert "Model registered in BentoML" in captured.out
