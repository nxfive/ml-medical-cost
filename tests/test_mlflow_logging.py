from unittest import mock

import pytest

from src.models.mlflow_logging import setup_mlflow


@pytest.mark.parametrize(
    "env, token, uri, exp_name",
    [
        ("local", "", "http://localhost:5000", "local-test"),
        ("remote", "token", "http://mlflow-remote", "ci-build"),
    ],
)
def test_setup_mlflow_local(monkeypatch, env, token, uri, exp_name):
    if env == "local":
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", token)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    with (
        mock.patch("src.models.mlflow_logging.mlflow.set_tracking_uri") as mock_set_uri,
        mock.patch("src.models.mlflow_logging.mlflow.set_experiment") as mock_set_exp,
        mock.patch("src.models.mlflow_logging.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value.strftime.return_value = "2025-11-04-2000"

        setup_mlflow()

    mock_set_uri.assert_called_once_with(uri)
    args = mock_set_exp.call_args[0]
    assert args[0].startswith(exp_name)
    assert "2025-11-04-2000" in args[0]
