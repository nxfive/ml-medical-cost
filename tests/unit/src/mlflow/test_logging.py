from unittest import mock

import numpy as np
import pytest

from src.mlflow.logging import log_model, setup_mlflow


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
        mock.patch("src.models.mlflow_logging.mlflow") as mock_mlflow,
        mock.patch("src.models.mlflow_logging.datetime") as mock_datetime
    ):
        mock_datetime.now.return_value.strftime.return_value = "2025-11-04-2000"

        setup_mlflow()

    if token and uri:
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.call_args[0][0].startswith(exp_name)
    else:
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.call_args[0][0].startswith(exp_name)
    mock_mlflow.set_tracking_uri.assert_called_once_with(uri)


def test_log_model(train_data):
    X_train, _ = train_data

    fake_estimator = mock.Mock()
    fake_estimator.predict.return_value = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    param_grid = {"n_estimators": 200, "max_depth": 5}
    model = mock.Mock(__name__="FakeModel")
    metrics = {"test_r2": 0.88}
    folds_scores = [0.90, 0.95]
    folds_scores_mean = 0.925
    transformer_name = "transformer"

    with (
        mock.patch("src.models.mlflow_logging.mlflow") as mock_mlflow,
        mock.patch(
            "src.models.mlflow_logging.mlflow.sklearn.log_model"
        ) as mock_log_model,
        mock.patch(
            "src.models.mlflow_logging.mlflow.models.infer_signature",
            return_value="signature",
        ) as mock_signature,
    ):
        log_model(
            fake_estimator,
            param_grid,
            X_train,
            model,
            metrics,
            folds_scores,
            folds_scores_mean,
            study=None,
            transformer_name=transformer_name,
        )

    mock_mlflow.log_param.assert_any_call("n_estimators", 200)
    mock_mlflow.log_param.assert_any_call("transformer", transformer_name)
    mock_mlflow.log_metric.assert_any_call("test_r2", 0.88, step=mock.ANY)
    mock_mlflow.log_metric.assert_any_call("fold_1_r2", 0.9, step=mock.ANY)
    mock_mlflow.log_metric.assert_any_call("folds_r2_mean", 0.925, step=mock.ANY)

    mock_log_model.assert_called_once_with(
        fake_estimator, name="FakeModel", signature="signature", input_example=mock.ANY
    )
    mock_signature.assert_called_once()
