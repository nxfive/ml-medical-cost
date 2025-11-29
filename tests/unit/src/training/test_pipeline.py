from unittest import mock

from src.training.pipeline import MODELS, models_pipeline, optuna_pipeline


def test_models_pipeline():
    X_train, X_test = mock.Mock(), mock.Mock()
    y_train, y_test = mock.Mock(), mock.Mock()

    mock_pipeline_results = (
        (mock.Mock(), [0.9, 0.8], [0.85], [1, 2], [1.5, 2.1]),
        {"param": [1, 2]},
        "transformer",
    )

    with (
        mock.patch(
            "src.models.pipeline.run_pipeline", return_value=[mock_pipeline_results]
        ) as mock_run_pipeline,
        mock.patch(
            "src.models.pipeline.get_metrics", return_value={"test_r2": 0.9}
        ) as mock_get_metrics,
        mock.patch("src.models.pipeline.check_model_results") as mock_check_results,
        mock.patch("src.models.pipeline.log_model") as mock_log_model,
    ):

        result = models_pipeline(X_train, X_test, y_train, y_test)

    assert isinstance(result, dict)

    assert mock_run_pipeline.call_count == len(MODELS)
    assert mock_get_metrics.call_count == len(MODELS)
    assert mock_check_results.call_count == len(MODELS)
    assert mock_log_model.call_count == len(MODELS)

    assert result["param_grid"] == {"param": [1, 2]}
    assert result["metrics"] == {"test_r2": 0.9}


def test_optuna_pipeline():
    X_train, X_test = mock.Mock(), mock.Mock()
    y_train, y_test = mock.Mock(), mock.Mock()

    fake_estimator = mock.Mock()

    mock_best_model_info = {
        "model": mock.Mock(__name__="FakeModel"),
        "estimator": fake_estimator,
        "param_grid": {"param": [1, 2]},
        "transformer": "transformer",
        "folds_scores_mean": [0.85],
        "metrics": {"test_r2": 0.9},
    }

    with (
        mock.patch(
            "src.models.pipeline.optimize_model",
            return_value=(mock.Mock(), {"param": 2}),
        ) as mock_optimize_model,
        mock.patch(
            "src.models.pipeline.update_params_with_optuna", return_value={"param": 2}
        ) as mock_update_params,
        mock.patch("src.models.pipeline.get_metrics") as mock_get_metrics,
        mock.patch("src.models.pipeline.log_model") as mock_log_model,
        mock.patch("src.models.pipeline.save_model_with_metadata") as mock_save_model,
    ):

        optuna_pipeline(mock_best_model_info, X_train, X_test, y_train, y_test)

    mock_optimize_model.assert_called_once_with(
        mock_best_model_info["model"], X_train, y_train
    )
    mock_update_params.assert_called_once_with(
        mock_best_model_info["param_grid"],
        optuna_params={"param": 2},
    )

    fake_estimator.set_params.assert_called_once_with(param=2)
    fake_estimator.fit.assert_called_once_with(X_train, y_train)
    fake_estimator.predict.assert_has_calls([mock.call(X_test), mock.call(X_train)])

    mock_get_metrics.assert_called_once_with(
        y_train,
        y_test,
        fake_estimator.predict.return_value,
        fake_estimator.predict.return_value,
    )

    mock_log_model.assert_called_once()
    mock_save_model.assert_called_once_with(
        fake_estimator,
        mock_best_model_info["model"].__name__,
        mock_get_metrics.return_value,
        {"param": 2},
    )
