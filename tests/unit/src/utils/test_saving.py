from pathlib import Path
from unittest import mock

from src.utils.saving import (build_metadata, save_metrics, save_model,
                              save_model_with_metadata, save_run)


def test_save_model_with_metadata(cfg_saving):
    cfg = cfg_saving
    fake_model = mock.Mock()
    model_name = "RandomForestRegressor"
    metrics = {"test_r2": 0.85, "train_r2": 0.90}
    params = {"n_estimators": 100, "max_depth": 5}
    mock_metadata = {
        "model_name": model_name,
        "version": "1.0",
        "date_trained": "2025-11-29",
        "features_processed": {
            "cat_features": ["sex", "region"],
            "num_features": ["age", "bmi"],
            "bin_features": ["smoker"],
        },
        "params": params,
        "metrics": metrics,
    }
    with (
        mock.patch("pathlib.Path.mkdir") as mock_mkdir,
        mock.patch(
            "src.utils.saving.build_metadata", return_value=mock_metadata
        ) as mock_build_metadata,
        mock.patch("src.utils.saving.save_model") as mock_save_model,
        mock.patch("src.utils.saving.save_metrics") as mock_save_metrics,
    ):
        save_model_with_metadata(fake_model, model_name, metrics, params, cfg)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_build_metadata.assert_called_once_with(model_name, cfg, params, metrics)

    expected_model_path = Path(cfg.models.output_dir) / f"{model_name.lower()}.pkl"
    mock_save_model.assert_called_once_with(fake_model, expected_model_path)

    expected_metadata_path = (
        Path(cfg.models.output_dir) / "metadata" / f"{model_name.lower()}.yml"
    )
    mock_save_metrics.assert_called_once_with(mock_metadata, expected_metadata_path)


def test_save_run(cfg_saving):
    results = {
        "model": "RandomForestRegressor",
        "param_grid": {"n_estimators": 100, "max_depth": 5},
        "transformer": None,
        "folds_scores_mean": 88.8,
        "metrics": {"test_r2": 0.85, "train_r2": 0.90},
    }
    pipeline = mock.Mock()
    cfg = cfg_saving

    fake_date = mock.Mock()
    fake_date.strftime.return_value = "2025-11-29"

    with (
        mock.patch("src.utils.saving.datetime") as mock_datetime,
        mock.patch("pathlib.Path.mkdir") as mock_path_mkdir,
        mock.patch("src.utils.saving.save_metrics") as mock_save_metrics,
        mock.patch("src.utils.saving.save_model") as mock_save_model,
    ):
        mock_datetime.now.return_value = fake_date
        save_run(results, pipeline, cfg)

    expected_path = Path(cfg.training.output_dir)

    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_save_metrics.assert_called_once_with(results, expected_path / "metrics.yaml")
    mock_save_model.assert_called_once_with(pipeline, expected_path / "pipeline.pkl")


def test_save_metrics():
    metrics = {"r2": 0.9}
    path = Path("/fake/path/metrics.yml")
    mock_open_func = mock.mock_open()

    with (
        mock.patch("builtins.open", mock_open_func),
        mock.patch("yaml.safe_dump") as mock_safe_dump,
    ):
        save_metrics(metrics, path)

    mock_open_func.assert_called_once_with(path, "w")

    handle = mock_open_func()  # handle: object returned from open()
    mock_safe_dump.assert_called_once_with(metrics, handle)


def test_save_model():
    model = mock.Mock()
    path = Path("/fake/path/pipeline.pkl")

    with mock.patch("joblib.dump") as mock_joblib_dump:
        save_model(model, path)

    mock_joblib_dump.assert_called_once_with(model, path)


def test_build_metadata(cfg_saving):
    model_name = "RandomForestRegressor"
    cfg = cfg_saving
    metrics = {"test_r2": 0.85, "train_r2": 0.90}
    params = {"n_estimators": 100, "max_depth": 5}

    fake_date = mock.Mock()
    fake_date.strftime.return_value = "2025-11-29"

    with mock.patch("src.utils.saving.datetime") as mock_datetime:
        mock_datetime.today.return_value = fake_date
        result = build_metadata(model_name, cfg, params, metrics)

    assert result["model_name"] == model_name
    assert result["version"] == "1.0"
    assert result["params"] == params
    assert result["metrics"] == metrics
    assert result["features_processed"]["cat_features"] == ["sex", "region"]
    assert result["features_processed"]["num_features"] == ["age", "bmi"]
    assert result["features_processed"]["bin_features"] == ["smoker"]
    assert result["date_trained"] == "2025-11-29"
