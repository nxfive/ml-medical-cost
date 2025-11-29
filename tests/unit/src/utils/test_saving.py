from datetime import datetime
from unittest import mock

from omegaconf import OmegaConf

from src.utils.saving import save_model_with_metadata


def test_save_model_with_metadata():
    cfg = OmegaConf.create(
        {
            "models": {"output_dir": "/fake/path"},
            "features": {
                "categorical": ["sex", "region"],
                "numeric": ["age", "bmi"],
                "binary": ["smoker"],
            },
        }
    )

    fake_model = mock.Mock()
    model_name = "RandomForestRegressor"
    metrics = {"test_r2": 0.85, "train_r2": 0.90}
    params = {"n_estimators": 100, "max_depth": 5}

    with (
        mock.patch("src.utils.utils.joblib.dump") as mock_dump,
        mock.patch("src.utils.utils.yaml.safe_dump") as mock_yaml_dump,
        mock.patch("builtins.open", mock.mock_open()) as mock_file,
        mock.patch("src.utils.utils.datetime") as mock_datetime,
        mock.patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        mock_datetime.today.return_value = datetime(2025, 11, 2)

        save_model_with_metadata(fake_model, model_name, metrics, params, cfg)

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_dump.assert_called_once_with(fake_model, mock.ANY)
    mock_file.assert_called_once_with(mock.ANY, "w")
    args, _ = mock_yaml_dump.call_args
    metadata_written = args[0]

    assert metadata_written["model_name"] == model_name
    assert metadata_written["version"] == "1.0"
    assert metadata_written["params"] == params
    assert metadata_written["metrics"] == metrics
    assert metadata_written["features_processed"]["num_features"] == ["age", "bmi"]
    assert metadata_written["features_processed"]["cat_features"] == ["sex", "region"]
    assert metadata_written["features_processed"]["bin_features"] == ["smoker"]
    assert metadata_written["date_trained"] == "2025-11-02"
