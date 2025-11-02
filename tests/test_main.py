from src.main import main

from unittest import mock


def test_main():

    with (
        mock.patch("src.main.setup_mlflow") as mock_mlflow,
        mock.patch(
            "src.main.data_pipeline",
            return_value=("X_train", "X_test", "y_train", "y_test"),
        ) as mock_data_pipeline,
        mock.patch(
            "src.main.models_pipeline", return_value="best_model_info"
        ) as mock_models_pipeline,
        mock.patch("src.main.optuna_pipeline") as mock_optuna_pipeline,
    ):

        main()

        mock_mlflow.assert_called_once()
        mock_data_pipeline.assert_called_once()
        mock_models_pipeline.assert_called_once_with(
            "X_train", "X_test", "y_train", "y_test"
        )
        mock_optuna_pipeline.assert_called_once_with(
            "best_model_info", "X_train", "X_test", "y_train", "y_test"
        )
