from unittest import mock

import pandas as pd
import pytest

from src.models.train import run_pipeline


@pytest.mark.parametrize(
    "has_params, target_transformations",
    [
        (True, False),  # GridSearch, No Target Transformation
        (False, False),  # Cross-Validation, No Target Transformation
        (True, True),  # GridSearch, Target Transformation
        (False, True),  # Cross-Validation, Target Transformation
    ]
)
def test_run_pipeline(has_params, target_transformations):
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    X_test = pd.DataFrame({"feature": [4, 5]})
    y_train = pd.Series([10, 20, 30])

    model_cls = mock.Mock(__name__="LinearRegression")
    fake_result = mock.Mock(name="PipelineResult")
    fake_pipeline = mock.Mock()

    fake_eval_generator = [
        (mock.Mock(name="pipe1"), {"param": [1]}, "log"),
        (mock.Mock(name="pipe2"), {"param": [2]}, "quantile"),
    ]

    with (
        mock.patch(
            "src.models.train.create_model_pipeline", return_value=fake_pipeline
        ),
        mock.patch(
            "src.models.train.prepare_grid",
            return_value={"param": [1, 2]} if has_params else {},
        ),
        mock.patch(
            "src.models.train.evaluate_target_transformers",
            return_value=fake_eval_generator if target_transformations else [],
        ),
        mock.patch(
            "src.models.train.perform_cross_validation", return_value=fake_result
        ) as mock_cv,
        mock.patch(
            "src.models.train.perform_grid_search", return_value=fake_result
        ) as mock_gs,
        mock.patch("src.models.train.pipeline_config") as mock_cfg,
    ):
        mock_cfg.models = {
            "LinearRegression": mock.Mock(
                params={"param": [1, 2]} if has_params else {},
                target_transformations=target_transformations,
            )
        }
        results = list(run_pipeline(model_cls, X_train, X_test, y_train))

        if target_transformations:
            assert len(results) == 2
            for (res, _, name), expected_name in zip(results, ["log", "quantile"]):
                assert res is fake_result
                assert name == expected_name

            expected_calls = len(fake_eval_generator)
            assert mock_gs.call_count == (expected_calls if has_params else 0)
            assert mock_cv.call_count == (expected_calls if not has_params else 0)

        else:
            assert len(results) == 1
            result, _, name = results[0]
            assert result is fake_result
            assert name is None

            if has_params:
                mock_gs.assert_called_once()
                mock_cv.assert_not_called()
            else:
                mock_cv.assert_called_once()
                mock_gs.assert_not_called()
