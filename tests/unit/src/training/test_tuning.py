from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.training.tuning import (
    evaluate_target_transformers,
    perform_cross_validation,
    perform_grid_search,
)


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame({"feature": [1, 2, 3]})
    X_test = pd.DataFrame({"feature": [4, 5]})
    y_train = pd.Series([10, 20, 30])
    return X_train, X_test, y_train


def test_perform_cross_validation(sample_data):
    X_train, X_test, y_train = sample_data

    fake_pipeline = mock.Mock()
    fake_pipeline.predict.side_effect = [
        np.array([11, 19, 31]),  # y_train_pred
        np.array([41, 56]),  # y_test_pred
    ]

    mock_cv_scores = np.array([0.8, 0.85, 0.9])

    with (
        mock.patch(
            "src.models.tuning.cross_val_score", return_value=mock_cv_scores
        ) as mock_cv,
        mock.patch("src.models.tuning.get_cv", return_value="fake_cv") as mock_get_cv,
    ):
        (trained_pipeline, folds_scores, folds_mean, y_train_pred, y_test_pred) = (
            perform_cross_validation(
                fake_pipeline, X_train=X_train, X_test=X_test, y_train=y_train
            )
        )

    mock_cv.assert_called_once_with(
        fake_pipeline, X_train, y_train, cv="fake_cv", scoring="r2"
    )
    mock_get_cv.assert_called_once()

    fake_pipeline.fit.assert_called_once_with(X_train, y_train)

    assert fake_pipeline.predict.call_count == 2

    np.testing.assert_array_equal(folds_scores, mock_cv_scores)
    assert folds_mean == np.mean(mock_cv_scores)

    np.testing.assert_array_equal(y_train_pred, np.array([11, 19, 31]))
    np.testing.assert_array_equal(y_test_pred, np.array([41, 56]))

    assert trained_pipeline is fake_pipeline


def test_perform_grid_search(sample_data):
    X_train, X_test, y_train = sample_data

    fake_pipeline = mock.Mock()

    fake_best_estimator = mock.Mock()
    fake_best_estimator.predict.side_effect = [
        np.array([41, 56]),  # y_test_pred
        np.array([11, 19, 31]),  # y_train_pred
    ]

    fake_gs = mock.Mock()
    fake_gs.best_estimator_ = fake_best_estimator
    fake_gs.cv_results_ = {
        "split0_test_score": np.array([0.8]),
        "split1_test_score": np.array([0.85]),
        "split2_test_score": np.array([0.9]),
    }
    fake_gs.best_index_ = 0
    fake_gs.cv.get_n_splits.return_value = 3

    with (
        mock.patch(
            "src.models.tuning.GridSearchCV", return_value=fake_gs
        ) as mock_gs_cls,
        mock.patch("src.models.tuning.get_cv", return_value="fake_cv"),
    ):
        result = perform_grid_search(
            pipeline=fake_pipeline,
            param_grid={"param": [1, 2, 3]},
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
        )

    mock_gs_cls.assert_called_once_with(
        fake_pipeline,
        {"param": [1, 2, 3]},
        cv="fake_cv",
        scoring="r2",
        return_train_score=True,
    )
    fake_gs.fit.assert_called_once_with(X_train, y_train)

    assert isinstance(result, tuple)
    best_estimator, folds_scores, folds_mean, y_train_pred, y_test_pred = result
    assert best_estimator == fake_best_estimator

    np.testing.assert_array_equal(folds_scores, [0.8, 0.85, 0.9])
    assert np.isclose(folds_mean, 0.85)

    np.testing.assert_array_equal(y_train_pred, [11, 19, 31])
    np.testing.assert_array_equal(y_test_pred, [41, 56])


def test_evaluate_target_transformers():
    fake_inner_pipeline = mock.Mock(spec=Pipeline)
    fake_param_grid = {"regressor__n_estimators": [50, 100]}

    fake_config = mock.Mock()
    fake_config.transformations = {
        "log": {"params": {"alpha": [0.1, 0.5]}},
        "quantile": {"params": {"quantile_range": [(0.1, 0.9)]}},
        "none": {"params": {}},
    }

    with (
        mock.patch("src.models.tuning.pipeline_config", fake_config),
        mock.patch(
            "src.models.tuning.update_param_grid",
            side_effect=lambda grid, prefix: {
                f"{prefix}__{k}": v for k, v in grid.items()
            },
        ) as mock_update,
    ):
        results = list(
            evaluate_target_transformers(fake_inner_pipeline, fake_param_grid)
        )

    assert len(results) == 3

    for ttr, _, name in results:
        if name != "none":
            assert isinstance(ttr, TransformedTargetRegressor)
        else:
            assert isinstance(ttr, Pipeline)
        assert name in fake_config.transformations
        mock_update.assert_any_call(fake_param_grid, "regressor")
        mock_update.assert_any_call({"alpha": [0.1, 0.5]}, "transformer")
