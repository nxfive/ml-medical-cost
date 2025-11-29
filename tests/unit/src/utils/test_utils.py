from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import KFold

from src.utils.utils import (check_fold_stability, check_model_results,
                             check_overfitting, get_cv, get_metrics,
                             prepare_grid, save_model_with_metadata,
                             update_param_grid, update_params_with_optuna)


@pytest.mark.parametrize(
    ("param_grid", "step_name", "results"),
    [
        ({"n_estimators": [100, 150]}, "model", {"model__n_estimators": [100, 150]}),
        ({}, "model", {}),
        ({"n_neighbors": [5, 10]}, " ", {"n_neighbors": [5, 10]}),
        ({"fit_intercept": [True, False]}, "", {"fit_intercept": [True, False]}),
        ({"max_depth": [5, 7]}, "model__", {"model__max_depth": [5, 7]}),
        (
            {"min_samples_split": [3, 5]},
            "__model",
            {"model__min_samples_split": [3, 5]},
        ),
    ],
)
def test_update_param_grid(param_grid, step_name, results):
    updated_grid = update_param_grid(param_grid, step_name)

    assert updated_grid == results


@pytest.mark.parametrize(
    ("cfg_dict", "expected"),
    [
        (
            {
                "model": {
                    "name": "rf",
                    "params": {"n_estimators": [50, 100], "max_depth": [3, 7]},
                },
            },
            {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 7],
            },
        ),
        (
            {
                "model": {
                    "name": "lr",
                    "params": {"fit_intercept": [True, False]},
                }
            },
            {"model__fit_intercept": [True, False]},
        ),
        (
            {"model": {"name": "rf", "params": {}}},
            {},
        ),
        (
            {"model": {"name": "lr", "params": None}},
            {},
        ),
    ],
)
def test_prepare_grid(cfg_dict, expected):
    cfg = OmegaConf.create(cfg_dict)
    grid = prepare_grid(cfg)

    for key in grid:
        assert key.startswith("model__")

    assert grid == expected


def test_get_cv():
    cfg_cv = OmegaConf.create(
        {"cv": {"n_splits": 10, "shuffle": True, "random_state": 42}}
    )
    kfold_cv = get_cv(cfg_cv)

    assert isinstance(kfold_cv, KFold)
    assert kfold_cv.get_n_splits() == 10
    assert kfold_cv.shuffle is True
    assert kfold_cv.random_state == 42


def test_get_metrics():
    y_train = pd.Series([1, 2, 3])
    y_test = pd.Series([4, 5, 6])

    y_train_pred = np.array([1.1, 2.1, 3.2])
    y_test_pred = np.array([4.8, 5.5, 5.4])

    with (
        mock.patch("src.utils.utils.mean_absolute_error", return_value=0.8) as mock_mae,
        mock.patch("src.utils.utils.r2_score", return_value=0.5) as mock_r2,
        mock.patch(
            "src.utils.utils.root_mean_squared_error", return_value=0.3
        ) as mock_rmse,
    ):

        metrics = get_metrics(y_train, y_test, y_train_pred, y_test_pred)

        assert mock_mae.call_count == 2
        assert mock_r2.call_count == 2
        assert mock_rmse.call_count == 2

        expected_keys = {
            "train_r2",
            "test_r2",
            "train_mae",
            "test_mae",
            "train_rmse",
            "test_rmse",
        }

        assert expected_keys == set(metrics.keys())

        mock_mae.assert_called()
        mock_r2.assert_called()
        mock_rmse.assert_called()


@pytest.mark.parametrize(
    "folds, threshold, expected",
    [
        ([0.8, 0.85, 0.79, 0.89], 0.1, True),
        ([0.77, 0.86, 0.91, 0.79], 0.1, False),
        ([0.8, 0.8, 0.8], 0.01, True),
        ([0.9, 0.88, 0.91], 0.05, True),
    ],
)
def test_check_fold_stability(folds, threshold, expected, capsys):
    result = check_fold_stability(folds, threshold)
    captured = capsys.readouterr()

    assert result == expected
    assert isinstance(result, bool)
    assert "Fold scores:" in captured.out
    assert "max-min difference:" in captured.out


@pytest.mark.parametrize(
    "train_r2, test_r2, threshold, expected",
    [
        (0.82, 0.6, 0.2, True),
        (0.88, 0.94, 0.2, False),
        (0.72, 0.52, 0.2, False),
    ],
)
def test_check_overfitting(train_r2, test_r2, threshold, expected, capsys):
    results = check_overfitting(train_r2, test_r2, threshold)
    captured = capsys.readouterr()

    assert results == expected
    assert isinstance(results, bool)
    assert "Difference:" in captured.out


@pytest.mark.parametrize(
    "metrics, folds_scores, model_name, cfs_return_value, co_return_value, expected",
    [
        (
            {"train_r2": 0.9, "test_r2": 0.62},
            [0.99, 0.88],
            "BadModel",
            False,
            True,
            ["potential instability", "may be overfitting"],
        ),  # not stable, overfit
        (
            {"train_r2": 0.92, "test_r2": 0.88},
            [0.91, 0.89],
            "GoodModel",
            True,
            False,
            [""],
        ),  # stable, not overfit
    ],
)
def test_check_model_results(
    metrics,
    folds_scores,
    model_name,
    cfs_return_value,
    co_return_value,
    expected,
    capsys,
):
    with (
        mock.patch(
            "src.utils.utils.check_fold_stability", return_value=cfs_return_value
        ) as mock_stable,
        mock.patch(
            "src.utils.utils.check_overfitting", return_value=co_return_value
        ) as mock_overfit,
    ):
        check_model_results(
            model=mock.Mock(__name__=model_name),
            metrics=metrics,
            folds_scores=folds_scores,
            fold_threshold=0.1,
            overfit_threshold=0.2,
        )

        mock_stable.assert_called_once_with(folds_scores, threshold=0.1)
        mock_overfit.assert_called_once_with(
            metrics["train_r2"], metrics["test_r2"], threshold=0.2
        )

        captured = capsys.readouterr()

        for exp in expected:
            assert exp in captured.out


@pytest.mark.parametrize(
    "model_params, optuna_params, expected",
    [
        (
            {
                "model__n_estimators": [25, 50, 100],
                "model__max_depth": [3, 5, 10],
            },
            {"n_estimators": 47, "max_depth": 6},
            {
                "model__n_estimators": 47,
                "model__max_depth": 6,
            },
        ),
        (
            {
                "model__n_estimators": [25, 50, 100],
                "model__max_depth": [3, 5, 10],
            },
            {"n_estimators": 47},
            {
                "model__n_estimators": 47,
                "model__max_depth": 3,
            },
        ),
        (
            {
                "model__n_estimators": [25, 50, 100],
                "model__max_depth": 10,
            },
            {"n_estimators": 47},
            {
                "model__n_estimators": 47,
                "model__max_depth": 10,
            },
        ),
    ],
)
def test_update_params_with_optuna(model_params, optuna_params, expected):
    results = update_params_with_optuna(model_params, optuna_params)

    assert isinstance(results, dict)
    assert results == expected


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
