import pytest
from omegaconf import OmegaConf

from src.utils.grid import prepare_grid, update_param_grid, update_params_with_optuna


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
