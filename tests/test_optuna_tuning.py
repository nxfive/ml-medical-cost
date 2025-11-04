from unittest import mock

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.models.optuna_tuning import objective


@pytest.mark.parametrize(
    "model, optuna_params",
    [
        (LinearRegression, {"fit_intercept": [True, False]}),
        (RandomForestRegressor, {"n_estimators": {"min": 50, "max": 100, "step": 10}}),
        (
            RandomForestRegressor,
            {
                "max_depth": {
                    "min": 2.0,
                    "max": 10.0,
                }
            },
        ),
        (
            RandomForestRegressor,
            {"criterion": {"choices": ["squared_error", "absolute_error"]}},
        ),
    ],
)
def test_objective(monkeypatch, train_data, fake_trial, fake_cv, model, optuna_params):
    X_train, y_train = train_data

    fake_trial = fake_trial
    fake_pipeline = mock.Mock()
    fake_pipeline.score.return_value = 0.9

    monkeypatch.setattr(
        "src.models.optuna_tuning.create_model_pipeline",
        lambda model_instance: fake_pipeline,
    )

    fake_cv = fake_cv
    result = objective(fake_trial, X_train, y_train, model, optuna_params, fake_cv)

    assert isinstance(result, float)
    assert result == 0.9

    fake_pipeline.fit.assert_called()
    fake_pipeline.score.assert_called()

    assert fake_trial.reported == [(0.9, 0)]


def test_objective_invalid_param_type(train_data, fake_trial, fake_cv):
    X_train, y_train = train_data

    fake_trial = fake_trial
    fake_cv = fake_cv

    with pytest.raises(ValueError, match="Invalid param format"):
        objective(
            fake_trial,
            X_train,
            y_train,
            model=LinearRegression,
            optuna_params={"param": {"foo": "bar"}},
            cv=fake_cv,
        )

    with pytest.raises(ValueError, match="Unexpected param type"):
        objective(
            fake_trial,
            X_train,
            y_train,
            model=LinearRegression,
            optuna_params={"param": 123},
            cv=fake_cv,
        )


def test_objective_pruning(monkeypatch, train_data, fake_trial_prune, fake_cv):
    import optuna

    X_train, y_train = train_data
    
    monkeypatch.setattr(
        "src.models.optuna_tuning.create_model_pipeline",
        lambda model_instance: mock.Mock(),
    )

    with pytest.raises(optuna.exceptions.TrialPruned):
        objective(
            trial=fake_trial_prune,
            X_train=X_train,
            y_train=y_train,
            model=LinearRegression,
            optuna_params={"fit_intercept": [True, False]},
            cv=fake_cv,
        )
