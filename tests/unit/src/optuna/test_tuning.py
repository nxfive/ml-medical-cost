from unittest import mock

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.optuna.tuning import objective, optimize_model


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


def test_optimize_model_no_params(capsys, train_data):
    model = mock.Mock(__name__="FakeModel")
    X_train, y_train = train_data

    optuna_config_instance = mock.Mock()
    optuna_config_instance.params = {}

    with mock.patch(
        "src.models.optuna_tuning.OptunaConfig", return_value=optuna_config_instance
    ):
        results = optimize_model(model, X_train, y_train)

    captured = capsys.readouterr()

    assert results is None
    assert model.__name__ in captured.out
    assert "skipping optimization" in captured.out


def test_optimize_model_with_params(capsys, train_data):
    X_train, y_train = train_data

    model = mock.Mock()

    mock_optuna_config = mock.Mock()
    mock_optuna_config.params = {"n_estimators": {"min": 50, "max": 150, "step": 10}}
    mock_optuna_config.trials = 5

    fake_study = mock.Mock()
    fake_study.best_params = {"n_estimators": 50}

    with (
        mock.patch(
            "src.models.optuna_tuning.OptunaConfig", return_value=mock_optuna_config
        ),
        mock.patch("optuna.create_study", return_value=fake_study),
    ):
        results = optimize_model(model, X_train, y_train)

    captured = capsys.readouterr()
    assert isinstance(results, tuple)
    study, best_params = results
    assert study is fake_study
    assert "skipping optimization" not in captured.out
