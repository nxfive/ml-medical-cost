from unittest import mock

import pytest

from src.utils.validation import (check_fold_stability, check_model_results,
                             check_overfitting)


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
