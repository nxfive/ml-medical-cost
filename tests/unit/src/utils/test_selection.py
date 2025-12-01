from pathlib import Path
from unittest import mock

import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.utils.selection import (find_best_run, get_model_class_and_short,
                                 pick_best)


def test_find_best_run():
    fake_file_1 = mock.Mock(spec=Path)
    fake_file_2 = mock.Mock(spec=Path)
    fake_file_1.parent = Path("/run1")
    fake_file_2.parent = Path("/run2")

    with (
        mock.patch("pathlib.Path.glob", return_value=[fake_file_1, fake_file_2]),
        mock.patch("src.utils.selection.load_metrics") as mock_load,
    ):
        mock_load.side_effect = [
            {"metrics": {"test_r2": 0.8}},
            {"metrics": {"test_r2": 0.95}},
        ]

        best_run, metrics = find_best_run("fake_dir")

    assert best_run == Path("/run2")
    assert metrics["metrics"]["test_r2"] == 0.95
    assert mock_load.call_count == 2


def test_find_best_run_no_metrics_raises():
    with mock.patch("pathlib.Path.glob", return_value=[]):
        with pytest.raises(ValueError, match="No valid metrics found"):
            find_best_run("fake_dir")


def test_pick_best():
    fake_run = Path("/fake_run")
    fake_metrics = {"metrics": {"test_r2": 0.9}}
    fake_pipeline = mock.Mock()

    with (
        mock.patch(
            "src.utils.selection.find_best_run", return_value=(fake_run, fake_metrics)
        ) as mock_find,
        mock.patch(
            "src.utils.selection.load_model", return_value=fake_pipeline
        ) as mock_load,
    ):
        pipeline, metrics = pick_best("results_dir")

    mock_find.assert_called_once_with("results_dir")
    mock_load.assert_called_once_with(fake_run / "pipeline.pkl")

    assert pipeline == fake_pipeline
    assert metrics == fake_metrics


@pytest.mark.parametrize(
    "name, output",
    [
        ("rf", (RandomForestRegressor, None)),
        ("RandomForestRegressor", (RandomForestRegressor, "rf")),
        ("LinearRegression", (LinearRegression, "linear")),
    ],
)
def test_get_model_class_and_short(name, output):
    results = get_model_class_and_short(name)
    assert results == output


@pytest.mark.parametrize("name", [("lr"), ("LinearRegressor"), (""), (None)])
def test_get_model_class_and_short_raise_error(name):
    with pytest.raises(ValueError, match=f"Model '{name}' not found"):
        get_model_class_and_short(name)
