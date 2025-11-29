from types import SimpleNamespace
from unittest import mock

import pytest
from omegaconf import OmegaConf

from src.main import run_stage


def test_run_stage_calls_correct_module():
    cfg = SimpleNamespace(stage="data") 
    mock_run = mock.Mock()

    with mock.patch("builtins.__import__") as mock_import:
        mock_module = mock.Mock(run=mock_run)
        mock_import.return_value = mock_module

        run_stage(cfg)

        mock_import.assert_called_once_with("src.data.pipeline", fromlist=["run"])
        mock_run.assert_called_once_with(cfg)


def test_run_stage_raises_on_unknown_stage():
    cfg = OmegaConf.create({"stage": "unknown"})
    
    with pytest.raises(ValueError):
        run_stage(cfg)
