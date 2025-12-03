from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator

from src.data.core import DataLoader
from src.io.file_ops import PathManager


@dataclass
class BestRun:
    path: Path
    metrics: dict[str, Any]
    model: BaseEstimator


class BestRunSelector:
    def __init__(
        self,
        results_dir: Path,
        data_loader: DataLoader,
        model_file: str,
        metrics_file: str,
    ):
        self.results_dir = results_dir
        self.data_loader = data_loader
        self.model_file = model_file
        self.metrics_file = metrics_file

    def _find_best_run(self) -> tuple[Path, dict[str, Any]]:
        """
        Finds run directory with the highest test R2.
        """
        best_run_dir = None
        best_score = float("-inf")
        best_metrics = {}

        for metrics_file in self.results_dir.glob(f"*/{self.metrics_file}"):
            data = self.data_loader.load_metrics(metrics_file)
            metrics = data.get("metrics", {})
            r2 = metrics.get("test_r2", float("-inf"))
            if r2 > best_score:
                best_score = r2
                best_run_dir = metrics_file.parent
                best_metrics = data

        if best_run_dir is None:
            raise ValueError(f"No valid runs found in {self.results_dir}")

        return best_run_dir, best_metrics

    def load(self) -> BestRun:
        """
        Loads the best trained pipeline and its metrics from a directory of runs.
        """
        run_dir, metrics = self._find_best_run()
        model_path = run_dir / self.model_file
        if PathManager.exists(model_path):
            model = self.data_loader.load_model(model_path)
            return BestRun(path=run_dir, metrics=metrics, model=model)
        raise FileNotFoundError(
            f"Model file {self.model_file} not found in {self.results_dir}"
        )
