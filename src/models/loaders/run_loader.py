from pathlib import Path

from src.conf.schema import TrainingDir
from src.containers.results import RunResult
from src.data.core import DataLoader
from src.io.file_ops import PathManager
from src.serializers.stage_result import StageResultSerializer


class RunLoader:
    def __init__(self, training_dir: TrainingDir, data_loader: DataLoader):
        self.training_dir = training_dir
        self.data_loader = data_loader

    def load(self, run_dir: Path) -> RunResult:
        """
        Loads a single run from the given directory.
        """
        metrics_path = run_dir / self.training_dir.metrics_file
        pipeline_path = run_dir / self.training_dir.model_file

        if not PathManager.exists(metrics_path):
            raise FileNotFoundError(f"Path: {metrics_path} not found")

        if not PathManager.exists(pipeline_path):
            raise FileNotFoundError(f"Path: {pipeline_path} not found")

        metrics = self.data_loader.load_metrics(metrics_path)
        pipeline = self.data_loader.load_model(pipeline_path)

        return StageResultSerializer.from_loader(metrics, pipeline)
