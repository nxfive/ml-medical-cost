from datetime import datetime

from src.conf.schema import TrainingDir
from src.containers.results import StageResult
from src.data.core import DataSaver
from src.io.file_ops import PathManager
from src.serializers.stage_result import StageResultSerializer


class RunSaver:
    def __init__(
        self,
        training_dir: TrainingDir,
        data_saver: DataSaver,
    ):
        self.training_dir = training_dir
        self.data_saver = data_saver

    def save(self, run_result: StageResult) -> None:
        """
        Saves estimator and metrics to a timestamped directory under
        the training output path.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = self.training_dir.output_dir / timestamp
        PathManager.ensure_dir(results_path)

        metrics_path = results_path / self.training_dir.metrics_file
        model_path = results_path / self.training_dir.model_file

        self.data_saver.save_metrics(
            metrics=StageResultSerializer.to_metrics(run_result),
            metrics_path=metrics_path,
        )
        self.data_saver.save_model(
            model=run_result.estimator,
            model_path=model_path,
        )
