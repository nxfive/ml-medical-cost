from datetime import datetime

from src.conf.schema import TrainingDir
from src.data.core import DataSaver
from src.io.file_ops import PathManager
from src.models.types import ModelRun


class RunSaver:
    def __init__(
        self,
        trainig_dir: TrainingDir,
        data_saver: DataSaver,
    ):
        self.training_dir = trainig_dir
        self.data_saver = data_saver

    def save(self, run: ModelRun) -> None:
        """
        Saves a ModelRun to a timestamped directory under the training output path.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = self.training_dir.output_dir / timestamp
        PathManager.ensure_dir(results_path)

        self.data_saver.save_metrics(
            metrics=run.result.to_dict(),
            metrics_path=results_path / self.training_dir.metrics_file,
        )
        self.data_saver.save_model(
            model=run.estimator, model_path=results_path / self.training_dir.model_file
        )
