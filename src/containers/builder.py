from dataclasses import dataclass

from src.data.core import DataLoader
from src.models.savers.model_saver import ModelSaver
from src.models.savers.run_saver import RunSaver
from src.models.spec import ModelSpec
from src.optuna.tuning import OptunaOptimize
from src.training.train import TrainModel
from src.tuning.runners import CrossValidationRunner, OptunaSearchRunner


@dataclass
class TrainingBuildResult:
    model_spec: ModelSpec
    run_saver: RunSaver
    loader: DataLoader
    training: TrainModel


@dataclass
class OptunaBuildResult:
    data_loader: DataLoader
    model_saver: ModelSaver
    model_spec: ModelSpec
    optimizer: OptunaOptimize
    cross_runner: CrossValidationRunner
    search_runner: OptunaSearchRunner
