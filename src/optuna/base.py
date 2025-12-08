from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator

from src.conf.schema import OptunaStageConfig
from src.data.core import DataLoader
from src.factories.io_factory import IOFactory
from src.factories.optuna_factory import OptunaConfigFactory, TrainingDir
from src.io.file_ops import PathManager
from src.models.loaders.run_loader import RunLoader
from src.models.registry import get_model_class_and_short
from src.models.selection import BestRunSelector
from src.patterns.base_pipeline import BasePipeline

from .types import DynamicConfig, LoadedModelResults, ModelRun


@dataclass
class OptunaBaseResult:
    static_config: OptunaStageConfig
    model_run: ModelRun


class OptunaBasePipeline(BasePipeline[RunLoader, OptunaBaseResult]):
    def __init__(self, dynamic_cfg: DictConfig):
        self.cfg = dynamic_cfg

    def select_best_run(self, run_loader: RunLoader) -> ModelRun:
        """
        Selects the best model run from all loaded results.
        """
        runs = self.load_all_model_results(run_loader)
        brs = BestRunSelector(runs)
        return brs.select()

    def load_optuna_config(
        self, model_class: type[BaseEstimator], dynamic_cfg: DynamicConfig
    ) -> OptunaStageConfig:
        """
        Builds OptunaStageConfig using static and dynamic config parts.
        """
        return OptunaConfigFactory.create(self.cfg, dynamic_cfg, model_class)

    def load_all_model_results(self, run_loader: RunLoader) -> LoadedModelResults:
        """
        Loads all model results from the training output directory.
        """
        runs = {}
        for run_dir in Path(self.cfg.training.output_dir).iterdir():
            if run_dir.is_dir():
                runs[run_dir.name] = run_loader.load(run_dir)

        return LoadedModelResults(runs=runs)

    def build(self) -> RunLoader:
        """
        Prepares readers and constructs a RunLoader for the pipeline.
        """
        PathManager.ensure_dir(Path(self.cfg.training.output_dir))
        readers = IOFactory.create_readers()
        return RunLoader(
            training_dir=TrainingDir(
                output_dir=Path(self.cfg.training.output_dir),
                model_file=self.cfg.training.model_file,
                metrics_file=self.cfg.training.metrics_file,
            ),
            data_loader=DataLoader(readers),
        )

    def run(self) -> OptunaBaseResult:
        """
        Builds loaders, selects best run, and creates full stage config.
        """
        run_loader = self.build()
        best_run = self.select_best_run(run_loader)
        model_class, short = get_model_class_and_short(best_run.result.model_name)

        dc = DynamicConfig(
            model=OmegaConf.load(f"src/conf/model/{short}.yaml"),
            optuna_model=OmegaConf.load(f"src/conf/optuna/{short}.yaml"),
        )
        return OptunaBaseResult(
            static_config=self.load_optuna_config(model_class, dynamic_cfg=dc),
            model_run=best_run,
        )
