import optuna
from src.conf.schema import OptunaStageConfig
from src.data.core import DataLoader, DataSaver
from src.factories.io_factory import IOFactory
from src.factories.pruner_factory import PrunerFactory
from src.io.file_ops import PathManager
from src.optuna.tuning import OptunaOptimize
from src.optuna.types import OptunaBuildResult
from src.training.cv import get_cv
from src.tuning.runners import CrossValidationRunner, OptunaSearchRunner


class OptunaPipelineBuilder:
    def __init__(self, cfg: OptunaStageConfig):
        self.cfg = cfg

    @staticmethod
    def _build_data_io() -> tuple[DataLoader, DataSaver]:
        """
        Builds the DataLoader and DataSaver using IOFactory readers/writers.
        """
        readers = IOFactory.create_readers()
        writers = IOFactory.create_writers()
        return DataLoader(readers), DataSaver(writers)

    def _build_optimizer(self) -> OptunaOptimize:
        """
        Builds an Optuna optimizer with the configured pruner.
        """
        pruner = PrunerFactory.create(
            cfg_pruner=self.cfg.pruner, cfg_patient=self.cfg.patient
        )
        return OptunaOptimize(
            optuna_cfg=self.cfg.optuna,
            pruner=pruner,
        )

    def _build_runners(
        self, study: optuna.Study
    ) -> tuple[CrossValidationRunner, OptunaSearchRunner]:
        """
        Builds CrossValidationRunner and OptunaSearchRunner for the given study.
        """
        cv = get_cv(self.cfg.cv)
        cross_runner = CrossValidationRunner(cv=cv, scoring=self.cfg.cv.scoring)
        search_runner = OptunaSearchRunner(
            study=study,
            cv=cv,
            scoring=self.cfg.cv.scoring,
            trials=self.cfg.optuna_config.trials,
            timeout=self.cfg.optuna_config.timeout,
        )
        return cross_runner, search_runner

    def build(self) -> OptunaBuildResult:
        """
        Builds all components for the Optuna experiment pipeline:
        - DataLoader and DataSaver
        - Optuna optimizer
        - Cross-validation and Optuna search runners
        """
        PathManager.ensure_dir(self.cfg.models_dir.output_dir)
        data_loader, data_saver = self._build_data_io()
        optimizer = self._build_optimizer()
        cross_runner, search_runner = self._build_runners(study=optimizer.study)

        return OptunaBuildResult(
            loader=data_loader,
            saver=data_saver,
            cross_runner=cross_runner,
            search_runner=search_runner,
        )
