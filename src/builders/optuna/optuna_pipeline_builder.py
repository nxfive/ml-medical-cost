import optuna
from src.conf.schema import OptunaStageConfig
from src.containers.builder import OptunaBuildResult
from src.data.core import DataLoader, DataSaver
from src.factories.io_factory import IOFactory
from src.factories.pruner_factory import PrunerFactory
from src.io.file_ops import PathManager
from src.models.savers.model_saver import ModelSaver
from src.optuna.tuning import OptunaOptimize
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

    def _build_model_saver(self, data_saver: DataSaver) -> ModelSaver:
        """
        Builds a ModelSaver that saves trained model.
        """
        return ModelSaver(
            models_dir=self.cfg.models_dir.output_dir,
            data_saver=data_saver,
        )

    def _build_optimizer(self) -> OptunaOptimize:
        """
        Builds an Optuna optimizer with the configured pruner.
        """
        pruner = PrunerFactory.create(
            cfg_pruner=self.cfg.pruner, cfg_patient=self.cfg.patient
        )
        return OptunaOptimize(
            optuna_cfg=self.cfg.optuna_config,
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
            optuna_cfg=self.cfg.optuna_config,
            study=study,
            cv=cv,
            scoring=self.cfg.cv.scoring,
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
        model_saver = self._build_model_saver(data_saver)
        optimizer = self._build_optimizer()
        cross_runner, search_runner = self._build_runners(study=optimizer.study)

        return OptunaBuildResult(
            data_loader=data_loader,
            model_saver=model_saver,
            optimizer=optimizer,
            cross_runner=cross_runner,
            search_runner=search_runner,
        )
