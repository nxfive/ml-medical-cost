from optuna.pruners import BasePruner
from src.conf.schema import PrunerConfig
from src.optuna.pruners.registry import PRUNERS


class PrunerFactory:
    @staticmethod
    def create(
        cfg_pruner: PrunerConfig, cfg_patient: PrunerConfig | None
    ) -> BasePruner:
        """
        Creates an Optuna pruner instance based on configuration.
        """
        pruner = PRUNERS[cfg_pruner.name](**cfg_pruner.params)

        if cfg_patient:
            pruner = PRUNERS[cfg_patient.name](base_pruner=pruner, **cfg_patient.params)

        return pruner
