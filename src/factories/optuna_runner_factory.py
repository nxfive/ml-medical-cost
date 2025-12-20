import optuna
from src.conf.schema import CVConfig, OptunaConfig
from src.training.cv import get_cv
from src.tuning.runners import CrossValidationRunner, OptunaSearchRunner


class OptunaRunnerFactory:
    """
    Factory for creating Optuna search runners.

    Provides methods to create either a wrapper runner or a direct runner
    with appropriate CV and study configuration.
    """

    @staticmethod
    def create_wrapper_runner(cv_cfg: CVConfig):
        return CrossValidationRunner(
            cv=get_cv(cv_cfg),
            scoring=cv_cfg.scoring,
        )

    @staticmethod
    def create_direct_runner(
        cv_cfg: CVConfig, optuna_cfg: OptunaConfig, study: optuna.Study
    ):
        return OptunaSearchRunner(
            study=study,
            cv=get_cv(cv_cfg),
            scoring=cv_cfg.scoring,
            trials=optuna_cfg.trials,
            timeout=optuna_cfg.timeout,
        )
