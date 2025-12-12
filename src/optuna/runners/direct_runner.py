import optuna
from src.factories.optuna_runner_factory import OptunaRunnerFactory
from src.optuna.types import ExperimentContext
from src.tuning.types import RunnerResult

from .base import BaseExperimentRunner


class DirectOptunaRunner(BaseExperimentRunner[RunnerResult]):
    def __init__(self, study: optuna.Study):
        self.study = study

    def run(self, context: ExperimentContext) -> RunnerResult:
        exp_setup = self.build(exp_config=context.to_experiment_config())
        search_runner = OptunaRunnerFactory.create_direct_runner(
            cv_cfg=context.cv_cfg, optuna_cfg=context.optuna_cfg, study=self.study
        )
        return search_runner.run(
            estimator=exp_setup.pipeline,
            param_grid=exp_setup.params,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
        )
