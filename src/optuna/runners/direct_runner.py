from src.containers.experiment import ExperimentContext
from src.containers.results import RunnerResult
from src.tuning.runners import OptunaSearchRunner

from .base import BaseExperimentRunner


class DirectOptunaRunner(BaseExperimentRunner[RunnerResult]):
    def __init__(self, runner: OptunaSearchRunner):
        self.runner = runner

    def run(self, context: ExperimentContext) -> RunnerResult:
        """
        Builds the experiment setup from the provided context and runs cross-validation
        with the configured pipeline and parameters.
        """
        exp_setup = self.build(exp_config=context.to_experiment_config())

        return self.runner.run(
            estimator=exp_setup.pipeline,
            param_grid=exp_setup.params,
            X_train=context.X_train,
            X_test=context.X_test,
            y_train=context.y_train,
        )
