from src.builders.pipeline_builder import PipelineBuilder
from src.optuna.runners import DirectOptunaRunner, WrapperOptunaRunner
from src.tuning.runners import SimpleRunner
from src.tuning.types import RunnerResult

from .tuning import OptunaOptimize
from .types import ExperimentContext


class OptunaExperimentManager:
    def __init__(self, context: ExperimentContext, optimizer: OptunaOptimize):
        self.context = context
        self.optimizer = optimizer

    def _check_transformation(self):
        return self.context.model_cfg.target_transformations

    def manage(self) -> RunnerResult:
        transform = self._check_transformation()

        if transform:
            runner = WrapperOptunaRunner(optimizer=self.optimizer)
            study = runner.run(self.context)

            estimator = PipelineBuilder.build(
                model_cfg=self.context.model_cfg,
                features_cfg=self.context.features_cfg,
                transformation=study.best_params["transformation"],
            )
            param_grid = {
                k: v for k, v in study.best_params.items() if k != "transformation"
            }

            simpler = SimpleRunner()
            return simpler.run(
                estimator=estimator,
                param_grid=param_grid,
                X_train=self.context.X_train,
                X_test=self.context.X_test,
                y_train=self.context.y_train,
                folds_scores=None,
            )

        else:
            runner = DirectOptunaRunner(self.optimizer.study)
            return runner.run(self.context)
