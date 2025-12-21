from abc import ABC, abstractmethod
from typing import Generic

import optuna
from src.builders.optuna.optuna_experiment_builder import \
    OptunaExperimentBuilder
from src.containers.experiment import ExperimentContext, ExperimentSetup
from src.containers.types import OptunaRunnerResult
from src.dto.config import OptunaExperimentConfig


class BaseExperimentRunner(ABC, Generic[OptunaRunnerResult]):
    def build(
        self, exp_config: OptunaExperimentConfig, trial: optuna.Trial | None = None
    ) -> ExperimentSetup:
        """
        Builds an ExperimentSetup, including the pipeline and parameters from the given
        experiment configuration and optional trial.
        """
        return OptunaExperimentBuilder.build(
            cfg=exp_config,
            trial=trial,
        )

    @abstractmethod
    def run(self, context: ExperimentContext) -> OptunaRunnerResult: ...
