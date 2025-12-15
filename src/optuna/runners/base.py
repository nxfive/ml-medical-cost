from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import optuna
from src.builders.optuna_experiment_builder import OptunaExperimentBuilder
from src.optuna.types import (ExperimentContext, ExperimentSetup,
                              OptunaExperimentConfig)

ORR = TypeVar("OptunaRunnerResult")


class BaseExperimentRunner(ABC, Generic[ORR]):
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
    def run(self, context: ExperimentContext) -> ORR: ...
