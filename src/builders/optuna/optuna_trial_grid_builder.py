from typing import Any

import optuna
from src.params.optuna_grid import OptunaGrid
from src.params.optuna_updater import OptunaParamUpdater
from src.params.prefixer import ParamGridPrefixer
from src.tuning.transformers.registry import TRANSFORMERS

from .optuna_space_builder import OptunaSpaceBuilder


class OptunaTrialGridBuilder:
    @staticmethod
    def _build_model_trial_params(
        trial: optuna.Trial,
        optuna_model_params: dict[str, Any],
        model_params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Samples model parameters for the given trial based on the Optuna space.
        """
        model_space = OptunaSpaceBuilder.build(
            optuna_params=optuna_model_params,
            model_params=model_params,
        )
        return OptunaGrid.create_trial_params(trial=trial, optuna_space=model_space)

    @staticmethod
    def _build_transformer_trial_params(
        trial: optuna.Trial, transformation: str, transformers: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Samples transformer parameters for the given trial and chosen transformation.
        """
        transformer_params = transformers[transformation]
        updated = OptunaParamUpdater.update(
            model_params=transformer_params, optuna_params=None
        )
        transformer_space = OptunaGrid.create_optuna_space(updated)
        return OptunaGrid.create_trial_params(
            trial=trial,
            optuna_space=transformer_space,
        )

    @staticmethod
    def _merge_and_prefix(
        model_trial_params: dict[str, Any],
        transformer_trial_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merges model and transformer parameters and applies pipeline/wrapper prefixes.
        """
        if transformer_trial_params is None:
            return ParamGridPrefixer().prepare_pipeline_grid(
                model_params=model_trial_params
            )
        return ParamGridPrefixer().prepare_wrapper_grid(
            model_params=model_trial_params,
            transformer_params=transformer_trial_params,
        )

    def build(
        self,
        trial: optuna.Trial,
        optuna_params: dict[str, Any],
        model_params: dict[str, Any],
        transformers: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Builds the full trial parameter grid.
        Includes transformer parameters only if a transformer is selected for optimization.
        """
        model_trial_params = self._build_model_trial_params(
            trial=trial,
            optuna_model_params=optuna_params,
            model_params=model_params,
        )
        chosen_transformer = trial.suggest_categorical(
            "transformation", list(transformers)
        )

        spec = TRANSFORMERS[chosen_transformer]
        if not spec.is_identity:
            transformer_trial_params = self._build_transformer_trial_params(
                trial=trial,
                transformation=chosen_transformer,
                transformers=transformers,
            )
            return self._merge_and_prefix(model_trial_params, transformer_trial_params)

        else:
            return self._merge_and_prefix(model_trial_params)
