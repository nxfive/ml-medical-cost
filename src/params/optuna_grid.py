from typing import Any

import optuna
from optuna.distributions import (BaseDistribution, CategoricalDistribution,
                                  FloatDistribution, IntDistribution)


class OptunaGrid:
    @staticmethod
    def create_optuna_space(
        optuna_params: dict[str, Any],
    ) -> dict[str, BaseDistribution]:
        """
        Converts a parameter dictionary into an Optuna search space.
        """
        space = {}
        for k, v in optuna_params.items():
            if isinstance(v, dict):
                if "min" in v and "max" in v:
                    space[k] = IntDistribution(
                        low=v["min"],
                        high=v["max"],
                        step=v.get("step", 1),
                    )
                elif "choices" in v:
                    space[k] = CategoricalDistribution(v["choices"])
                else:
                    raise ValueError(f"Invalid optuna param dict: {k}")

            elif isinstance(v, list):
                space[k] = CategoricalDistribution(v)
            else:
                raise ValueError(f"Invalid optuna param: {k}")
        return space

    @staticmethod
    def create_trial_params(
        trial: optuna.Trial, optuna_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        """
        Suggests trial parameters for a given Optuna trial based on the provided distributions.
        """
        trial_params = {}

        for name, distribution in optuna_space.items():
            if isinstance(distribution, IntDistribution):
                trial_params[name] = trial.suggest_int(
                    name, distribution.low, distribution.high, step=distribution.step
                )
            elif isinstance(distribution, FloatDistribution):
                trial_params[name] = trial.suggest_float(
                    name, distribution.low, distribution.high, step=distribution.step
                )
            elif isinstance(distribution, CategoricalDistribution):
                trial_params[name] = trial.suggest_categorical(
                    name, distribution.choices
                )
            else:
                raise TypeError(f"Unknown distribution type for {name}")

        return trial_params
