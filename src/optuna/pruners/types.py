from typing import TypedDict

from optuna.pruners import BasePruner


class PrunersDict(TypedDict):
    median: BasePruner
    hyperband: BasePruner
    sha: BasePruner
    patient: BasePruner
    nop: BasePruner
