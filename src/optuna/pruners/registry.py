import optuna

from .types import PrunersDict

PRUNERS: PrunersDict = {
    "median": optuna.pruners.MedianPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "nop": optuna.pruners.NopPruner,
    "sha": optuna.pruners.SuccessiveHalvingPruner,
    "patient": optuna.pruners.PatientPruner,
}
