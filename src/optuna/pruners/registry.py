import optuna
from optuna.pruners import BasePruner

PRUNERS: dict[str, BasePruner] = {
    "median": optuna.pruners.MedianPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
    "nop": optuna.pruners.NopPruner,
    "sha": optuna.pruners.SuccessiveHalvingPruner,
    "patient": optuna.pruners.PatientPruner,
}
