from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class ModelSpec:
    model_class: type[BaseEstimator]
    alias: str | None = None
