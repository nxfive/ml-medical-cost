from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class ModelResultDict(TypedDict):
    model_name: str
    param_grid: dict[str, list]
    folds_scores_mean: float
    metrics: dict[str, float]
    transformation: str | None = None


@dataclass
class ModelResult:
    model_name: str
    param_grid: dict[str, list]
    folds_scores_mean: float
    metrics: dict[str, float]
    transformation: str | None = None

    def to_dict(self) -> ModelResultDict:
        return {
            "model_name": self.model_name,
            "param_grid": self.param_grid,
            "transformation": self.transformation,
            "folds_scores_mean": self.folds_scores_mean,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, result: ModelResultDict) -> ModelResult:
        return ModelResult(
            model_name=result["model_name"],
            param_grid=result["param_grid"],
            transformation=result["transformation"],
            folds_scores_mean=result["folds_scores_mean"],
            metrics=result["metrics"],
        )
