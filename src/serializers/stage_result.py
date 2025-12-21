from typing import Any

from sklearn.base import BaseEstimator

from src.containers.results import RunResult, StageResult

from .sanitizer import sanitize_params


class StageResultSerializer:
    @staticmethod
    def to_metrics(stage_result: StageResult) -> dict[str, Any]:
        return {
            "model_name": stage_result.model_name,
            "params": sanitize_params(stage_result.params),
            "param_grid": stage_result.param_grid,
            "folds_scores": [float(fold) for fold in stage_result.folds_scores],
            "folds_scores_mean": float(stage_result.folds_scores_mean),
            "metrics": stage_result.metrics,
            "transformation": stage_result.transformation,
        }

    @staticmethod
    def from_loader(metrics: dict[str, Any], pipeline: BaseEstimator) -> StageResult:
        return StageResult(
            model_name=metrics["model_name"],
            params=metrics["params"],
            param_grid=metrics["param_grid"],
            folds_scores=metrics["folds_scores"],
            folds_scores_mean=metrics["folds_scores_mean"],
            metrics=metrics["metrics"],
            transformation=metrics["transformation"],
            estimator=pipeline,
        )

    @staticmethod
    def from_stage(
        result: RunResult, metrics: dict[str, Any], model_name: str
    ) -> StageResult:
        return StageResult(
            model_name=model_name,
            estimator=result.runner_result.trained,
            params=result.runner_result.params,
            param_grid=result.param_grid,
            folds_scores=result.runner_result.folds_scores,
            folds_scores_mean=result.runner_result.folds_scores_mean,
            transformation=result.transformation,
            metrics=metrics,
        )
