from dataclasses import dataclass

from src.tuning.types import RunnerResult


@dataclass
class TrainResult:
    runner_result: RunnerResult
    param_grid: dict[str, list]
    transformation: str | None
