from dataclasses import dataclass


@dataclass
class SplitMetrics:
    r2: float
    mae: float
    rmse: float


@dataclass
class AllMetrics:
    train: SplitMetrics
    test: SplitMetrics
