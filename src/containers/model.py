from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ModelMetadata:
    model_name: str
    features: dict[str, list[str]]
    params: dict[str, Any]
    param_grid: dict[str, Any]
    metrics: dict[str, float]
    version: str = "1.0"
    date_trained: str = field(
        default_factory=lambda: datetime.today().strftime("%Y-%m-%d")
    )
