import datetime
from typing import Any


class ModelMetadata:
    def __init__(
        self,
        model_name: str,
        features: dict[str, list],
        params: dict[str, list],
        metrics: dict[str, float],
    ):
        self.model_name = model_name
        self.features = features
        self.params = params
        self.metrics = metrics
        self.version = "1.0"
        self.date_trained = datetime.today().strftime("%Y-%m-%d")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "date_trained": self.date_trained,
            "features_processed": {
                "cat_features": list(self.features.categorical),
                "num_features": list(self.features.numeric),
                "bin_features": list(self.features.binary),
            },
            "params": self.params or {},
            "metrics": self.metrics,
        }
