from typing import Any

from src.containers.model import ModelMetadata


class ModelMetadataSerializer:
    @staticmethod
    def to_dict(metadata: ModelMetadata) -> dict[str, Any]:
        return {
            "model_name": metadata.model_name,
            "version": metadata.version,
            "date_trained": metadata.date_trained,
            "features_processed": {
                "cat_features": metadata.features.categorical,
                "num_features": metadata.features.numeric,
                "bin_features": metadata.features.binary,
            },
            "params": metadata.params or {},
            "metrics": metadata.metrics,
        }
