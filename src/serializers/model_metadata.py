from typing import Any

from src.conf.schema import FeaturesConfig
from src.containers.model import ModelMetadata
from src.containers.results import StageResult

from .sanitizer import sanitize_params


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
            "params": sanitize_params(metadata.params) or {},
            "metrics": metadata.metrics,
        }

    @staticmethod
    def from_stage(result: StageResult, features: FeaturesConfig) -> ModelMetadata:
        return ModelMetadata(
            model_name=result.model_name,
            features=features,
            params=result.params,
            param_grid=result.param_grid,
            metrics=result.metrics,
        )
