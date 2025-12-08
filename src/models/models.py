from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.conf.schema import FeaturesConfig


def get_preprocessor(
    preprocess_num_features: bool, cfg: FeaturesConfig
) -> ColumnTransformer:
    """
    Returns a ColumnTransformer with preprocessing steps based on preprocess_num_features config.
    """
    if preprocess_num_features:
        return ColumnTransformer(
            [
                ("num", StandardScaler(), cfg.numeric),
                ("bin", "passthrough", cfg.binary),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    cfg.categorical,
                ),
            ]
        )
    else:
        return ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    cfg.categorical,
                ),
                (
                    "rest",
                    "passthrough",
                    cfg.numeric + cfg.binary,
                ),
            ]
        )


def create_model_pipeline(
    preprocess_num_features: bool,
    features_cfg: FeaturesConfig,
    model: type[BaseEstimator] | None = None,
    model_instance: BaseEstimator | None = None,
) -> Pipeline:
    """
    Creates a pipeline for a given model class or instance with custom preprocessing.
    One of `model` or `model_instance` must be provided.
    """
    if model_instance is not None:
        return Pipeline(
            [
                (
                    "preprocessor",
                    get_preprocessor(preprocess_num_features, features_cfg),
                ),
                ("model", model_instance),
            ]
        )
    elif model is not None:
        return Pipeline(
            [
                (
                    "preprocessor",
                    get_preprocessor(preprocess_num_features, features_cfg),
                ),
                ("model", model()),
            ]
        )
    else:
        raise ValueError("Provide either `model` class or `model_instance`.")
