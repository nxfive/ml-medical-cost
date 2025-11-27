from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_preprocessor(model: type, cfg: DictConfig) -> ColumnTransformer:
    """
    Returns a ColumnTransformer with preprocessing steps based on preprocess_num_features config.
    """
    if cfg.model.preprocess_num_features:
        return ColumnTransformer(
            [
                ("num", StandardScaler(), list(cfg.features.numeric)),
                ("bin", "passthrough", list(cfg.features.binary)),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    list(cfg.features.categorical),
                ),
            ]
        )
    elif model.__name__ in ["DecisionTreeRegressor", "RandomForestRegressor"]:
        return ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(cfg.features.categorical),
                ),
                (
                    "rest",
                    "passthrough",
                    list(cfg.features.numeric) + list(cfg.features.binary),
                ),
            ]
        )


def create_model_pipeline(
    cfg: DictConfig,
    model: type[BaseEstimator] = None,
    model_instance: BaseEstimator = None,
) -> Pipeline:
    """
    Creates a pipeline for a given model class or instance with custom preprocessing.
    One of `model` or `model_instance` must be provided.
    """
    if model_instance is not None:
        return Pipeline(
            [
                ("preprocessor", get_preprocessor(type(model_instance), cfg)),
                ("model", model_instance),
            ]
        )
    elif model is not None:
        return Pipeline(
            [("preprocessor", get_preprocessor(model, cfg)), ("model", model())]
        )
    else:
        raise ValueError("Provide either `model` class or `model_instance`.")
