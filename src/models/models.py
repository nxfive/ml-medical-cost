from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.models.utils import pipeline_config


def get_preprocessor(model: type) -> ColumnTransformer:
    """
    Returns a ColumnTransformer with preprocessing steps based on preprocess_num_features config.
    """
    features = pipeline_config.features
    if pipeline_config.models[model.__name__].preprocess_num_features:
        return ColumnTransformer([
            ("num", StandardScaler(), features.numeric),
            ("bin", "passthrough", features.binary),
            ("cat", OneHotEncoder(
                drop="first", 
                sparse_output=False, 
                handle_unknown="ignore"
            ), features.categorical)
        ])
    elif model.__name__ in ["DecisionTreeRegressor", "RandomForestRegressor"]:
        return ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), features.categorical),
            ("rest", "passthrough", features.numeric + features.binary),
        ])


def create_model_pipeline(model: type[BaseEstimator] = None, model_instance: BaseEstimator = None) -> Pipeline:
    """
    Creates a pipeline for a given model class or instance with custom preprocessing.
    One of `model` or `model_instance` must be provided.
    """
    if model_instance is not None:
        return Pipeline([
            ("preprocessor", get_preprocessor(type(model_instance))),
            ("model", model_instance)
        ])
    elif model is not None:
        return Pipeline([
            ("preprocessor", get_preprocessor(model)),
            ("model", model())
        ])
    else:
        raise ValueError("Provide either `model` class or `model_instance`.")