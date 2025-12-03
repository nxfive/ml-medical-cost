from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

MODELS_MAPPING: dict[str, tuple[type[BaseEstimator], str]] = {
    "RandomForestRegressor": (RandomForestRegressor, "rf"),
    "LinearRegression": (LinearRegression, "linear"),
    "DecisionTreeRegressor": (DecisionTreeRegressor, "tree"),
    "KNeighborsRegressor": (KNeighborsRegressor, "knn"),
}


def get_model_class_and_short(name: str) -> tuple[type[BaseEstimator], str | None]:
    """
    Map model name or alias to actual model class.
    """
    for full_name, (model_class, short_alias) in MODELS_MAPPING.items():
        if name == full_name:
            return model_class, short_alias
        elif name == short_alias:
            return model_class, None
    else:
        raise ValueError(f"Model '{name}' not found in MODELS_MAPPING")
