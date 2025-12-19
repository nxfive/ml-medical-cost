from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from .spec import ModelSpec

MODELS: dict[str, ModelSpec] = {
    "RandomForestRegressor": ModelSpec(RandomForestRegressor, "rf"),
    "LinearRegression": ModelSpec(LinearRegression, "linear"),
    "DecisionTreeRegressor": ModelSpec(DecisionTreeRegressor, "tree"),
    "KNeighborsRegressor": ModelSpec(KNeighborsRegressor, "knn"),
}
