import numpy as np
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import QuantileTransformer

from src.training.types import TransformersDict

TRANSFORMERS: TransformersDict = {
    "log": FunctionTransformer(np.log, inverse_func=np.exp),
    "quantile": QuantileTransformer(output_distribution="normal", n_quantiles=100),
    "none": None,
}
