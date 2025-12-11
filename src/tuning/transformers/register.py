import numpy as np
from sklearn.pipeline import FunctionTransformer

from src.tuning.types import TransformersDict

TRANSFORMERS: TransformersDict = {
    "log": FunctionTransformer(np.log, inverse_func=np.exp),
    "sqrt": FunctionTransformer(np.sqrt, inverse_func=np.square),
    "square": FunctionTransformer(np.square, inverse_func=np.sqrt),
    "none": None,
}
