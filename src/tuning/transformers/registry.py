import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from .identity import IdentityTransformer
from .spec import TransformerSpec

# NOTE:
# Certain non-linear target transformations (e.g. sqrt, square) are intentionally excluded.
# Regression models (e.g. linear regression, kNN) may predict negative values, which makes
# inverse transformation invalid. Additionally, power-based transforms can excessively
# amplify target variance and outliers, leading to numerical instability and NaN/inf values.
# These decisions are backed by tests confirming the potential issues.

TRANSFORMERS: dict[str, TransformerSpec] = {
    "log": TransformerSpec(
        name="log",
        spec_class=FunctionTransformer,
        spec_params={
            "func": np.log,
            "inverse_func": np.exp,
        },
    ),
    "power": TransformerSpec(
        name="power",
        spec_class=PowerTransformer,
        spec_params={},
    ),
    "none": TransformerSpec(
        name="none",
        spec_class=IdentityTransformer,
        spec_params={},
    ),
}
