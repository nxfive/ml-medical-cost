import inspect
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModelPipelineBuilder:
    @staticmethod
    def build(
        preprocessor: ColumnTransformer,
        model: type[BaseEstimator],
        params: dict[str, Any] | None = None,
    ) -> Pipeline:
        """
        Builds a pipeline consisting of a preprocessor and a model.
        """
        params = params or {}

        signature = inspect.signature(model.__init__)
        if "random_state" in signature.parameters:
            params.setdefault("random_state", 42)

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model()),
            ]
        )
