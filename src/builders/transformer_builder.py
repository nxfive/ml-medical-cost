from typing import Any

from sklearn.base import BaseEstimator

from src.models.transformers import get_transformer


class TransformerBuilder:
    @staticmethod
    def build(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
        """
        Builds a target transformer instance.
        """
        transformer = get_transformer(name=name)

        if params:
            transformer.set_params(**params)

        return transformer
