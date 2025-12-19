from sklearn.base import TransformerMixin

from src.tuning.transformers.registry import TRANSFORMERS


class TargetTransformerFactory:
    @staticmethod
    def create(transformation: str) -> TransformerMixin:
        """
        Creates an instance of a target transformer with given parameters.

        Only includes technical parameters that do not affect model results.
        """
        spec = TRANSFORMERS.get(transformation)
        if spec is None:
            raise ValueError(f"Transformer '{transformation}' not found")
        return spec.spec_class(**spec.spec_params)
