from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


class TransformerPipelineBuilder:
    @staticmethod
    def build(
        pipeline: Pipeline, transformer: BaseEstimator
    ) -> TransformedTargetRegressor:
        """
        Wraps a pipeline in a TransformedTargetRegressor using the provided transformer.
        """
        return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)
