from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class ModelPipelineBuilder:
    @staticmethod
    def build(
        preprocessor: ColumnTransformer,
        model: type[BaseEstimator],
    ) -> Pipeline:
        """
        Builds a pipeline consisting of a preprocessor and a model.
        """
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model()),
            ]
        )
