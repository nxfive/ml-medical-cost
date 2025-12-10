from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.conf.schema import FeaturesConfig


class PreprocessorBuilder:
    @staticmethod
    def build(preprocess_num_features: bool, cfg: FeaturesConfig) -> ColumnTransformer:
        """
        Builds a ColumnTransformer for feature preprocessing.
        """
        if preprocess_num_features:
            return ColumnTransformer(
                [
                    ("num", StandardScaler(), cfg.numeric),
                    ("bin", "passthrough", cfg.binary),
                    (
                        "cat",
                        OneHotEncoder(
                            drop="first", sparse_output=False, handle_unknown="ignore"
                        ),
                        cfg.categorical,
                    ),
                ]
            )
        else:
            return ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore"),
                        cfg.categorical,
                    ),
                    (
                        "rest",
                        "passthrough",
                        cfg.numeric + cfg.binary,
                    ),
                ]
            )
