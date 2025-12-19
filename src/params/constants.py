from enum import Enum


class Prefixes(str, Enum):
    PIPELINE_MODEL = "model__"
    WRAPPER_REGRESSOR = "regressor__"
    WRAPPER_TRANSFORMER = "transformer__"
