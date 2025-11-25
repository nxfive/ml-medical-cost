from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class RegionEnum(str, Enum):
    northeast = "northeast"
    northwest = "northwest"
    southeast = "southeast"
    southwest = "southwest"


class SexEnum(str, Enum):
    female = "female"
    male = "male"


class SmokerEnum(str, Enum):
    yes = "yes"
    no = "no"


class MedicalCostFeatures(BaseModel):
    age: Annotated[int, Field(ge=18, le=100)]
    sex: SexEnum
    bmi: Annotated[float, Field(ge=10.0, le=50.0)]
    children: Annotated[int, Field(ge=0, le=15)]
    smoker: SmokerEnum
    region: RegionEnum


class MedicalCostDTO(MedicalCostFeatures):
    predicted_charge: float

    @classmethod
    def from_features(cls, features: MedicalCostFeatures, predicted_charge: float):
        return cls(**features.model_dump(), predicted_charge=predicted_charge)
