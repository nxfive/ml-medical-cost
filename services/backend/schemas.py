from pydantic import BaseModel


class MedicalCostFeatures(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
