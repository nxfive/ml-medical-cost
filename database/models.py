from sqlalchemy import Column, Float, Integer, String

from .db import Base


class MedicalPrediction(Base):
    __tablename__ = "medical_predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    bmi = Column(Float, nullable=False)
    children = Column(Integer, nullable=False)
    smoker = Column(String, nullable=False)
    region = Column(String, nullable=False)
    predicted_charge = Column(Float, nullable=False)
