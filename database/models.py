from sqlalchemy import Column, Float, Integer, String

from .db import Base


class MedicalPrediction(Base):
    __tablename__ = "medical_predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    sex = Column(String)
    bmi = Column(Float)
    children = Column(Integer)
    smoker = Column(String)
    region = Column(String)
    predicted_charge = Column(Float)
