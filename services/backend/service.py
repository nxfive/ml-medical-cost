import bentoml
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import MedicalPrediction
from src.features.features import convert_features_type


class MedicalCostFeatures(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@bentoml.service(name="medical_regressor_service")
class MedicalRegressorService:

    def __init__(self):
        self.model = bentoml.models.get("medical_regressor:latest").load_model()

    def save_prediction_to_db(
        self, db: Session, prediction, features: MedicalCostFeatures
    ):
        new_prediction = MedicalPrediction(
            age=features.age,
            sex=features.sex,
            bmi=features.bmi,
            children=features.children,
            smoker=features.smoker,
            region=features.region,
            predicted_charge=float(prediction),
        )
        try:
            db.add(new_prediction)
            db.commit()
            db.refresh(new_prediction)
            return new_prediction
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error saving prediction: {e}")
            return None

    @bentoml.api()
    def predict(self, medical_features: MedicalCostFeatures):
        medical_features_df = pd.DataFrame([medical_features.model_dump()])
        medical_features_df = convert_features_type(medical_features_df)

        prediction = self.model.predict(medical_features_df)[0]

        db = next(get_db())
        self.save_prediction_to_db(db, prediction, medical_features)

        return {"charges": prediction}

    @bentoml.api()
    def predict_multiple(self, medical_features_list: list[MedicalCostFeatures]):
        medical_features_dicts = [
            features.model_dump() for features in medical_features_list
        ]
        medical_features_df = pd.DataFrame(medical_features_dicts)
        medical_features_df = convert_features_type(medical_features_df)

        predictions = self.model.predict(medical_features_df)

        db = next(get_db())
        for prediction, features in zip(predictions, medical_features_list):
            self.save_prediction_to_db(db, prediction, features)

        return {"charges": predictions.tolist()}
