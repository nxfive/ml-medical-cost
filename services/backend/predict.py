import pandas as pd
import requests
from sqlalchemy.orm import Session

from src.features.features import convert_features_type
from .schemas import MedicalCostFeatures
from .db import Database


class PredictService:

    def __init__(self, bento_url: str, db: Session):
        self.bento_url = bento_url
        self.database = Database(db)

    def predict(self, features: MedicalCostFeatures) -> float:
        payload = features.model_dump()
        df = convert_features_type(pd.DataFrame([payload]))
        converted = df.to_dict(orient="records")[0]

        response = requests.post(
            f"{self.bento_url}/predict", json={"input_data": converted}, timeout=10
        )
        response.raise_for_status()
        prediction = response.json()["charges"]

        self.database.create_record(data=payload, predicted_charge=prediction)
        return prediction

    def predict_many(self, features_list: list[MedicalCostFeatures]) -> list[float]:
        payload = [f.model_dump() for f in features_list]
        df = convert_features_type(pd.DataFrame(payload))
        converted = df.to_dict(orient="records")

        response = requests.post(
            f"{self.bento_url}/predict_multiple",
            json={"input_data": converted},
            timeout=10,
        )
        response.raise_for_status()
        predictions = response.json()["charges"]

        self.database.create_records(data_list=payload, predictions=predictions)
        return predictions
