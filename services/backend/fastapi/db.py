from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.models import MedicalPrediction
from .schemas import MedicalCostDTO


class Database:

    def __init__(self, db: Session):
        self.db = db

    def create_record(self, data: dict, prediction: float):
        if prediction is None:
            raise ValueError("Prediction value cannot be None")

        try:
            dto = MedicalCostDTO(**data, predicted_charge=round(prediction, 2))
            record = MedicalPrediction(**dto.model_dump())
            self.db.add(record)
            self.db.commit()

        except ValidationError as e:
            raise ValueError(f"Error validating data: {e}")
        
        except SQLAlchemyError as e:
            self.db.rollback()
            raise RuntimeError(f"Error saving prediction: {e}")

    def create_records(self, data_list: list[dict], predictions: list[float]):
        if predictions is None:
            raise ValueError("Predictions values cannot be None")
        
        if len(data_list) != len(predictions):
            raise ValueError("Length of data_list and predictions must match")
        
        if len(predictions) == 0:
            raise ValueError("Predictions cannot be empty")

        try:
            dtos = [
                MedicalCostDTO(**data, predicted_charge=round(pred, 2))
                for data, pred in zip(data_list, predictions)
            ]
            records = [MedicalPrediction(**dto.model_dump()) for dto in dtos]
            self.db.add_all(records)
            self.db.commit()

        except ValidationError as e:
            raise ValueError(f"Error validating data: {e}")
        
        except SQLAlchemyError as e:
            self.db.rollback()
            raise RuntimeError(f"Error saving predictions: {e}")
