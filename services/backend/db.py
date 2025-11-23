from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.models import MedicalPrediction


class Database:

    def __init__(self, db: Session):
        self.db = db

    def create_record(self, data: dict, prediction: float):
        try:
            record = MedicalPrediction(**data, predicted_charge=round(prediction, 2))
            self.db.add(record)
            self.db.commit()

        except SQLAlchemyError as e:
            self.db.rollback()
            print(f"Error saving prediction: {e}")
        
    def create_records(self, data_list: list[dict], predictions: list[float]):
        try:
            records = [
                MedicalPrediction(**data, predicted_charge=round(pred, 2))
                for data, pred in zip(data_list, predictions)
            ]
            self.db.add_all(records)
            self.db.commit()

        except SQLAlchemyError as e:
            self.db.rollback()
            print(f"Error saving predictions: {e}")
