import os
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from database.db import get_db

from .predict import PredictService
from .schemas import MedicalCostFeatures

app = FastAPI()

bento_host = os.getenv("BENTO_HOST", "127.0.0.1")
bento_port = os.getenv("BENTO_PORT", 3000)

ps = PredictService(bento_url=f"http://{bento_host}:{bento_port}")


@app.post("/predict")
def predict(features: MedicalCostFeatures, db: Annotated[Session, Depends(get_db)]) -> dict:
    result = ps.predict(db, features)
    return {"charges": result}


@app.post("/predict_many")
def predict_many(
    features_list: list[MedicalCostFeatures], db: Annotated[Session, Depends(get_db)]) -> dict:
    result = ps.predict_many(db, features_list)
    return {"charges": result}
