from typing import List
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from services.backend.service import (MedicalCostFeatures,
                                      MedicalRegressorService)


@pytest.fixture
def client():
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array([1234.56])

    converted_df = pd.DataFrame(
        [[30.0, 0.0, 25.0, 0.0, 0.0, 0.0]],
        columns=["age", "sex", "bmi", "children", "smoker", "region"],
    )

    with mock.patch(
        "services.backend.service.convert_features_type", return_value=converted_df
    ):
        service = MedicalRegressorService()
        service.model = fake_model

        app = FastAPI()

        @app.post("/predict")
        def predict(features: MedicalCostFeatures):
            return service.predict(features)

        return TestClient(app)


def test_predict_endpoint(client):
    payload = {
        "age": 30,
        "sex": "male",
        "bmi": 25.0,
        "children": 0,
        "smoker": "no",
        "region": "northwest",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"charges": 1234.56}


@pytest.fixture
def client_many():
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array([1000.0, 2000.0])

    converted_df = pd.DataFrame(
        [
            [40.0, 1.0, 22.0, 1.0, 0.0, 1.0],
            [35.0, 0.0, 28.0, 2.0, 1.0, 0.0],
        ],
        columns=["age", "sex", "bmi", "children", "smoker", "region"],
    )

    with mock.patch(
        "services.backend.service.convert_features_type", return_value=converted_df
    ):
        service = MedicalRegressorService()
        service.model = fake_model

        app = FastAPI()

        @app.post("/predict_multiple")
        def predict_multiple(features_list: List[MedicalCostFeatures]):
            return service.predict_multiple(features_list)

        return TestClient(app)


def test_predict_multiple_endpoint(client_many):
    payload = [
        {
            "age": 40,
            "sex": "female",
            "bmi": 22.0,
            "children": 1,
            "smoker": "no",
            "region": "southwest",
        },
        {
            "age": 35,
            "sex": "male",
            "bmi": 28.0,
            "children": 2,
            "smoker": "yes",
            "region": "northwest",
        },
    ]

    response = client_many.post("/predict_multiple", json=payload)
    assert response.status_code == 200
    assert response.json() == {"charges": [1000.0, 2000.0]}
