from unittest import mock

import numpy as np
import pytest

from services.backend.bento.service import MedicalRegressorService


@pytest.mark.parametrize(
    "input_data, prediction",
    [
        (
            {
                "age": 30.0,
                "sex": 0.0,
                "bmi": 25.0,
                "children": 0.0,
                "smoker": 0.0,
                "region": "northwest"
            },
            [1234.56],
        ),
        (
            [
                {
                    "age": 40.0,
                    "sex": 1.0,
                    "bmi": 22.0,
                    "children": 1.0,
                    "smoker": 0.0,
                    "region": "southwest"
                },
                {
                    "age": 35.0,
                    "sex": 0.0,
                    "bmi": 28.0,
                    "children": 2.0,
                    "smoker": 1.0,
                    "region": "northwest"
                },
            ],
            [1000.0, 2000.0],
        ),
    ],
)
def test_medical_service_predict(input_data, prediction):
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array(prediction)

    service = MedicalRegressorService()
    service.model = fake_model

    if isinstance(input_data, list):
        result = service.predict_multiple(input_data)
        assert result == {"charges": prediction}
    else:
        result = service.predict(input_data)
        assert result == {"charges": prediction[0]}

    fake_model.predict.assert_called_once()
