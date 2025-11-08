from unittest import mock

import numpy as np
import pandas as pd
import pytest

from services.backend.service import MedicalCostFeatures, MedicalRegressorService


@pytest.mark.parametrize(
    "data, prediction, convert",
    [
        (
            MedicalCostFeatures(
                age=30,
                sex="male",
                bmi=25.0,
                children=0,
                smoker="no",
                region="northwest",
            ),
            [1234.56],
            [[30.0, 0.0, 25.0, 0.0, 0.0, 0.0]],
        ),
        (
            [
                MedicalCostFeatures(
                    age=40,
                    sex="female",
                    bmi=22.0,
                    children=1,
                    smoker="no",
                    region="southwest",
                ),
                MedicalCostFeatures(
                    age=35,
                    sex="male",
                    bmi=28.0,
                    children=2,
                    smoker="yes",
                    region="northwest",
                ),
            ],
            [1000.0, 2000.0],
            [
                [40.0, 1.0, 22.0, 1.0, 0.0, 1.0],
                [35.0, 0.0, 28.0, 2.0, 1.0, 0.0],
            ],
        ),
    ],
)
def test_medical_service_predict(data, prediction, convert, fake_db):
    fake_model = mock.Mock()
    fake_model.predict.return_value = np.array(prediction)

    converted_df = pd.DataFrame(
        convert, columns=["age", "sex", "bmi", "children", "smoker", "region"]
    )

    with (
        mock.patch(
            "services.backend.service.convert_features_type", return_value=converted_df
        ),
        mock.patch("services.backend.service.get_db", return_value=iter([fake_db])),
    ):
        service = MedicalRegressorService()
        service.model = fake_model

        if isinstance(data, list):
            result = service.predict_multiple(data)
            assert result == {"charges": prediction}
            assert fake_db.add.call_count == len(data)
            assert fake_db.commit.call_count == len(data)
        else:
            result = service.predict(data)
            assert result == {"charges": prediction[0]}
            fake_db.add.assert_called_once()
            fake_db.commit.assert_called_once()

        fake_model.predict.assert_called_once()
