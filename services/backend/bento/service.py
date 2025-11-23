import bentoml
import pandas as pd


@bentoml.service(name="medical_regressor_service")
class MedicalRegressorService:

    def __init__(self):
        self.model = bentoml.models.get("medical_regressor:latest").load_model()

    @bentoml.api()
    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        prediction = self.model.predict(df)

        return {"charges": prediction[0]}

    @bentoml.api()
    def predict_multiple(self, input_data: list[dict]):
        df = pd.DataFrame(input_data)
        predictions = self.model.predict(df)

        return {"charges": predictions.tolist()}
