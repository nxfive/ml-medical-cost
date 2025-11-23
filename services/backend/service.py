import bentoml
import pandas as pd

with bentoml.importing():
    from src.features.features import convert_features_type
    from .schemas import MedicalCostFeatures


@bentoml.service(name="medical_regressor_service")
class MedicalRegressorService:

    def __init__(self):
        self.model = bentoml.models.get("medical_regressor:latest").load_model()

    @bentoml.api()
    def predict(self, medical_features: MedicalCostFeatures):
        medical_features_df = pd.DataFrame([medical_features.model_dump()])
        medical_features_df = convert_features_type(medical_features_df)

        prediction = self.model.predict(medical_features_df)[0]

        return {"charges": prediction}

    @bentoml.api()
    def predict_multiple(self, medical_features_list: list[MedicalCostFeatures]):
        medical_features_dicts = [
            features.model_dump() for features in medical_features_list
        ]
        medical_features_df = pd.DataFrame(medical_features_dicts)
        medical_features_df = convert_features_type(medical_features_df)

        predictions = self.model.predict(medical_features_df)

        return {"charges": predictions.tolist()}
