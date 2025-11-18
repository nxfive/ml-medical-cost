from datetime import datetime

import bentoml
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator

from src.models.mlflow_logging import setup_mlflow


BENTO_MODEL_NAME = "medical_regressor"
MODEL_NAME = "MedicalRegressor"


def get_model_version(model_name: str) -> BaseEstimator:
    """
    Fetches the latest version of a registered MLflow model and loads it as a Scikit-learn model.
    """
    client = MlflowClient()
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    if not all_versions:
        raise ValueError(f"No versions found for model {model_name}")
    
    latest_version = max(int(v.version) for v in all_versions)
    
    return mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")


def register_bento_model(model_name: str, bento_name: str) -> None:
    """
    Loads the latest MLflow model version and registers it in the BentoML.
    """
    timestamp_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    model = get_model_version(model_name)
    
    bento_model = bentoml.sklearn.save_model(
        name=f"{bento_name}:{timestamp_tag}",
        model=model,
        signatures={"predict": {"batchable": True, "batch_dim": 0}},
        metadata={
            "description": f"Model {model_name} for medical cost prediction",
            "source": "MLflow Registry",
            "model_name": model_name,
        },
    )
    print(f"Model registered in BentoML: {bento_model.tag}")


if __name__ == "__main__":
    setup_mlflow(create_experiment=False)
    register_bento_model(model_name=MODEL_NAME, bento_name=BENTO_MODEL_NAME)
