from datetime import datetime

import bentoml
from sklearn.base import BaseEstimator

from src.mlflow.service import MLFLOW_REGISTER_NAME, MLflowService
from src.logger.setup import logger

BENTO_MODEL_NAME = "medical_regressor"


def load_latest_model(model_name: str, service: MLflowService) -> BaseEstimator:
    """
    Loads the latest registered version of a model from MLflow.
    """
    latest_version = service.get_latest_model_version(model_name)
    return service.load_model(model_name, version=latest_version)


def register_bento_model(
    model_name: str, bento_name: str, service: MLflowService
) -> None:
    """
    Registers the latest version of a model from MLflow in BentoML.
    """
    timestamp_tag = datetime.now().strftime("%Y%m%d%H%M%S")
    model = load_latest_model(model_name, service)

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
    logger.info(f"Model registered in BentoML: {bento_model.tag}")


if __name__ == "__main__":
    service = MLflowService()
    service.setup(create_experiment=False)
    register_bento_model(
        model_name=MLFLOW_REGISTER_NAME, bento_name=BENTO_MODEL_NAME, service=service
    )
