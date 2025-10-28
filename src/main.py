from src.data.pipeline import data_pipeline
from src.models.pipeline import models_pipeline, optuna_pipeline
from src.models.mlflow_logging import setup_mlflow


def main():
    setup_mlflow()

    X_train, X_test, y_train, y_test = data_pipeline()

    best_model_info = models_pipeline(X_train, X_test, y_train, y_test)

    optuna_pipeline(best_model_info, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
