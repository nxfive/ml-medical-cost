import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


class Settings:
    ENV: str = os.getenv("ENV", "dev")
    GITHUB_SHA: str | None = os.getenv("GITHUB_SHA")
    LOCAL_MLFLOW_URI: str = "http://localhost:5000"
    MLFLOW_TRACKING_URI: str | None = os.getenv("MLFLOW_TRACKING_URI")

    @classmethod
    def commit_hash(cls) -> str:
        """
        Returns the short commit hash (7 characters).
        """
        if cls.GITHUB_SHA:
            return cls.GITHUB_SHA[:7]
        return ""

    @classmethod
    def experiment_name(cls) -> str:
        """
        Generates an MLflow experiment name based on the environment.
        """
        prefix = "ci-build" if cls.MLFLOW_TRACKING_URI else "local-test"
        identifier = cls.commit_hash() or datetime.now().strftime("%Y-%m-%d-%H%M")
        return f"{prefix}-{identifier}"

    @classmethod
    def tracking_uri(cls) -> str:
        """
        Returns the MLflow URI (remote or local).
        """
        return cls.MLFLOW_TRACKING_URI or cls.LOCAL_MLFLOW_URI
