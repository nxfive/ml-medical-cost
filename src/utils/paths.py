import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "model_config.yml")
OPTUNA_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "optuna.yml")

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")