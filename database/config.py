import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

ENV = os.getenv("ENV", "dev")

DATABASES = {
    "prod": os.getenv("DATABASE_URL_PROD"),
    "dev": os.getenv("DATABASE_URL_LOCAL"),
    "test": os.getenv("DATABASE_URL_TEST"),
}

DATABASE_URL = DATABASES.get(ENV)

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///:memory:"
    print(f"DATABASE_URL not set, using fallback: {DATABASE_URL}")
