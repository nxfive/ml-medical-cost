import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

ENV = os.getenv("ENV", "dev")

if ENV == "prod":
    DATABASE_URL = os.getenv("DATABASE_URL_PROD")
else:
    DATABASE_URL = os.getenv("DATABASE_URL_LOCAL")
