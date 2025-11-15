FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow psycopg2

WORKDIR /mlflow

COPY scripts/mlflow-entrypoint.sh ./entrypoint.sh

ENTRYPOINT ["/bin/bash", "/mlflow/entrypoint.sh"]