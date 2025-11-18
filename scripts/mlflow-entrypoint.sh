#!/bin/bash
set -e

POSTGRES_USER=$(cat /run/secrets/POSTGRES_USER)
POSTGRES_PASS=$(cat /run/secrets/POSTGRES_PASS)
POSTGRES_PORT=$(cat /run/secrets/POSTGRES_PORT)
POSTGRES_DB=$(cat /run/secrets/POSTGRES_DB)
MLFLOW_HOST=$(cat /run/secrets/MLFLOW_HOST)

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "postgresql://$POSTGRES_USER:$POSTGRES_PASS@$POSTGRES_SERVICE:$POSTGRES_PORT/$POSTGRES_DB?sslmode=verify-full&sslrootcert=/certs/ca.crt&sslcert=/certs/client.crt&sslkey=/certs/client.key" \
    --artifacts-destination ./mlflow_artifacts \
    --cors-allowed-origins "https://mlflow.nxfive.pl" \
    --allowed-hosts "$MLFLOW_HOST,mlflow:5000,localhost:*,127.0.0.1:*"
