#!/bin/bash

set -e

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "postgresql://$POSTGRES_USER:$POSTGRES_PASS@$POSTGRES_SERVICE:$POSTGRES_PORT/$POSTGRES_DB?sslmode=verify-full&sslrootcert=/certs/ca.crt&sslcert=/certs/client.crt&sslkey=/certs/client.key" \
    --artifacts-destination ./mlflow_artifacts \
    --cors-allowed-origins "https://mlflow.nxfive.pl" \
    --allowed-hosts "$MLFLOW_HOST,mlflow:5000,localhost:*,127.0.0.1:*"
