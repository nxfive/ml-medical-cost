#!/bin/bash

set -e

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASS@$POSTGRES_SERVICE:$POSTGRES_PORT/$POSTGRES_DB \
    --default-artifact-root /mlflow/artifacts \
    --cors-allowed-origins "https://mlflow.nxfive.pl" \
    --allowed-hosts $MLFLOW_HOST
