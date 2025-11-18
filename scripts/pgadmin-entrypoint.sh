#!/bin/bash
set -e

export PGADMIN_DEFAULT_EMAIL=$(cat /run/secrets/PGADMIN_EMAIL)
export PGADMIN_DEFAULT_PASSWORD=$(cat /run/secrets/PGADMIN_PASSWORD)

PGADMIN_EMAIL_PATH=$(cat /run/secrets/PGADMIN_EMAIL_PATH)

mkdir -p /var/lib/pgadmin/storage/${PGADMIN_EMAIL_PATH}/certs/backend/
mkdir -p /var/lib/pgadmin/storage/${PGADMIN_EMAIL_PATH}/certs/mlflow/

chown -R 5050:5050 /var/lib/pgadmin/storage

exec /entrypoint.sh "$@"
