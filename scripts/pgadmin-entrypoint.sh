#!/bin/bash

set -eu

mkdir -p /var/lib/pgadmin/storage/${PGADMIN_EMAIL_PATH}/certs/backend/
mkdir -p /var/lib/pgadmin/storage/${PGADMIN_EMAIL_PATH}/certs/mlflow/

chown -R 5050:5050 /var/lib/pgadmin/storage

exec /entrypoint.sh "$@"
