#!/bin/bash
set -eu

export POSTGRES_USER=$(cat /run/secrets/BACKEND_POSTGRES_USER)
export POSTGRES_PASSWORD=$(cat /run/secrets/BACKEND_POSTGRES_PASSWORD)
export POSTGRES_DB=$(cat /run/secrets/BACKEND_POSTGRES_DB)

echo "[INFO] Postgres env loaded from secrets."

exec docker-entrypoint.sh postgres
