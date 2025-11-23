#!/bin/bash
set -e

export DATABASE_URL_PROD=$(cat /run/secrets/DATABASE_URL_PROD)
export MLFLOW_TRACKING_URI=$(cat /run/secrets/MLFLOW_TRACKING_PROD)
export BENTO_PORT=$(cat /run/secrets/BENTO_PORT)

DB_HOST=$(cat /run/secrets/BACKEND_POSTGRES_HOST)
DB_PORT=$(cat /run/secrets/POSTGRES_PORT)

until nc -z $DB_HOST $DB_PORT; do
  echo "Waiting for database..."
  sleep 2
done

echo "Create DB migrations..."
alembic upgrade head

uvicorn server.main:app --host 0.0.0.0 --port 8000