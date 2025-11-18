#!/bin/bash
set -e

export DATABASE_URL_PROD=$(cat /run/secrets/DATABASE_URL_PROD)
export MLFLOW_TRACKING_URI=$(cat /run/secrets/MLFLOW_TRACKING_PROD)

DB_HOST=$(cat /run/secrets/BACKEND_POSTGRES_HOST)
DB_PORT=$(cat /run/secrets/POSTGRES_PORT)

until nc -z $DB_HOST $DB_PORT; do
  echo "Waiting for database..."
  sleep 2
done

echo "Create DB migrations..."
alembic upgrade head

echo "Build BentoML service..."
cd server
bentoml build

cd ..
echo "Register model from MLflow"
python -m server.bento

echo "Start BentoML service..."
python -m bentoml serve server.service:MedicalRegressorService
