#!/bin/bash
set -e

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
