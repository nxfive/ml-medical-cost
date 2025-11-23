#!/bin/bash
set -e

export BACKEND_PORT=$(cat /run/secrets/BACKEND_PORT)

echo "Waiting for server..."

while ! curl -s http://$BACKEND_HOST:$BACKEND_PORT > /dev/null; do
  sleep 2
done

echo "Server ready"

cd client
streamlit run app.py --server.port 8501 --server.address 0.0.0.0