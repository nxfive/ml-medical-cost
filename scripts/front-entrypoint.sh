#!/bin/bash
set -e

BENTO_PORT=$(cat /run/secrets/BACKEND_PORT)

echo "Waiting for server..."

while ! curl -s http://$BENTO_HOST:$BENTO_PORT > /dev/null; do
  sleep 2
done

echo "Server ready"

cd client
streamlit run app.py --server.port 8501 --server.address 0.0.0.0