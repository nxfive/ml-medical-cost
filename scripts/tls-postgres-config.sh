#!/bin/bash
set -eu

PG_SSL_ADDRESS=$(cat /run/secrets/PG_SSL_ADDRESS)
PG_MOUNT=$(cat /run/secrets/PG_MOUNT)

PATHS=("/config/backend" "/config/mlflow")

for p in "${PATHS[@]}"; do      
  echo "Create ssl/tls configuration file for postgres-${p#/config/}..."
  echo "hostssl all all ${PG_SSL_ADDRESS} cert" > "${p}/pg_hba.conf"
  
  cat > "${p}/postgresql.conf" <<EOF
listen_addresses = '*'
port = 5432
ssl = on
ssl_cert_file = '${PG_MOUNT}/certs/server.crt'
ssl_key_file = '${PG_MOUNT}/certs/server.key'
ssl_ca_file = '${PG_MOUNT}/certs/ca.crt'
EOF

  echo "Done."

done