#!/bin/bash
set -eu

# Generate certs for each service
SERVICE="${SERVICE:-postgres}"   
DAYS="${DAYS:-365}"
OUT_DIR="/out"
CA_KEY="/ca/ca.key"

if [ ! -f "$CA_KEY" ]; then
  echo "ERROR: CA key not found at $CA_KEY"
  exit 2
fi

mkdir -p "$OUT_DIR/server" "$OUT_DIR/client"
cd "$OUT_DIR"

if [ ! -f "ca.crt" ]; then
  openssl req -x509 -new -nodes -key "$CA_KEY" -sha256 -days 3650 -out ca.crt -subj "/CN=${SERVICE}-CA"
fi

echo "Generating server key/cert for postgres-${SERVICE}..."

openssl genrsa -out server/server.key 2048
openssl req -new -key server/server.key -out server/server.csr -subj "/CN=postgres-${SERVICE}"
openssl x509 -req -in server/server.csr -CA ca.crt -CAkey "$CA_KEY" -CAcreateserial -out server/server.crt -days "$DAYS" -sha256

rm -f server/server.csr

chmod 600 server/server.key
chmod 644 server/server.crt
chown -R 999:999 server    # postgres user

echo "Generating key/cert for $SERVICE client..."

openssl genrsa -out "client/client.key" 2048
openssl req -new -key "client/client.key" -out "client/client.csr" -subj "/CN=${SERVICE}-user"
openssl x509 -req -in "client/client.csr" -CA ca.crt -CAkey "$CA_KEY" -CAcreateserial -out "client/client.crt" -days "$DAYS" -sha256

rm -f "client/client.csr"

chmod 600 "client/client.key"
chmod 644 "client/client.crt"

echo "Done. Files in $OUT_DIR:"
ls -R "$OUT_DIR"

# Distribute certs to the specific service
SRC="/out"

echo "Copying server certs to $DST_SERVER"
mkdir -p "$DST_SERVER"
cp -a "$SRC/server/"* "$DST_SERVER/" 
cp -a "$SRC/ca.crt" "$DST_SERVER/"

echo "Copying client-$SERVICE cert to $DST_CLIENT"
mkdir -p "$DST_CLIENT"
cp -a "$SRC/client/"* "$DST_CLIENT/" 
cp -a "$SRC/ca.crt" "$DST_CLIENT/"

echo "Copying client-pgadmin cert to $DST_PGADMIN"
mkdir -p "$DST_PGADMIN"
cp -a "$SRC/client/"* "$DST_PGADMIN/" 
cp -a "$SRC/ca.crt" "$DST_PGADMIN/"

chown -R 5050:5050 "$DST_PGADMIN/"

echo "Done. Listing:"
ls -R "$DST_SERVER" 
ls -R "$DST_CLIENT" 
ls -R "$DST_PGADMIN"
