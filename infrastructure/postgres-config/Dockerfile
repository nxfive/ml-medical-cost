FROM alpine:latest

RUN apk add --no-cache bash

WORKDIR work/

COPY scripts/tls-postgres-config.sh /work/tls-postgres-config.sh

RUN chmod +x /work/tls-postgres-config.sh

ENTRYPOINT ["/work/tls-postgres-config.sh"]
