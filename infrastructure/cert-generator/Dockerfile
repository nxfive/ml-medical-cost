FROM alpine:3.19

RUN apk add --no-cache openssl bash

WORKDIR /work

COPY scripts/generate-certs.sh /work/generate-certs.sh

RUN chmod +x /work/generate-certs.sh

ENTRYPOINT ["/work/generate-certs.sh"]