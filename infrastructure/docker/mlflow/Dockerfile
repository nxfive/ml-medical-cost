FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir mlflow psycopg2

WORKDIR /mlflow

COPY scripts/mlflow-entrypoint.sh ./entrypoint.sh


FROM python:3.12-slim

WORKDIR /mlflow

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 netcat-openbsd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/mlflow /usr/local/bin/mlflow
COPY --from=builder /mlflow/entrypoint.sh ./
RUN chmod +x /mlflow/entrypoint.sh

ENTRYPOINT ["/mlflow/entrypoint.sh"]