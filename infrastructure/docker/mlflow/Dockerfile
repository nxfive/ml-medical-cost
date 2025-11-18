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

COPY --from=builder /mlflow/entrypoint.sh ./
RUN chmod +x /mlflow/entrypoint.sh

RUN useradd -m user
USER user

ENTRYPOINT ["/mlflow/entrypoint.sh"]