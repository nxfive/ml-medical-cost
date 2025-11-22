FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /backend

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        libffi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY pyproject.toml uv.lock alembic.ini ./
COPY alembic ./alembic
COPY database ./database
COPY services/backend/ ./server/
COPY src ./src/
COPY scripts/back-entrypoint.sh ./entrypoint.sh

RUN pip install --no-cache-dir uv && \
    uv sync --group server --frozen --no-dev


FROM python:3.12-slim

WORKDIR /backend

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libffi-dev \
        libpq5 \
        netcat-openbsd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /backend/.venv ./.venv
COPY --from=builder /backend/alembic ./alembic
COPY --from=builder /backend/database ./database
COPY --from=builder /backend/server ./server
COPY --from=builder /backend/src ./src
COPY --from=builder /backend/alembic.ini ./ 
COPY --from=builder /backend/entrypoint.sh ./

RUN chmod +x /backend/entrypoint.sh

ENV VIRTUAL_ENV="/backend/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV ENV="prod"

ENTRYPOINT ["/backend/entrypoint.sh"]