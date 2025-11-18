FROM python:3.12-slim AS builder

WORKDIR /frontend

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
COPY services/frontend/ ./client/
COPY scripts/front-entrypoint.sh ./entrypoint.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    pip install --no-cache-dir uv && \
    uv sync --group client --frozen --no-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


FROM python:3.12-slim

WORKDIR /frontend

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /frontend/.venv ./.venv
COPY --from=builder /frontend/client ./client
COPY --from=builder /frontend/entrypoint.sh ./

RUN chmod +x /frontend/entrypoint.sh

RUN useradd -m user
USER user

ENV VIRTUAL_ENV="/frontend/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV ENV="prod"

ENTRYPOINT ["/frontend/entrypoint.sh"]