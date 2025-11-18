FROM postgres:15

RUN useradd -m user

COPY postgres/mlflow/entrypoint.sh /docker-entrypoint-init/entrypoint.sh

RUN chmod +x /docker-entrypoint-init/entrypoint.sh

USER user

ENTRYPOINT ["/docker-entrypoint-init/entrypoint.sh"]
