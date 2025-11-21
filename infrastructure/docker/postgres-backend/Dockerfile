FROM postgres:15

COPY scripts/pg-backend-entrypoint.sh /docker-entrypoint-init/entrypoint.sh

RUN chmod +x /docker-entrypoint-init/entrypoint.sh

ENTRYPOINT ["/docker-entrypoint-init/entrypoint.sh"]
