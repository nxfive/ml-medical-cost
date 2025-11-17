FROM dpage/pgadmin4:9.8

COPY pgadmin/entrypoint.sh /init-entrypoint.sh

USER root

RUN chmod +x /init-entrypoint.sh

ENTRYPOINT ["/init-entrypoint.sh"]