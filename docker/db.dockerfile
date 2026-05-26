FROM pgvector/pgvector:pg16

COPY docker/db-init/ /docker-entrypoint-initdb.d/
