# IcBerg governance gateway — self-hostable REST + SSE API in front of a governed
# database (see ARCHITECTURE.md's "Integration surfaces" #2). This is a SEPARATE image
# from backend/Dockerfile, which packages the project's original Titanic Q&A demo app
# (backend/main.py); this one packages backend/gateway_app.py's `create_gateway_app`
# factory instead.
FROM python:3.12-slim

WORKDIR /app

# curl: used by the HEALTHCHECK below to probe GET /health.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install the project (pyproject.toml's dependency set already includes fastapi,
# uvicorn, sqlglot, and pyyaml) before copying the rest of the source, so the dependency
# layer is cached across source-only changes. `frontend/` is intentionally not copied —
# it is a separate process this image never runs.
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY icberg ./icberg
RUN pip install --no-cache-dir .

# Non-root runtime user. /data holds the gateway's SQLite database, audit log, and
# approval queue (see ICBERG_GATEWAY_DB_PATH below) when the default SQLite DSN is used.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin icberg \
    && mkdir -p /data \
    && chown -R icberg:icberg /app /data
USER icberg

# `backend/gateway_app.py`'s module-level `app` object is built lazily by that module's
# own `__getattr__` — for `uvicorn backend.gateway_app:app` below — which calls the
# `create_gateway_app(dsn, ...)` factory internally, configured entirely from these two
# environment variables (see that module's docstring):
#   ICBERG_GATEWAY_DB_PATH        - DSN for the governed database: a SQLite path/URI
#                                    (default here), postgres://..., or mysql://... .
#   ICBERG_GATEWAY_RATE_LIMIT_PER_MIN - per-actor request cap (default: 60).
ENV ICBERG_GATEWAY_DB_PATH=/data/icberg.sqlite \
    ICBERG_GATEWAY_RATE_LIMIT_PER_MIN=60 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.gateway_app:app", "--host", "0.0.0.0", "--port", "8000"]
