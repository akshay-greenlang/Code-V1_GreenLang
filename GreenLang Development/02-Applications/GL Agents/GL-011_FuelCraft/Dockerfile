# GL-011 FUELCRAFT - Multi-stage Dockerfile
ARG PYTHON_VERSION=3.11
FROM python:3.11-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates tini && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 1000 gluser && useradd --uid 1000 --gid 1000 --shell /usr/sbin/nologin --create-home gluser

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gfortran libopenblas-dev && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:PATH
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt

FROM base AS production
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:PATH
WORKDIR /app
COPY --chown=gluser:gluser . .
RUN mkdir -p /app/logs /app/data /app/cache /app/audit /app/solver && chown -R gluser:gluser /app && chmod 700 /app/audit
USER gluser
EXPOSE 8080 50051 9090
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 CMD curl -sf http://localhost:8080/health || exit 1
ENV GL_AGENT_ID=GL-011 GL_AGENT_NAME=FuelCraft GL_LOG_LEVEL=INFO GL_AUDIT_RETENTION_DAYS=2555
ENTRYPOINT [/usr/bin/tini, --]
CMD [uvicorn, api.rest_api:app, --host, 0.0.0.0, --port, 8080, --workers, 4]

FROM production AS development
USER root
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio hypothesis black ruff mypy httpx
COPY --chown=gluser:gluser tests/ /app/tests/
USER gluser
CMD [uvicorn, api.rest_api:app, --host, 0.0.0.0, --port, 8080, --reload]

FROM development AS test
ENV GL_TEST_MODE=true GL_LOG_LEVEL=DEBUG PYTHONPATH=/app
CMD [pytest, tests/, -v, --cov=., --cov-fail-under=85]
