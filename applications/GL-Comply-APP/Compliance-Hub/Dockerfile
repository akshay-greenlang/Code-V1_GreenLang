# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip wheel --no-cache-dir -w /wheels -e ".[server,security,monitoring]"

FROM python:3.11-slim AS production
ARG APP_USER=gl
RUN useradd -m -u 10001 ${APP_USER}
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY applications/GL-Comply-APP/Compliance-Hub /app
ENV PYTHONPATH=/app \
    PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
EXPOSE 8080
USER ${APP_USER}
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
