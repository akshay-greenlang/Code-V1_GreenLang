# =============================================================================
# GreenLang Agent Registry - Multi-Stage Dockerfile
# =============================================================================
#
# This Dockerfile builds a production-ready container for the Agent Registry API.
#
# Features:
# - Multi-stage build for minimal image size
# - Non-root user for security (uid 1000)
# - Health check endpoint
# - Optimized Python dependencies
# - Production-grade gunicorn configuration
#
# Build:
#   docker build -t greenlang-registry:latest .
#
# Run:
#   docker run -p 8000:8000 \
#     -e DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/db" \
#     greenlang-registry:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Labels for container registry
LABEL org.opencontainers.image.title="GreenLang Agent Registry" \
      org.opencontainers.image.description="Agent Registry API for GreenLang" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.source="https://github.com/greenlang/agent-registry"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    APP_HOME=/app \
    APP_USER=appuser \
    APP_UID=1000 \
    APP_GID=1000 \
    # Default configuration
    PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=info

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/health/live || exit 1

# Default command - production with gunicorn
CMD ["sh", "-c", "gunicorn greenlang_registry.api.app:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL} \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional - for local development)
# -----------------------------------------------------------------------------
FROM production AS development

# Switch to root for installing dev dependencies
USER root

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    black \
    ruff \
    mypy

# Switch back to non-root user
USER ${APP_USER}

# Development command with hot reload
CMD ["uvicorn", "greenlang_registry.api.app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--reload", \
    "--log-level", "debug"]
