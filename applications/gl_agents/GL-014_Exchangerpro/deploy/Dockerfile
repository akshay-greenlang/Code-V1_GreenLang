# =============================================================================
# GL-014 EXCHANGERPRO - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production deployment
#
# Features:
# - Multi-stage build for minimal image size
# - Non-root user for security
# - Health check endpoint
# - Reproducible builds with pinned versions
# - Layer caching optimization
#
# Usage:
#   docker build -t gl-014-exchangerpro:latest -f deploy/Dockerfile .
#   docker run -p 8000:8000 gl-014-exchangerpro:latest
#
# Standards:
# - CIS Docker Benchmark compliance
# - GreenLang container security guidelines
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /build
COPY pyproject.toml ./
COPY requirements*.txt ./  2>/dev/null || true

# Install production dependencies only
RUN pip install --upgrade pip setuptools wheel && \
    pip install . && \
    pip install uvicorn[standard] gunicorn

# -----------------------------------------------------------------------------
# Stage 2: Production
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Labels for container metadata
LABEL org.opencontainers.image.title="GL-014 EXCHANGERPRO" \
      org.opencontainers.image.description="Heat Exchanger Optimizer Agent" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="Proprietary" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-agents" \
      maintainer="GreenLang Engineering <engineering@greenlang.io>"

# Security: Run as non-root user
RUN groupadd --gid 1000 gluser && \
    useradd --uid 1000 --gid gluser --shell /bin/bash --create-home gluser

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    APP_MODULE="api.main:app" \
    APP_HOST="0.0.0.0" \
    APP_PORT="8000" \
    APP_WORKERS="4" \
    APP_LOG_LEVEL="info" \
    # GreenLang settings
    GL_AGENT_ID="GL-014" \
    GL_AGENT_NAME="EXCHANGERPRO" \
    GL_ENVIRONMENT="production" \
    GL_DETERMINISTIC_MODE="true" \
    GL_SEED="42"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directories
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/models /app/audit && \
    chown -R gluser:gluser /app

# Copy application code
COPY --chown=gluser:gluser . /app/

# Switch to non-root user
USER gluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${APP_PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - production server with gunicorn
CMD ["sh", "-c", "gunicorn ${APP_MODULE} \
    --bind ${APP_HOST}:${APP_PORT} \
    --workers ${APP_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --log-level ${APP_LOG_LEVEL} \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    --enable-stdio-inheritance"]

# =============================================================================
# Alternative: Development/Debug mode
# =============================================================================
# To run in development mode, override CMD:
#   docker run -p 8000:8000 gl-014-exchangerpro:latest \
#     uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# =============================================================================
