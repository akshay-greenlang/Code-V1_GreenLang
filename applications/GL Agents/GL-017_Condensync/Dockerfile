# =============================================================================
# GL-017 CONDENSYNC - Production Dockerfile
# Multi-stage build for minimal, secure production image
# =============================================================================
# Build: docker build -t gl017-condensync:latest .
# Run:   docker run -p 8017:8017 gl017-condensync:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and build wheels
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Security labels
LABEL org.opencontainers.image.title="GL-017 CONDENSYNC" \
      org.opencontainers.image.description="Condenser Optimization Agent for GreenLang Platform" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.authors="GL-BackendDeveloper" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-017-condensync" \
      org.opencontainers.image.licenses="Proprietary"

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application configuration
    GL017_AGENT_ID=GL-017 \
    GL017_AGENT_NAME=CONDENSYNC \
    GL017_AGENT_MODE=monitoring \
    GL017_API_HOST=0.0.0.0 \
    GL017_API_PORT=8017 \
    GL017_LOG_LEVEL=INFO \
    GL017_METRICS_ENABLED=true \
    GL017_METRICS_PORT=9017 \
    # Virtual environment path
    PATH="/opt/venv/bin:$PATH" \
    # Application paths
    APP_HOME=/app \
    APP_USER=condensync

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_USER} && \
    useradd --uid 1000 --gid ${APP_USER} --shell /bin/bash --create-home ${APP_USER}

WORKDIR ${APP_HOME}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root user
USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${GL017_API_PORT}/health || exit 1

# Expose ports
# 8017 - REST API
# 9017 - Prometheus metrics
EXPOSE 8017 9017

# Default command - run with uvicorn
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8017", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]

# -----------------------------------------------------------------------------
# Stage 3: Development - Includes dev dependencies and tools
# -----------------------------------------------------------------------------
FROM production AS development

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER ${APP_USER}

# Override command for development (with reload)
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8017", \
     "--reload", \
     "--reload-dir", "/app"]

# -----------------------------------------------------------------------------
# Stage 4: Testing - For CI/CD pipeline
# -----------------------------------------------------------------------------
FROM development AS testing

USER root

# Set testing environment
ENV GL017_AGENT_MODE=diagnostic \
    GL017_LOG_LEVEL=DEBUG \
    PYTEST_ADDOPTS="--tb=short -q"

USER ${APP_USER}

# Default command runs tests
CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=xml", "--cov-report=term"]
