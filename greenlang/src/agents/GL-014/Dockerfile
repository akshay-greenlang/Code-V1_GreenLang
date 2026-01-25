# ==============================================================================
# GL-014 EXCHANGER-PRO - Production Dockerfile
# Multi-stage build for optimized container image
# ==============================================================================
# Build optimizations:
#   - Multi-stage build reduces final image size by ~70%
#   - Non-root user for security compliance
#   - Health checks for Kubernetes probes
#   - Minimal runtime dependencies
#   - Optimized layer caching for faster builds
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile extensions
# ------------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Build arguments for customization
ARG PYTHON_VERSION=3.11
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Labels for image metadata
LABEL maintainer="GreenLang DevOps Team <devops@greenlang.io>"
LABEL stage="builder"
LABEL application="gl-014-exchanger-pro"

# Set working directory for build
WORKDIR /build

# Install build dependencies required for compiling Python packages
# - build-essential: gcc, g++, make for native extensions
# - libpq-dev: PostgreSQL client library headers for asyncpg
# - libffi-dev: Foreign function interface for cryptography
# - libssl-dev: OpenSSL headers for secure connections
# - cargo: Rust compiler for cryptography package
# - gfortran: Fortran compiler for scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    liblapack-dev \
    libblas-dev \
    cargo \
    rustc \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create Python virtual environment
# Using venv ensures clean separation of dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file first for better Docker layer caching
# This allows Docker to cache the dependency installation layer
COPY requirements.txt .

# Install Python dependencies into virtual environment
# Using --no-cache-dir to reduce image size
# Using --compile to pre-compile .pyc files
RUN pip install --no-cache-dir --compile -r requirements.txt

# Verify critical packages are installed correctly
RUN python -c "import numpy; import pandas; import scipy; print('Scientific computing dependencies OK')"
RUN python -c "import fastapi; import uvicorn; print('Web framework OK')"
RUN python -c "import prometheus_client; print('Monitoring OK')"
RUN python -c "import redis; import asyncpg; print('Database/Cache dependencies OK')"

# ------------------------------------------------------------------------------
# Stage 2: Test Runner - Run tests to validate build (optional)
# ------------------------------------------------------------------------------
FROM builder as tester

# Copy application code for testing
COPY . /build/

# Install test dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov

# Run tests (uncomment in CI/CD pipeline)
# RUN pytest tests/ -v --cov=. --cov-report=term-missing

# ------------------------------------------------------------------------------
# Stage 3: Runtime - Minimal production image
# ------------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Image labels following OCI specification
LABEL org.opencontainers.image.title="GL-014 EXCHANGER-PRO"
LABEL org.opencontainers.image.description="Heat Exchanger Optimizer Agent for GreenLang Platform"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.source="https://github.com/greenlang/gl-014-exchanger-pro"
LABEL org.opencontainers.image.licenses="Proprietary"

# Create non-root user and group for security
# Using fixed UID/GID for Kubernetes security context compatibility
RUN groupadd -g 1000 greenlang && \
    useradd -u 1000 -g greenlang -m -s /bin/bash -d /home/greenlang greenlang

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies only
# - libpq5: PostgreSQL client library runtime
# - curl: Health check endpoint testing
# - ca-certificates: TLS certificate validation
# - tini: Proper init system for signal handling
# - liblapack3/libblas3: Runtime libraries for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    tini \
    dumb-init \
    liblapack3 \
    libblas3 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set Python environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONHASHSEED=random

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/tmp /app/config && \
    chown -R greenlang:greenlang /app

# Copy application code with correct ownership
COPY --chown=greenlang:greenlang . /app/

# Remove unnecessary files to reduce image size
RUN rm -rf /app/.git /app/.github /app/tests /app/docs \
    /app/*.md /app/Makefile /app/.coverage 2>/dev/null || true

# Switch to non-root user for security
USER greenlang

# Set environment variables for runtime
ENV APP_HOME=/app
ENV APP_USER=greenlang
ENV LOG_LEVEL=INFO
ENV METRICS_PORT=9090
ENV SERVICE_PORT=8080
ENV GRPC_PORT=50051

# Health check for container orchestration
# - interval: How often to check
# - timeout: Max time for check to complete
# - start-period: Grace period after container start
# - retries: Number of consecutive failures before unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT}/health || exit 1

# Expose application ports
# 8080: Main REST API
# 9090: Prometheus metrics endpoint
# 50051: gRPC endpoint
EXPOSE 8080 9090 50051

# Set stop signal for graceful shutdown
STOPSIGNAL SIGTERM

# Use tini as init system for proper signal handling
# This ensures proper handling of SIGTERM for graceful shutdown
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run application with uvicorn
# - host 0.0.0.0: Bind to all interfaces
# - workers: Set via environment variable for scaling
# - access-log: Enable for debugging
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
