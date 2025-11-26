# GL-006 HeatRecoveryMaximizer Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

# Labels
LABEL maintainer="GreenLang Team <engineering@greenlang.io>"
LABEL org.opencontainers.image.title="GL-006 HeatRecoveryMaximizer"
LABEL org.opencontainers.image.description="Heat Recovery Optimization Agent"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.source="https://github.com/greenlang/gl-006-heatreclaim"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=42 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    APP_USER=greenlang \
    APP_UID=1000 \
    APP_GID=1000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Create application directories
RUN mkdir -p ${APP_HOME} \
    && mkdir -p /var/log/greenlang \
    && mkdir -p /var/data/greenlang \
    && mkdir -p /var/reports/heat_recovery \
    && chown -R ${APP_USER}:${APP_USER} ${APP_HOME} \
    && chown -R ${APP_USER}:${APP_USER} /var/log/greenlang \
    && chown -R ${APP_USER}:${APP_USER} /var/data/greenlang \
    && chown -R ${APP_USER}:${APP_USER} /var/reports/heat_recovery

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . ${APP_HOME}

# Switch to non-root user
USER ${APP_USER}

# Expose ports
# 8000: Main API
# 9090: Prometheus metrics
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# Stage 3: Development (optional, for local development)
# =============================================================================
FROM runtime as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    mypy \
    ipython

USER ${APP_USER}

# Override command for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Stage 4: Testing (for CI/CD pipelines)
# =============================================================================
FROM runtime as testing

USER root

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    pytest-mock>=3.12.0 \
    hypothesis>=6.92.0 \
    faker>=22.0.0 \
    respx>=0.20.0

USER ${APP_USER}

# Set test environment
ENV ENVIRONMENT=testing \
    GL006_ENVIRONMENT=testing

# Run tests by default
CMD ["pytest", "--cov=.", "--cov-report=xml", "--cov-report=term", "-v"]
