# =============================================================================
# GL-017 CONDENSYNC - CondenserOptimizationAgent
# Multi-stage Docker build for production deployment
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile requirements
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Labels for container metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.io>" \
      version="1.0.0" \
      description="GL-017 CONDENSYNC - Condenser Optimization Agent" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-017-condensync"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=condensync \
    APP_UID=1000 \
    APP_GID=1000 \
    HTTP_PORT=8017 \
    METRICS_PORT=9017

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user and group for security
RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories with proper permissions
RUN mkdir -p ${APP_HOME}/data ${APP_HOME}/logs ${APP_HOME}/config ${APP_HOME}/models && \
    chown -R ${APP_USER}:${APP_USER} ${APP_HOME}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . ${APP_HOME}/

# Security hardening: Remove unnecessary files and set permissions
RUN chmod -R 755 ${APP_HOME} && \
    find ${APP_HOME} -type f -name "*.py" -exec chmod 644 {} \; && \
    rm -rf ${APP_HOME}/.git ${APP_HOME}/.gitignore ${APP_HOME}/tests

# Switch to non-root user
USER ${APP_USER}

# Expose ports for HTTP API and Prometheus metrics
EXPOSE ${HTTP_PORT} ${METRICS_PORT}

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${HTTP_PORT}/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8017", "--workers", "4"]
