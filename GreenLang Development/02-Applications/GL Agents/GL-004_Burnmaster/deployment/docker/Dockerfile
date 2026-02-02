# GL-004 BURNMASTER - Multi-stage Dockerfile
# Industrial Combustion Monitoring and Optimization System
# ============================================================

# -----------------------------
# Stage 1: Build Dependencies
# -----------------------------
FROM python:3.11-slim AS builder

# Build-time arguments
ARG POETRY_VERSION=1.7.1
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Install additional ML/analytics dependencies
RUN pip install \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    torch>=2.0.0 \
    influxdb-client>=1.36.0 \
    confluent-kafka>=2.2.0 \
    asyncua>=1.0.0 \
    prometheus-client>=0.17.0 \
    opentelemetry-api>=1.20.0 \
    opentelemetry-sdk>=1.20.0

# -----------------------------
# Stage 2: Runtime Image
# -----------------------------
FROM python:3.11-slim AS runtime

# Labels for container metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.io>" \
      version="1.0.0" \
      description="GL-004 BURNMASTER - Industrial Combustion Monitoring Agent" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-004-burnmaster" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.licenses="MIT"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    GL_ENVIRONMENT=production \
    GL_LOG_LEVEL=INFO \
    GL_METRICS_PORT=9090 \
    GL_GRPC_PORT=50051 \
    GL_HTTP_PORT=8080

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    curl \
    dumb-init \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security (following CIS Docker Benchmark)
RUN groupadd -r -g 1000 greenlang && \
    useradd -r -u 1000 -g greenlang -d /app -s /sbin/nologin greenlang && \
    mkdir -p /app /app/logs /app/data /app/config /var/run/greenlang && \
    chown -R greenlang:greenlang /app /var/run/greenlang

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=greenlang:greenlang . /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R greenlang:greenlang /app

# Security: Remove unnecessary files and set permissions
RUN find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete && \
    chmod -R 550 /app && \
    chmod -R 770 /app/logs /app/data /app/cache

# Switch to non-root user
USER greenlang

# Expose ports
# 8080 - HTTP API
# 50051 - gRPC
# 9090 - Prometheus metrics
EXPOSE 8080 50051 9090

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Use dumb-init for proper signal handling (PID 1 zombie reaping)
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Default command - run the BURNMASTER agent
CMD ["python", "-m", "gl004_burnmaster.main", \
     "--config", "/app/config/config.yaml", \
     "--log-level", "${GL_LOG_LEVEL}"]

# -----------------------------
# Stage 3: Development Image
# -----------------------------
FROM runtime AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    htop \
    iputils-ping \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.5.0 \
    ipython>=8.14.0

USER greenlang

ENV GL_ENVIRONMENT=development \
    GL_LOG_LEVEL=DEBUG

CMD ["python", "-m", "gl004_burnmaster.main", "--dev-mode"]

# -----------------------------
# Stage 4: Testing Image
# -----------------------------
FROM development AS testing

USER root

# Install additional test dependencies
RUN pip install \
    pytest-xdist>=3.3.0 \
    hypothesis>=6.82.0 \
    faker>=19.0.0 \
    locust>=2.16.0 \
    testcontainers>=3.7.0

USER greenlang

ENV GL_ENVIRONMENT=testing

CMD ["pytest", "-v", "--cov=/app", "--cov-report=xml"]
