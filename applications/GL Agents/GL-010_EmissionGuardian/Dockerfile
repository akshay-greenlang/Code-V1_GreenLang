# GL-010 EmissionsGuardian Dockerfile
# Multi-stage build for production deployment
#
# Standards Compliance:
# - EPA 40 CFR Part 75: Continuous Emissions Monitoring
# - SOC 2 Type II: Security controls
#
# Build: docker build -t gl-010-emissionguardian:1.0.0 .
# Run:   docker run -d -p 8080:8080 gl-010-emissionguardian:1.0.0

# =============================================================================
# Stage 1: Base image with Python and system dependencies
# =============================================================================
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security (SOC 2 compliance)
RUN groupadd --gid 1000 gluser \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home gluser

# =============================================================================
# Stage 2: Builder for dependencies
# =============================================================================
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=gluser:gluser . .

# Create directories for runtime data
RUN mkdir -p /app/logs /app/data /app/cache \
    && chown -R gluser:gluser /app

# Switch to non-root user
USER gluser

# Expose ports
# 8080: REST API
# 9090: Prometheus metrics
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default environment variables
ENV GL_AGENT_ID="GL-010" \
    GL_AGENT_NAME="EmissionsGuardian" \
    GL_LOG_LEVEL="INFO" \
    GL_METRICS_PORT="9090" \
    GL_API_PORT="8080" \
    GL_DETERMINISTIC_MODE="true" \
    GL_FAIL_CLOSED="true" \
    GL_ENFORCE_READ_ONLY="true"

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

# =============================================================================
# Stage 4: Development image (optional)
# =============================================================================
FROM production AS development

# Switch back to root for dev setup
USER root

# Install development dependencies
RUN pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    hypothesis \
    black \
    ruff \
    mypy \
    ipython

# Switch back to non-root user
USER gluser

# Override CMD for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# =============================================================================
# Stage 5: Test runner image
# =============================================================================
FROM development AS test

# Set test environment
ENV GL_TEST_MODE="true"

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=.", "--cov-report=term-missing"]
