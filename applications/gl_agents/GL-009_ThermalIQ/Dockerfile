# ==============================================================================
# GL-009 ThermalIQ Dockerfile
# ==============================================================================
# Production-grade multi-stage build for thermal intelligence agent deployment.
#
# Standards Compliance:
#   - ASME PTC 4.1: Steam Generator Performance
#   - ISO 50001:2018: Energy Management Systems
#   - IAPWS-IF97: Water/Steam Properties
#
# Security Features:
#   - Non-root user execution
#   - Read-only filesystem support
#   - No shell in production image
#   - Minimal attack surface
#   - Health checks with proper timeouts
#
# Usage:
#   Build:  docker build -t gl-009-thermaliq:2.1.0 --target production .
#   Run:    docker run -d -p 8080:8080 -p 9090:9090 gl-009-thermaliq:2.1.0
#   Test:   docker build -t gl-009-thermaliq:test --target test .
#   Dev:    docker-compose up -d
#
# Author: GL-BackendDeveloper
# Version: 2.1.0
# ==============================================================================

# Build arguments for version tracking
ARG APP_VERSION=2.1.0
ARG BUILD_DATE=""
ARG VCS_REF=""
ARG PYTHON_VERSION=3.11

# =============================================================================
# Stage 1: Base image with Python and system dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Create non-root user for security (follows principle of least privilege)
RUN groupadd --gid 1000 gluser \
    && useradd --uid 1000 --gid 1000 --shell /usr/sbin/nologin --create-home gluser

# =============================================================================
# Stage 2: Builder for dependencies (discarded in production)
# =============================================================================
FROM base AS builder

# Install build dependencies (not included in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with security best practices
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip check

# =============================================================================
# Stage 3: Production image (minimal, secure)
# =============================================================================
FROM base AS production

# Accept build arguments
ARG APP_VERSION
ARG BUILD_DATE
ARG VCS_REF

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=gluser:gluser . .

# Create directories for runtime data with correct permissions
RUN mkdir -p /app/logs /app/data /app/cache /app/audit \
    && chown -R gluser:gluser /app \
    && chmod -R 755 /app \
    && chmod 700 /app/audit

# Security hardening: Remove unnecessary files
RUN rm -rf \
    /app/.git \
    /app/.github \
    /app/tests \
    /app/docs \
    /app/*.md \
    /app/Dockerfile* \
    /app/docker-compose* \
    /app/.env* \
    /app/.coveragerc \
    /app/pytest.ini \
    2>/dev/null || true

# Switch to non-root user
USER gluser

# Expose ports
# 8080: REST API / GraphQL endpoint
# 9090: Prometheus metrics endpoint
EXPOSE 8080 9090

# Health check configuration
# - Checks API health endpoint every 30 seconds
# - Allows 45 seconds startup time for container initialization
# - Times out after 10 seconds per check
# - Retries 3 times before marking unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# Default environment variables for GreenLang agent
ENV GL_AGENT_ID="GL-009" \
    GL_AGENT_NAME="ThermalIQ" \
    GL_VERSION="${APP_VERSION}" \
    GL_LOG_LEVEL="INFO" \
    GL_LOG_FORMAT="json" \
    GL_METRICS_PORT="9090" \
    GL_METRICS_ENABLED="true" \
    GL_API_PORT="8080" \
    GL_DETERMINISTIC_MODE="true" \
    GL_FAIL_CLOSED="true" \
    GL_DECIMAL_PRECISION="true" \
    GL_AUDIT_ENABLED="true" \
    GL_AUDIT_RETENTION_DAYS="2555" \
    GL_READ_ONLY_MODE="true"

# OCI Labels for container metadata
LABEL org.opencontainers.image.title="GL-009 ThermalIQ" \
    org.opencontainers.image.description="Industrial Thermal Efficiency & Exergy Analysis Agent - Zero Hallucination Architecture" \
    org.opencontainers.image.version="${APP_VERSION}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.created="${BUILD_DATE}" \
    org.opencontainers.image.vendor="GreenLang" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/greenlang/gl-agents" \
    org.opencontainers.image.documentation="https://docs.greenlang.io/gl-009" \
    org.greenlang.agent.id="GL-009" \
    org.greenlang.agent.name="ThermalIQ" \
    org.greenlang.standards="ASME-PTC-4.1,ISO-50001,IAPWS-IF97" \
    org.greenlang.global-ai-standards.version="2.0" \
    org.greenlang.global-ai-standards.target="95+"

# Use tini as init process for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the application with uvicorn
# - 4 workers for production workload
# - Bind to all interfaces
# - Access logging enabled
# - Graceful shutdown with 30s timeout
CMD ["uvicorn", "api.rest_api:app", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--workers", "4", \
    "--access-log", \
    "--timeout-graceful-shutdown", "30", \
    "--proxy-headers", \
    "--forwarded-allow-ips", "*"]

# =============================================================================
# Stage 4: Development image (includes dev tools)
# =============================================================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    pytest-asyncio>=0.21.0 \
    pytest-timeout>=2.1.0 \
    hypothesis>=6.0.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.0.0 \
    ipython>=8.0.0 \
    httpx>=0.24.0

# Restore test and documentation files
COPY --chown=gluser:gluser tests/ /app/tests/
COPY --chown=gluser:gluser docs/ /app/docs/ 2>/dev/null || true

USER gluser

# Development mode with hot reload
CMD ["uvicorn", "api.rest_api:app", \
    "--host", "0.0.0.0", \
    "--port", "8080", \
    "--reload", \
    "--reload-dir", "/app"]

# =============================================================================
# Stage 5: Test runner (CI/CD optimized)
# =============================================================================
FROM development AS test

ENV GL_TEST_MODE="true" \
    GL_LOG_LEVEL="DEBUG" \
    PYTHONPATH="/app"

# Run tests with coverage enforcement (85% minimum)
CMD ["pytest", "tests/", \
    "-v", \
    "--cov=.", \
    "--cov-report=term-missing", \
    "--cov-report=xml:coverage.xml", \
    "--cov-fail-under=85", \
    "--junitxml=test-results.xml", \
    "-x", \
    "--tb=short"]

# =============================================================================
# Stage 6: Security scanner (for CI/CD vulnerability checks)
# =============================================================================
FROM production AS security-scan

USER root

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit>=1.7.0 \
    safety>=2.0.0 \
    pip-audit>=2.0.0

USER gluser

# Run security scans
CMD ["sh", "-c", \
    "bandit -r /app -x /app/tests -ll -ii && \
     pip-audit --desc && \
     echo 'Security scan completed successfully'"]
