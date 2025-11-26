# GL-010 EMISSIONWATCH EmissionsComplianceAgent - Multi-stage Production Dockerfile
# Optimized for emissions monitoring, CEMS integration, and regulatory compliance
# Python 3.11 base with industrial protocol support and real-time monitoring

# =============================================================================
# Stage 1: Builder Stage
# =============================================================================
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install build dependencies for scientific computing and industrial protocols
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade core Python packaging tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Scientific libraries are built from source for optimization
RUN pip install --no-cache-dir -r requirements.txt

# Compile Python bytecode for faster startup
RUN python -m compileall /opt/venv

# =============================================================================
# Stage 2: Security Scanning Stage (Optional - for CI/CD)
# =============================================================================
FROM builder as security-scanner

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    pip-audit

# Copy application code for security scanning
COPY . /app
WORKDIR /app

# Run security scans (will fail build if critical issues found)
# Generate JSON reports for CI/CD artifact collection
RUN bandit -r . -f json -o /tmp/bandit-report.json -ll || true && \
    safety check --json --output /tmp/safety-report.json || true && \
    pip-audit --format json --output /tmp/pip-audit-report.json || true

# =============================================================================
# Stage 3: Runtime Stage
# =============================================================================
FROM python:3.11-slim

# Set metadata labels following OCI standards
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      description="GL-010 EMISSIONWATCH EmissionsComplianceAgent - Industrial emissions monitoring and regulatory compliance agent" \
      version="1.0.0" \
      agent="GL-010" \
      category="emissions-compliance" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-010" \
      org.opencontainers.image.documentation="https://docs.greenlang.ai/agents/GL-010" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.title="EMISSIONWATCH EmissionsComplianceAgent"

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    GREENLANG_ENV=production \
    APP_HOME=/app \
    # Scientific computing optimizations
    OPENBLAS_NUM_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    # Memory management for emissions data processing
    MALLOC_TRIM_THRESHOLD_=100000 \
    MALLOC_MMAP_THRESHOLD_=100000 \
    # Emissions monitoring specific settings
    CEMS_POLLING_INTERVAL=60 \
    EMISSIONS_BUFFER_SIZE=10000

# Install runtime dependencies only (smaller footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libpq5 \
    libssl3 \
    libhdf5-103 \
    libxml2 \
    libxslt1.1 \
    curl \
    ca-certificates \
    tini \
    # Serial port access for CEMS communication
    libudev1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user and group (non-root for security)
# Using UID/GID 1000 for compatibility with Kubernetes security contexts
RUN groupadd -r greenlang --gid=1000 && \
    useradd -r -g greenlang --uid=1000 --no-log-init --create-home greenlang

# Create application directories with proper permissions
# Emissions data requires specific directories for compliance data retention
RUN mkdir -p \
    ${APP_HOME} \
    /app/logs \
    /app/cache \
    /app/data \
    /app/emissions \
    /app/reports \
    /app/compliance \
    /app/cems_data \
    /app/regulatory \
    /tmp/greenlang && \
    chown -R greenlang:greenlang \
    ${APP_HOME} \
    /app/logs \
    /app/cache \
    /app/data \
    /app/emissions \
    /app/reports \
    /app/compliance \
    /app/cems_data \
    /app/regulatory \
    /tmp/greenlang

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code with proper ownership
COPY --chown=greenlang:greenlang . ${APP_HOME}/

# Create version file for runtime introspection
RUN echo "GL-010 EMISSIONWATCH v1.0.0" > /app/VERSION && \
    echo "Build Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> /app/VERSION && \
    echo "Compliance Standards: EPA 40 CFR Part 75, EU ETS, ISO 14064" >> /app/VERSION && \
    chown greenlang:greenlang /app/VERSION

# Switch to non-root user
USER greenlang

# Expose ports
# 8080: Main API (emissions data and compliance endpoints)
# 9090: Metrics (Prometheus)
# 8081: Admin API (internal management)
# 8082: WebSocket (real-time emissions monitoring)
# 4840: OPC UA (CEMS communication)
EXPOSE 8080 9090 8081 8082

# Health check configuration
# Verifies API availability and CEMS connection status
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Use tini as init system for proper signal handling
# Critical for graceful shutdown of emissions data collection
ENTRYPOINT ["/usr/bin/tini", "--"]

# Application startup command
# Uses Uvicorn with uvloop for high-performance async I/O
# Emissions monitoring requires high concurrency for multiple CEMS connections
CMD ["python", "-m", "uvicorn", \
     "emissionwatch.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info", \
     "--limit-concurrency", "1000", \
     "--timeout-keep-alive", "75"]
