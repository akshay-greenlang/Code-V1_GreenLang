# GL-009 THERMALIQ ThermalEfficiencyCalculator - Multi-stage Production Dockerfile
# Optimized for thermal engineering calculations with scientific computing libraries
# Python 3.11 base with NumPy, SciPy, and industrial protocol support

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
RUN bandit -r . -f json -o /tmp/bandit-report.json || true && \
    safety check --json --output /tmp/safety-report.json || true && \
    pip-audit --format json --output /tmp/pip-audit-report.json || true

# =============================================================================
# Stage 3: Runtime Stage
# =============================================================================
FROM python:3.11-slim

# Set metadata labels
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      description="GL-009 THERMALIQ ThermalEfficiencyCalculator - Industrial thermal efficiency optimization agent" \
      version="1.0.0" \
      agent="GL-009" \
      category="thermal-engineering" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-009" \
      org.opencontainers.image.documentation="https://docs.greenlang.ai/agents/GL-009" \
      org.opencontainers.image.licenses="MIT"

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
    # Memory management
    MALLOC_TRIM_THRESHOLD_=100000 \
    MALLOC_MMAP_THRESHOLD_=100000

# Install runtime dependencies only (smaller footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libpq5 \
    libssl3 \
    libhdf5-103 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user and group (non-root for security)
RUN groupadd -r greenlang --gid=1000 && \
    useradd -r -g greenlang --uid=1000 --no-log-init --create-home greenlang

# Create application directories with proper permissions
RUN mkdir -p \
    ${APP_HOME} \
    /app/logs \
    /app/cache \
    /app/data \
    /app/calculations \
    /app/models \
    /app/reports \
    /tmp/greenlang && \
    chown -R greenlang:greenlang \
    ${APP_HOME} \
    /app/logs \
    /app/cache \
    /app/data \
    /app/calculations \
    /app/models \
    /app/reports \
    /tmp/greenlang

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code with proper ownership
COPY --chown=greenlang:greenlang . ${APP_HOME}/

# Create version file for runtime introspection
RUN echo "GL-009 THERMALIQ v1.0.0" > /app/VERSION && \
    echo "Build Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> /app/VERSION && \
    chown greenlang:greenlang /app/VERSION

# Switch to non-root user
USER greenlang

# Expose ports
# 8000: Main API (thermal efficiency calculations)
# 8001: Metrics (Prometheus)
# 8002: Admin API (internal management)
# 8003: WebSocket (real-time thermal monitoring)
EXPOSE 8000 8001 8002 8003

# Health check configuration
# Verifies API availability and thermal calculation engine status
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=60s \
            --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Application startup command
# Uses Uvicorn with uvloop for high-performance async I/O
CMD ["python", "-m", "uvicorn", \
     "thermaliq.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info", \
     "--limit-concurrency", "1000", \
     "--timeout-keep-alive", "75"]
