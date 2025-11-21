# GL-001 ProcessHeatOrchestrator - Multi-stage Production Dockerfile
# Optimized for size, security, and performance
# Python 3.11 base with production-grade configurations

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.10-slim as builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.10-slim

# Set metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      description="GL-001 ProcessHeatOrchestrator - Industrial process heat optimization agent" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/greenlang/gl-001" \
      org.opencontainers.image.documentation="https://docs.greenlang.ai/agents/GL-001"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    GREENLANG_ENV=production \
    APP_HOME=/app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libssl3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user (non-root for security)
RUN groupadd -r greenlang && useradd -r -g greenlang -u 1000 greenlang

# Create application directory with proper permissions
RUN mkdir -p ${APP_HOME} /app/logs /app/cache /app/data && \
    chown -R greenlang:greenlang ${APP_HOME} /app/logs /app/cache /app/data

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=greenlang:greenlang . ${APP_HOME}/

# Create necessary directories for runtime
RUN mkdir -p /tmp/greenlang && chown -R greenlang:greenlang /tmp/greenlang

# Switch to non-root user
USER greenlang

# Expose ports
# 8000: Main API
# 8001: Metrics (Prometheus)
# 8002: Admin API (internal)
EXPOSE 8000 8001 8002

# Health check for Kubernetes probes
# Checks the /health endpoint which should return 200 OK
HEALTHCHECK --interval=30s \
            --timeout=10s \
            --start-period=40s \
            --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Entry point for the application
# Uses gunicorn for production-grade ASGI serving
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["process_heat_orchestrator:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info"]
