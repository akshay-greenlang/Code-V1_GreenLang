# GL-005 CombustionControlAgent - Production Dockerfile
# Multi-stage build for optimized, secure, production-ready container
# Real-time control system with low latency requirements (<100ms)
# Target: < 500MB | Security: Non-root user | High Availability: 99.9% uptime

# =============================================================================
# Stage 1: Builder - Compile dependencies and prepare environment
# =============================================================================
FROM python:3.11-slim-bullseye AS builder

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL description="GL-005 CombustionControlAgent - Builder Stage"
LABEL version="1.0.0"

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir gunicorn[gevent]==21.2.0

# =============================================================================
# Stage 2: Security Scanner - Scan for vulnerabilities
# =============================================================================
FROM builder AS security-scanner

# Install security scanning tools
RUN pip install --no-cache-dir bandit safety

# Copy application code for scanning
COPY . /app
WORKDIR /app

# Run security scans (non-blocking in build, log results)
RUN bandit -r /app -ll -i || true && \
    safety check --short-report || true && \
    echo "Security scan completed - review logs for issues"

# =============================================================================
# Stage 3: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim-bullseye AS runtime

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL description="GL-005 CombustionControlAgent - Production Runtime"
LABEL version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/greenlang/Code-V1_GreenLang"
LABEL org.opencontainers.image.title="GL-005 CombustionControlAgent"
LABEL org.opencontainers.image.description="AI-powered real-time combustion control system"

# Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    ENVIRONMENT=production \
    PORT=8000 \
    METRICS_PORT=8001

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libssl1.1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security (UID/GID 1000)
RUN groupadd -r greenlang --gid=1000 && \
    useradd -r -g greenlang --uid=1000 --shell=/bin/bash --create-home greenlang && \
    mkdir -p /app /app/logs /app/data /app/cache /app/config && \
    chown -R greenlang:greenlang /app

# Switch to application directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=greenlang:greenlang . /app/

# Create necessary directories for runtime
RUN mkdir -p \
    /app/logs \
    /app/data \
    /app/cache \
    /app/monitoring \
    /app/control_logs \
    /app/sensor_data \
    && chown -R greenlang:greenlang /app

# Set read-only filesystem for security (except writable dirs)
# Note: This is enforced at runtime via Kubernetes securityContext

# Switch to non-root user
USER greenlang

# Expose application ports
EXPOSE 8000 8001

# Health check configuration
# Critical for real-time control - fast failure detection
HEALTHCHECK --interval=10s \
            --timeout=5s \
            --start-period=30s \
            --retries=3 \
            CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Volume mounts for persistent data
VOLUME ["/app/logs", "/app/data", "/app/cache", "/app/control_logs"]

# Set entrypoint and default command
# Using uvicorn for async performance with real-time requirements
ENTRYPOINT ["python", "-m"]
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log", \
     "--use-colors", \
     "--loop", "uvloop", \
     "--no-server-header"]

# Metadata
ARG BUILD_VERSION=1.0.0
ENV APP_VERSION=${BUILD_VERSION}

# Runtime labels
LABEL com.greenlang.agent.id="GL-005"
LABEL com.greenlang.agent.name="CombustionControlAgent"
LABEL com.greenlang.agent.version="${APP_VERSION}"
LABEL com.greenlang.agent.type="real-time-control"
LABEL com.greenlang.deployment.environment="production"
LABEL com.greenlang.sla.uptime="99.9%"
LABEL com.greenlang.sla.latency="<100ms"

# =============================================================================
# Build Instructions:
# =============================================================================
# Build image:
#   docker build -f Dockerfile \
#     --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
#     --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#     --build-arg VERSION=1.0.0 \
#     -t greenlang/gl-005:latest \
#     -t greenlang/gl-005:1.0.0 \
#     .
#
# Run container locally:
#   docker run -d \
#     --name gl-005-combustion-control \
#     -p 8000:8000 \
#     -p 8001:8001 \
#     -e DATABASE_URL=postgresql://user:pass@host/db \
#     -e REDIS_URL=redis://host:6379/0 \
#     -v gl005-logs:/app/logs \
#     -v gl005-data:/app/data \
#     greenlang/gl-005:latest
#
# Security scan:
#   docker scan greenlang/gl-005:latest
#   trivy image greenlang/gl-005:latest
#   grype greenlang/gl-005:latest
#
# Image size check:
#   docker images greenlang/gl-005:latest
#   dive greenlang/gl-005:latest
# =============================================================================
