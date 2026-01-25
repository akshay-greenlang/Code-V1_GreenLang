# GL-003 UNIFIEDSTEAM - Docker Image
# Steam System Optimization Agent
# Multi-stage build for optimized production image

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
FROM python:3.11-slim as builder

# Build arguments
ARG VERSION=1.0.0
ARG BUILD_DATE
ARG GIT_COMMIT

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Production - Minimal runtime image
# ==============================================================================
FROM python:3.11-slim

# Labels for container metadata
LABEL maintainer="GreenLang <support@greenlang.io>"
LABEL version="${VERSION}"
LABEL description="GL-003 UNIFIEDSTEAM Steam System Optimization Agent"
LABEL org.opencontainers.image.title="GL-003 UNIFIEDSTEAM"
LABEL org.opencontainers.image.description="Steam System Optimization with Real-time Control"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user and group
RUN groupadd -r -g 1000 unifiedsteam && \
    useradd -r -u 1000 -g unifiedsteam -d /app -s /sbin/nologin unifiedsteam

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set Python environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY --chown=unifiedsteam:unifiedsteam . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/config /app/models && \
    chown -R unifiedsteam:unifiedsteam /app

# Switch to non-root user
USER unifiedsteam

# Environment variables for the application
ENV UNIFIEDSTEAM_LOG_LEVEL=INFO
ENV UNIFIEDSTEAM_API_PORT=8080
ENV UNIFIEDSTEAM_GRPC_PORT=50051
ENV UNIFIEDSTEAM_METRICS_PORT=9090
ENV UNIFIEDSTEAM_ENV=production

# Health check - verify API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
# 8080 - REST/GraphQL API
# 50051 - gRPC API
# 9090 - Prometheus metrics
EXPOSE 8080 50051 9090

# Default entrypoint - can be overridden
ENTRYPOINT ["python", "-m"]

# Default command - run the API server
CMD ["uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
