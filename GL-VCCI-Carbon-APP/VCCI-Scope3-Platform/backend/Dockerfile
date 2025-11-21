# ==============================================================================
# GL-VCCI Backend API - Production Dockerfile
# Multi-stage build for optimized image size and security
# ==============================================================================

# ==============================================================================
# Stage 1: Dependencies Builder
# ==============================================================================
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Stage 2: Runtime Image
# ==============================================================================
FROM python:3.10-slim

# Metadata
LABEL maintainer="GL-VCCI Team <support@greenlang.ai>" \
      version="2.0.0" \
      description="GL-VCCI Scope 3 Carbon Intelligence Platform - Backend API"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app \
    PATH="/home/appuser/.local/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR ${APP_HOME}

# Copy Python dependencies from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Copy services and agents from parent directory
COPY --chown=appuser:appuser ../services ./services
COPY --chown=appuser:appuser ../agents ./agents
COPY --chown=appuser:appuser ../connectors ./connectors
COPY --chown=appuser:appuser ../utils ./utils
COPY --chown=appuser:appuser ../config ./config
COPY --chown=appuser:appuser ../compliance ./compliance
COPY --chown=appuser:appuser ../entity_mdm ./entity_mdm
COPY --chown=appuser:appuser ../policy ./policy
COPY --chown=appuser:appuser ../schemas ./schemas
COPY --chown=appuser:appuser ../specs ./specs

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /tmp && \
    chown -R appuser:appuser /app/logs /app/cache /tmp

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Start command (using uvicorn for FastAPI)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
