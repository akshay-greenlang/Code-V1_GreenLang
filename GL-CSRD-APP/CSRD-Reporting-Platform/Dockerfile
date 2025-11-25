# ============================================================================
# Multi-stage Dockerfile for CSRD/ESRS Digital Reporting Platform
# ============================================================================
#
# Production-ready build with:
# - Multi-stage build for minimal image size
# - Security hardening (non-root user, minimal packages)
# - Layer caching optimization
# - Health checks
# - Python 3.11 base
#
# Build:   docker build -t greenlang/csrd-app:latest .
# Run:     docker run -p 8000:8000 greenlang/csrd-app:latest
# ============================================================================

# ----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies
# ----------------------------------------------------------------------------
# NOTE: Python 3.11 is the standardized version for GreenLang core infrastructure.
# See pyproject.toml for supported versions (3.10, 3.11, 3.12).
FROM python:3.11-slim AS builder

LABEL stage="builder"

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Install production server
RUN pip install --no-cache-dir --user \
    gunicorn[gevent]==21.2.0 \
    uvicorn[standard]==0.27.0

# ----------------------------------------------------------------------------
# Stage 2: Runtime - Production image
# ----------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="GreenLang CSRD Team <csrd@greenlang.io>"
LABEL description="CSRD/ESRS Digital Reporting Platform - Zero-hallucination EU sustainability reporting"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/akshay-greenlang/Code-V1_GreenLang"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=1000:1000 . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    PORT=8000

# Create non-root user for security
RUN groupadd -r -g 1000 csrd && \
    useradd -r -u 1000 -g csrd -m -s /bin/bash csrd && \
    chown -R csrd:csrd /app

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs /app/tmp && \
    chown -R csrd:csrd /app/data /app/output /app/logs /app/tmp

# Switch to non-root user
USER csrd

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - FastAPI with uvicorn
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
