# ============================================================================
# GL-CBAM-APP - Production Dockerfile
# ============================================================================
# Multi-stage build for optimized production container
# Base: Python 3.11 (slim)
# Features: Security hardening, minimal attack surface, health checks
# ============================================================================

# ----------------------------------------------------------------------------
# STAGE 1: Builder - Install dependencies
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="GreenLang CBAM Team <cbam@greenlang.com>"
LABEL description="GL-CBAM-APP Backend API - Builder Stage"
LABEL version="1.0.0"

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn==21.2.0 \
    uvicorn[standard]==0.24.0 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1 \
    python-multipart==0.0.6 \
    python-jose[cryptography]==3.3.0 \
    passlib[bcrypt]==1.7.4

# ----------------------------------------------------------------------------
# STAGE 2: Runtime - Production image
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="GreenLang CBAM Team <cbam@greenlang.com>"
LABEL description="GL-CBAM-APP Backend API - Production"
LABEL version="1.0.0"

# Security: Create non-root user
RUN groupadd -r cbamuser && useradd -r -g cbamuser cbamuser

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=cbamuser:cbamuser . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/output /app/uploads && \
    chown -R cbamuser:cbamuser /app

# Security: Set file permissions
RUN chmod -R 755 /app && \
    chmod -R 700 /app/logs /app/data /app/output /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER cbamuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_ENV=production \
    LOG_LEVEL=INFO

# Default command: Run with gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info", \
     "api.main:app"]

# Alternative command for development (override in docker-compose)
# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# BUILD INSTRUCTIONS
# ============================================================================
#
# Build:
#   docker build -t gl-cbam-app:latest .
#   docker build -t gl-cbam-app:1.0.0 .
#
# Run (standalone):
#   docker run -d -p 8000:8000 \
#     -e DATABASE_URL=postgresql://user:pass@host:5432/cbam \
#     -e REDIS_URL=redis://host:6379 \
#     --name cbam-api gl-cbam-app:latest
#
# Build with build args:
#   docker build \
#     --build-arg VERSION=1.0.0 \
#     --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
#     -t gl-cbam-app:1.0.0 .
#
# Security scan:
#   docker scan gl-cbam-app:latest
#   trivy image gl-cbam-app:latest
#
# ============================================================================
