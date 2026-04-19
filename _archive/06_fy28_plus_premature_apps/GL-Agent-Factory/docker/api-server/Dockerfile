# GreenLang API Server - Production Dockerfile
# FastAPI application with Gunicorn/Uvicorn

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

# Build arguments
ARG ENVIRONMENT=production

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY backend/requirements.txt /tmp/requirements.txt

# Install Python dependencies including gunicorn
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir gunicorn[gevent]>=21.2.0

# ==============================================================================
# Stage 2: Runtime - Production image
# ==============================================================================
FROM python:3.11-slim AS runtime

# Labels for image metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      version="1.0.0" \
      description="GreenLang API Server - FastAPI Gateway for Agent Factory"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    APP_USER=greenlang \
    APP_GROUP=greenlang \
    # Gunicorn settings
    GUNICORN_WORKERS=4 \
    GUNICORN_THREADS=2 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_KEEPALIVE=5

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    dumb-init \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 ${APP_GROUP} && \
    useradd --uid 1000 --gid ${APP_GROUP} --shell /bin/bash --create-home ${APP_USER}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
RUN mkdir -p ${APP_HOME} && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code
COPY --chown=${APP_USER}:${APP_GROUP} backend/ ${APP_HOME}/

# Copy gunicorn configuration
COPY --chown=${APP_USER}:${APP_GROUP} docker/api-server/gunicorn.conf.py ${APP_HOME}/gunicorn.conf.py

# Create necessary directories
RUN mkdir -p ${APP_HOME}/logs ${APP_HOME}/data ${APP_HOME}/tmp && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8000

# Health check - API server specific endpoints
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Temp directory for read-only root filesystem
ENV TMPDIR=/app/tmp

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Run with gunicorn and uvicorn workers
CMD ["gunicorn", "app.main:app", \
     "--config", "gunicorn.conf.py"]
