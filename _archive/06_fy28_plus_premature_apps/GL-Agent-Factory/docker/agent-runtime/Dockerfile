# GreenLang Agent Runtime - Production Dockerfile
# Multi-stage build for optimized image size and security

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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# Stage 2: Runtime - Production image
# ==============================================================================
FROM python:3.11-slim AS runtime

# Labels for image metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      version="1.0.0" \
      description="GreenLang Agent Runtime - Zero-hallucination AI agents for sustainability"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    APP_USER=greenlang \
    APP_GROUP=greenlang

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    dumb-init \
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

# Create necessary directories with correct permissions
RUN mkdir -p ${APP_HOME}/logs ${APP_HOME}/data ${APP_HOME}/tmp && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME}

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security: Read-only root filesystem compatible settings
ENV TMPDIR=/app/tmp

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Run application with gunicorn for production
CMD ["gunicorn", "app.main:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "--enable-stdio-inheritance"]
