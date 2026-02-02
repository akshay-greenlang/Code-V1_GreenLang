# GreenLang Registry - Production Dockerfile
# PostgreSQL integration with Alembic migrations

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
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir gunicorn[gevent]>=21.2.0

# ==============================================================================
# Stage 2: Runtime - Production image
# ==============================================================================
FROM python:3.11-slim AS runtime

# Labels for image metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.ai>" \
      version="1.0.0" \
      description="GreenLang Registry - Agent Registry with PostgreSQL"

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

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libffi8 \
    libssl3 \
    curl \
    dumb-init \
    ca-certificates \
    postgresql-client \
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

# Copy migration scripts and entrypoint
COPY --chown=${APP_USER}:${APP_GROUP} docker/registry/entrypoint.sh ${APP_HOME}/entrypoint.sh
COPY --chown=${APP_USER}:${APP_GROUP} docker/registry/migrate.sh ${APP_HOME}/migrate.sh

# Create necessary directories
RUN mkdir -p ${APP_HOME}/logs ${APP_HOME}/data ${APP_HOME}/tmp ${APP_HOME}/migrations && \
    chown -R ${APP_USER}:${APP_GROUP} ${APP_HOME} && \
    chmod +x ${APP_HOME}/entrypoint.sh ${APP_HOME}/migrate.sh

# Switch to non-root user
USER ${APP_USER}

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Temp directory for read-only root filesystem
ENV TMPDIR=/app/tmp

# Use dumb-init for proper signal handling
ENTRYPOINT ["dumb-init", "--", "/app/entrypoint.sh"]

# Default command - run registry service
CMD ["gunicorn", "app.main:app", \
     "--bind", "0.0.0.0:8001", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
