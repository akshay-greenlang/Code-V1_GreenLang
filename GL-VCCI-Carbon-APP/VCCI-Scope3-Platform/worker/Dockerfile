# ==============================================================================
# GL-VCCI Worker - Celery Worker Dockerfile
# Based on backend image with additional ML dependencies
# ==============================================================================

# Use backend as base
FROM python:3.10-slim

# Metadata
LABEL maintainer="GL-VCCI Team <support@greenlang.ai>" \
      version="2.0.0" \
      description="GL-VCCI Scope 3 Platform - Celery Worker (ML & Background Tasks)"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    C_FORCE_ROOT=true \
    APP_HOME=/app \
    PATH="/home/appuser/.local/bin:$PATH"

# Install runtime dependencies + optional GPU support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Set working directory
WORKDIR ${APP_HOME}

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (including ML libraries)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        celery[redis]==5.3.4 \
        flower==2.0.1 \
        scikit-learn==1.3.2 \
        transformers==4.35.2 \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu \
        sentence-transformers==2.2.2

# Copy application code
COPY --chown=appuser:appuser . .

# Copy shared code from parent
COPY --chown=appuser:appuser ../services ./services
COPY --chown=appuser:appuser ../agents ./agents
COPY --chown=appuser:appuser ../connectors ./connectors
COPY --chown=appuser:appuser ../utils ./utils
COPY --chown=appuser:appuser ../config ./config
COPY --chown=appuser:appuser ../entity_mdm ./entity_mdm

# Create directories
RUN mkdir -p /app/logs /app/cache /tmp && \
    chown -R appuser:appuser /app/logs /app/cache /tmp

# Switch to non-root user
USER appuser

# Health check (check if Celery worker is responding)
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD celery -A celery_app inspect ping || exit 1

# Start Celery worker
CMD ["celery", "-A", "celery_app", "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--pool=prefork", \
     "--max-tasks-per-child=100", \
     "--task-events", \
     "--time-limit=3600", \
     "--soft-time-limit=3000"]
