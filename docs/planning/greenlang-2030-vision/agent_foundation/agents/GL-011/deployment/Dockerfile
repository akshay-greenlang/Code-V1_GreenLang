# GL-011 FUELCRAFT - Dockerfile
# Multi-stage build for optimized image size and security
# Compatible with Python 3.11+

# =============================================================================
# Stage 1: Builder Stage
# =============================================================================
FROM python:3.11-slim AS builder

# Set build-time labels
LABEL maintainer="GreenLang Platform Team <platform@greenlang.ai>"
LABEL stage="builder"

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

# Set working directory
WORKDIR /build

# Copy requirements first (for better layer caching)
COPY requirements.txt requirements-prod.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# If production requirements exist, install those too
RUN if [ -f requirements-prod.txt ]; then \
        pip install --no-cache-dir -r requirements-prod.txt; \
    fi

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runtime

# Set runtime labels
LABEL maintainer="GreenLang Platform Team <platform@greenlang.ai>"
LABEL agent="GL-011"
LABEL agent_name="FUELCRAFT"
LABEL description="Fuel mix optimization and emissions management agent"
LABEL version="1.0.0"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libssl3 \
    ca-certificates \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment path
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH="/app:$PYTHONPATH"

# Create non-root user for security
RUN groupadd --gid 1000 greenlang && \
    useradd --uid 1000 --gid greenlang --shell /bin/bash --create-home greenlang

# Create application directories
RUN mkdir -p /app /app/logs /app/cache /app/data /app/fuel-data && \
    chown -R greenlang:greenlang /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=greenlang:greenlang src/ ./src/
COPY --chown=greenlang:greenlang config/ ./config/
COPY --chown=greenlang:greenlang main.py ./
COPY --chown=greenlang:greenlang alembic.ini ./
COPY --chown=greenlang:greenlang alembic/ ./alembic/

# Switch to non-root user
USER greenlang

# Expose ports
# 8080: HTTP API
# 9090: Prometheus metrics
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
