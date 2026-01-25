# GL-006 HEATRECLAIM - Docker Image
# Heat Recovery Maximizer Agent
#
# Build: docker build -t gl-006-heatreclaim:latest .
# Run:   docker run -p 8006:8006 gl-006-heatreclaim:latest

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ========================================
# Builder stage - install dependencies
# ========================================
FROM base AS builder

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ========================================
# Production stage
# ========================================
FROM base AS production

# Create non-root user for security
RUN groupadd --gid 1000 gluser \
    && useradd --uid 1000 --gid gluser --shell /bin/bash --create-home gluser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=gluser:gluser . /app

# Set ownership
RUN chown -R gluser:gluser /app

# Switch to non-root user
USER gluser

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8006/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8006"]

# ========================================
# Development stage
# ========================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    mypy \
    ruff

USER gluser

# Mount point for development
VOLUME ["/app"]

CMD ["uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8006", "--reload"]
