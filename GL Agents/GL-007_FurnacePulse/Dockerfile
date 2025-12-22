# GL-007 FURNACEPULSE - Docker Image
# Furnace Performance Monitoring Agent
#
# Build: docker build -t gl-007-furnacepulse:latest .
# Run:   docker run -p 8080:8080 -p 9090:9090 gl-007-furnacepulse:latest

# ========================================
# Base stage - common configuration
# ========================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ========================================
# Builder stage - install dependencies
# ========================================
FROM base AS builder

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========================================
# Production stage
# ========================================
FROM base AS production

# Create non-root user for security
RUN groupadd --gid 1000 gluser \
    && useradd --uid 1000 --gid gluser --shell /bin/bash --create-home gluser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/gluser/.local

# Update PATH for user-installed packages
ENV PATH="/home/gluser/.local/bin:$PATH"

# Copy application code
COPY --chown=gluser:gluser . /app

# Set ownership
RUN chown -R gluser:gluser /app

# Switch to non-root user
USER gluser

# Expose ports: 8080 for API, 9090 for metrics
EXPOSE 8080 9090

# Health check - check API health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8080"]

# ========================================
# Development stage
# ========================================
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.3.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    mypy>=1.3.0 \
    ruff>=0.0.270 \
    hypothesis>=6.80.0 \
    ipython>=8.0.0

USER gluser

# Mount point for development
VOLUME ["/app"]

# Development command with hot reload
CMD ["uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# ========================================
# Metrics sidecar stage (optional)
# ========================================
FROM production AS metrics

# Run metrics exporter alongside main app
CMD ["sh", "-c", "uvicorn api.rest_api:app --host 0.0.0.0 --port 8080 & uvicorn api.metrics:app --host 0.0.0.0 --port 9090 && wait"]
