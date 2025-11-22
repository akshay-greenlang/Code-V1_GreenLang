# GL-007 FurnacePerformanceMonitor - Production Dockerfile
# Multi-stage build for optimal image size and security
# Base: Python 3.11 slim

# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

# Build arguments
ARG PYTHON_VERSION=3.11
ARG GREENLANG_VERSION=2.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# ============================================================================
# STAGE 2: Runtime
# ============================================================================
FROM python:3.11-slim

# Metadata labels
LABEL maintainer="GreenLang Industrial Team <furnace-performance@greenlang.ai>"
LABEL version="1.0.0"
LABEL description="GL-007 FurnacePerformanceMonitor - Industrial Furnace Monitoring & Optimization"
LABEL org.opencontainers.image.source="https://github.com/greenlang/gl-007"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    AGENT_ID="GL-007" \
    AGENT_VERSION="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd --create-home --shell /bin/bash greenlang && \
    mkdir -p /app /app/data /app/logs && \
    chown -R greenlang:greenlang /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=greenlang:greenlang . /app/

# Switch to non-root user
USER greenlang

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["uvicorn", "furnace_performance_orchestrator:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
