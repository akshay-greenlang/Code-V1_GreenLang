# GreenLang Agent Factory - Production Dockerfile
# Multi-stage build for optimized image size and security

# =============================================================================
# Stage 1: Build Stage
# =============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG VERSION=0.0.0
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Install the application
WORKDIR /app
RUN pip install --no-cache-dir -e .

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM python:3.11-slim as runtime

# Set labels
LABEL org.opencontainers.image.title="GreenLang Agent Factory"
LABEL org.opencontainers.image.description="Climate and Industrial Decarbonization Agents at Scale"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.source="https://github.com/greenlang/agent-factory"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Create non-root user for security
RUN groupadd --gid 1000 greenlang \
    && useradd --uid 1000 --gid greenlang --shell /bin/bash --create-home greenlang \
    && mkdir -p /app /app/logs /app/data \
    && chown -R greenlang:greenlang /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --from=builder --chown=greenlang:greenlang /app /app

# Switch to non-root user
USER greenlang

# Expose port
EXPOSE 8000

# Set version environment variable
ARG VERSION=0.0.0
ENV APP_VERSION=${VERSION}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# =============================================================================
# Stage 3: Development Stage (optional)
# =============================================================================
FROM runtime as development

# Switch to root for development setup
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Switch back to non-root user
USER greenlang

# Development command with hot reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
