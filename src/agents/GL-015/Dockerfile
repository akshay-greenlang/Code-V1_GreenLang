# ==============================================================================
# GL-015 INSULSCAN - Production Dockerfile
# ==============================================================================
# Multi-stage build for insulation inspection agent with thermal image processing
# Optimized for minimal image size and security
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and compile packages
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Build metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL stage="builder"

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies for image processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Image metadata
LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL org.opencontainers.image.title="GL-015 INSULSCAN"
LABEL org.opencontainers.image.description="Insulation Inspection Agent with Thermal Image Processing"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.source="https://github.com/greenlang/gl-015-insulscan"

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    APP_USER=insulscan \
    APP_UID=1000 \
    APP_GID=1000

# Install runtime dependencies only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Create application directories
RUN mkdir -p ${APP_HOME} \
    ${APP_HOME}/data \
    ${APP_HOME}/logs \
    ${APP_HOME}/cache \
    ${APP_HOME}/thermal_images \
    ${APP_HOME}/reports \
    && chown -R ${APP_USER}:${APP_USER} ${APP_HOME}

# Set working directory
WORKDIR ${APP_HOME}

# Copy application code with correct ownership
COPY --chown=${APP_USER}:${APP_USER} . ${APP_HOME}/

# Create health check script
RUN echo '#!/bin/bash\ncurl -sf http://localhost:8000/api/v1/health || exit 1' > /usr/local/bin/healthcheck.sh && \
    chmod +x /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER ${APP_USER}

# Expose application port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD ["/usr/local/bin/healthcheck.sh"]

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - run uvicorn with production settings
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

# ------------------------------------------------------------------------------
# Stage 3: GPU Runtime (Optional) - For accelerated image processing
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime-gpu

# Image metadata for GPU variant
LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL org.opencontainers.image.title="GL-015 INSULSCAN (GPU)"
LABEL org.opencontainers.image.description="Insulation Inspection Agent with GPU-accelerated Thermal Image Processing"
LABEL org.opencontainers.image.version="1.0.0-gpu"

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME=/app \
    APP_USER=insulscan \
    APP_UID=1000 \
    APP_GID=1000 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libpq5 \
    libjpeg62 \
    libpng16-16 \
    libtiff5 \
    libwebp7 \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Create application directories
RUN mkdir -p ${APP_HOME} \
    ${APP_HOME}/data \
    ${APP_HOME}/logs \
    ${APP_HOME}/cache \
    ${APP_HOME}/thermal_images \
    ${APP_HOME}/reports \
    && chown -R ${APP_USER}:${APP_USER} ${APP_HOME}

WORKDIR ${APP_HOME}

COPY --chown=${APP_USER}:${APP_USER} . ${APP_HOME}/

USER ${APP_USER}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools"]
