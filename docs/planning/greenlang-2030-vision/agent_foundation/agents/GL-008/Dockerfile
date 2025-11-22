# GL-008 TRAPCATCHER - SteamTrapInspector
# Production-grade Docker container

# Base image: Python 3.10 slim (Debian-based)
FROM python:3.10-slim-bullseye AS base

# Metadata
LABEL maintainer="GreenLang Foundation <engineering@greenlang.org>"
LABEL agent.id="GL-008"
LABEL agent.name="TRAPCATCHER"
LABEL agent.version="1.0.0"
LABEL agent.description="Automated steam trap failure detection and diagnostics"
LABEL org.opencontainers.image.source="https://github.com/greenlang/agents/gl-008"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    AGENT_ID=GL-008-TRAPCATCHER \
    AGENT_VERSION=1.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash greenlang && \
    mkdir -p /app /app/logs /app/data /app/models && \
    chown -R greenlang:greenlang /app

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY --chown=greenlang:greenlang requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=greenlang:greenlang config.py .
COPY --chown=greenlang:greenlang tools.py .
COPY --chown=greenlang:greenlang steam_trap_inspector.py .
COPY --chown=greenlang:greenlang pack.yaml .
COPY --chown=greenlang:greenlang gl.yaml .
COPY --chown=greenlang:greenlang run.json .

# Copy ML models directory (if exists)
COPY --chown=greenlang:greenlang ml_models/ ./ml_models/

# Switch to non-root user
USER greenlang

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose metrics port
EXPOSE 9090

# Default environment
ENV LOG_LEVEL=INFO \
    ENABLE_METRICS=true \
    ENABLE_CACHING=true \
    CACHE_TTL_SECONDS=300

# Volume mounts for persistence
VOLUME ["/app/logs", "/app/data", "/app/models"]

# Entry point
ENTRYPOINT ["python", "-m", "steam_trap_inspector"]

# Default command (can be overridden)
CMD ["--mode", "monitor", "--config", "run.json"]
