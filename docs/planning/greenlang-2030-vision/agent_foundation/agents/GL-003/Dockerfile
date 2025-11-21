# GL-003 SteamSystemAnalyzer - Development Dockerfile
# Multi-stage build optimized for development workflow
# Features: Fast rebuilds, live reload, debugging support

# =============================================================================
# Stage 1: Builder - Compile dependencies
# =============================================================================
FROM python:3.10-slim-bullseye AS builder

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL description="GL-003 SteamSystemAnalyzer - Development Builder"
LABEL version="dev"

# Build arguments
ARG BUILD_DATE
ARG VCS_REF

# Build-time environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies (including dev dependencies)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-asyncio>=0.21.0 \
    black>=23.7.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    mypy>=1.4.0 \
    bandit>=1.7.5 \
    ipdb>=0.13.13 \
    watchdog>=3.0.0 \
    uvicorn[standard]>=0.23.0

# =============================================================================
# Stage 2: Development Runtime
# =============================================================================
FROM python:3.10-slim-bullseye AS development

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL description="GL-003 SteamSystemAnalyzer - Development Runtime"
LABEL version="dev"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/greenlang/Code-V1_GreenLang"
LABEL org.opencontainers.image.title="GL-003 SteamSystemAnalyzer (Dev)"
LABEL org.opencontainers.image.description="AI-powered steam system analysis agent - Development environment"

# Development environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive \
    ENVIRONMENT=development \
    PORT=8000 \
    LOG_LEVEL=debug \
    RELOAD=true

# Install runtime dependencies + dev tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libssl1.1 \
    ca-certificates \
    curl \
    wget \
    git \
    vim \
    nano \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r greenlang --gid=1000 && \
    useradd -r -g greenlang --uid=1000 --shell=/bin/bash --create-home greenlang && \
    mkdir -p /app /app/logs /app/data /app/cache && \
    chown -R greenlang:greenlang /app

# Switch to application directory
WORKDIR /app

# Copy application code (can be overridden by volume mount)
COPY --chown=greenlang:greenlang . /app/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/data \
    /app/cache \
    /app/monitoring \
    /app/tests \
    && chown -R greenlang:greenlang /app

# Switch to non-root user
USER greenlang

# Expose application port
EXPOSE 8000

# Expose debugger port
EXPOSE 5678

# Health check configuration (more lenient for dev)
HEALTHCHECK --interval=60s \
            --timeout=15s \
            --start-period=60s \
            --retries=5 \
            CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Volume mounts for development
VOLUME ["/app", "/app/logs", "/app/data", "/app/cache"]

# Development command with auto-reload
CMD ["uvicorn", "steam_system_orchestrator:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app", \
     "--log-level", "debug", \
     "--access-log", \
     "--use-colors"]

# Metadata
ENV APP_VERSION=dev

# Runtime labels
LABEL com.greenlang.agent.id="GL-003"
LABEL com.greenlang.agent.name="SteamSystemAnalyzer"
LABEL com.greenlang.agent.version="dev"
LABEL com.greenlang.deployment.environment="development"

# =============================================================================
# Development Build Instructions:
# =============================================================================
# Build development image:
#   docker build -f Dockerfile \
#     --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
#     --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#     -t greenlang/gl-003:dev \
#     .
#
# Run development container with live reload:
#   docker run -d \
#     --name gl-003-steam-dev \
#     -p 8000:8000 \
#     -p 5678:5678 \
#     -e ENVIRONMENT=development \
#     -e LOG_LEVEL=debug \
#     -v $(pwd):/app \
#     -v gl003-dev-logs:/app/logs \
#     greenlang/gl-003:dev
#
# Run with Docker Compose (recommended):
#   docker-compose up -d
#
# Access logs:
#   docker logs -f gl-003-steam-dev
#
# Execute tests:
#   docker exec gl-003-steam-dev pytest -v
#
# Access shell:
#   docker exec -it gl-003-steam-dev bash
#
# Debug with ipdb:
#   Add 'import ipdb; ipdb.set_trace()' in code
#   docker attach gl-003-steam-dev
# =============================================================================
