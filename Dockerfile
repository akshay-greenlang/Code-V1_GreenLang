# syntax=docker/dockerfile:1.7
# Production-ready GreenLang CLI Docker image
# This image provides a minimal, secure runtime for GreenLang CLI

# Multi-stage build for optimized layer caching and security
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS builder

# Build stage environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install GreenLang CLI
ARG GL_VERSION=0.3.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install "greenlang-cli==${GL_VERSION}"

# Final stage - minimal production image
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    GL_HOME=/var/lib/greenlang \
    GL_CACHE_DIR=/var/cache/greenlang \
    GL_LOG_DIR=/var/log/greenlang

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tini \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* \
    && rm -rf /tmp/*

# Copy virtual environment from builder
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Verify installation
ARG GL_VERSION=0.3.0
RUN python -c "import greenlang; print(f'GreenLang CLI v${GL_VERSION} installed successfully')" \
    && gl --version

# Create non-root user with proper UID for security
RUN groupadd -g 10001 appuser \
    && useradd -u 10001 -g 10001 -m -s /bin/bash appuser \
    && mkdir -p ${GL_HOME} ${GL_CACHE_DIR} ${GL_LOG_DIR} /workspace \
    && chown -R appuser:appuser ${GL_HOME} ${GL_CACHE_DIR} ${GL_LOG_DIR} /workspace

# Security: Drop all capabilities from Python binary
RUN setcap -r /opt/venv/bin/python3.11 2>/dev/null || true

# Switch to non-root user
USER appuser
WORKDIR /workspace

# OCI standard labels
ARG VCS_REF
ARG BUILD_DATE
ARG VERSION=${GL_VERSION}
LABEL org.opencontainers.image.title="GreenLang CLI" \
      org.opencontainers.image.description="Production-ready GreenLang CLI for climate intelligence applications" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/greenlang/greenlang" \
      org.opencontainers.image.url="https://greenlang.io" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="GreenLang Team" \
      org.opencontainers.image.documentation="https://greenlang.io/docs" \
      security.capabilities.drop="ALL" \
      security.no-new-privileges="true"

# Volume for workspace and caches
VOLUME ["${GL_CACHE_DIR}", "${GL_LOG_DIR}", "/workspace"]

# Healthcheck using gl CLI
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD gl --help >/dev/null 2>&1 || exit 1

# Default entrypoint with Tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "gl"]
CMD ["--help"]