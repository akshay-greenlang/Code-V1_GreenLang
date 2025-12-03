# =============================================================================
# EUDR Deforestation Compliance Agent - Docker Image
# =============================================================================
# Agent: regulatory/eudr_compliance_v1
# Version: 1.0.0
# Description: Validates supply chain compliance with EU Deforestation Regulation (EU) 2023/1115
# =============================================================================

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 2: Runtime stage
# =============================================================================
FROM python:3.11-slim as runtime

# Labels
LABEL org.opencontainers.image.title="EUDR Deforestation Compliance Agent" \
      org.opencontainers.image.description="Validates compliance with EU Deforestation Regulation (EU) 2023/1115" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="GreenLang" \
      org.opencontainers.image.licenses="Apache-2.0" \
      agent.id="regulatory/eudr_compliance_v1" \
      agent.version="1.0.0" \
      agent.deadline="2025-12-30" \
      agent.tier="1-extreme-urgency"

# Runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Agent configuration
    AGENT_NAME="eudr-compliance" \
    AGENT_VERSION="1.0.0" \
    AGENT_ID="regulatory/eudr_compliance_v1" \
    # GreenLang configuration
    GREENLANG_ENV="production" \
    GREENLANG_LOG_LEVEL="INFO" \
    GREENLANG_LOG_FORMAT="json" \
    # Server configuration
    SERVER_HOST="0.0.0.0" \
    SERVER_PORT="8000" \
    # Metrics
    METRICS_PORT="9090" \
    # EUDR configuration
    EUDR_CUTOFF_DATE="2020-12-31" \
    EUDR_DEADLINE="2025-12-30"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user (UID 1000)
RUN groupadd --gid 1000 greenlang \
    && useradd --uid 1000 --gid greenlang --shell /bin/bash --create-home greenlang \
    && mkdir -p /app /app/data /app/logs /tmp \
    && chown -R greenlang:greenlang /app /tmp

# Set working directory
WORKDIR /app

# Copy core greenlang libraries (for EUDR databases)
COPY --chown=greenlang:greenlang core /app/core

# Copy SDK (required dependency)
COPY --chown=greenlang:greenlang sdks/python/greenlang_sdk /app/greenlang_sdk

# Copy agent code
COPY --chown=greenlang:greenlang generated/eudr_compliance_v1 /app/agent

# Copy entrypoint
COPY --chown=greenlang:greenlang generated/eudr_compliance_v1/entrypoint.py /app/

# Switch to non-root user
USER greenlang

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run agent
CMD ["python", "/app/entrypoint.py"]
