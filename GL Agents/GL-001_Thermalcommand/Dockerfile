# GL-001_Thermalcommand Docker Image
# Multi-stage build for production deployment
# Compliant with GreenLang Framework standards

# ============================================================================
# Stage 1: Builder - Install dependencies and build wheel
# ============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build wheel (optional)
RUN pip install --no-cache-dir -e .

# ============================================================================
# Stage 2: Production - Minimal runtime image
# ============================================================================
FROM python:3.11-slim AS production

LABEL maintainer="GreenLang Platform <platform@greenlang.io>"
LABEL org.opencontainers.image.title="GL-001_Thermalcommand"
LABEL org.opencontainers.image.description="Multi-Equipment Thermal Asset Optimization"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="GreenLang"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 thermalcmd && \
    useradd --uid 1000 --gid thermalcmd --shell /bin/bash --create-home thermalcmd

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=thermalcmd:thermalcmd . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GL_001_LOG_LEVEL=INFO
ENV GL_001_TRACK_PROVENANCE=true

# Switch to non-root user
USER thermalcmd

# Expose ports (REST, GraphQL, gRPC, Metrics)
EXPOSE 8000 8001 50051 9090

# Health check - verify API responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - run FastAPI with uvicorn
CMD ["python", "-m", "uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
