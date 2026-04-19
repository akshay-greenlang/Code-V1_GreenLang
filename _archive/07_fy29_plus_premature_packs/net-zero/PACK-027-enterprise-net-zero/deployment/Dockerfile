# PACK-027: Enterprise Net Zero Pack - Docker Image
# Multi-stage build for optimized production image

# Stage 1: Builder
FROM python:3.11-slim as builder

LABEL maintainer="GreenLang Platform Team <enterprise-team@greenlang.io>"
LABEL pack.id="PACK-027"
LABEL pack.name="Enterprise Net Zero Pack"
LABEL pack.version="1.0.0"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd -m -u 1000 greenlang && \
    mkdir -p /app && \
    chown -R greenlang:greenlang /app

# Set working directory
WORKDIR /app

# Copy pack files
COPY --chown=greenlang:greenlang . /app/

# Switch to non-root user
USER greenlang

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PACK_ID=PACK-027 \
    PACK_NAME="Enterprise Net Zero Pack" \
    PACK_VERSION=1.0.0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "deployment/app.py"]
