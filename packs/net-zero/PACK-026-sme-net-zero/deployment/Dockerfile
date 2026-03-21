# PACK-026: SME Net Zero Pack - Production Dockerfile
# Multi-stage build for optimal image size and security

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r greenlang && useradd -r -g greenlang greenlang

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy pack files
COPY engines/ ./engines/
COPY workflows/ ./workflows/
COPY templates/ ./templates/
COPY integrations/ ./integrations/
COPY config/ ./config/
COPY data/ ./data/
COPY __init__.py .
COPY pack.yaml .

# Copy main application entrypoint
COPY deployment/app.py ./app.py

# Create required directories
RUN mkdir -p /app/cache /app/logs /tmp && \
    chown -R greenlang:greenlang /app /tmp

# Switch to non-root user
USER greenlang

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
