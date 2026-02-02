# GL-002 FLAMEGUARD - Docker Image
# Boiler Efficiency Optimization Agent

FROM python:3.11-slim as builder

# Build arguments
ARG VERSION=1.0.0

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

# Labels
LABEL maintainer="GreenLang <support@greenlang.io>"
LABEL version="${VERSION}"
LABEL description="GL-002 FLAMEGUARD Boiler Efficiency Optimization Agent"

# Security: Run as non-root user
RUN groupadd -r flameguard && useradd -r -g flameguard flameguard

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/flameguard/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R flameguard:flameguard /app

# Switch to non-root user
USER flameguard

# Add local packages to path
ENV PATH=/home/flameguard/.local/bin:$PATH
ENV PYTHONPATH=/app

# Environment variables
ENV FLAMEGUARD_LOG_LEVEL=INFO
ENV FLAMEGUARD_API_PORT=8080
ENV FLAMEGUARD_METRICS_PORT=9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080 9090

# Default command
CMD ["python", "-m", "uvicorn", "api.rest_api:app", "--host", "0.0.0.0", "--port", "8080"]
