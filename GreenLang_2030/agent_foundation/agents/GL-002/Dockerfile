# GL-002 BoilerEfficiencyOptimizer - Production Dockerfile
# Multi-stage build for minimal image size and security
# Production-grade with health checks and non-root user

# Stage 1: Builder
# Install build dependencies and Python packages in isolated layer
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Set PATH to use virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements from parent project
COPY requirements.txt .

# Install Python dependencies (use minimal layers approach)
RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete

# Stage 2: Runtime
# Minimal production image with only runtime dependencies
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=42 \
    GREENLANG_ENV=production

# Create non-root user for security (principle of least privilege)
RUN groupadd -r boiler && useradd -r -g boiler -u 1000 -d /home/boiler -s /sbin/nologin boiler && \
    mkdir -p /app /home/boiler && \
    chown -R boiler:boiler /app /home/boiler

# Set working directory
WORKDIR /app

# Copy application code from build context
COPY --chown=boiler:boiler . /app

# Create directories for logs and data with proper permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R boiler:boiler /app/logs /app/data && \
    chmod 755 /app/logs /app/data

# Switch to non-root user
USER boiler

# Expose application port
EXPOSE 8000

# Health check - Critical for Kubernetes liveness probe
# Checks application health every 30 seconds
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Application startup command
# Use uvicorn for async Python web framework
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
