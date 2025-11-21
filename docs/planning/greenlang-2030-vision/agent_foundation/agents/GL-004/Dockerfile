# GL-004 BurnerOptimizationAgent - Production Docker Image
# Multi-stage build for optimized image size and security

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.10-slim-bullseye as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to user site-packages
RUN pip install --user --no-cache-dir --no-warn-script-location -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.10-slim-bullseye

# Set labels
LABEL maintainer="greenlang@example.com"
LABEL description="GL-004 BurnerOptimizationAgent - Combustion optimization"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r greenlang && useradd -r -g greenlang -u 1000 greenlang

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Change ownership
RUN chown -R greenlang:greenlang /app

# Switch to non-root user
USER greenlang

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
