# GL-016 WATERGUARD - Multi-stage Production Dockerfile
# Project: GreenLang Industrial Sustainability Framework
# Agent: Water Treatment Quality Control

# Stage 1: Builder
FROM python:3.10-slim as builder

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL project="GL-016 WATERGUARD"
LABEL description="Intelligent water treatment quality control agent"
LABEL version="1.0.0"

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

LABEL maintainer="GreenLang DevOps <devops@greenlang.io>"
LABEL project="GL-016 WATERGUARD"
LABEL description="Intelligent water treatment quality control agent"
LABEL version="1.0.0"
LABEL org.opencontainers.image.title="GL-016 WATERGUARD"
LABEL org.opencontainers.image.description="Water treatment quality control with predictive analytics"
LABEL org.opencontainers.image.vendor="GreenLang"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.schema-version="1.0"

# Create non-root user
RUN groupadd -r waterguard && \
    useradd -r -g waterguard -u 1000 -m -s /bin/bash waterguard

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/waterguard/.local

# Copy application code
COPY --chown=waterguard:waterguard . .

# Update PATH for user-installed packages
ENV PATH=/home/waterguard/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENT_ID=GL-016
ENV AGENT_NAME=WATERGUARD

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R waterguard:waterguard /app

# Switch to non-root user
USER waterguard

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
