# GreenLang Executor Docker Image
FROM python:3.11-slim

# Build arguments for version
ARG GL_VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 greenlang

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY greenlang/ ./greenlang/

# Create necessary directories
RUN mkdir -p /var/cache/greenlang /var/log/greenlang \
    && chown -R greenlang:greenlang /app /var/cache/greenlang /var/log/greenlang

# Switch to non-root user
USER greenlang

# Environment variables
ENV PYTHONPATH=/app \
    GL_BACKEND=kubernetes \
    GL_LOG_LEVEL=INFO

# Add version labels
LABEL org.opencontainers.image.version="${GL_VERSION}" \
      org.opencontainers.image.title="GreenLang" \
      org.opencontainers.image.description="Infrastructure for Climate Intelligence" \
      org.opencontainers.image.source="https://github.com/greenlang/greenlang" \
      org.opencontainers.image.vendor="GreenLang Team"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose ports
EXPOSE 8080 9090

# Entry point
ENTRYPOINT ["python", "-m", "greenlang.runtime.server"]

# Default command
CMD ["--host", "0.0.0.0", "--port", "8080"]