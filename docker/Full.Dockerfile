# GreenLang Full Dockerfile
# Full development image with build tools for GreenLang v0.3.0
FROM python:3.11

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GREENLANG_VERSION=0.3.0

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r gluser && useradd -r -g gluser -m gluser

# Create app directory and set ownership
RUN mkdir -p /app && chown gluser:gluser /app
WORKDIR /app

# Copy wheel file
COPY dist/greenlang-0.3.0-py3-none-any.whl /tmp/greenlang-0.3.0-py3-none-any.whl

# Install development tools and GreenLang from local wheel
RUN pip install --no-cache-dir --upgrade pip build twine pytest && \
    pip install --no-cache-dir /tmp/greenlang-0.3.0-py3-none-any.whl && \
    rm /tmp/greenlang-0.3.0-py3-none-any.whl

# Switch to non-root user
USER gluser

# Set working directory to user home
WORKDIR /home/gluser

# Set entrypoint
ENTRYPOINT ["gl"]
CMD ["--help"]

# Metadata
LABEL maintainer="GreenLang Team <team@greenlang.io>"
LABEL version="0.3.0"
LABEL description="GreenLang Full - Development environment with build tools for Climate Intelligence"