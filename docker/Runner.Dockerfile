# GreenLang Runner Dockerfile
# Minimal runtime image for GreenLang v0.3.0
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV GREENLANG_VERSION=0.3.0

# Create non-root user
RUN groupadd -r gluser && useradd -r -g gluser gluser

# Create app directory and set ownership
RUN mkdir -p /app && chown gluser:gluser /app
WORKDIR /app

# Copy wheel file
COPY dist/greenlang-0.3.0-py3-none-any.whl /tmp/greenlang-0.3.0-py3-none-any.whl

# Install GreenLang from local wheel
RUN pip install --no-cache-dir /tmp/greenlang-0.3.0-py3-none-any.whl && \
    rm /tmp/greenlang-0.3.0-py3-none-any.whl

# Switch to non-root user
USER gluser

# Set entrypoint
ENTRYPOINT ["gl"]
CMD ["--help"]

# Metadata
LABEL maintainer="GreenLang Team <team@greenlang.io>"
LABEL version="0.3.0"
LABEL description="GreenLang Runner - Infrastructure for Climate Intelligence"