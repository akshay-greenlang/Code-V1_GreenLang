FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Create directories
RUN mkdir -p data logs cache

# Set environment variables
ENV PYTHONPATH=/app
ENV GREENLANG_ENV=production

# Run application
CMD ["python", "src/main.py"]
