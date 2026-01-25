#!/bin/bash

# GreenLang Emission Factor API - Startup Script
# Usage: ./run.sh [dev|prod|docker]

set -e

MODE=${1:-dev}

case $MODE in
  dev)
    echo "Starting API in development mode..."
    uvicorn greenlang.api.main:app --reload --host 0.0.0.0 --port 8000
    ;;

  prod)
    echo "Starting API in production mode..."
    uvicorn greenlang.api.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers 4 \
      --log-level info \
      --access-log \
      --proxy-headers
    ;;

  docker)
    echo "Starting API with Docker Compose..."
    docker-compose up -d
    echo "API running at http://localhost:8000"
    echo "Docs at http://localhost:8000/api/docs"
    ;;

  test)
    echo "Running tests..."
    pytest greenlang/api/tests/ -v --cov=greenlang.api --cov-report=term-missing
    ;;

  *)
    echo "Usage: ./run.sh [dev|prod|docker|test]"
    echo ""
    echo "  dev    - Development mode with auto-reload"
    echo "  prod   - Production mode with 4 workers"
    echo "  docker - Run with Docker Compose"
    echo "  test   - Run test suite"
    exit 1
    ;;
esac
