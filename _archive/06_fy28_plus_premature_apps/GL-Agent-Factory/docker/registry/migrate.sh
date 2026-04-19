#!/bin/bash
# GreenLang Registry - Migration Script
# Standalone script for running database migrations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
COMMAND=${1:-upgrade}
REVISION=${2:-head}

case $COMMAND in
    upgrade)
        log_info "Running upgrade to revision: $REVISION"
        alembic upgrade $REVISION
        ;;
    downgrade)
        log_info "Running downgrade to revision: $REVISION"
        alembic downgrade $REVISION
        ;;
    current)
        log_info "Showing current revision"
        alembic current
        ;;
    history)
        log_info "Showing migration history"
        alembic history --verbose
        ;;
    revision)
        log_info "Creating new revision: $REVISION"
        alembic revision --autogenerate -m "$REVISION"
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        echo "Usage: migrate.sh [upgrade|downgrade|current|history|revision] [revision]"
        exit 1
        ;;
esac

log_info "Migration command completed successfully"
