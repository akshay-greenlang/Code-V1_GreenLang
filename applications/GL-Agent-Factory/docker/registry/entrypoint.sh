#!/bin/bash
# GreenLang Registry - Container Entrypoint Script
# Handles database migrations and service startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    log_info "Waiting for PostgreSQL to be ready..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if pg_isready -h "${DATABASE_HOST:-localhost}" -p "${DATABASE_PORT:-5432}" -U "${DATABASE_USER:-postgres}" > /dev/null 2>&1; then
            log_info "PostgreSQL is ready!"
            return 0
        fi

        log_warn "PostgreSQL not ready, attempt $attempt/$max_attempts..."
        sleep 2
        attempt=$((attempt + 1))
    done

    log_error "PostgreSQL failed to become ready after $max_attempts attempts"
    return 1
}

# Run database migrations
run_migrations() {
    if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
        log_info "Running database migrations..."

        if [ -d "migrations" ] && [ -f "alembic.ini" ]; then
            alembic upgrade head
            log_info "Migrations completed successfully"
        else
            log_warn "No migrations directory or alembic.ini found, skipping migrations"
        fi
    else
        log_info "Skipping migrations (RUN_MIGRATIONS not set to 'true')"
    fi
}

# Initialize database if needed
init_database() {
    if [ "${INIT_DATABASE:-false}" = "true" ]; then
        log_info "Initializing database..."
        python -c "from db.connection import init_db; init_db()" || true
        log_info "Database initialization completed"
    fi
}

# Main entrypoint logic
main() {
    log_info "Starting GreenLang Registry..."
    log_info "Environment: ${GREENLANG_APP_ENV:-production}"

    # Wait for PostgreSQL
    if [ "${WAIT_FOR_POSTGRES:-true}" = "true" ]; then
        wait_for_postgres || exit 1
    fi

    # Run migrations
    run_migrations

    # Initialize database
    init_database

    log_info "Starting application..."

    # Execute the main command
    exec "$@"
}

# Run main function with all arguments
main "$@"
