#!/usr/bin/env bash
# ==============================================================================
# GreenLang Agent Registry - PostgreSQL Database Setup Script
# ==============================================================================
# This script sets up the PostgreSQL database for the GreenLang Agent Registry.
# It creates the main database, test database, applies schema, and adds seed data.
#
# Usage:
#   ./setup_registry_db.sh [--reset] [--seed] [--test-only]
#
# Options:
#   --reset     Drop and recreate databases (DESTRUCTIVE)
#   --seed      Add seed data after schema creation
#   --test-only Only create test database
#
# Environment Variables:
#   POSTGRES_HOST     PostgreSQL host (default: localhost)
#   POSTGRES_PORT     PostgreSQL port (default: 5432)
#   POSTGRES_USER     PostgreSQL superuser (default: postgres)
#   POSTGRES_PASSWORD PostgreSQL superuser password
#
# Author: GreenLang Team
# Date: December 2025
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Database connection settings
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"

# Database names
MAIN_DB="greenlang_registry"
TEST_DB="greenlang_registry_test"

# Application user credentials
APP_USER="greenlang"
APP_PASSWORD="${APP_PASSWORD:-greenlang_secure_password}"

# Script directory (for relative paths to schema.sql)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCHEMA_FILE="$PROJECT_ROOT/core/greenlang/registry/schema.sql"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Execute SQL command as postgres superuser
psql_exec() {
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -c "$1" 2>&1
}

# Execute SQL command on specific database
psql_exec_db() {
    local db="$1"
    local sql="$2"
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$db" \
        -c "$sql" 2>&1
}

# Execute SQL file on specific database
psql_exec_file() {
    local db="$1"
    local file="$2"
    PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$db" \
        -f "$file" 2>&1
}

# Check if database exists
db_exists() {
    local db="$1"
    local result
    result=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -tAc "SELECT 1 FROM pg_database WHERE datname='$db'" 2>&1)
    [[ "$result" == "1" ]]
}

# Check if user exists
user_exists() {
    local user="$1"
    local result
    result=$(PGPASSWORD="$POSTGRES_PASSWORD" psql \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -tAc "SELECT 1 FROM pg_roles WHERE rolname='$user'" 2>&1)
    [[ "$result" == "1" ]]
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    local max_attempts=30
    local attempt=1

    log_info "Waiting for PostgreSQL to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if PGPASSWORD="$POSTGRES_PASSWORD" psql \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -c "SELECT 1" >/dev/null 2>&1; then
            log_success "PostgreSQL is ready!"
            return 0
        fi

        log_info "Attempt $attempt/$max_attempts - waiting for PostgreSQL..."
        sleep 2
        ((attempt++))
    done

    log_error "PostgreSQL did not become ready in time"
    exit 1
}

# ==============================================================================
# Database Creation Functions
# ==============================================================================

create_app_user() {
    log_info "Creating application user '$APP_USER'..."

    if user_exists "$APP_USER"; then
        log_warning "User '$APP_USER' already exists, updating password..."
        psql_exec "ALTER USER $APP_USER WITH PASSWORD '$APP_PASSWORD';"
    else
        psql_exec "CREATE USER $APP_USER WITH PASSWORD '$APP_PASSWORD';"
        log_success "Created user '$APP_USER'"
    fi
}

create_database() {
    local db="$1"
    local description="$2"

    log_info "Creating database '$db' ($description)..."

    if db_exists "$db"; then
        if [[ "$RESET_MODE" == "true" ]]; then
            log_warning "Dropping existing database '$db'..."
            psql_exec "DROP DATABASE $db;"
        else
            log_warning "Database '$db' already exists, skipping creation"
            return 0
        fi
    fi

    psql_exec "CREATE DATABASE $db OWNER $APP_USER;"
    log_success "Created database '$db'"
}

apply_schema() {
    local db="$1"

    log_info "Applying schema to database '$db'..."

    if [[ ! -f "$SCHEMA_FILE" ]]; then
        log_error "Schema file not found: $SCHEMA_FILE"
        exit 1
    fi

    psql_exec_file "$db" "$SCHEMA_FILE"
    log_success "Schema applied to '$db'"
}

grant_permissions() {
    local db="$1"

    log_info "Granting permissions on '$db' to '$APP_USER'..."

    psql_exec_db "$db" "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $APP_USER;"
    psql_exec_db "$db" "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $APP_USER;"
    psql_exec_db "$db" "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $APP_USER;"
    psql_exec_db "$db" "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $APP_USER;"

    log_success "Permissions granted on '$db'"
}

add_seed_data() {
    local db="$1"

    log_info "Adding seed data to '$db'..."

    # Insert sample agents for testing
    psql_exec_db "$db" "
    INSERT INTO agents (name, namespace, description, author, repository_url, spec_hash, status)
    VALUES
        ('waterguard', 'greenlang', 'Boiler water treatment optimization with zero-hallucination guarantees', 'GreenLang Team', 'https://github.com/greenlang/GL-016', '$(echo -n 'GL-016-WATERGUARD-v1.0.0' | sha256sum | cut -d' ' -f1)', 'active'),
        ('condensync', 'greenlang', 'Condenser optimization with zero-hallucination guarantees for steam systems', 'GreenLang Team', 'https://github.com/greenlang/GL-017', '$(echo -n 'GL-017-CONDENSYNC-v1.0.0' | sha256sum | cut -d' ' -f1)', 'active'),
        ('flueflow', 'greenlang', 'Flue gas analysis and combustion optimization with zero-hallucination guarantees', 'GreenLang Team', 'https://github.com/greenlang/GL-018', '$(echo -n 'GL-018-FLUEFLOW-v1.0.0' | sha256sum | cut -d' ' -f1)', 'active')
    ON CONFLICT (namespace, name) DO NOTHING;
    "

    # Add versions for each agent
    psql_exec_db "$db" "
    INSERT INTO agent_versions (agent_id, version, pack_path, pack_hash, metadata, capabilities, dependencies, size_bytes, status, published_by)
    SELECT
        id,
        '1.0.0',
        '/packages/greenlang/' || name || '-1.0.0.glpack',
        encode(sha256(('pack-' || name || '-1.0.0')::bytea), 'hex'),
        jsonb_build_object(
            'license', 'MIT',
            'python_version', '>=3.11',
            'framework', 'greenlang'
        ),
        CASE name
            WHEN 'waterguard' THEN jsonb_build_array('water-chemistry-analysis', 'blowdown-optimization', 'chemical-dosing', 'scale-prevention', 'corrosion-control')
            WHEN 'condensync' THEN jsonb_build_array('vacuum-optimization', 'heat-transfer-analysis', 'fouling-detection', 'air-inleakage-monitoring', 'cooling-water-optimization')
            WHEN 'flueflow' THEN jsonb_build_array('combustion-analysis', 'emissions-monitoring', 'efficiency-optimization', 'air-fuel-ratio', 'compliance-reporting')
            ELSE '[]'::jsonb
        END,
        '[]'::jsonb,
        CASE name
            WHEN 'waterguard' THEN 2048576
            WHEN 'condensync' THEN 2560000
            WHEN 'flueflow' THEN 1843200
            ELSE 1024000
        END,
        'published',
        'GreenLang Team'
    FROM agents
    WHERE namespace = 'greenlang'
    ON CONFLICT (agent_id, version) DO NOTHING;
    "

    # Add tags for each agent
    psql_exec_db "$db" "
    INSERT INTO agent_tags (agent_id, tag)
    SELECT id, unnest(ARRAY['water-treatment', 'boiler', 'chemistry', 'scale-prevention', 'corrosion', 'scope1', 'industrial', 'zero-hallucination'])
    FROM agents WHERE name = 'waterguard' AND namespace = 'greenlang'
    ON CONFLICT (agent_id, tag) DO NOTHING;

    INSERT INTO agent_tags (agent_id, tag)
    SELECT id, unnest(ARRAY['condenser', 'steam', 'vacuum', 'heat-transfer', 'fouling', 'cooling', 'efficiency', 'zero-hallucination'])
    FROM agents WHERE name = 'condensync' AND namespace = 'greenlang'
    ON CONFLICT (agent_id, tag) DO NOTHING;

    INSERT INTO agent_tags (agent_id, tag)
    SELECT id, unnest(ARRAY['combustion', 'flue-gas', 'emissions', 'efficiency', 'air-fuel-ratio', 'nox', 'co', 'compliance', 'zero-hallucination'])
    FROM agents WHERE name = 'flueflow' AND namespace = 'greenlang'
    ON CONFLICT (agent_id, tag) DO NOTHING;
    "

    # Add sample certifications
    psql_exec_db "$db" "
    INSERT INTO agent_certifications (agent_id, version, dimension, status, score, evidence, certified_by)
    SELECT
        id,
        '1.0.0',
        'security',
        'passed',
        95.0,
        jsonb_build_object(
            'test_suite', 'GL-CERT-SECURITY-v1.0',
            'vulnerabilities_found', 0,
            'audit_date', CURRENT_TIMESTAMP
        ),
        'GL-CERT'
    FROM agents WHERE namespace = 'greenlang'
    ON CONFLICT (agent_id, version, dimension) DO NOTHING;

    INSERT INTO agent_certifications (agent_id, version, dimension, status, score, evidence, certified_by)
    SELECT
        id,
        '1.0.0',
        'performance',
        'passed',
        92.5,
        jsonb_build_object(
            'test_suite', 'GL-CERT-PERFORMANCE-v1.0',
            'latency_p50_ms', 25,
            'latency_p99_ms', 150,
            'throughput_rps', 500
        ),
        'GL-CERT'
    FROM agents WHERE namespace = 'greenlang'
    ON CONFLICT (agent_id, version, dimension) DO NOTHING;

    INSERT INTO agent_certifications (agent_id, version, dimension, status, score, evidence, certified_by)
    SELECT
        id,
        '1.0.0',
        'determinism',
        'passed',
        100.0,
        jsonb_build_object(
            'test_suite', 'GL-CERT-DETERMINISM-v1.0',
            'zero_hallucination', true,
            'calculation_consistency', '100%',
            'provenance_tracking', true
        ),
        'GL-CERT'
    FROM agents WHERE namespace = 'greenlang'
    ON CONFLICT (agent_id, version, dimension) DO NOTHING;
    "

    log_success "Seed data added to '$db'"
}

# ==============================================================================
# Main Script
# ==============================================================================

main() {
    local RESET_MODE="false"
    local SEED_MODE="false"
    local TEST_ONLY="false"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --reset)
                RESET_MODE="true"
                shift
                ;;
            --seed)
                SEED_MODE="true"
                shift
                ;;
            --test-only)
                TEST_ONLY="true"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--reset] [--seed] [--test-only]"
                echo ""
                echo "Options:"
                echo "  --reset     Drop and recreate databases (DESTRUCTIVE)"
                echo "  --seed      Add seed data after schema creation"
                echo "  --test-only Only create test database"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo "=============================================="
    echo "GreenLang Agent Registry - Database Setup"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  PostgreSQL Host: $POSTGRES_HOST:$POSTGRES_PORT"
    echo "  Main Database:   $MAIN_DB"
    echo "  Test Database:   $TEST_DB"
    echo "  App User:        $APP_USER"
    echo "  Schema File:     $SCHEMA_FILE"
    echo "  Reset Mode:      $RESET_MODE"
    echo "  Seed Mode:       $SEED_MODE"
    echo ""

    # Export RESET_MODE for use in functions
    export RESET_MODE

    # Wait for PostgreSQL
    wait_for_postgres

    # Create application user
    create_app_user

    if [[ "$TEST_ONLY" == "false" ]]; then
        # Create main database
        create_database "$MAIN_DB" "Main registry database"
        apply_schema "$MAIN_DB"
        grant_permissions "$MAIN_DB"

        if [[ "$SEED_MODE" == "true" ]]; then
            add_seed_data "$MAIN_DB"
        fi
    fi

    # Create test database
    create_database "$TEST_DB" "Test registry database"
    apply_schema "$TEST_DB"
    grant_permissions "$TEST_DB"

    # Always add seed data to test database
    add_seed_data "$TEST_DB"

    echo ""
    echo "=============================================="
    log_success "Database setup completed successfully!"
    echo "=============================================="
    echo ""
    echo "Connection strings:"
    echo "  Main:  postgresql://$APP_USER:****@$POSTGRES_HOST:$POSTGRES_PORT/$MAIN_DB"
    echo "  Test:  postgresql://$APP_USER:****@$POSTGRES_HOST:$POSTGRES_PORT/$TEST_DB"
    echo ""
    echo "Environment variables to set:"
    echo "  export DB_HOST=$POSTGRES_HOST"
    echo "  export DB_PORT=$POSTGRES_PORT"
    echo "  export DB_USER=$APP_USER"
    echo "  export DB_PASSWORD=<your_password>"
    echo "  export DB_NAME=$MAIN_DB"
    echo ""
}

# Run main function
main "$@"
