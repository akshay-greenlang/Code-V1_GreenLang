#!/bin/bash
# =============================================================================
# Apply EUDR Migrations V118-V128
# =============================================================================
# This script applies database migrations V118 through V128 for EUDR agents
# 030-040 (Documentation & Reporting category)
#
# EUDR Agents covered:
#   V118 - AGENT-EUDR-030: Documentation Generator
#   V119 - AGENT-EUDR-031: Stakeholder Engagement Tool
#   V120 - AGENT-EUDR-032: Grievance Mechanism Manager
#   V121 - AGENT-EUDR-033: Continuous Monitoring Agent
#   V122 - AGENT-EUDR-034: Annual Review Scheduler
#   V123 - AGENT-EUDR-035: Improvement Plan Creator
#   V124 - AGENT-EUDR-036: EU Information System Interface
#   V125 - AGENT-EUDR-037: Due Diligence Statement Creator
#   V126 - AGENT-EUDR-038: Reference Number Generator
#   V127 - AGENT-EUDR-039: Customs Declaration Support
#   V128 - AGENT-EUDR-040: Authority Communication Manager
#
# Prerequisites:
#   - Docker installed and running
#   - PostgreSQL database accessible
#   - Flyway Docker image available
#
# Usage:
#   ./apply_v118_v128.sh [environment]
#
# Examples:
#   ./apply_v118_v128.sh dev        # Apply to development database
#   ./apply_v118_v128.sh staging    # Apply to staging database
#   ./apply_v118_v128.sh prod       # Apply to production database
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default environment
ENV="${1:-dev}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="${SCRIPT_DIR}/sql"
CONFIG_FILE="${SCRIPT_DIR}/flyway.conf"

# =============================================================================
# Configuration
# =============================================================================

# Database connection settings (override with environment variables)
DB_HOST="${FLYWAY_DB_HOST:-localhost}"
DB_PORT="${FLYWAY_DB_PORT:-5432}"
DB_NAME="${FLYWAY_DB_NAME:-greenlang_platform}"
DB_USER="${FLYWAY_DB_USER:-greenlang_admin}"
DB_PASSWORD="${FLYWAY_DB_PASSWORD:-greenlang_secure_2024}"

# JDBC URL
JDBC_URL="jdbc:postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Flyway Docker image
FLYWAY_IMAGE="${FLYWAY_IMAGE:-flyway/flyway:9.22.3}"

# =============================================================================
# Helper Functions
# =============================================================================

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

check_docker() {
    if ! docker info &>/dev/null; then
        log_error "Docker is not running. Please start Docker Desktop and try again."
        exit 1
    fi
    log_success "Docker is running"
}

check_database() {
    log_info "Checking database connectivity..."

    if docker run --rm \
        --network host \
        postgres:15-alpine \
        pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" &>/dev/null; then
        log_success "Database is accessible"
    else
        log_error "Cannot connect to database at ${DB_HOST}:${DB_PORT}"
        log_info "Make sure PostgreSQL is running and accessible"
        exit 1
    fi
}

run_flyway() {
    local command="$1"

    log_info "Running Flyway ${command}..."

    docker run --rm \
        --network host \
        -v "${MIGRATIONS_DIR}:/flyway/sql:ro" \
        -v "${CONFIG_FILE}:/flyway/conf/flyway.conf:ro" \
        "${FLYWAY_IMAGE}" \
        -url="${JDBC_URL}" \
        -user="${DB_USER}" \
        -password="${DB_PASSWORD}" \
        -locations="filesystem:/flyway/sql" \
        -configFiles="/flyway/conf/flyway.conf" \
        "${command}"
}

# =============================================================================
# Main Execution
# =============================================================================

echo "============================================================================="
echo "GreenLang Database Migrations V118-V128"
echo "============================================================================="
echo "Environment: ${ENV}"
echo "Database: ${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "============================================================================="
echo ""

# Pre-flight checks
log_info "Running pre-flight checks..."
check_docker
check_database

# Show current migration status
echo ""
log_info "Current migration status:"
run_flyway "info"

# Show pending migrations
echo ""
log_info "Pending migrations:"
run_flyway "info -pending" || true

# Validate migrations
echo ""
log_info "Validating migrations..."
if run_flyway "validate"; then
    log_success "Migration validation passed"
else
    log_error "Migration validation failed"
    exit 1
fi

# Prompt for confirmation (skip in non-interactive mode)
if [ -t 0 ]; then
    echo ""
    echo -e "${YELLOW}Ready to apply migrations V118-V128${NC}"
    read -p "Do you want to proceed? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log_warning "Migration cancelled by user"
        exit 0
    fi
fi

# Apply migrations
echo ""
log_info "Applying migrations V118-V128..."
if run_flyway "migrate"; then
    log_success "Migrations applied successfully"
else
    log_error "Migration failed"
    exit 1
fi

# Show final migration status
echo ""
log_info "Final migration status:"
run_flyway "info"

echo ""
echo "============================================================================="
log_success "Migration complete! Database is now at version V128"
echo "============================================================================="
echo ""
echo "EUDR Agents 030-040 (Documentation & Reporting) are now ready:"
echo "  - V118: Documentation Generator"
echo "  - V119: Stakeholder Engagement Tool"
echo "  - V120: Grievance Mechanism Manager"
echo "  - V121: Continuous Monitoring Agent"
echo "  - V122: Annual Review Scheduler"
echo "  - V123: Improvement Plan Creator"
echo "  - V124: EU Information System Interface"
echo "  - V125: Due Diligence Statement Creator"
echo "  - V126: Reference Number Generator"
echo "  - V127: Customs Declaration Support"
echo "  - V128: Authority Communication Manager"
echo "============================================================================="
