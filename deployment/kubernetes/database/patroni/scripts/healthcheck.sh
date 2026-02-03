#!/bin/bash
# ==============================================================================
# Patroni Health Check Script
# Comprehensive health checks for PostgreSQL/TimescaleDB HA cluster
# ==============================================================================

set -euo pipefail

# Configuration
PATRONI_API_URL="${PATRONI_API_URL:-http://localhost:8008}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
MAX_REPLICATION_LAG_BYTES="${MAX_REPLICATION_LAG_BYTES:-104857600}"  # 100MB
MAX_REPLICATION_LAG_SECONDS="${MAX_REPLICATION_LAG_SECONDS:-60}"

# Exit codes
EXIT_OK=0
EXIT_WARNING=1
EXIT_CRITICAL=2
EXIT_UNKNOWN=3

# Logging
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warning() {
    echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Check if Patroni API is responsive
check_patroni_api() {
    local response
    local http_code

    response=$(curl -s -w "\n%{http_code}" "${PATRONI_API_URL}/health" 2>/dev/null) || {
        log_error "Failed to connect to Patroni API at ${PATRONI_API_URL}"
        return ${EXIT_CRITICAL}
    }

    http_code=$(echo "${response}" | tail -n1)
    local body=$(echo "${response}" | head -n -1)

    if [[ "${http_code}" == "200" ]]; then
        log_info "Patroni API is healthy"
        return ${EXIT_OK}
    else
        log_error "Patroni API returned HTTP ${http_code}"
        return ${EXIT_CRITICAL}
    fi
}

# Check if this node is the primary
is_primary() {
    local response
    response=$(curl -s "${PATRONI_API_URL}/primary" 2>/dev/null) || return 1

    if [[ -n "${response}" ]]; then
        return 0
    fi
    return 1
}

# Check if this node is a replica
is_replica() {
    local response
    response=$(curl -s "${PATRONI_API_URL}/replica" 2>/dev/null) || return 1

    if [[ -n "${response}" ]]; then
        return 0
    fi
    return 1
}

# Get node role from Patroni
get_role() {
    local role
    role=$(curl -s "${PATRONI_API_URL}/patroni" 2>/dev/null | jq -r '.role // "unknown"')
    echo "${role}"
}

# Check PostgreSQL connectivity
check_postgres_connection() {
    if pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" >/dev/null 2>&1; then
        log_info "PostgreSQL is accepting connections"
        return ${EXIT_OK}
    else
        log_error "PostgreSQL is not accepting connections"
        return ${EXIT_CRITICAL}
    fi
}

# Check if PostgreSQL is running
check_postgres_running() {
    local pg_ctl_status

    if pgrep -x "postgres" > /dev/null 2>&1; then
        log_info "PostgreSQL process is running"
        return ${EXIT_OK}
    else
        log_error "PostgreSQL process is not running"
        return ${EXIT_CRITICAL}
    fi
}

# Check replication status (for primary)
check_replication_primary() {
    local replication_count

    replication_count=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "SELECT count(*) FROM pg_stat_replication WHERE state = 'streaming';" 2>/dev/null | tr -d ' ')

    if [[ -z "${replication_count}" ]]; then
        log_warning "Could not check replication status"
        return ${EXIT_WARNING}
    fi

    if [[ "${replication_count}" -ge 1 ]]; then
        log_info "Primary has ${replication_count} streaming replicas"
        return ${EXIT_OK}
    else
        log_warning "Primary has no streaming replicas"
        return ${EXIT_WARNING}
    fi
}

# Check replication lag (for replica)
check_replication_lag() {
    local lag_bytes
    local lag_seconds

    # Get lag in bytes
    lag_bytes=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "
        SELECT CASE
            WHEN pg_is_in_recovery() THEN
                COALESCE(pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()), 0)
            ELSE 0
        END;" 2>/dev/null | tr -d ' ')

    # Get lag in seconds
    lag_seconds=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "
        SELECT CASE
            WHEN pg_is_in_recovery() THEN
                COALESCE(EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())), 0)
            ELSE 0
        END;" 2>/dev/null | tr -d ' ')

    if [[ -z "${lag_bytes}" ]] || [[ -z "${lag_seconds}" ]]; then
        log_warning "Could not determine replication lag"
        return ${EXIT_WARNING}
    fi

    log_info "Replication lag: ${lag_bytes} bytes, ${lag_seconds} seconds"

    # Check thresholds
    if (( $(echo "${lag_bytes} > ${MAX_REPLICATION_LAG_BYTES}" | bc -l) )); then
        log_error "Replication lag (${lag_bytes} bytes) exceeds threshold (${MAX_REPLICATION_LAG_BYTES} bytes)"
        return ${EXIT_CRITICAL}
    fi

    if (( $(echo "${lag_seconds} > ${MAX_REPLICATION_LAG_SECONDS}" | bc -l) )); then
        log_error "Replication lag (${lag_seconds}s) exceeds threshold (${MAX_REPLICATION_LAG_SECONDS}s)"
        return ${EXIT_CRITICAL}
    fi

    return ${EXIT_OK}
}

# Check disk space
check_disk_space() {
    local data_dir="${PGDATA:-/var/lib/postgresql/data/pgdata}"
    local threshold_percent="${DISK_SPACE_THRESHOLD:-90}"
    local usage_percent

    usage_percent=$(df "${data_dir}" 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')

    if [[ -z "${usage_percent}" ]]; then
        log_warning "Could not determine disk usage"
        return ${EXIT_WARNING}
    fi

    log_info "Disk usage: ${usage_percent}%"

    if [[ "${usage_percent}" -ge "${threshold_percent}" ]]; then
        log_error "Disk usage (${usage_percent}%) exceeds threshold (${threshold_percent}%)"
        return ${EXIT_CRITICAL}
    fi

    return ${EXIT_OK}
}

# Check WAL archive status
check_wal_archive() {
    local failed_count

    failed_count=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "SELECT failed_count FROM pg_stat_archiver;" 2>/dev/null | tr -d ' ')

    if [[ -z "${failed_count}" ]]; then
        log_warning "Could not check WAL archive status"
        return ${EXIT_WARNING}
    fi

    if [[ "${failed_count}" -gt 0 ]]; then
        log_warning "WAL archiver has ${failed_count} failed attempts"
        return ${EXIT_WARNING}
    fi

    log_info "WAL archiver is healthy"
    return ${EXIT_OK}
}

# Check active connections
check_connections() {
    local max_connections
    local current_connections
    local threshold_percent="${CONNECTION_THRESHOLD:-90}"

    max_connections=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "SHOW max_connections;" 2>/dev/null | tr -d ' ')

    current_connections=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ')

    if [[ -z "${max_connections}" ]] || [[ -z "${current_connections}" ]]; then
        log_warning "Could not check connection count"
        return ${EXIT_WARNING}
    fi

    local usage_percent=$((current_connections * 100 / max_connections))
    log_info "Connections: ${current_connections}/${max_connections} (${usage_percent}%)"

    if [[ "${usage_percent}" -ge "${threshold_percent}" ]]; then
        log_error "Connection usage (${usage_percent}%) exceeds threshold (${threshold_percent}%)"
        return ${EXIT_CRITICAL}
    fi

    return ${EXIT_OK}
}

# Check TimescaleDB health
check_timescaledb() {
    local ts_version

    ts_version=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';" 2>/dev/null | tr -d ' ')

    if [[ -z "${ts_version}" ]]; then
        log_warning "TimescaleDB extension not installed"
        return ${EXIT_WARNING}
    fi

    log_info "TimescaleDB version: ${ts_version}"

    # Check for failed background jobs
    local failed_jobs
    failed_jobs=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
        -d postgres -t -c "
        SELECT count(*) FROM timescaledb_information.job_stats
        WHERE last_run_status = 'Failed'
        AND last_run_started_at > now() - interval '1 hour';" 2>/dev/null | tr -d ' ')

    if [[ "${failed_jobs}" -gt 0 ]]; then
        log_warning "TimescaleDB has ${failed_jobs} failed jobs in the last hour"
        return ${EXIT_WARNING}
    fi

    return ${EXIT_OK}
}

# Liveness check - is the service alive?
check_liveness() {
    local exit_code=${EXIT_OK}

    # Check Patroni API
    check_patroni_api || exit_code=${EXIT_CRITICAL}

    # Check PostgreSQL is running
    check_postgres_running || exit_code=${EXIT_CRITICAL}

    return ${exit_code}
}

# Readiness check - is the service ready to accept traffic?
check_readiness() {
    local exit_code=${EXIT_OK}
    local role

    # Check Patroni API
    check_patroni_api || return ${EXIT_CRITICAL}

    # Check PostgreSQL connection
    check_postgres_connection || return ${EXIT_CRITICAL}

    # Get role
    role=$(get_role)
    log_info "Node role: ${role}"

    case "${role}" in
        master|primary)
            # Primary should have at least one replica for HA
            check_replication_primary || exit_code=${EXIT_WARNING}
            ;;
        replica|standby)
            # Replica should have acceptable lag
            check_replication_lag || exit_code=${EXIT_CRITICAL}
            ;;
        *)
            log_warning "Unknown role: ${role}"
            exit_code=${EXIT_WARNING}
            ;;
    esac

    return ${exit_code}
}

# Full health check
check_full() {
    local exit_code=${EXIT_OK}
    local role

    log_info "Starting full health check..."

    # Basic checks
    check_patroni_api || exit_code=${EXIT_CRITICAL}
    check_postgres_running || exit_code=${EXIT_CRITICAL}
    check_postgres_connection || exit_code=${EXIT_CRITICAL}

    # Get role
    role=$(get_role)
    log_info "Node role: ${role}"

    # Role-specific checks
    case "${role}" in
        master|primary)
            check_replication_primary || exit_code=${EXIT_WARNING}
            check_wal_archive || exit_code=${EXIT_WARNING}
            ;;
        replica|standby)
            check_replication_lag || exit_code=${EXIT_CRITICAL}
            ;;
    esac

    # Resource checks
    check_disk_space || exit_code=${EXIT_WARNING}
    check_connections || exit_code=${EXIT_WARNING}

    # TimescaleDB check
    check_timescaledb || exit_code=${EXIT_WARNING}

    log_info "Health check completed with exit code: ${exit_code}"
    return ${exit_code}
}

# Output health status as JSON
output_json() {
    local role
    local is_ready="false"
    local is_live="false"
    local lag_bytes="0"
    local lag_seconds="0"

    role=$(get_role)

    if check_liveness >/dev/null 2>&1; then
        is_live="true"
    fi

    if check_readiness >/dev/null 2>&1; then
        is_ready="true"
    fi

    if [[ "${role}" == "replica" ]] || [[ "${role}" == "standby" ]]; then
        lag_bytes=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
            -d postgres -t -c "
            SELECT CASE
                WHEN pg_is_in_recovery() THEN
                    COALESCE(pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn()), 0)
                ELSE 0
            END;" 2>/dev/null | tr -d ' ' || echo "0")

        lag_seconds=$(psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" \
            -d postgres -t -c "
            SELECT CASE
                WHEN pg_is_in_recovery() THEN
                    COALESCE(EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())), 0)
                ELSE 0
            END;" 2>/dev/null | tr -d ' ' || echo "0")
    fi

    cat << EOF
{
  "role": "${role}",
  "is_live": ${is_live},
  "is_ready": ${is_ready},
  "replication_lag_bytes": ${lag_bytes:-0},
  "replication_lag_seconds": ${lag_seconds:-0},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# Print usage
print_usage() {
    cat << EOF
Patroni Health Check Script

Usage: $0 <command>

Commands:
  liveness     Check if service is alive (for Kubernetes liveness probe)
  readiness    Check if service is ready (for Kubernetes readiness probe)
  full         Perform full health check
  primary      Exit 0 if this node is primary, 1 otherwise
  replica      Exit 0 if this node is replica, 1 otherwise
  json         Output health status as JSON
  help         Show this help message

Exit Codes:
  0 - OK
  1 - Warning
  2 - Critical
  3 - Unknown

Environment Variables:
  PATRONI_API_URL               Patroni API URL (default: http://localhost:8008)
  POSTGRES_HOST                 PostgreSQL host (default: localhost)
  POSTGRES_PORT                 PostgreSQL port (default: 5432)
  POSTGRES_USER                 PostgreSQL user (default: postgres)
  MAX_REPLICATION_LAG_BYTES     Max acceptable lag in bytes (default: 104857600)
  MAX_REPLICATION_LAG_SECONDS   Max acceptable lag in seconds (default: 60)
  DISK_SPACE_THRESHOLD          Disk usage threshold percent (default: 90)
  CONNECTION_THRESHOLD          Connection usage threshold percent (default: 90)

EOF
}

# Main
main() {
    local command="${1:-help}"

    case "${command}" in
        liveness)
            check_liveness
            ;;
        readiness)
            check_readiness
            ;;
        full)
            check_full
            ;;
        primary)
            if is_primary; then
                echo "This node is the primary"
                exit ${EXIT_OK}
            else
                echo "This node is not the primary"
                exit 1
            fi
            ;;
        replica)
            if is_replica; then
                echo "This node is a replica"
                exit ${EXIT_OK}
            else
                echo "This node is not a replica"
                exit 1
            fi
            ;;
        json)
            output_json
            ;;
        help|--help|-h)
            print_usage
            exit ${EXIT_OK}
            ;;
        *)
            log_error "Unknown command: ${command}"
            print_usage
            exit ${EXIT_UNKNOWN}
            ;;
    esac
}

main "$@"
