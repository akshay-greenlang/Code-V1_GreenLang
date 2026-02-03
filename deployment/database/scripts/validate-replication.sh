#!/bin/bash
#===============================================================================
# validate-replication.sh
#
# Validates PostgreSQL replication health including streaming replication,
# WAL shipping, replication slots, and overall cluster health.
#
# Usage: ./validate-replication.sh [--json] [--verbose] [--alert-threshold <seconds>]
#
# Author: GreenLang Database Operations Team
# Version: 1.0.0
# Date: 2026-02-03
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
PATRONI_CONFIG="${PATRONI_CONFIG:-/etc/patroni/patroni.yml}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-greenlang}"

# Alert thresholds
LAG_THRESHOLD_BYTES="${LAG_THRESHOLD_BYTES:-10485760}"  # 10MB
LAG_THRESHOLD_SECONDS="${LAG_THRESHOLD_SECONDS:-60}"     # 60 seconds
WAL_ARCHIVE_LAG_THRESHOLD="${WAL_ARCHIVE_LAG_THRESHOLD:-10}"  # 10 WAL files
SLOT_INACTIVE_THRESHOLD="${SLOT_INACTIVE_THRESHOLD:-3600}"    # 1 hour

# Output options
OUTPUT_JSON=false
VERBOSE=false

#-------------------------------------------------------------------------------
# Color Output
#-------------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# Output Functions
#-------------------------------------------------------------------------------
print_header() {
    if [ "$OUTPUT_JSON" != "true" ]; then
        echo ""
        echo "========================================"
        echo "$1"
        echo "========================================"
    fi
}

print_status() {
    local status="$1"
    local message="$2"

    if [ "$OUTPUT_JSON" != "true" ]; then
        case "$status" in
            "OK")      echo -e "[${GREEN}OK${NC}] $message" ;;
            "WARNING") echo -e "[${YELLOW}WARNING${NC}] $message" ;;
            "ERROR")   echo -e "[${RED}ERROR${NC}] $message" ;;
            *)         echo "[$status] $message" ;;
        esac
    fi
}

print_verbose() {
    if [ "$VERBOSE" == "true" ] && [ "$OUTPUT_JSON" != "true" ]; then
        echo "    $1"
    fi
}

#-------------------------------------------------------------------------------
# Database Connection
#-------------------------------------------------------------------------------
run_query() {
    local query="$1"
    local host="${2:-$DB_HOST}"

    psql -h "$host" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "$query" 2>/dev/null
}

run_query_json() {
    local query="$1"
    local host="${2:-$DB_HOST}"

    psql -h "$host" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -A -c "
        SELECT json_agg(row_to_json(t)) FROM ($query) t;
    " 2>/dev/null | sed 's/^$/[]/'
}

#-------------------------------------------------------------------------------
# Cluster Status Check
#-------------------------------------------------------------------------------
check_cluster_status() {
    print_header "Cluster Status"

    local cluster_status
    cluster_status=$(patronictl -c "$PATRONI_CONFIG" list -f json 2>/dev/null || echo "[]")

    if [ "$cluster_status" == "[]" ]; then
        print_status "ERROR" "Cannot retrieve cluster status"
        return 1
    fi

    local primary_count
    primary_count=$(echo "$cluster_status" | jq '[.[] | select(.Role == "Leader")] | length')

    local total_nodes
    total_nodes=$(echo "$cluster_status" | jq 'length')

    local running_nodes
    running_nodes=$(echo "$cluster_status" | jq '[.[] | select(.State == "running")] | length')

    if [ "$primary_count" -eq 1 ]; then
        print_status "OK" "Cluster has exactly one primary"
    elif [ "$primary_count" -eq 0 ]; then
        print_status "ERROR" "No primary node found"
        return 1
    else
        print_status "ERROR" "Multiple primary nodes detected (split-brain!)"
        return 1
    fi

    if [ "$running_nodes" -eq "$total_nodes" ]; then
        print_status "OK" "All $total_nodes nodes are running"
    else
        print_status "WARNING" "$running_nodes of $total_nodes nodes are running"
    fi

    # Display node details
    if [ "$VERBOSE" == "true" ] && [ "$OUTPUT_JSON" != "true" ]; then
        echo ""
        echo "Node Details:"
        patronictl -c "$PATRONI_CONFIG" list
    fi

    echo "$cluster_status"
}

#-------------------------------------------------------------------------------
# Replication Lag Check
#-------------------------------------------------------------------------------
check_replication_lag() {
    print_header "Replication Lag"

    local replication_info
    replication_info=$(run_query_json "
        SELECT
            client_addr::text,
            usename,
            application_name,
            state,
            sync_state,
            pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
            pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS total_lag_bytes,
            sent_lsn::text,
            write_lsn::text,
            flush_lsn::text,
            replay_lsn::text,
            EXTRACT(EPOCH FROM (now() - reply_time))::int AS reply_age_seconds
        FROM pg_stat_replication
    ")

    if [ -z "$replication_info" ] || [ "$replication_info" == "null" ] || [ "$replication_info" == "[]" ]; then
        print_status "WARNING" "No active replication connections"
        return 0
    fi

    local replica_count
    replica_count=$(echo "$replication_info" | jq 'length')

    print_status "OK" "Found $replica_count active replication connections"

    local has_issues=false

    # Check each replica
    echo "$replication_info" | jq -c '.[]' | while read -r replica; do
        local client_addr
        client_addr=$(echo "$replica" | jq -r '.client_addr // "unknown"')

        local app_name
        app_name=$(echo "$replica" | jq -r '.application_name // "unknown"')

        local lag_bytes
        lag_bytes=$(echo "$replica" | jq -r '.lag_bytes // 0')

        local sync_state
        sync_state=$(echo "$replica" | jq -r '.sync_state // "unknown"')

        local state
        state=$(echo "$replica" | jq -r '.state // "unknown"')

        # Format lag
        local lag_display
        if [ "$lag_bytes" -lt 1024 ]; then
            lag_display="${lag_bytes} bytes"
        elif [ "$lag_bytes" -lt 1048576 ]; then
            lag_display="$(echo "scale=2; $lag_bytes/1024" | bc) KB"
        else
            lag_display="$(echo "scale=2; $lag_bytes/1048576" | bc) MB"
        fi

        # Check if lag exceeds threshold
        if [ "$lag_bytes" -gt "$LAG_THRESHOLD_BYTES" ]; then
            print_status "WARNING" "$app_name ($client_addr): lag=$lag_display, state=$state, sync=$sync_state"
            has_issues=true
        else
            print_status "OK" "$app_name ($client_addr): lag=$lag_display, state=$state, sync=$sync_state"
        fi

        print_verbose "  sent_lsn=$(echo "$replica" | jq -r '.sent_lsn')"
        print_verbose "  replay_lsn=$(echo "$replica" | jq -r '.replay_lsn')"
    done

    echo "$replication_info"
}

#-------------------------------------------------------------------------------
# WAL Archive Check
#-------------------------------------------------------------------------------
check_wal_archive() {
    print_header "WAL Archive Status"

    local archive_info
    archive_info=$(run_query_json "
        SELECT
            archived_count,
            last_archived_wal,
            last_archived_time::text,
            failed_count,
            last_failed_wal,
            last_failed_time::text,
            EXTRACT(EPOCH FROM (now() - last_archived_time))::int AS archive_age_seconds
        FROM pg_stat_archiver
    ")

    if [ -z "$archive_info" ] || [ "$archive_info" == "null" ] || [ "$archive_info" == "[]" ]; then
        print_status "WARNING" "WAL archiving may not be enabled"
        return 0
    fi

    local archived_count
    archived_count=$(echo "$archive_info" | jq -r '.[0].archived_count // 0')

    local failed_count
    failed_count=$(echo "$archive_info" | jq -r '.[0].failed_count // 0')

    local last_archived_time
    last_archived_time=$(echo "$archive_info" | jq -r '.[0].last_archived_time // "never"')

    local archive_age
    archive_age=$(echo "$archive_info" | jq -r '.[0].archive_age_seconds // 0')

    print_status "OK" "Archived WAL files: $archived_count"

    if [ "$failed_count" -gt 0 ]; then
        print_status "WARNING" "Failed archive attempts: $failed_count"
        print_verbose "Last failed WAL: $(echo "$archive_info" | jq -r '.[0].last_failed_wal // "unknown"')"
    else
        print_status "OK" "No failed archive attempts"
    fi

    if [ "$archive_age" -gt 300 ]; then
        print_status "WARNING" "Last archive was $archive_age seconds ago"
    else
        print_status "OK" "Last archived at: $last_archived_time"
    fi

    # Check WAL ready for archive
    local wal_ready
    wal_ready=$(run_query "SELECT count(*) FROM pg_ls_archive_statusdir() WHERE name LIKE '%.ready';")

    if [ -n "$wal_ready" ] && [ "$wal_ready" -gt "$WAL_ARCHIVE_LAG_THRESHOLD" ]; then
        print_status "WARNING" "$wal_ready WAL files waiting to be archived"
    else
        print_status "OK" "WAL archive queue: ${wal_ready:-0} files"
    fi

    echo "$archive_info"
}

#-------------------------------------------------------------------------------
# Replication Slot Check
#-------------------------------------------------------------------------------
check_replication_slots() {
    print_header "Replication Slots"

    local slot_info
    slot_info=$(run_query_json "
        SELECT
            slot_name,
            slot_type,
            active,
            restart_lsn::text,
            confirmed_flush_lsn::text,
            pg_wal_lsn_diff(pg_current_wal_lsn(), restart_lsn) AS retained_bytes,
            wal_status
        FROM pg_replication_slots
    ")

    if [ -z "$slot_info" ] || [ "$slot_info" == "null" ] || [ "$slot_info" == "[]" ]; then
        print_status "OK" "No replication slots configured"
        return 0
    fi

    local slot_count
    slot_count=$(echo "$slot_info" | jq 'length')

    local active_slots
    active_slots=$(echo "$slot_info" | jq '[.[] | select(.active == true)] | length')

    local inactive_slots
    inactive_slots=$(echo "$slot_info" | jq '[.[] | select(.active == false)] | length')

    print_status "OK" "Total replication slots: $slot_count (active: $active_slots, inactive: $inactive_slots)"

    # Check each slot
    echo "$slot_info" | jq -c '.[]' | while read -r slot; do
        local slot_name
        slot_name=$(echo "$slot" | jq -r '.slot_name')

        local slot_type
        slot_type=$(echo "$slot" | jq -r '.slot_type')

        local active
        active=$(echo "$slot" | jq -r '.active')

        local retained_bytes
        retained_bytes=$(echo "$slot" | jq -r '.retained_bytes // 0')

        local wal_status
        wal_status=$(echo "$slot" | jq -r '.wal_status // "unknown"')

        # Format retained size
        local retained_display
        if [ "$retained_bytes" -lt 1048576 ]; then
            retained_display="$(echo "scale=2; $retained_bytes/1024" | bc) KB"
        elif [ "$retained_bytes" -lt 1073741824 ]; then
            retained_display="$(echo "scale=2; $retained_bytes/1048576" | bc) MB"
        else
            retained_display="$(echo "scale=2; $retained_bytes/1073741824" | bc) GB"
        fi

        if [ "$active" == "false" ]; then
            print_status "WARNING" "Slot '$slot_name' ($slot_type) is INACTIVE - retained: $retained_display"
        elif [ "$wal_status" != "normal" ] && [ "$wal_status" != "reserved" ]; then
            print_status "WARNING" "Slot '$slot_name' ($slot_type) has WAL status: $wal_status"
        else
            print_status "OK" "Slot '$slot_name' ($slot_type): active, retained: $retained_display"
        fi
    done

    echo "$slot_info"
}

#-------------------------------------------------------------------------------
# Synchronous Replication Check
#-------------------------------------------------------------------------------
check_synchronous_replication() {
    print_header "Synchronous Replication"

    local sync_config
    sync_config=$(run_query "SHOW synchronous_standby_names;")

    if [ -z "$sync_config" ] || [ "$sync_config" == "" ]; then
        print_status "WARNING" "Synchronous replication is not configured"
        return 0
    fi

    print_status "OK" "synchronous_standby_names = $sync_config"

    # Check for sync standby
    local sync_standbys
    sync_standbys=$(run_query "SELECT count(*) FROM pg_stat_replication WHERE sync_state = 'sync';")

    local quorum
    quorum=$(run_query "SELECT count(*) FROM pg_stat_replication WHERE sync_state IN ('sync', 'quorum');")

    if [ "${sync_standbys:-0}" -gt 0 ]; then
        print_status "OK" "Synchronous standbys: $sync_standbys"
    elif [ "${quorum:-0}" -gt 0 ]; then
        print_status "OK" "Quorum standbys: $quorum"
    else
        print_status "ERROR" "No synchronous standbys available!"
        return 1
    fi

    # Check synchronous_commit setting
    local sync_commit
    sync_commit=$(run_query "SHOW synchronous_commit;")
    print_verbose "synchronous_commit = $sync_commit"

    return 0
}

#-------------------------------------------------------------------------------
# Timeline Check
#-------------------------------------------------------------------------------
check_timeline() {
    print_header "Timeline Status"

    local primary_timeline
    primary_timeline=$(run_query "SELECT timeline_id FROM pg_control_checkpoint();")

    print_status "OK" "Primary timeline: $primary_timeline"

    # Check replica timelines via Patroni
    local cluster_info
    cluster_info=$(patronictl -c "$PATRONI_CONFIG" list -f json 2>/dev/null || echo "[]")

    local timeline_issues=false

    echo "$cluster_info" | jq -c '.[]' | while read -r node; do
        local member
        member=$(echo "$node" | jq -r '.Member')

        local timeline
        timeline=$(echo "$node" | jq -r '.TL // "unknown"')

        if [ "$timeline" != "$primary_timeline" ] && [ "$timeline" != "unknown" ]; then
            print_status "WARNING" "Node $member has different timeline: $timeline"
            timeline_issues=true
        else
            print_verbose "Node $member: timeline $timeline"
        fi
    done

    if [ "$timeline_issues" == "false" ]; then
        print_status "OK" "All nodes on same timeline"
    fi
}

#-------------------------------------------------------------------------------
# Connection Check
#-------------------------------------------------------------------------------
check_connections() {
    print_header "Connection Status"

    local conn_info
    conn_info=$(run_query_json "
        SELECT
            sum(numbackends)::int AS total_connections,
            (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_connections,
            (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') AS active_queries,
            (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') AS idle_connections,
            (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction') AS idle_in_transaction
        FROM pg_stat_database
        WHERE datname = '$DB_NAME'
    ")

    local total
    total=$(echo "$conn_info" | jq -r '.[0].total_connections // 0')

    local max
    max=$(echo "$conn_info" | jq -r '.[0].max_connections // 100')

    local active
    active=$(echo "$conn_info" | jq -r '.[0].active_queries // 0')

    local idle_txn
    idle_txn=$(echo "$conn_info" | jq -r '.[0].idle_in_transaction // 0')

    local usage_pct
    usage_pct=$(echo "scale=1; $total * 100 / $max" | bc)

    if (( $(echo "$usage_pct > 90" | bc -l) )); then
        print_status "WARNING" "Connection usage: ${usage_pct}% ($total/$max)"
    elif (( $(echo "$usage_pct > 75" | bc -l) )); then
        print_status "WARNING" "Connection usage: ${usage_pct}% ($total/$max)"
    else
        print_status "OK" "Connection usage: ${usage_pct}% ($total/$max)"
    fi

    print_verbose "Active queries: $active"
    print_verbose "Idle in transaction: $idle_txn"

    if [ "$idle_txn" -gt 10 ]; then
        print_status "WARNING" "$idle_txn connections idle in transaction"
    fi

    echo "$conn_info"
}

#-------------------------------------------------------------------------------
# Generate Health Report
#-------------------------------------------------------------------------------
generate_report() {
    local cluster_status="$1"
    local replication_info="$2"
    local archive_info="$3"
    local slot_info="$4"
    local conn_info="$5"

    local overall_status="healthy"
    local issues=()

    # Check for issues
    local primary_count
    primary_count=$(echo "$cluster_status" | jq '[.[] | select(.Role == "Leader")] | length')

    if [ "$primary_count" -ne 1 ]; then
        overall_status="critical"
        issues+=("Primary count: $primary_count")
    fi

    # Check replication lag
    local max_lag
    max_lag=$(echo "$replication_info" | jq '[.[] | .lag_bytes // 0] | max // 0')

    if [ "$max_lag" -gt "$LAG_THRESHOLD_BYTES" ]; then
        overall_status="warning"
        issues+=("Replication lag: $max_lag bytes")
    fi

    # Generate JSON report
    cat << EOF
{
    "timestamp": "$(date -Iseconds)",
    "overall_status": "$overall_status",
    "checks": {
        "cluster": $(echo "$cluster_status" | jq -c '.'),
        "replication": $(echo "$replication_info" | jq -c '.'),
        "wal_archive": $(echo "$archive_info" | jq -c '.'),
        "replication_slots": $(echo "$slot_info" | jq -c '.'),
        "connections": $(echo "$conn_info" | jq -c '.')
    },
    "thresholds": {
        "lag_bytes": $LAG_THRESHOLD_BYTES,
        "lag_seconds": $LAG_THRESHOLD_SECONDS,
        "wal_archive_lag": $WAL_ARCHIVE_LAG_THRESHOLD
    },
    "issues": $(printf '%s\n' "${issues[@]:-}" | jq -R . | jq -s .)
}
EOF
}

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
main() {
    if [ "$OUTPUT_JSON" != "true" ]; then
        echo "=========================================="
        echo "PostgreSQL Replication Health Check"
        echo "Timestamp: $(date)"
        echo "Host: $DB_HOST:$DB_PORT"
        echo "=========================================="
    fi

    # Run all checks
    local cluster_status
    cluster_status=$(check_cluster_status)

    local replication_info
    replication_info=$(check_replication_lag)

    local archive_info
    archive_info=$(check_wal_archive)

    local slot_info
    slot_info=$(check_replication_slots)

    check_synchronous_replication
    check_timeline

    local conn_info
    conn_info=$(check_connections)

    # Generate report
    if [ "$OUTPUT_JSON" == "true" ]; then
        generate_report "$cluster_status" "$replication_info" "$archive_info" "$slot_info" "$conn_info"
    else
        print_header "Summary"
        echo "Replication validation complete."
        echo ""
    fi
}

#-------------------------------------------------------------------------------
# Argument Parsing
#-------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Validates PostgreSQL replication health

OPTIONS:
    --json                    Output results in JSON format
    --verbose, -v             Enable verbose output
    --lag-threshold <bytes>   Replication lag threshold (default: 10485760)
    --host <host>             Database host (default: localhost)
    --port <port>             Database port (default: 5432)
    -h, --help                Show this help message

ENVIRONMENT VARIABLES:
    PATRONI_CONFIG            Path to Patroni config
    DB_HOST                   Database host
    DB_PORT                   Database port
    DB_USER                   Database user
    DB_NAME                   Database name
    LAG_THRESHOLD_BYTES       Replication lag threshold in bytes

EXAMPLES:
    # Basic check
    $(basename "$0")

    # JSON output
    $(basename "$0") --json

    # Verbose mode
    $(basename "$0") --verbose

    # Custom threshold
    $(basename "$0") --lag-threshold 52428800

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            OUTPUT_JSON=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --lag-threshold)
            LAG_THRESHOLD_BYTES="$2"
            shift 2
            ;;
        --host)
            DB_HOST="$2"
            shift 2
            ;;
        --port)
            DB_PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main
