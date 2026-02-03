#!/bin/bash
# ==============================================================================
# Patronictl Wrapper Script
# Provides convenient commands for managing Patroni PostgreSQL HA cluster
# ==============================================================================

set -euo pipefail

# Configuration
CLUSTER_NAME="${PATRONI_SCOPE:-greenlang-cluster}"
NAMESPACE="${POD_NAMESPACE:-greenlang-db}"
PATRONI_API_PORT="${PATRONI_API_PORT:-8008}"
PATRONI_USER="${PATRONI_USER:-patroni}"
PATRONI_PASSWORD="${PATRONI_RESTAPI_PASSWORD:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if patronictl is available
check_patronictl() {
    if ! command -v patronictl &> /dev/null; then
        log_error "patronictl command not found. Make sure Patroni is installed."
        exit 1
    fi
}

# Get cluster status
show_cluster_status() {
    log_info "Fetching cluster status for ${CLUSTER_NAME}..."
    echo ""
    patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
    echo ""

    # Additional status information
    log_info "Cluster topology:"
    patronictl -c /etc/patroni/patroni.yml topology "${CLUSTER_NAME}" 2>/dev/null || true

    echo ""
    log_info "Replication status:"
    psql -U postgres -c "SELECT * FROM pg_stat_replication;" 2>/dev/null || log_warning "Could not fetch replication status (not primary or no replicas)"
}

# Show cluster history
show_history() {
    log_info "Fetching cluster history for ${CLUSTER_NAME}..."
    patronictl -c /etc/patroni/patroni.yml history "${CLUSTER_NAME}"
}

# Perform switchover (planned failover)
do_switchover() {
    local target_leader="${1:-}"
    local current_leader="${2:-}"
    local scheduled_at="${3:-now}"

    log_info "Preparing switchover for cluster ${CLUSTER_NAME}..."

    # Get current cluster state
    echo ""
    show_cluster_status
    echo ""

    if [[ -z "${target_leader}" ]]; then
        log_info "Available members:"
        patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
        echo ""
        read -p "Enter the name of the new leader: " target_leader
    fi

    if [[ -z "${target_leader}" ]]; then
        log_error "Target leader must be specified"
        exit 1
    fi

    log_warning "About to perform switchover to ${target_leader}"
    log_warning "This will cause a brief interruption to write operations"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        log_info "Switchover cancelled"
        exit 0
    fi

    log_info "Performing switchover to ${target_leader}..."

    if [[ -n "${current_leader}" ]]; then
        patronictl -c /etc/patroni/patroni.yml switchover "${CLUSTER_NAME}" \
            --leader "${current_leader}" \
            --candidate "${target_leader}" \
            --scheduled "${scheduled_at}" \
            --force
    else
        patronictl -c /etc/patroni/patroni.yml switchover "${CLUSTER_NAME}" \
            --candidate "${target_leader}" \
            --scheduled "${scheduled_at}" \
            --force
    fi

    log_success "Switchover initiated. Monitoring progress..."
    sleep 5
    show_cluster_status
}

# Perform failover (unplanned/emergency failover)
do_failover() {
    local target_leader="${1:-}"

    log_warning "WARNING: Failover is an emergency operation!"
    log_warning "Use switchover for planned leader changes."
    echo ""

    # Get current cluster state
    show_cluster_status
    echo ""

    if [[ -z "${target_leader}" ]]; then
        log_info "Available members for failover:"
        patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
        echo ""
        read -p "Enter the name of the new leader: " target_leader
    fi

    if [[ -z "${target_leader}" ]]; then
        log_error "Target leader must be specified"
        exit 1
    fi

    log_error "DANGER: This will force a failover to ${target_leader}"
    log_error "Data loss may occur if the current leader has uncommitted transactions"
    echo ""
    read -p "Type 'FAILOVER' to confirm: " confirm

    if [[ "${confirm}" != "FAILOVER" ]]; then
        log_info "Failover cancelled"
        exit 0
    fi

    log_info "Performing failover to ${target_leader}..."
    patronictl -c /etc/patroni/patroni.yml failover "${CLUSTER_NAME}" \
        --candidate "${target_leader}" \
        --force

    log_success "Failover initiated. Monitoring progress..."
    sleep 5
    show_cluster_status
}

# Reinitialize a replica
reinit_replica() {
    local member_name="${1:-}"

    if [[ -z "${member_name}" ]]; then
        log_info "Current cluster members:"
        patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
        echo ""
        read -p "Enter the name of the replica to reinitialize: " member_name
    fi

    if [[ -z "${member_name}" ]]; then
        log_error "Member name must be specified"
        exit 1
    fi

    log_warning "About to reinitialize replica ${member_name}"
    log_warning "This will delete all data on the replica and resync from primary"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        log_info "Reinit cancelled"
        exit 0
    fi

    log_info "Reinitializing replica ${member_name}..."
    patronictl -c /etc/patroni/patroni.yml reinit "${CLUSTER_NAME}" "${member_name}" --force

    log_success "Reinit started. Use 'show-status' to monitor progress."
}

# Restart PostgreSQL on a member
restart_member() {
    local member_name="${1:-}"
    local scheduled_at="${2:-now}"

    if [[ -z "${member_name}" ]]; then
        log_info "Current cluster members:"
        patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
        echo ""
        read -p "Enter the name of the member to restart: " member_name
    fi

    if [[ -z "${member_name}" ]]; then
        log_error "Member name must be specified"
        exit 1
    fi

    log_info "Restarting PostgreSQL on ${member_name}..."
    patronictl -c /etc/patroni/patroni.yml restart "${CLUSTER_NAME}" "${member_name}" \
        --scheduled "${scheduled_at}" \
        --force

    log_success "Restart scheduled."
}

# Reload PostgreSQL configuration
reload_config() {
    local member_name="${1:-}"

    log_info "Reloading PostgreSQL configuration..."

    if [[ -n "${member_name}" ]]; then
        patronictl -c /etc/patroni/patroni.yml reload "${CLUSTER_NAME}" "${member_name}" --force
    else
        patronictl -c /etc/patroni/patroni.yml reload "${CLUSTER_NAME}" --force
    fi

    log_success "Configuration reload initiated."
}

# Pause/Resume automatic failover
pause_cluster() {
    log_info "Pausing automatic failover for cluster ${CLUSTER_NAME}..."
    patronictl -c /etc/patroni/patroni.yml pause "${CLUSTER_NAME}" --wait
    log_success "Cluster paused. Automatic failover is now disabled."
    log_warning "Remember to resume the cluster after maintenance!"
}

resume_cluster() {
    log_info "Resuming automatic failover for cluster ${CLUSTER_NAME}..."
    patronictl -c /etc/patroni/patroni.yml resume "${CLUSTER_NAME}" --wait
    log_success "Cluster resumed. Automatic failover is now enabled."
}

# Edit cluster configuration
edit_config() {
    log_info "Opening cluster configuration editor..."
    patronictl -c /etc/patroni/patroni.yml edit-config "${CLUSTER_NAME}"
}

# Show cluster configuration
show_config() {
    log_info "Current cluster configuration:"
    patronictl -c /etc/patroni/patroni.yml show-config "${CLUSTER_NAME}"
}

# Remove a member from cluster
remove_member() {
    local member_name="${1:-}"

    if [[ -z "${member_name}" ]]; then
        log_info "Current cluster members:"
        patronictl -c /etc/patroni/patroni.yml list "${CLUSTER_NAME}"
        echo ""
        read -p "Enter the name of the member to remove: " member_name
    fi

    if [[ -z "${member_name}" ]]; then
        log_error "Member name must be specified"
        exit 1
    fi

    log_warning "About to remove member ${member_name} from cluster"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        log_info "Remove cancelled"
        exit 0
    fi

    log_info "Removing member ${member_name}..."
    patronictl -c /etc/patroni/patroni.yml remove "${CLUSTER_NAME}" "${member_name}"

    log_success "Member removed from cluster."
}

# Check replication lag
check_lag() {
    log_info "Checking replication lag..."
    echo ""

    # Try to get lag from patroni API
    for pod in $(kubectl get pods -n "${NAMESPACE}" -l app=patroni -o jsonpath='{.items[*].metadata.name}'); do
        local lag=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- curl -s "http://localhost:${PATRONI_API_PORT}/patroni" 2>/dev/null | jq -r '.xlog.replayed_location // "N/A"')
        local role=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- curl -s "http://localhost:${PATRONI_API_PORT}/patroni" 2>/dev/null | jq -r '.role // "unknown"')
        echo "${pod}: role=${role}, xlog=${lag}"
    done

    echo ""
    log_info "Detailed replication status (from primary):"
    psql -U postgres -c "
        SELECT
            client_addr,
            usename,
            application_name,
            state,
            sync_state,
            pg_wal_lsn_diff(pg_current_wal_lsn(), sent_lsn) AS sent_lag_bytes,
            pg_wal_lsn_diff(pg_current_wal_lsn(), write_lsn) AS write_lag_bytes,
            pg_wal_lsn_diff(pg_current_wal_lsn(), flush_lsn) AS flush_lag_bytes,
            pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS replay_lag_bytes
        FROM pg_stat_replication;
    " 2>/dev/null || log_warning "Could not fetch replication status (not primary or no replicas)"
}

# Backup operations
trigger_backup() {
    local backup_type="${1:-full}"

    log_info "Triggering ${backup_type} backup with pgBackRest..."

    case "${backup_type}" in
        full)
            pgbackrest --stanza=greenlang backup --type=full
            ;;
        diff)
            pgbackrest --stanza=greenlang backup --type=diff
            ;;
        incr)
            pgbackrest --stanza=greenlang backup --type=incr
            ;;
        *)
            log_error "Unknown backup type: ${backup_type}"
            log_info "Valid types: full, diff, incr"
            exit 1
            ;;
    esac

    log_success "Backup completed."
}

show_backups() {
    log_info "Available backups:"
    pgbackrest --stanza=greenlang info
}

# Print usage
print_usage() {
    cat << EOF
Patroni Cluster Management Script

Usage: $0 <command> [options]

Commands:
  status, show-status       Show cluster status and topology
  history                   Show cluster history
  switchover [target] [current] [scheduled]
                           Perform planned switchover to target leader
  failover [target]        Perform emergency failover (USE WITH CAUTION)
  reinit [member]          Reinitialize a replica from primary
  restart [member] [scheduled]
                           Restart PostgreSQL on a member
  reload [member]          Reload PostgreSQL configuration
  pause                    Pause automatic failover
  resume                   Resume automatic failover
  edit-config              Edit cluster configuration
  show-config              Show cluster configuration
  remove [member]          Remove a member from cluster
  check-lag                Check replication lag
  backup [full|diff|incr]  Trigger backup
  show-backups             Show available backups
  help                     Show this help message

Examples:
  $0 status                           # Show cluster status
  $0 switchover patroni-1             # Switchover to patroni-1
  $0 switchover patroni-1 patroni-0   # Switchover from patroni-0 to patroni-1
  $0 reinit patroni-2                 # Reinitialize patroni-2
  $0 backup full                      # Trigger full backup

Environment Variables:
  PATRONI_SCOPE            Cluster name (default: greenlang-cluster)
  POD_NAMESPACE            Kubernetes namespace (default: greenlang-db)
  PATRONI_API_PORT         Patroni API port (default: 8008)
  PATRONI_RESTAPI_PASSWORD Patroni REST API password

EOF
}

# Main command handler
main() {
    check_patronictl

    local command="${1:-help}"
    shift || true

    case "${command}" in
        status|show-status)
            show_cluster_status
            ;;
        history)
            show_history
            ;;
        switchover)
            do_switchover "$@"
            ;;
        failover)
            do_failover "$@"
            ;;
        reinit|reinitialize)
            reinit_replica "$@"
            ;;
        restart)
            restart_member "$@"
            ;;
        reload)
            reload_config "$@"
            ;;
        pause)
            pause_cluster
            ;;
        resume)
            resume_cluster
            ;;
        edit-config)
            edit_config
            ;;
        show-config)
            show_config
            ;;
        remove)
            remove_member "$@"
            ;;
        check-lag|lag)
            check_lag
            ;;
        backup)
            trigger_backup "$@"
            ;;
        show-backups|backups)
            show_backups
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: ${command}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
