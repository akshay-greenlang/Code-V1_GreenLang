#!/bin/bash
#
# redis-manual-failover.sh
# Manual Redis Failover Script using Sentinel
#
# This script performs a controlled failover of the Redis master
# using Sentinel's FAILOVER command.
#
# Usage: ./redis-manual-failover.sh [--force] [--sentinel-host HOST] [--master-name NAME]
#
# Options:
#   --force         Skip confirmation prompts
#   --sentinel-host Sentinel host (default: sentinel.greenlang.internal)
#   --master-name   Master name (default: mymaster)
#
# Exit codes:
#   0 - Failover successful
#   1 - Failover failed
#   2 - Pre-flight checks failed
#   3 - User cancelled

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SENTINEL_HOST="${SENTINEL_HOST:-sentinel.greenlang.internal}"
SENTINEL_PORT="${SENTINEL_PORT:-26379}"
MASTER_NAME="${MASTER_NAME:-mymaster}"
SENTINEL_PASSWORD="${SENTINEL_PASSWORD:-}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
FORCE_MODE=false
FAILOVER_TIMEOUT=60
CHECK_INTERVAL=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Functions
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

print_banner() {
    echo "=============================================="
    echo "       Redis Manual Failover Script          "
    echo "=============================================="
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Sentinel:  $SENTINEL_HOST:$SENTINEL_PORT"
    echo "Master:    $MASTER_NAME"
    echo "=============================================="
}

sentinel_cmd() {
    local cmd="$1"
    if [ -n "$SENTINEL_PASSWORD" ]; then
        redis-cli -h "$SENTINEL_HOST" -p "$SENTINEL_PORT" -a "$SENTINEL_PASSWORD" --no-auth-warning $cmd 2>/dev/null
    else
        redis-cli -h "$SENTINEL_HOST" -p "$SENTINEL_PORT" $cmd 2>/dev/null
    fi
}

redis_cmd() {
    local host="$1"
    local port="$2"
    local cmd="$3"
    if [ -n "$REDIS_PASSWORD" ]; then
        redis-cli -h "$host" -p "$port" -a "$REDIS_PASSWORD" --no-auth-warning $cmd 2>/dev/null
    else
        redis-cli -h "$host" -p "$port" $cmd 2>/dev/null
    fi
}

get_master_info() {
    local info
    info=$(sentinel_cmd "SENTINEL master $MASTER_NAME")
    if [ -z "$info" ]; then
        return 1
    fi
    echo "$info"
}

get_master_address() {
    sentinel_cmd "SENTINEL get-master-addr-by-name $MASTER_NAME" | head -2 | tr '\n' ':' | sed 's/:$//'
}

get_replicas() {
    sentinel_cmd "SENTINEL slaves $MASTER_NAME"
}

check_quorum() {
    sentinel_cmd "SENTINEL CKQUORUM $MASTER_NAME"
}

check_sentinel_connectivity() {
    local ping_result
    ping_result=$(sentinel_cmd "PING")
    if [ "$ping_result" == "PONG" ]; then
        return 0
    fi
    return 1
}

check_master_connectivity() {
    local master_addr
    master_addr=$(get_master_address)
    local master_ip="${master_addr%:*}"
    local master_port="${master_addr#*:}"

    local ping_result
    ping_result=$(redis_cmd "$master_ip" "$master_port" "PING")
    if [ "$ping_result" == "PONG" ]; then
        return 0
    fi
    return 1
}

get_replica_count() {
    get_master_info | grep -A 1 "num-slaves" | tail -1
}

get_master_flags() {
    get_master_info | grep -A 1 "^flags" | tail -1
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check 1: Sentinel connectivity
    echo -n "  [1/5] Sentinel connectivity... "
    if check_sentinel_connectivity; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "Cannot connect to Sentinel at $SENTINEL_HOST:$SENTINEL_PORT"
        return 2
    fi

    # Check 2: Quorum
    echo -n "  [2/5] Sentinel quorum... "
    quorum_result=$(check_quorum)
    if echo "$quorum_result" | grep -q "OK"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "Quorum check failed: $quorum_result"
        return 2
    fi

    # Check 3: Master status
    echo -n "  [3/5] Master status... "
    master_flags=$(get_master_flags)
    if echo "$master_flags" | grep -q "master"; then
        echo -e "${GREEN}OK${NC} (flags: $master_flags)"
    else
        echo -e "${YELLOW}WARNING${NC} (flags: $master_flags)"
        log_warning "Master has unexpected flags"
    fi

    # Check 4: Replicas available
    echo -n "  [4/5] Replica availability... "
    replica_count=$(get_replica_count)
    if [ "$replica_count" -gt 0 ]; then
        echo -e "${GREEN}OK${NC} ($replica_count replicas)"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "No replicas available for failover"
        return 2
    fi

    # Check 5: Replication lag
    echo -n "  [5/5] Replication lag... "
    local master_addr
    master_addr=$(get_master_address)
    local master_ip="${master_addr%:*}"
    local master_port="${master_addr#*:}"

    master_offset=$(redis_cmd "$master_ip" "$master_port" "INFO replication" | grep master_repl_offset | cut -d: -f2 | tr -d '\r')

    lag_ok=true
    while IFS= read -r line; do
        if echo "$line" | grep -q "slave_repl_offset"; then
            slave_offset=$(echo "$line" | cut -d: -f2 | tr -d '\r')
            lag=$((master_offset - slave_offset))
            if [ "$lag" -gt 10000 ]; then
                lag_ok=false
            fi
        fi
    done < <(redis_cmd "$master_ip" "$master_port" "INFO replication")

    if [ "$lag_ok" = true ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}WARNING${NC} (high replication lag detected)"
        log_warning "High replication lag may result in data loss"
    fi

    log_success "Pre-flight checks passed"
    return 0
}

# =============================================================================
# Failover Execution
# =============================================================================

execute_failover() {
    local current_master
    local new_master
    local start_time
    local elapsed

    # Get current master
    current_master=$(get_master_address)
    log_info "Current master: $current_master"

    # Trigger failover
    log_info "Triggering Sentinel FAILOVER command..."
    failover_result=$(sentinel_cmd "SENTINEL FAILOVER $MASTER_NAME")

    if [ "$failover_result" != "OK" ]; then
        log_error "Failover command failed: $failover_result"
        return 1
    fi

    log_info "Failover initiated. Waiting for completion..."

    # Wait for failover to complete
    start_time=$(date +%s)
    while true; do
        sleep $CHECK_INTERVAL

        new_master=$(get_master_address)
        elapsed=$(($(date +%s) - start_time))

        echo -n "  Elapsed: ${elapsed}s - Current master: $new_master"

        if [ "$new_master" != "$current_master" ]; then
            echo -e " ${GREEN}[CHANGED]${NC}"
            log_success "Failover completed!"
            log_info "Old master: $current_master"
            log_info "New master: $new_master"
            return 0
        fi

        echo " [waiting...]"

        if [ $elapsed -ge $FAILOVER_TIMEOUT ]; then
            log_error "Failover timeout after ${FAILOVER_TIMEOUT}s"
            return 1
        fi
    done
}

# =============================================================================
# Post-Failover Validation
# =============================================================================

post_failover_validation() {
    log_info "Running post-failover validation..."

    local new_master
    new_master=$(get_master_address)
    local master_ip="${new_master%:*}"
    local master_port="${new_master#*:}"

    # Check 1: New master is responding
    echo -n "  [1/4] New master responding... "
    ping_result=$(redis_cmd "$master_ip" "$master_port" "PING")
    if [ "$ping_result" == "PONG" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "New master is not responding"
        return 1
    fi

    # Check 2: New master role
    echo -n "  [2/4] New master role... "
    role=$(redis_cmd "$master_ip" "$master_port" "INFO replication" | grep "^role:" | cut -d: -f2 | tr -d '\r')
    if [ "$role" == "master" ]; then
        echo -e "${GREEN}OK${NC} (role: master)"
    else
        echo -e "${RED}FAILED${NC} (role: $role)"
        log_error "New master has incorrect role"
        return 1
    fi

    # Check 3: Replicas connected
    echo -n "  [3/4] Replicas connected... "
    connected_slaves=$(redis_cmd "$master_ip" "$master_port" "INFO replication" | grep "connected_slaves:" | cut -d: -f2 | tr -d '\r')
    if [ "$connected_slaves" -gt 0 ]; then
        echo -e "${GREEN}OK${NC} ($connected_slaves replicas)"
    else
        echo -e "${YELLOW}WARNING${NC} (no replicas connected yet)"
        log_warning "Replicas may still be reconnecting"
    fi

    # Check 4: Write test
    echo -n "  [4/4] Write test... "
    test_key="_failover_validation_$(date +%s)"
    set_result=$(redis_cmd "$master_ip" "$master_port" "SET $test_key test_value EX 60")
    get_result=$(redis_cmd "$master_ip" "$master_port" "GET $test_key")
    redis_cmd "$master_ip" "$master_port" "DEL $test_key" > /dev/null

    if [ "$get_result" == "test_value" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "Write test failed on new master"
        return 1
    fi

    log_success "Post-failover validation passed"
    return 0
}

# =============================================================================
# Update Application Configs (optional)
# =============================================================================

update_application_configs() {
    log_info "Checking if application config update is needed..."

    # If using Sentinel-aware clients, no update needed
    log_info "Applications using Sentinel-aware clients will auto-discover the new master"

    # Optional: If you need to update Kubernetes ConfigMaps or environment variables
    # Uncomment and customize the following:

    # log_info "Updating Kubernetes ConfigMap..."
    # kubectl patch configmap redis-config -n greenlang \
    #   --type merge \
    #   -p "{\"data\":{\"REDIS_MASTER\": \"$new_master_ip\"}}"

    # log_info "Restarting application pods..."
    # kubectl rollout restart deployment/greenlang-app -n greenlang

    log_success "Application configuration update complete"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --force              Skip confirmation prompts"
    echo "  --sentinel-host HOST Sentinel host (default: $SENTINEL_HOST)"
    echo "  --sentinel-port PORT Sentinel port (default: $SENTINEL_PORT)"
    echo "  --master-name NAME   Master name (default: $MASTER_NAME)"
    echo "  --timeout SECONDS    Failover timeout (default: $FAILOVER_TIMEOUT)"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SENTINEL_HOST        Sentinel host"
    echo "  SENTINEL_PORT        Sentinel port"
    echo "  SENTINEL_PASSWORD    Sentinel password (if authentication enabled)"
    echo "  REDIS_PASSWORD       Redis password (if authentication enabled)"
    echo "  MASTER_NAME          Master name"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE_MODE=true
                shift
                ;;
            --sentinel-host)
                SENTINEL_HOST="$2"
                shift 2
                ;;
            --sentinel-port)
                SENTINEL_PORT="$2"
                shift 2
                ;;
            --master-name)
                MASTER_NAME="$2"
                shift 2
                ;;
            --timeout)
                FAILOVER_TIMEOUT="$2"
                shift 2
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    print_banner

    # Pre-flight checks
    if ! preflight_checks; then
        log_error "Pre-flight checks failed. Aborting failover."
        exit 2
    fi

    # Confirmation
    if [ "$FORCE_MODE" = false ]; then
        echo ""
        log_warning "This will trigger a failover of the Redis master."
        log_warning "Applications may experience brief connection interruptions."
        read -p "Do you want to proceed? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Failover cancelled by user"
            exit 3
        fi
    fi

    echo ""

    # Execute failover
    if ! execute_failover; then
        log_error "Failover execution failed"
        exit 1
    fi

    echo ""

    # Post-failover validation
    if ! post_failover_validation; then
        log_error "Post-failover validation failed"
        log_warning "Manual investigation required"
        exit 1
    fi

    echo ""

    # Update application configs
    update_application_configs

    echo ""
    echo "=============================================="
    log_success "FAILOVER COMPLETED SUCCESSFULLY"
    echo "=============================================="
    echo ""
    echo "Summary:"
    echo "  New Master: $(get_master_address)"
    echo "  Replicas:   $(get_replica_count)"
    echo "  Status:     $(get_master_flags)"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor application logs for Redis connection errors"
    echo "  2. Check Grafana dashboard for metric anomalies"
    echo "  3. Investigate the old master if it failed unexpectedly"
    echo ""
}

main "$@"
