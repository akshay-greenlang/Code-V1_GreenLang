#!/bin/bash
#
# redis-sentinel-reset.sh
# Redis Sentinel State Reset Script
#
# This script resets the Sentinel cluster state to recover from
# desynchronization, stale data, or configuration issues.
#
# Usage: ./redis-sentinel-reset.sh [--soft|--hard] [--sentinel-hosts HOSTS]
#
# Options:
#   --soft    Soft reset using SENTINEL RESET command (default)
#   --hard    Hard reset: stop, clear state, restart
#   --force   Skip confirmation prompts
#
# Exit codes:
#   0 - Reset successful
#   1 - Reset failed
#   2 - Pre-checks failed

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SENTINEL_HOSTS="${SENTINEL_HOSTS:-sentinel-1.greenlang.internal sentinel-2.greenlang.internal sentinel-3.greenlang.internal}"
SENTINEL_PORT="${SENTINEL_PORT:-26379}"
SENTINEL_PASSWORD="${SENTINEL_PASSWORD:-}"
MASTER_NAME="${MASTER_NAME:-mymaster}"
MASTER_IP="${MASTER_IP:-}"
MASTER_PORT="${MASTER_PORT:-6379}"
QUORUM="${QUORUM:-2}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

RESET_MODE="soft"
FORCE_MODE=false

# Sentinel configuration template for hard reset
SENTINEL_CONFIG_DIR="${SENTINEL_CONFIG_DIR:-/etc/redis}"
SENTINEL_DATA_DIR="${SENTINEL_DATA_DIR:-/var/lib/redis}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

sentinel_cmd() {
    local host="$1"
    local cmd="$2"
    if [ -n "$SENTINEL_PASSWORD" ]; then
        timeout 5 redis-cli -h "$host" -p "$SENTINEL_PORT" -a "$SENTINEL_PASSWORD" --no-auth-warning $cmd 2>/dev/null || echo ""
    else
        timeout 5 redis-cli -h "$host" -p "$SENTINEL_PORT" $cmd 2>/dev/null || echo ""
    fi
}

redis_cmd() {
    local host="$1"
    local port="$2"
    local cmd="$3"
    if [ -n "$REDIS_PASSWORD" ]; then
        timeout 5 redis-cli -h "$host" -p "$port" -a "$REDIS_PASSWORD" --no-auth-warning $cmd 2>/dev/null || echo ""
    else
        timeout 5 redis-cli -h "$host" -p "$port" $cmd 2>/dev/null || echo ""
    fi
}

print_banner() {
    echo "=============================================="
    echo "      Redis Sentinel Reset Script            "
    echo "=============================================="
    echo "Timestamp:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Reset Mode: $RESET_MODE"
    echo "Sentinels:  $SENTINEL_HOSTS"
    echo "Master:     $MASTER_NAME"
    echo "=============================================="
}

# =============================================================================
# Pre-Reset Checks
# =============================================================================

discover_master() {
    log_info "Discovering current master..."

    # Try to get master from any responsive Sentinel
    for sentinel in $SENTINEL_HOSTS; do
        local master_addr
        master_addr=$(sentinel_cmd "$sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME")

        if [ -n "$master_addr" ]; then
            MASTER_IP=$(echo "$master_addr" | head -1)
            MASTER_PORT=$(echo "$master_addr" | tail -1)
            log_info "Found master from $sentinel: $MASTER_IP:$MASTER_PORT"
            return 0
        fi
    done

    # If no Sentinel responded, try to find master manually
    if [ -z "$MASTER_IP" ]; then
        log_warning "Could not discover master from Sentinels"
        log_info "Attempting to find master by checking Redis instances..."

        # This would need to be customized for your environment
        # For now, prompt for manual input
        read -p "Enter master IP: " MASTER_IP
        read -p "Enter master port [6379]: " input_port
        MASTER_PORT="${input_port:-6379}"
    fi

    # Verify master
    local role
    role=$(redis_cmd "$MASTER_IP" "$MASTER_PORT" "INFO replication" | grep "^role:" | cut -d: -f2 | tr -d '\r')

    if [ "$role" != "master" ]; then
        log_error "$MASTER_IP:$MASTER_PORT is not a master (role: $role)"
        return 1
    fi

    log_success "Verified master: $MASTER_IP:$MASTER_PORT"
    return 0
}

pre_reset_checks() {
    log_info "Running pre-reset checks..."

    # Check 1: At least one Sentinel is reachable
    echo -n "  [1/3] Sentinel reachability... "
    local reachable=0
    for sentinel in $SENTINEL_HOSTS; do
        if [ "$(sentinel_cmd "$sentinel" "PING")" == "PONG" ]; then
            reachable=$((reachable + 1))
        fi
    done

    if [ $reachable -gt 0 ]; then
        echo -e "${GREEN}OK${NC} ($reachable reachable)"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "No Sentinels are reachable"
        return 2
    fi

    # Check 2: Master discovery
    echo -n "  [2/3] Master discovery... "
    if discover_master; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        return 2
    fi

    # Check 3: Master is responsive
    echo -n "  [3/3] Master connectivity... "
    local ping_result
    ping_result=$(redis_cmd "$MASTER_IP" "$MASTER_PORT" "PING")
    if [ "$ping_result" == "PONG" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        log_error "Master is not responding"
        return 2
    fi

    log_success "Pre-reset checks passed"
    return 0
}

# =============================================================================
# Soft Reset
# =============================================================================

perform_soft_reset() {
    log_info "Performing soft reset..."
    echo ""

    local success_count=0
    local fail_count=0

    for sentinel in $SENTINEL_HOSTS; do
        echo -n "  Resetting $sentinel... "

        # Check if Sentinel is reachable
        if [ "$(sentinel_cmd "$sentinel" "PING")" != "PONG" ]; then
            echo -e "${YELLOW}SKIPPED (unreachable)${NC}"
            fail_count=$((fail_count + 1))
            continue
        fi

        # Execute SENTINEL RESET
        local reset_result
        reset_result=$(sentinel_cmd "$sentinel" "SENTINEL RESET $MASTER_NAME")

        if [ -n "$reset_result" ]; then
            echo -e "${GREEN}OK${NC} (reset $reset_result masters)"
            success_count=$((success_count + 1))
        else
            echo -e "${RED}FAILED${NC}"
            fail_count=$((fail_count + 1))
        fi
    done

    echo ""
    log_info "Reset complete: $success_count succeeded, $fail_count failed"

    if [ $success_count -eq 0 ]; then
        log_error "No Sentinels were reset successfully"
        return 1
    fi

    return 0
}

# =============================================================================
# Hard Reset
# =============================================================================

perform_hard_reset() {
    log_info "Performing hard reset..."
    log_warning "This will stop Sentinel services, clear state, and restart"
    echo ""

    if [ "$FORCE_MODE" = false ]; then
        read -p "Are you sure you want to perform a hard reset? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Hard reset cancelled"
            return 1
        fi
    fi

    for sentinel in $SENTINEL_HOSTS; do
        echo ""
        echo "--- Processing $sentinel ---"

        # Step 1: Check if we can SSH to the sentinel
        echo -n "  [1/5] Checking SSH access... "
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$sentinel" "echo ok" &>/dev/null; then
            echo -e "${RED}FAILED${NC}"
            log_warning "Cannot SSH to $sentinel, skipping"
            continue
        fi
        echo -e "${GREEN}OK${NC}"

        # Step 2: Stop Sentinel service
        echo -n "  [2/5] Stopping Sentinel service... "
        ssh "$sentinel" "sudo systemctl stop redis-sentinel" 2>/dev/null || true
        sleep 2

        if ssh "$sentinel" "pgrep -x redis-sentinel" &>/dev/null; then
            echo -e "${RED}FAILED${NC}"
            log_warning "Sentinel service still running on $sentinel"
            continue
        fi
        echo -e "${GREEN}OK${NC}"

        # Step 3: Backup and remove old Sentinel state
        echo -n "  [3/5] Clearing Sentinel state... "
        local timestamp
        timestamp=$(date +%Y%m%d-%H%M%S)

        ssh "$sentinel" "
            sudo cp $SENTINEL_CONFIG_DIR/sentinel.conf $SENTINEL_CONFIG_DIR/sentinel.conf.bak-$timestamp 2>/dev/null || true
            sudo rm -f $SENTINEL_DATA_DIR/sentinel-*.conf 2>/dev/null || true
        "
        echo -e "${GREEN}OK${NC}"

        # Step 4: Create fresh Sentinel configuration
        echo -n "  [4/5] Creating fresh configuration... "

        # Generate configuration
        local sentinel_config="
port $SENTINEL_PORT
daemonize yes
pidfile /var/run/redis/redis-sentinel.pid
logfile /var/log/redis/sentinel.log
dir $SENTINEL_DATA_DIR

sentinel monitor $MASTER_NAME $MASTER_IP $MASTER_PORT $QUORUM
sentinel down-after-milliseconds $MASTER_NAME 5000
sentinel parallel-syncs $MASTER_NAME 1
sentinel failover-timeout $MASTER_NAME 60000
"

        # Add authentication if configured
        if [ -n "$REDIS_PASSWORD" ]; then
            sentinel_config+="
sentinel auth-pass $MASTER_NAME $REDIS_PASSWORD
"
        fi

        if [ -n "$SENTINEL_PASSWORD" ]; then
            sentinel_config+="
requirepass $SENTINEL_PASSWORD
"
        fi

        # Write configuration
        echo "$sentinel_config" | ssh "$sentinel" "sudo tee $SENTINEL_CONFIG_DIR/sentinel.conf > /dev/null"
        ssh "$sentinel" "sudo chown redis:redis $SENTINEL_CONFIG_DIR/sentinel.conf"
        echo -e "${GREEN}OK${NC}"

        # Step 5: Start Sentinel service
        echo -n "  [5/5] Starting Sentinel service... "
        ssh "$sentinel" "sudo systemctl start redis-sentinel"
        sleep 3

        if ssh "$sentinel" "redis-cli -p $SENTINEL_PORT ${SENTINEL_PASSWORD:+-a $SENTINEL_PASSWORD} --no-auth-warning PING" 2>/dev/null | grep -q "PONG"; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
            log_warning "Sentinel on $sentinel may not have started correctly"
        fi
    done

    return 0
}

# =============================================================================
# Post-Reset Validation
# =============================================================================

post_reset_validation() {
    log_info "Running post-reset validation..."
    echo ""

    # Wait for Sentinels to discover each other
    log_info "Waiting 30 seconds for Sentinel discovery..."
    sleep 30

    local healthy_sentinels=0
    local master_agreement=true
    local expected_master="$MASTER_IP:$MASTER_PORT"
    local first_seen_master=""

    for sentinel in $SENTINEL_HOSTS; do
        echo -n "  Checking $sentinel... "

        # Check connectivity
        if [ "$(sentinel_cmd "$sentinel" "PING")" != "PONG" ]; then
            echo -e "${RED}DOWN${NC}"
            continue
        fi

        # Check master view
        local seen_master
        seen_master=$(sentinel_cmd "$sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME" | tr '\n' ':' | sed 's/:$//')

        if [ -z "$first_seen_master" ]; then
            first_seen_master="$seen_master"
        elif [ "$seen_master" != "$first_seen_master" ]; then
            master_agreement=false
        fi

        # Check quorum
        local quorum_ok
        quorum_ok=$(sentinel_cmd "$sentinel" "SENTINEL CKQUORUM $MASTER_NAME" 2>&1 | grep -c "OK" || echo 0)

        if [ "$quorum_ok" -gt 0 ]; then
            echo -e "${GREEN}OK${NC} (sees master: $seen_master)"
            healthy_sentinels=$((healthy_sentinels + 1))
        else
            echo -e "${YELLOW}WARNING${NC} (quorum issue)"
        fi
    done

    echo ""

    # Summary
    echo "  Summary:"
    echo "    Healthy Sentinels: $healthy_sentinels / $(echo "$SENTINEL_HOSTS" | wc -w)"
    echo -n "    Master agreement:  "
    if [ "$master_agreement" = true ]; then
        echo -e "${GREEN}YES${NC}"
    else
        echo -e "${RED}NO${NC}"
    fi
    echo "    Observed master:   $first_seen_master"

    echo ""

    if [ $healthy_sentinels -ge $QUORUM ] && [ "$master_agreement" = true ]; then
        log_success "Sentinel cluster is healthy"
        return 0
    else
        log_warning "Sentinel cluster may need additional attention"
        return 1
    fi
}

# =============================================================================
# Re-configure Quorum
# =============================================================================

reconfigure_quorum() {
    log_info "Reconfiguring quorum across all Sentinels..."

    for sentinel in $SENTINEL_HOSTS; do
        if [ "$(sentinel_cmd "$sentinel" "PING")" == "PONG" ]; then
            sentinel_cmd "$sentinel" "SENTINEL SET $MASTER_NAME quorum $QUORUM"
            log_info "Set quorum to $QUORUM on $sentinel"
        fi
    done

    # Flush configuration to disk
    for sentinel in $SENTINEL_HOSTS; do
        if [ "$(sentinel_cmd "$sentinel" "PING")" == "PONG" ]; then
            sentinel_cmd "$sentinel" "SENTINEL FLUSHCONFIG"
        fi
    done

    log_success "Quorum reconfigured"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Reset modes:"
    echo "  --soft     Soft reset using SENTINEL RESET (default)"
    echo "  --hard     Hard reset: stop, clear state, restart"
    echo ""
    echo "Options:"
    echo "  --force              Skip confirmation prompts"
    echo "  --sentinel-hosts     Space-separated list of sentinel hosts"
    echo "  --master-ip IP       Master IP (auto-discovered if not set)"
    echo "  --master-port PORT   Master port (default: 6379)"
    echo "  --quorum N           Quorum value (default: 2)"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SENTINEL_HOSTS       Space-separated list of sentinel hosts"
    echo "  SENTINEL_PORT        Sentinel port (default: 26379)"
    echo "  SENTINEL_PASSWORD    Sentinel password"
    echo "  REDIS_PASSWORD       Redis password"
    echo "  MASTER_NAME          Master name (default: mymaster)"
    echo ""
    echo "Examples:"
    echo "  $0 --soft                    # Soft reset all sentinels"
    echo "  $0 --hard --force            # Hard reset without prompts"
    echo "  $0 --soft --quorum 3         # Reset and set quorum to 3"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --soft)
                RESET_MODE="soft"
                shift
                ;;
            --hard)
                RESET_MODE="hard"
                shift
                ;;
            --force)
                FORCE_MODE=true
                shift
                ;;
            --sentinel-hosts)
                SENTINEL_HOSTS="$2"
                shift 2
                ;;
            --master-ip)
                MASTER_IP="$2"
                shift 2
                ;;
            --master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --quorum)
                QUORUM="$2"
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

    # Pre-reset checks
    if ! pre_reset_checks; then
        log_error "Pre-reset checks failed"
        exit 2
    fi

    echo ""

    # Confirmation for non-force mode
    if [ "$FORCE_MODE" = false ]; then
        log_warning "This will reset the Sentinel cluster state."
        log_warning "This may cause brief monitoring interruption."
        read -p "Do you want to proceed? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Reset cancelled"
            exit 0
        fi
    fi

    echo ""

    # Perform reset based on mode
    case $RESET_MODE in
        soft)
            if ! perform_soft_reset; then
                log_error "Soft reset failed"
                exit 1
            fi
            ;;
        hard)
            if ! perform_hard_reset; then
                log_error "Hard reset failed"
                exit 1
            fi
            ;;
    esac

    echo ""

    # Reconfigure quorum
    reconfigure_quorum

    echo ""

    # Post-reset validation
    if ! post_reset_validation; then
        log_warning "Post-reset validation found issues"
        echo ""
        echo "Recommended actions:"
        echo "  1. Check Sentinel logs: tail -f /var/log/redis/sentinel.log"
        echo "  2. Verify network connectivity between Sentinels"
        echo "  3. Check if master is accessible from all Sentinels"
        echo "  4. Wait a few more minutes for full discovery"
        exit 1
    fi

    echo ""
    echo "=============================================="
    log_success "SENTINEL RESET COMPLETED SUCCESSFULLY"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Monitor Sentinel logs for any issues"
    echo "  2. Verify application connectivity"
    echo "  3. Test failover with: redis-cli -p $SENTINEL_PORT SENTINEL FAILOVER $MASTER_NAME"
    echo ""
}

main "$@"
