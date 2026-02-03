#!/bin/bash
#
# redis-health-check.sh
# Comprehensive Redis Health Check Script
#
# This script performs health checks on all Redis instances and Sentinels,
# checking replication status, memory usage, and cluster health.
#
# Usage: ./redis-health-check.sh [--json] [--alert] [--verbose]
#
# Options:
#   --json     Output results in JSON format
#   --alert    Send alerts for critical issues
#   --verbose  Show detailed output
#
# Exit codes:
#   0 - All checks passed
#   1 - Warning (non-critical issues)
#   2 - Critical (service degradation)

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Sentinel configuration
SENTINEL_HOSTS="${SENTINEL_HOSTS:-sentinel-1.greenlang.internal sentinel-2.greenlang.internal sentinel-3.greenlang.internal}"
SENTINEL_PORT="${SENTINEL_PORT:-26379}"
SENTINEL_PASSWORD="${SENTINEL_PASSWORD:-}"

# Redis configuration
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
MASTER_NAME="${MASTER_NAME:-mymaster}"

# Thresholds
MEMORY_WARN_PERCENT=80
MEMORY_CRIT_PERCENT=95
REPLICATION_LAG_WARN=10000
REPLICATION_LAG_CRIT=100000
CONNECTED_CLIENTS_WARN=8000
CONNECTED_CLIENTS_CRIT=9500

# Alert configuration
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"

# Output options
JSON_OUTPUT=false
ALERT_MODE=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
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

send_alert() {
    local severity="$1"
    local message="$2"

    if [ "$ALERT_MODE" = true ] && [ -n "$ALERT_WEBHOOK" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"severity\": \"$severity\", \"message\": \"$message\", \"source\": \"redis-health-check\"}" \
            "$ALERT_WEBHOOK" > /dev/null
    fi
}

# =============================================================================
# Health Check Functions
# =============================================================================

check_sentinel_health() {
    local sentinel_status=()
    local sentinels_up=0
    local sentinels_down=0
    local quorum_status="UNKNOWN"

    echo "=== Sentinel Cluster Health ==="
    echo ""

    for sentinel in $SENTINEL_HOSTS; do
        log_verbose "Checking sentinel: $sentinel"

        local ping_result
        ping_result=$(sentinel_cmd "$sentinel" "PING")

        if [ "$ping_result" == "PONG" ]; then
            sentinels_up=$((sentinels_up + 1))

            # Get master info from this sentinel
            local master_ip
            master_ip=$(sentinel_cmd "$sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME" | head -1)

            # Get sentinel info
            local sentinel_info
            sentinel_info=$(sentinel_cmd "$sentinel" "INFO sentinel")
            local masters_monitored
            masters_monitored=$(echo "$sentinel_info" | grep "sentinel_masters:" | cut -d: -f2 | tr -d '\r')

            echo -e "  $sentinel: ${GREEN}UP${NC}"
            echo "    - Masters monitored: ${masters_monitored:-1}"
            echo "    - Sees master: $master_ip"

            sentinel_status+=("{\"host\": \"$sentinel\", \"status\": \"up\", \"master\": \"$master_ip\"}")
        else
            sentinels_down=$((sentinels_down + 1))
            echo -e "  $sentinel: ${RED}DOWN${NC}"
            sentinel_status+=("{\"host\": \"$sentinel\", \"status\": \"down\"}")
        fi
    done

    echo ""

    # Check quorum
    if [ $sentinels_up -gt 0 ]; then
        local first_sentinel
        first_sentinel=$(echo "$SENTINEL_HOSTS" | awk '{print $1}')
        local quorum_result
        quorum_result=$(sentinel_cmd "$first_sentinel" "SENTINEL CKQUORUM $MASTER_NAME" 2>&1)

        if echo "$quorum_result" | grep -q "OK"; then
            quorum_status="OK"
            echo -e "  Quorum: ${GREEN}OK${NC}"
        else
            quorum_status="FAILED"
            echo -e "  Quorum: ${RED}FAILED${NC} - $quorum_result"
            send_alert "critical" "Redis Sentinel quorum lost"
        fi
    fi

    echo ""
    echo "  Summary: $sentinels_up up, $sentinels_down down"

    # Return status
    if [ $sentinels_down -gt 0 ] || [ "$quorum_status" != "OK" ]; then
        return 2
    fi
    return 0
}

check_redis_master() {
    echo "=== Redis Master Health ==="
    echo ""

    # Get master address from sentinel
    local first_sentinel
    first_sentinel=$(echo "$SENTINEL_HOSTS" | awk '{print $1}')
    local master_addr
    master_addr=$(sentinel_cmd "$first_sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME")
    local master_ip
    master_ip=$(echo "$master_addr" | head -1)
    local master_port
    master_port=$(echo "$master_addr" | tail -1)

    if [ -z "$master_ip" ]; then
        echo -e "  Master: ${RED}UNKNOWN${NC} (cannot get address from Sentinel)"
        return 2
    fi

    echo "  Master: $master_ip:$master_port"

    # Check connectivity
    local ping_result
    ping_result=$(redis_cmd "$master_ip" "$master_port" "PING")

    if [ "$ping_result" != "PONG" ]; then
        echo -e "  Status: ${RED}DOWN${NC}"
        send_alert "critical" "Redis master is not responding"
        return 2
    fi

    echo -e "  Status: ${GREEN}UP${NC}"

    # Get detailed info
    local info
    info=$(redis_cmd "$master_ip" "$master_port" "INFO all")

    # Role verification
    local role
    role=$(echo "$info" | grep "^role:" | cut -d: -f2 | tr -d '\r')
    echo "  Role: $role"

    if [ "$role" != "master" ]; then
        echo -e "    ${RED}WARNING: Expected master role${NC}"
        return 2
    fi

    # Connected replicas
    local connected_slaves
    connected_slaves=$(echo "$info" | grep "^connected_slaves:" | cut -d: -f2 | tr -d '\r')
    echo "  Connected replicas: $connected_slaves"

    # Memory usage
    local used_memory
    used_memory=$(echo "$info" | grep "^used_memory:" | cut -d: -f2 | tr -d '\r')
    local used_memory_human
    used_memory_human=$(echo "$info" | grep "^used_memory_human:" | cut -d: -f2 | tr -d '\r')
    local maxmemory
    maxmemory=$(echo "$info" | grep "^maxmemory:" | cut -d: -f2 | tr -d '\r')

    if [ "$maxmemory" -gt 0 ]; then
        local memory_percent
        memory_percent=$((used_memory * 100 / maxmemory))
        echo -n "  Memory: $used_memory_human / $(echo "$info" | grep "^maxmemory_human:" | cut -d: -f2 | tr -d '\r') ($memory_percent%)"

        if [ $memory_percent -ge $MEMORY_CRIT_PERCENT ]; then
            echo -e " ${RED}CRITICAL${NC}"
            send_alert "critical" "Redis master memory at ${memory_percent}%"
        elif [ $memory_percent -ge $MEMORY_WARN_PERCENT ]; then
            echo -e " ${YELLOW}WARNING${NC}"
            send_alert "warning" "Redis master memory at ${memory_percent}%"
        else
            echo -e " ${GREEN}OK${NC}"
        fi
    else
        echo "  Memory: $used_memory_human (no limit set)"
    fi

    # Connected clients
    local connected_clients
    connected_clients=$(echo "$info" | grep "^connected_clients:" | cut -d: -f2 | tr -d '\r')
    echo -n "  Connected clients: $connected_clients"

    if [ "$connected_clients" -ge $CONNECTED_CLIENTS_CRIT ]; then
        echo -e " ${RED}CRITICAL${NC}"
    elif [ "$connected_clients" -ge $CONNECTED_CLIENTS_WARN ]; then
        echo -e " ${YELLOW}WARNING${NC}"
    else
        echo -e " ${GREEN}OK${NC}"
    fi

    # Ops per second
    local ops_per_sec
    ops_per_sec=$(echo "$info" | grep "^instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d '\r')
    echo "  Ops/sec: $ops_per_sec"

    # Blocked clients
    local blocked_clients
    blocked_clients=$(echo "$info" | grep "^blocked_clients:" | cut -d: -f2 | tr -d '\r')
    echo -n "  Blocked clients: $blocked_clients"

    if [ "$blocked_clients" -gt 0 ]; then
        echo -e " ${YELLOW}WARNING${NC}"
    else
        echo -e " ${GREEN}OK${NC}"
    fi

    # Fragmentation ratio
    local frag_ratio
    frag_ratio=$(echo "$info" | grep "^mem_fragmentation_ratio:" | cut -d: -f2 | tr -d '\r')
    echo -n "  Fragmentation ratio: $frag_ratio"

    # Check if frag_ratio is numeric and do comparison
    if [[ "$frag_ratio" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        if (( $(echo "$frag_ratio > 1.5" | bc -l 2>/dev/null || echo 0) )); then
            echo -e " ${YELLOW}WARNING (high)${NC}"
        elif (( $(echo "$frag_ratio < 1.0" | bc -l 2>/dev/null || echo 0) )); then
            echo -e " ${YELLOW}WARNING (low)${NC}"
        else
            echo -e " ${GREEN}OK${NC}"
        fi
    else
        echo ""
    fi

    echo ""
    return 0
}

check_replication_status() {
    echo "=== Replication Status ==="
    echo ""

    # Get master address
    local first_sentinel
    first_sentinel=$(echo "$SENTINEL_HOSTS" | awk '{print $1}')
    local master_addr
    master_addr=$(sentinel_cmd "$first_sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME")
    local master_ip
    master_ip=$(echo "$master_addr" | head -1)
    local master_port
    master_port=$(echo "$master_addr" | tail -1)

    if [ -z "$master_ip" ]; then
        echo -e "  ${RED}Cannot determine master address${NC}"
        return 2
    fi

    # Get master replication info
    local master_info
    master_info=$(redis_cmd "$master_ip" "$master_port" "INFO replication")
    local master_offset
    master_offset=$(echo "$master_info" | grep "^master_repl_offset:" | cut -d: -f2 | tr -d '\r')

    echo "  Master offset: $master_offset"
    echo ""

    # Get replicas from Sentinel
    local slaves_info
    slaves_info=$(sentinel_cmd "$first_sentinel" "SENTINEL slaves $MASTER_NAME")

    local status_code=0
    local replica_count=0

    # Parse slave info (format: name, ip, port, etc. in pairs)
    while IFS= read -r line; do
        if echo "$line" | grep -q "^ip"; then
            replica_count=$((replica_count + 1))

            # Get IP from next line
            read -r ip_line
            local replica_ip
            replica_ip=$(echo "$ip_line" | tr -d '\r')

            # Skip to port
            while IFS= read -r line; do
                if echo "$line" | grep -q "^port"; then
                    read -r port_line
                    local replica_port
                    replica_port=$(echo "$port_line" | tr -d '\r')
                    break
                fi
            done

            echo "  Replica $replica_count: $replica_ip:$replica_port"

            # Check replica health
            local replica_ping
            replica_ping=$(redis_cmd "$replica_ip" "$replica_port" "PING")

            if [ "$replica_ping" != "PONG" ]; then
                echo -e "    Status: ${RED}DOWN${NC}"
                status_code=2
                continue
            fi

            echo -e "    Status: ${GREEN}UP${NC}"

            # Get replica replication info
            local replica_info
            replica_info=$(redis_cmd "$replica_ip" "$replica_port" "INFO replication")

            local replica_offset
            replica_offset=$(echo "$replica_info" | grep "^slave_repl_offset:" | cut -d: -f2 | tr -d '\r')
            local master_link_status
            master_link_status=$(echo "$replica_info" | grep "^master_link_status:" | cut -d: -f2 | tr -d '\r')

            echo -n "    Master link: "
            if [ "$master_link_status" == "up" ]; then
                echo -e "${GREEN}UP${NC}"
            else
                echo -e "${RED}DOWN${NC}"
                status_code=2
            fi

            # Calculate lag
            if [ -n "$replica_offset" ] && [ -n "$master_offset" ]; then
                local lag
                lag=$((master_offset - replica_offset))
                echo -n "    Replication lag: $lag bytes"

                if [ $lag -ge $REPLICATION_LAG_CRIT ]; then
                    echo -e " ${RED}CRITICAL${NC}"
                    status_code=2
                    send_alert "critical" "Redis replica $replica_ip has critical replication lag: $lag bytes"
                elif [ $lag -ge $REPLICATION_LAG_WARN ]; then
                    echo -e " ${YELLOW}WARNING${NC}"
                    [ $status_code -lt 2 ] && status_code=1
                else
                    echo -e " ${GREEN}OK${NC}"
                fi
            fi

            echo ""
        fi
    done < <(echo "$slaves_info")

    if [ $replica_count -eq 0 ]; then
        echo -e "  ${YELLOW}No replicas configured${NC}"
        status_code=1
    fi

    return $status_code
}

check_persistence_status() {
    echo "=== Persistence Status ==="
    echo ""

    # Get master address
    local first_sentinel
    first_sentinel=$(echo "$SENTINEL_HOSTS" | awk '{print $1}')
    local master_addr
    master_addr=$(sentinel_cmd "$first_sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME")
    local master_ip
    master_ip=$(echo "$master_addr" | head -1)
    local master_port
    master_port=$(echo "$master_addr" | tail -1)

    local info
    info=$(redis_cmd "$master_ip" "$master_port" "INFO persistence")

    # RDB status
    local rdb_bgsave_in_progress
    rdb_bgsave_in_progress=$(echo "$info" | grep "^rdb_bgsave_in_progress:" | cut -d: -f2 | tr -d '\r')
    local rdb_last_bgsave_status
    rdb_last_bgsave_status=$(echo "$info" | grep "^rdb_last_bgsave_status:" | cut -d: -f2 | tr -d '\r')
    local rdb_last_save_time
    rdb_last_save_time=$(echo "$info" | grep "^rdb_last_save_time:" | cut -d: -f2 | tr -d '\r')

    echo "  RDB:"
    echo "    Last save: $(date -d @$rdb_last_save_time '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')"
    echo -n "    Last status: "
    if [ "$rdb_last_bgsave_status" == "ok" ]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi
    echo "    Save in progress: $rdb_bgsave_in_progress"

    # AOF status
    local aof_enabled
    aof_enabled=$(echo "$info" | grep "^aof_enabled:" | cut -d: -f2 | tr -d '\r')

    echo ""
    echo "  AOF:"
    if [ "$aof_enabled" == "1" ]; then
        local aof_rewrite_in_progress
        aof_rewrite_in_progress=$(echo "$info" | grep "^aof_rewrite_in_progress:" | cut -d: -f2 | tr -d '\r')
        local aof_last_rewrite_status
        aof_last_rewrite_status=$(echo "$info" | grep "^aof_last_bgrewrite_status:" | cut -d: -f2 | tr -d '\r')
        local aof_current_size
        aof_current_size=$(echo "$info" | grep "^aof_current_size:" | cut -d: -f2 | tr -d '\r')

        echo -e "    Enabled: ${GREEN}YES${NC}"
        echo -n "    Last rewrite status: "
        if [ "$aof_last_rewrite_status" == "ok" ]; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
        fi
        echo "    Current size: $((aof_current_size / 1024 / 1024)) MB"
        echo "    Rewrite in progress: $aof_rewrite_in_progress"
    else
        echo -e "    Enabled: ${YELLOW}NO${NC}"
    fi

    echo ""
    return 0
}

generate_health_summary() {
    echo "=============================================="
    echo "           HEALTH CHECK SUMMARY              "
    echo "=============================================="
    echo ""
    echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo ""

    local overall_status="HEALTHY"
    local overall_code=0

    # Run all checks and collect results
    if ! check_sentinel_health; then
        overall_status="DEGRADED"
        overall_code=2
    fi
    echo ""

    if ! check_redis_master; then
        overall_status="DEGRADED"
        overall_code=2
    fi
    echo ""

    local repl_result
    if ! check_replication_status; then
        repl_result=$?
        if [ $repl_result -eq 2 ]; then
            overall_status="DEGRADED"
            overall_code=2
        elif [ $repl_result -eq 1 ] && [ "$overall_status" == "HEALTHY" ]; then
            overall_status="WARNING"
            overall_code=1
        fi
    fi
    echo ""

    check_persistence_status
    echo ""

    echo "=============================================="
    if [ "$overall_status" == "HEALTHY" ]; then
        echo -e "  Overall Status: ${GREEN}$overall_status${NC}"
    elif [ "$overall_status" == "WARNING" ]; then
        echo -e "  Overall Status: ${YELLOW}$overall_status${NC}"
    else
        echo -e "  Overall Status: ${RED}$overall_status${NC}"
    fi
    echo "=============================================="

    return $overall_code
}

# =============================================================================
# JSON Output
# =============================================================================

generate_json_output() {
    local first_sentinel
    first_sentinel=$(echo "$SENTINEL_HOSTS" | awk '{print $1}')

    # Get master info
    local master_addr
    master_addr=$(sentinel_cmd "$first_sentinel" "SENTINEL get-master-addr-by-name $MASTER_NAME")
    local master_ip
    master_ip=$(echo "$master_addr" | head -1)
    local master_port
    master_port=$(echo "$master_addr" | tail -1)

    local info
    info=$(redis_cmd "$master_ip" "$master_port" "INFO all")

    # Extract metrics
    local used_memory=$(echo "$info" | grep "^used_memory:" | cut -d: -f2 | tr -d '\r')
    local maxmemory=$(echo "$info" | grep "^maxmemory:" | cut -d: -f2 | tr -d '\r')
    local connected_clients=$(echo "$info" | grep "^connected_clients:" | cut -d: -f2 | tr -d '\r')
    local connected_slaves=$(echo "$info" | grep "^connected_slaves:" | cut -d: -f2 | tr -d '\r')
    local ops_per_sec=$(echo "$info" | grep "^instantaneous_ops_per_sec:" | cut -d: -f2 | tr -d '\r')

    # Count healthy sentinels
    local sentinels_up=0
    for sentinel in $SENTINEL_HOSTS; do
        if [ "$(sentinel_cmd "$sentinel" "PING")" == "PONG" ]; then
            sentinels_up=$((sentinels_up + 1))
        fi
    done

    cat << EOF
{
  "timestamp": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "master": {
    "host": "$master_ip",
    "port": $master_port,
    "status": "up"
  },
  "metrics": {
    "used_memory": $used_memory,
    "maxmemory": ${maxmemory:-0},
    "connected_clients": $connected_clients,
    "connected_replicas": $connected_slaves,
    "ops_per_sec": $ops_per_sec
  },
  "sentinel": {
    "healthy_count": $sentinels_up,
    "total_count": $(echo "$SENTINEL_HOSTS" | wc -w)
  }
}
EOF
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --json     Output results in JSON format"
    echo "  --alert    Send alerts for critical issues"
    echo "  --verbose  Show detailed debug output"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SENTINEL_HOSTS    Space-separated list of sentinel hosts"
    echo "  SENTINEL_PORT     Sentinel port (default: 26379)"
    echo "  SENTINEL_PASSWORD Sentinel password"
    echo "  REDIS_PASSWORD    Redis password"
    echo "  MASTER_NAME       Master name (default: mymaster)"
    echo "  ALERT_WEBHOOK     Webhook URL for alerts"
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --json)
                JSON_OUTPUT=true
                shift
                ;;
            --alert)
                ALERT_MODE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
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

    if [ "$JSON_OUTPUT" = true ]; then
        generate_json_output
        exit 0
    fi

    generate_health_summary
    exit $?
}

main "$@"
