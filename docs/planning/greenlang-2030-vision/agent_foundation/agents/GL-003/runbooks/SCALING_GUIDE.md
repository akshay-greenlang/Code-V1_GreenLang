# GL-003 SteamSystemAnalyzer Scaling Guide

## Document Control

**Document Version:** 1.0.0
**Last Updated:** 2025-11-17
**Owner:** GL-003 Platform Engineering Team
**Reviewers:** Infrastructure, Performance Engineering
**Next Review:** 2025-12-17

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scaling Architecture Overview](#scaling-architecture-overview)
3. [Horizontal Scaling](#horizontal-scaling)
4. [Vertical Scaling](#vertical-scaling)
5. [Database Scaling](#database-scaling)
6. [Steam Meter Count Scaling](#steam-meter-count-scaling)
7. [High-Frequency Monitoring](#high-frequency-monitoring)
8. [Large Facility Support](#large-facility-support)
9. [Distributed Steam Analysis](#distributed-steam-analysis)
10. [Performance Optimization](#performance-optimization)
11. [Cost Optimization](#cost-optimization)
12. [Capacity Planning](#capacity-planning)
13. [Auto-Scaling Configuration](#auto-scaling-configuration)
14. [Scaling Procedures](#scaling-procedures)
15. [Monitoring and Metrics](#monitoring-and-metrics)
16. [Troubleshooting](#troubleshooting)
17. [Appendices](#appendices)

---

## Executive Summary

This guide provides comprehensive procedures for scaling GL-003 SteamSystemAnalyzer to meet growing demands in steam system monitoring. Whether you're adding more meters, increasing sampling frequency, or expanding to larger facilities, this guide covers:

**Scaling Dimensions:**
- **Horizontal Scaling:** 3-10 replicas based on meter count
- **Vertical Scaling:** CPU/memory optimization for high-frequency monitoring
- **Database Scaling:** PostgreSQL + TimescaleDB for time-series data
- **Geographic Distribution:** Multi-region deployment for global facilities

**Scaling Triggers:**
- Steam meter count: >500 meters per facility
- Sampling frequency: >10Hz (10 samples/second)
- Facility size: >50 monitoring zones
- Geographic distribution: Multi-region deployments
- High availability: 99.95% uptime requirements

**Performance Targets:**
| Metric | Baseline | Target (Scaled) |
|--------|----------|-----------------|
| Response Time | <200ms | <500ms |
| Throughput | 1K readings/sec | 50K readings/sec |
| Meter Support | 100 meters | 5,000 meters |
| Data Retention | 30 days | 2 years |
| Availability | 99.9% | 99.95% |

---

## Scaling Architecture Overview

### Current Architecture (Baseline)

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                           │
│                    (Kubernetes Service)                      │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬───────────┐
        ▼                       ▼           ▼
    ┌───────┐              ┌───────┐   ┌───────┐
    │  Pod  │              │  Pod  │   │  Pod  │
    │  #1   │              │  #2   │   │  #3   │
    └───┬───┘              └───┬───┘   └───┬───┘
        │                      │           │
        └──────────┬───────────┴───────────┘
                   ▼
          ┌─────────────────┐
          │   PostgreSQL     │
          │  + TimescaleDB   │
          └─────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
    ┌──────────┐      ┌──────────┐
    │  Redis   │      │  Steam   │
    │  Cache   │      │  Meters  │
    └──────────┘      └──────────┘
```

**Baseline Configuration:**
- Pods: 3 replicas
- CPU: 500m per pod
- Memory: 1Gi per pod
- Database: Single PostgreSQL instance
- Meter Capacity: ~100 meters
- Sampling Rate: 1Hz (1 sample/second)

### Scaled Architecture (Target: 5,000 meters)

```
┌─────────────────────────────────────────────────────────────┐
│              Global Load Balancer (Cloud LB)                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────────────────────────┐
        │                                           │
        ▼ Region 1                                  ▼ Region 2
┌─────────────────┐                        ┌─────────────────┐
│ Regional LB     │                        │ Regional LB     │
└────────┬────────┘                        └────────┬────────┘
         │                                          │
    ┌────┴────┬────┬────┬────┬────┐           ┌────┴────┬────┐
    ▼    ▼    ▼    ▼    ▼    ▼    ▼           ▼    ▼    ▼    ▼
  Pod  Pod  Pod  Pod  Pod  Pod  Pod         Pod  Pod  Pod  Pod
   #1   #2   #3   #4   #5   #6   #7          #1   #2   #3   #4
    │    │    │    │    │    │    │           │    │    │    │
    └────┴────┴────┴────┴────┴────┴───────────┴────┴────┴────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  ┌──────────────┐    ┌──────────────┐
  │ PostgreSQL   │◄──►│ PostgreSQL   │
  │ Primary      │    │ Replica (RO) │
  │+ TimescaleDB │    │+ TimescaleDB │
  └──────┬───────┘    └──────────────┘
         │
         ▼
  ┌──────────────┐
  │  TimescaleDB │
  │  Compression │
  │   + Archive  │
  └──────────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ Redis  │ │SCADA │ │ ERP  │ │ BMS  │
│ Cluster│ │System│ │Conn. │ │Conn. │
└────────┘ └──────┘ └──────┘ └──────┘
```

**Scaled Configuration:**
- Pods: 10 replicas (auto-scaling 3-10)
- CPU: 2000m (2 cores) per pod
- Memory: 4Gi per pod
- Database: PostgreSQL primary + 2 read replicas
- Redis: 3-node cluster
- Meter Capacity: 5,000 meters
- Sampling Rate: 10Hz (10 samples/second)
- Geographic: Multi-region (active-active)

---

## Horizontal Scaling

### Replica Scaling Guidelines

**Scaling Formula:**

```
Replicas = CEILING(Total_Meters / Meters_Per_Pod) + Safety_Buffer

Where:
- Meters_Per_Pod = 150 (baseline capacity per pod)
- Safety_Buffer = 1 (minimum extra pod for HA)
```

**Recommended Replica Count:**

| Steam Meters | Replicas | CPU per Pod | Memory per Pod | Notes |
|--------------|----------|-------------|----------------|-------|
| 1-150 | 3 | 500m | 1Gi | Baseline HA configuration |
| 151-300 | 3 | 1000m | 2Gi | Increase pod resources |
| 301-600 | 4 | 1000m | 2Gi | Add 1 replica |
| 601-900 | 5 | 1500m | 3Gi | Scale resources + replicas |
| 901-1500 | 7 | 2000m | 4Gi | Significant scaling |
| 1501-3000 | 10 | 2000m | 4Gi | Maximum single-cluster |
| 3001-5000 | 10 + Sharding | 2000m | 4Gi | Require meter sharding |
| 5000+ | Multi-cluster | 2000m | 4Gi | Geographic distribution |

### Horizontal Scaling Procedure

**1. Assess Current Load**

```bash
#!/bin/bash
# assess_horizontal_scaling.sh

echo "========================================="
echo "   GL-003 Horizontal Scaling Assessment"
echo "========================================="
echo ""

# Current replica count
CURRENT_REPLICAS=$(kubectl get deployment gl-003-steam-analyzer -n greenlang-agents \
  -o jsonpath='{.spec.replicas}')
echo "Current Replicas: $CURRENT_REPLICAS"

# Current pod resource usage
echo ""
echo "Pod Resource Usage:"
kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer

# Average CPU and memory across pods
AVG_CPU=$(kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  | awk 'NR>1 {sum+=$2} END {print sum/NR}')
AVG_MEMORY=$(kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  | awk 'NR>1 {sum+=$3} END {print sum/NR}')

echo ""
echo "Average CPU: ${AVG_CPU}"
echo "Average Memory: ${AVG_MEMORY}"

# Steam meter count
TOTAL_METERS=$(psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -t -c "SELECT COUNT(*) FROM steam_meters WHERE status = 'active';")
echo ""
echo "Total Active Meters: $TOTAL_METERS"

# Meters per pod
METERS_PER_POD=$((TOTAL_METERS / CURRENT_REPLICAS))
echo "Meters per Pod: $METERS_PER_POD"

# Calculate recommended replicas
RECOMMENDED_REPLICAS=$(echo "scale=0; ($TOTAL_METERS / 150) + 1" | bc)
if [ $RECOMMENDED_REPLICAS -lt 3 ]; then
  RECOMMENDED_REPLICAS=3  # Minimum for HA
fi
echo ""
echo "Recommended Replicas: $RECOMMENDED_REPLICAS"

# Scaling recommendation
echo ""
echo "========================================="
if [ $CURRENT_REPLICAS -lt $RECOMMENDED_REPLICAS ]; then
  SCALE_UP=$((RECOMMENDED_REPLICAS - CURRENT_REPLICAS))
  echo "RECOMMENDATION: Scale UP by $SCALE_UP replicas"
  echo "Command: kubectl scale deployment gl-003-steam-analyzer -n greenlang-agents --replicas=$RECOMMENDED_REPLICAS"
elif [ $CURRENT_REPLICAS -gt $RECOMMENDED_REPLICAS ]; then
  SCALE_DOWN=$((CURRENT_REPLICAS - RECOMMENDED_REPLICAS))
  echo "RECOMMENDATION: Scale DOWN by $SCALE_DOWN replicas"
  echo "Command: kubectl scale deployment gl-003-steam-analyzer -n greenlang-agents --replicas=$RECOMMENDED_REPLICAS"
else
  echo "RECOMMENDATION: Current replica count is optimal"
fi
echo "========================================="
```

**2. Execute Horizontal Scaling**

```bash
#!/bin/bash
# horizontal_scale.sh

TARGET_REPLICAS="${1:-}"

if [ -z "$TARGET_REPLICAS" ]; then
  echo "ERROR: Target replica count required"
  echo "Usage: $0 <target_replicas>"
  exit 1
fi

echo "========================================="
echo "   GL-003 Horizontal Scaling"
echo "========================================="
echo "Target Replicas: $TARGET_REPLICAS"
echo ""

# Pre-scaling checks
echo "[Step 1/5] Pre-scaling validation..."

# Check cluster capacity
CLUSTER_CPU_AVAILABLE=$(kubectl describe nodes | grep -A 5 "Allocated resources" | grep "cpu" | awk '{print $4}' | sed 's/[()%]//g' | awk '{sum+=$1; count+=1} END {print (100-sum/count)}')
CLUSTER_MEM_AVAILABLE=$(kubectl describe nodes | grep -A 5 "Allocated resources" | grep "memory" | awk '{print $4}' | sed 's/[()%]//g' | awk '{sum+=$1; count+=1} END {print (100-sum/count)}')

echo "Cluster CPU Available: ${CLUSTER_CPU_AVAILABLE}%"
echo "Cluster Memory Available: ${CLUSTER_MEM_AVAILABLE}%"

if (( $(echo "$CLUSTER_CPU_AVAILABLE < 20" | bc -l) )); then
  echo "WARNING: Low cluster CPU capacity - consider adding nodes"
fi

if (( $(echo "$CLUSTER_MEM_AVAILABLE < 20" | bc -l) )); then
  echo "WARNING: Low cluster memory capacity - consider adding nodes"
fi

# Scale deployment
echo ""
echo "[Step 2/5] Scaling deployment to $TARGET_REPLICAS replicas..."
kubectl scale deployment gl-003-steam-analyzer -n greenlang-agents \
  --replicas=$TARGET_REPLICAS

# Wait for pods to be ready
echo ""
echo "[Step 3/5] Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod \
  -l app=gl-003-steam-analyzer \
  -n greenlang-agents \
  --timeout=300s

# Verify pod distribution across nodes
echo ""
echo "[Step 4/5] Verifying pod distribution..."
kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  -o custom-columns=POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase

# Validate scaling
echo ""
echo "[Step 5/5] Validating scaled deployment..."
RUNNING_PODS=$(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

echo "Running Pods: $RUNNING_PODS / $TARGET_REPLICAS"

if [ "$RUNNING_PODS" -eq "$TARGET_REPLICAS" ]; then
  echo "✓ Horizontal scaling successful"
else
  echo "✗ Horizontal scaling incomplete - Review pod status"
fi

echo ""
echo "========================================="
echo "Post-Scaling Actions:"
echo "1. Monitor pod resource usage for 15 minutes"
echo "2. Verify load distribution across pods"
echo "3. Check steam meter connectivity"
echo "4. Update capacity planning documentation"
echo "========================================="
```

**3. Verify Load Distribution**

```bash
#!/bin/bash
# verify_load_distribution.sh

echo "========================================="
echo "   Load Distribution Analysis"
echo "========================================="
echo ""

# Get all pod names
PODS=($(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[*].metadata.name}'))

echo "Analyzing load across ${#PODS[@]} pods..."
echo ""

# For each pod, check metrics
for POD in "${PODS[@]}"; do
  echo "Pod: $POD"

  # CPU and memory usage
  METRICS=$(kubectl top pod $POD -n greenlang-agents)
  echo "  Resources: $METRICS"

  # Request rate (from application metrics)
  REQUEST_RATE=$(kubectl exec -n greenlang-agents $POD -- \
    curl -s http://localhost:8080/metrics | grep "gl003_http_requests_total" | tail -1)
  echo "  Request Rate: $REQUEST_RATE"

  # Active meter connections
  METER_COUNT=$(kubectl exec -n greenlang-agents $POD -- \
    curl -s http://localhost:8080/api/v1/internal/meter-count)
  echo "  Connected Meters: $METER_COUNT"

  echo ""
done

# Calculate load distribution variance
echo "Load Distribution Analysis:"
echo "- Ideal: Even distribution across all pods"
echo "- Monitor for: Hot pods (>80% CPU/memory) or cold pods (<20% utilization)"
echo ""
```

### Pod Affinity and Anti-Affinity

**Ensure pods distributed across nodes:**

```yaml
# deployment-with-affinity.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-003-steam-analyzer
  namespace: greenlang-agents
spec:
  replicas: 10
  selector:
    matchLabels:
      app: gl-003-steam-analyzer
  template:
    metadata:
      labels:
        app: gl-003-steam-analyzer
    spec:
      # Pod anti-affinity: Spread pods across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - gl-003-steam-analyzer
              topologyKey: kubernetes.io/hostname

        # Node affinity: Prefer nodes with specific labels
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 50
            preference:
              matchExpressions:
              - key: node-type
                operator: In
                values:
                - compute-optimized

      # Topology spread constraints: Even distribution
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: gl-003-steam-analyzer
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: gl-003-steam-analyzer

      containers:
      - name: steam-analyzer
        image: gcr.io/greenlang/gl-003-steam-analyzer:2.3.1
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi
```

---

## Vertical Scaling

### CPU and Memory Sizing

**Resource Sizing Guidelines:**

```
CPU Requirements:
- Base: 100m (idle)
- Per Meter: 2m (at 1Hz sampling)
- Per Hz: 5m additional (high-frequency)
- Calculation overhead: 50m per 100 meters

Memory Requirements:
- Base: 200Mi (application runtime)
- Per Meter: 5Mi (connection state + buffers)
- Per Hz: 10Mi additional (buffering)
- Database connection pool: 100Mi
- Cache: 500Mi (Redis client)

Formula:
  CPU_Request = 100m + (Meters * 2m) + (Hz * 5m) + (Meters/100 * 50m)
  Memory_Request = 200Mi + (Meters * 5Mi) + (Hz * 10Mi) + 600Mi
```

**Sizing Examples:**

| Meters | Frequency | CPU Request | CPU Limit | Memory Request | Memory Limit |
|--------|-----------|-------------|-----------|----------------|--------------|
| 50 | 1Hz | 250m | 500m | 1Gi | 2Gi |
| 150 | 1Hz | 500m | 1000m | 1.5Gi | 3Gi |
| 150 | 10Hz | 1000m | 2000m | 3Gi | 6Gi |
| 500 | 1Hz | 1500m | 3000m | 3.5Gi | 7Gi |
| 500 | 10Hz | 3000m | 6000m | 7Gi | 14Gi |

### Vertical Scaling Procedure

**1. Calculate Optimal Resources**

```bash
#!/bin/bash
# calculate_vertical_resources.sh

METERS="${1:-150}"
FREQUENCY_HZ="${2:-1}"

echo "========================================="
echo "   GL-003 Vertical Resource Calculator"
echo "========================================="
echo "Meters: $METERS"
echo "Sampling Frequency: ${FREQUENCY_HZ}Hz"
echo ""

# Calculate CPU
CPU_BASE=100
CPU_PER_METER=2
CPU_PER_HZ=5
CPU_CALC_OVERHEAD=$((METERS / 100 * 50))

CPU_TOTAL=$((CPU_BASE + (METERS * CPU_PER_METER) + (FREQUENCY_HZ * CPU_PER_HZ) + CPU_CALC_OVERHEAD))
CPU_REQUEST="${CPU_TOTAL}m"
CPU_LIMIT="$((CPU_TOTAL * 2))m"

echo "CPU Calculation:"
echo "  Base: ${CPU_BASE}m"
echo "  Meters (${METERS} * ${CPU_PER_METER}m): $((METERS * CPU_PER_METER))m"
echo "  Frequency (${FREQUENCY_HZ}Hz * ${CPU_PER_HZ}m): $((FREQUENCY_HZ * CPU_PER_HZ))m"
echo "  Calculation Overhead: ${CPU_CALC_OVERHEAD}m"
echo "  ---"
echo "  Total Request: $CPU_REQUEST"
echo "  Total Limit: $CPU_LIMIT"

# Calculate Memory
MEMORY_BASE=200
MEMORY_PER_METER=5
MEMORY_PER_HZ=10
MEMORY_FIXED=600

MEMORY_TOTAL=$((MEMORY_BASE + (METERS * MEMORY_PER_METER) + (FREQUENCY_HZ * MEMORY_PER_HZ) + MEMORY_FIXED))
MEMORY_REQUEST="${MEMORY_TOTAL}Mi"
MEMORY_LIMIT="$((MEMORY_TOTAL * 2))Mi"

echo ""
echo "Memory Calculation:"
echo "  Base: ${MEMORY_BASE}Mi"
echo "  Meters (${METERS} * ${MEMORY_PER_METER}Mi): $((METERS * MEMORY_PER_METER))Mi"
echo "  Frequency (${FREQUENCY_HZ}Hz * ${MEMORY_PER_HZ}Mi): $((FREQUENCY_HZ * MEMORY_PER_HZ))Mi"
echo "  Fixed (DB pool + cache): ${MEMORY_FIXED}Mi"
echo "  ---"
echo "  Total Request: $MEMORY_REQUEST"
echo "  Total Limit: $MEMORY_LIMIT"

echo ""
echo "========================================="
echo "Recommended Resource Configuration:"
echo "========================================="
echo ""
cat <<EOF
resources:
  requests:
    cpu: $CPU_REQUEST
    memory: $MEMORY_REQUEST
  limits:
    cpu: $CPU_LIMIT
    memory: $MEMORY_LIMIT
EOF
echo ""
```

**2. Update Deployment Resources**

```bash
#!/bin/bash
# vertical_scale.sh

CPU_REQUEST="${1:-1000m}"
CPU_LIMIT="${2:-2000m}"
MEMORY_REQUEST="${3:-2Gi}"
MEMORY_LIMIT="${4:-4Gi}"

echo "========================================="
echo "   GL-003 Vertical Scaling"
echo "========================================="
echo "CPU: $CPU_REQUEST (request) / $CPU_LIMIT (limit)"
echo "Memory: $MEMORY_REQUEST (request) / $MEMORY_LIMIT (limit)"
echo ""

# Update deployment resources
echo "[Step 1/4] Updating deployment resources..."
kubectl patch deployment gl-003-steam-analyzer -n greenlang-agents --type='json' -p="[
  {
    \"op\": \"replace\",
    \"path\": \"/spec/template/spec/containers/0/resources/requests/cpu\",
    \"value\": \"$CPU_REQUEST\"
  },
  {
    \"op\": \"replace\",
    \"path\": \"/spec/template/spec/containers/0/resources/limits/cpu\",
    \"value\": \"$CPU_LIMIT\"
  },
  {
    \"op\": \"replace\",
    \"path\": \"/spec/template/spec/containers/0/resources/requests/memory\",
    \"value\": \"$MEMORY_REQUEST\"
  },
  {
    \"op\": \"replace\",
    \"path\": \"/spec/template/spec/containers/0/resources/limits/memory\",
    \"value\": \"$MEMORY_LIMIT\"
  }
]"

# Wait for rollout
echo ""
echo "[Step 2/4] Waiting for rollout to complete..."
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

# Verify new resources
echo ""
echo "[Step 3/4] Verifying new resource configuration..."
kubectl get deployment gl-003-steam-analyzer -n greenlang-agents \
  -o jsonpath='{.spec.template.spec.containers[0].resources}' | jq .

# Monitor resource usage
echo ""
echo "[Step 4/4] Monitoring resource usage..."
sleep 30  # Allow pods to stabilize
kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer

echo ""
echo "========================================="
echo "Vertical scaling complete"
echo ""
echo "Monitor resource usage for 30 minutes:"
echo "watch -n 30 'kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer'"
echo "========================================="
```

**3. Optimize Resource Usage**

```bash
#!/bin/bash
# optimize_resources.sh

echo "========================================="
echo "   Resource Optimization Analysis"
echo "========================================="
echo ""

# Analyze actual vs requested resources
echo "Actual vs Requested Resources:"
echo ""

PODS=($(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer \
  -o jsonpath='{.items[*].metadata.name}'))

for POD in "${PODS[@]}"; do
  echo "Pod: $POD"

  # Get requested resources
  CPU_REQUEST=$(kubectl get pod $POD -n greenlang-agents \
    -o jsonpath='{.spec.containers[0].resources.requests.cpu}')
  MEMORY_REQUEST=$(kubectl get pod $POD -n greenlang-agents \
    -o jsonpath='{.spec.containers[0].resources.requests.memory}')

  echo "  Requested: CPU=$CPU_REQUEST, Memory=$MEMORY_REQUEST"

  # Get actual usage
  ACTUAL=$(kubectl top pod $POD -n greenlang-agents --no-headers)
  CPU_ACTUAL=$(echo $ACTUAL | awk '{print $2}')
  MEMORY_ACTUAL=$(echo $ACTUAL | awk '{print $3}')

  echo "  Actual: CPU=$CPU_ACTUAL, Memory=$MEMORY_ACTUAL"

  # Calculate utilization percentage
  # Note: This is simplified - actual calculation would need unit conversion
  echo "  Utilization: Monitor actual vs requested ratio"

  echo ""
done

echo "Optimization Recommendations:"
echo "1. Over-provisioned: Actual usage <50% of request → reduce requests"
echo "2. Under-provisioned: Actual usage >80% of request → increase requests"
echo "3. Optimal: Actual usage 50-80% of request → no change needed"
echo ""
```

---

## Database Scaling

### PostgreSQL + TimescaleDB Scaling

**Database Architecture Scaling:**

```
Stage 1: Single Instance (0-100 meters)
┌─────────────────┐
│   PostgreSQL    │
│  + TimescaleDB  │
│   (Primary)     │
└─────────────────┘

Stage 2: Primary + Replica (100-500 meters)
┌─────────────────┐      ┌─────────────────┐
│   PostgreSQL    │─────►│   PostgreSQL    │
│  + TimescaleDB  │      │  + TimescaleDB  │
│   (Primary)     │      │  (Read Replica) │
└─────────────────┘      └─────────────────┘

Stage 3: Primary + 2 Replicas + Connection Pool (500-2000 meters)
                  ┌─────────────────┐
                  │   PgBouncer     │
                  │ Connection Pool │
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
│   PostgreSQL    │ │ PostgreSQL  │ │ PostgreSQL  │
│  + TimescaleDB  │ │+TimescaleDB │ │+TimescaleDB │
│   (Primary)     │ │  (Replica1) │ │  (Replica2) │
└─────────────────┘ └─────────────┘ └─────────────┘

Stage 4: Distributed TimescaleDB (2000+ meters)
                  ┌─────────────────┐
                  │   PgBouncer     │
                  │ Connection Pool │
                  └────────┬────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────┐
│  TimescaleDB    │ │ TimescaleDB │ │ TimescaleDB │
│  Node 1         │ │  Node 2     │ │  Node 3     │
│  (Data Node)    │ │ (Data Node) │ │ (Data Node) │
└─────────────────┘ └─────────────┘ └─────────────┘
         │                 │                 │
         └─────────────────┴─────────────────┘
                           │
                  ┌────────▼────────┐
                  │  Access Node    │
                  │  (Coordinator)  │
                  └─────────────────┘
```

### Database Scaling Procedures

**1. Add Read Replica**

```bash
#!/bin/bash
# add_read_replica.sh

REPLICA_NAME="${1:-postgres-replica-1}"

echo "========================================="
echo "   Add PostgreSQL Read Replica"
echo "========================================="
echo "Replica Name: $REPLICA_NAME"
echo ""

# Create replica using pg_basebackup
echo "[Step 1/6] Creating base backup..."
pg_basebackup -h $POSTGRES_PRIMARY_HOST -U replication_user \
  -D /var/lib/postgresql/data/$REPLICA_NAME \
  -P -R --wal-method=stream

# Configure replica
echo ""
echo "[Step 2/6] Configuring replica..."
cat > /var/lib/postgresql/data/$REPLICA_NAME/postgresql.auto.conf <<EOF
# Replica Configuration
primary_conninfo = 'host=$POSTGRES_PRIMARY_HOST port=5432 user=replication_user password=$REPLICATION_PASSWORD'
primary_slot_name = '$REPLICA_NAME'
hot_standby = on
max_standby_streaming_delay = 30s
wal_receiver_status_interval = 10s
EOF

# Start replica
echo ""
echo "[Step 3/6] Starting replica..."
pg_ctl -D /var/lib/postgresql/data/$REPLICA_NAME start

# Verify replication
echo ""
echo "[Step 4/6] Verifying replication status..."
sleep 10

psql -h $POSTGRES_PRIMARY_HOST -U $POSTGRES_USER -d postgres <<EOF
SELECT
  client_addr,
  application_name,
  state,
  sync_state,
  replay_lag
FROM pg_stat_replication
WHERE application_name = '$REPLICA_NAME';
EOF

# Configure application to use replica for reads
echo ""
echo "[Step 5/6] Updating application configuration..."
kubectl patch configmap gl-003-config -n greenlang-agents --type=merge -p "{
  \"data\": {
    \"DATABASE_READ_HOST\": \"$REPLICA_NAME.postgres.svc.cluster.local\"
  }
}"

# Restart application to pick up new config
echo ""
echo "[Step 6/6] Restarting application..."
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents
kubectl rollout status deployment/gl-003-steam-analyzer -n greenlang-agents

echo ""
echo "========================================="
echo "Read replica added successfully"
echo ""
echo "Verify replication lag:"
echo "psql -h $POSTGRES_PRIMARY_HOST -U $POSTGRES_USER -d postgres \\"
echo "  -c 'SELECT application_name, replay_lag FROM pg_stat_replication;'"
echo "========================================="
```

**2. Implement Connection Pooling (PgBouncer)**

```bash
#!/bin/bash
# deploy_pgbouncer.sh

echo "========================================="
echo "   Deploy PgBouncer Connection Pool"
echo "========================================="
echo ""

# Create PgBouncer configuration
echo "[Step 1/4] Creating PgBouncer configuration..."
cat > /tmp/pgbouncer.ini <<EOF
[databases]
gl003_steam = host=$POSTGRES_PRIMARY_HOST port=5432 dbname=gl003_steam
gl003_steam_ro = host=$POSTGRES_REPLICA_HOST port=5432 dbname=gl003_steam

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
min_pool_size = 10
reserve_pool_size = 10
reserve_pool_timeout = 5
max_db_connections = 100
max_user_connections = 100
server_lifetime = 3600
server_idle_timeout = 600
server_connect_timeout = 15
query_timeout = 0
query_wait_timeout = 120
client_idle_timeout = 0
idle_transaction_timeout = 0
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
stats_period = 60
EOF

# Create PgBouncer userlist
cat > /tmp/userlist.txt <<EOF
"gl003_admin" "$POSTGRES_PASSWORD_MD5"
EOF

# Create Kubernetes ConfigMap
kubectl create configmap pgbouncer-config -n greenlang-agents \
  --from-file=pgbouncer.ini=/tmp/pgbouncer.ini \
  --from-file=userlist.txt=/tmp/userlist.txt \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy PgBouncer
echo ""
echo "[Step 2/4] Deploying PgBouncer..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: greenlang-agents
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: pgbouncer/pgbouncer:1.21
        ports:
        - containerPort: 6432
        volumeMounts:
        - name: config
          mountPath: /etc/pgbouncer
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
      volumes:
      - name: config
        configMap:
          name: pgbouncer-config
---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: greenlang-agents
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 6432
    targetPort: 6432
  type: ClusterIP
EOF

# Wait for PgBouncer to be ready
echo ""
echo "[Step 3/4] Waiting for PgBouncer to be ready..."
kubectl wait --for=condition=available --timeout=60s \
  deployment/pgbouncer -n greenlang-agents

# Update application to use PgBouncer
echo ""
echo "[Step 4/4] Updating application configuration..."
kubectl patch configmap gl-003-config -n greenlang-agents --type=merge -p '{
  "data": {
    "DATABASE_HOST": "pgbouncer.greenlang-agents.svc.cluster.local",
    "DATABASE_PORT": "6432"
  }
}'

kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang-agents

echo ""
echo "========================================="
echo "PgBouncer deployed successfully"
echo ""
echo "Monitor connection pool stats:"
echo "psql -h pgbouncer.greenlang-agents.svc.cluster.local -p 6432 -U gl003_admin -d pgbouncer -c 'SHOW POOLS;'"
echo "========================================="
```

**3. Optimize TimescaleDB Settings**

```bash
#!/bin/bash
# optimize_timescaledb.sh

echo "========================================="
echo "   TimescaleDB Optimization"
echo "========================================="
echo ""

# Configure optimal chunk time intervals
echo "[Step 1/5] Optimizing chunk time intervals..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Adjust chunk interval based on data volume
-- For high-frequency data (10Hz): 1 hour chunks
-- For standard data (1Hz): 1 day chunks

SELECT set_chunk_time_interval('steam_readings', INTERVAL '1 hour');

-- View current chunk intervals
SELECT hypertable_name, chunk_time_interval
FROM timescaledb_information.hypertables;
EOF

# Configure compression
echo ""
echo "[Step 2/5] Configuring compression..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Enable compression for data older than 7 days
ALTER TABLE steam_readings SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'meter_id',
  timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy
SELECT add_compression_policy('steam_readings', INTERVAL '7 days');

-- View compression stats
SELECT
  hypertable_name,
  compression_status,
  uncompressed_heap_size,
  compressed_heap_size,
  ROUND(100.0 * compressed_heap_size / NULLIF(uncompressed_heap_size, 0), 2) AS compression_ratio
FROM timescaledb_information.compressed_hypertable_stats;
EOF

# Configure continuous aggregates
echo ""
echo "[Step 3/5] Optimizing continuous aggregates..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Refresh continuous aggregates more frequently
SELECT alter_job(
  (SELECT job_id FROM timescaledb_information.jobs
   WHERE proc_name = 'policy_refresh_continuous_aggregate'
   AND hypertable_name = 'steam_readings_hourly'),
  schedule_interval => INTERVAL '15 minutes'
);

-- Add new continuous aggregate for 5-minute intervals (high-frequency monitoring)
DROP MATERIALIZED VIEW IF EXISTS steam_readings_5min CASCADE;

CREATE MATERIALIZED VIEW steam_readings_5min
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('5 minutes', timestamp) AS bucket,
  meter_id,
  AVG(pressure_psi) AS avg_pressure,
  MAX(pressure_psi) AS max_pressure,
  MIN(pressure_psi) AS min_pressure,
  AVG(temperature_f) AS avg_temperature,
  AVG(flow_rate_lbh) AS avg_flow_rate,
  COUNT(*) AS reading_count
FROM steam_readings
GROUP BY bucket, meter_id;

-- Add refresh policy for 5-minute aggregates
SELECT add_continuous_aggregate_policy('steam_readings_5min',
  start_offset => INTERVAL '1 hour',
  end_offset => INTERVAL '5 minutes',
  schedule_interval => INTERVAL '5 minutes');
EOF

# Configure data retention
echo ""
echo "[Step 4/5] Configuring data retention..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Remove old retention policy if exists
SELECT remove_retention_policy('steam_readings', if_exists => true);

-- Add retention policy based on requirements
-- Raw data: 90 days
-- Aggregated data: 2 years

SELECT add_retention_policy('steam_readings', INTERVAL '90 days');

-- Keep aggregated data longer
-- (Continuous aggregates are separate from raw data retention)
EOF

# Optimize vacuum and analyze
echo ""
echo "[Step 5/5] Configuring autovacuum..."
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam <<EOF
-- Optimize autovacuum for time-series data
ALTER TABLE steam_readings SET (
  autovacuum_vacuum_scale_factor = 0.0,
  autovacuum_vacuum_threshold = 5000,
  autovacuum_analyze_scale_factor = 0.0,
  autovacuum_analyze_threshold = 5000
);

-- Run manual vacuum analyze
VACUUM ANALYZE steam_readings;
EOF

echo ""
echo "========================================="
echo "TimescaleDB optimization complete"
echo ""
echo "Monitor query performance:"
echo "psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam -c '\\"
echo "  SELECT * FROM pg_stat_statements \\"
echo "  ORDER BY total_exec_time DESC LIMIT 10;'"
echo "========================================="
```

**4. Database Resource Scaling**

```yaml
# postgresql-scaled.yaml
# Example: PostgreSQL with increased resources for 2000+ meters

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: greenlang-db
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi  # Increased from 100Gi
  storageClassName: ssd-storage  # High-performance SSD

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: greenlang-db
spec:
  serviceName: postgresql
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgresql
        image: timescale/timescaledb:2.13-pg15
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: gl003_steam
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata

        # Increased resources for scaled workload
        resources:
          requests:
            cpu: 4000m      # 4 cores
            memory: 16Gi    # 16GB RAM
          limits:
            cpu: 8000m      # 8 cores
            memory: 32Gi    # 32GB RAM

        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data

        # PostgreSQL configuration optimizations
        args:
        - "-c"
        - "max_connections=500"                    # Increased from 100
        - "-c"
        - "shared_buffers=8GB"                     # 25% of RAM
        - "-c"
        - "effective_cache_size=24GB"              # 75% of RAM
        - "-c"
        - "maintenance_work_mem=2GB"
        - "-c"
        - "checkpoint_completion_target=0.9"
        - "-c"
        - "wal_buffers=16MB"
        - "-c"
        - "default_statistics_target=100"
        - "-c"
        - "random_page_cost=1.1"                   # SSD optimization
        - "-c"
        - "effective_io_concurrency=200"           # SSD optimization
        - "-c"
        - "work_mem=20MB"                          # Per connection
        - "-c"
        - "min_wal_size=1GB"
        - "-c"
        - "max_wal_size=4GB"
        - "-c"
        - "max_worker_processes=8"
        - "-c"
        - "max_parallel_workers_per_gather=4"
        - "-c"
        - "max_parallel_workers=8"
        - "-c"
        - "max_parallel_maintenance_workers=4"

        # TimescaleDB specific settings
        - "-c"
        - "timescaledb.max_background_workers=16"
        - "-c"
        - "timescaledb.compress_chunk_time_threshold=1d"

      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
```

---

## Steam Meter Count Scaling

### Meter Sharding Strategy

When scaling beyond 1,000 meters, implement meter sharding:

```
Sharding Strategy: By Facility or Geographic Region

Shard 1: Facility A (Meters 1-500)
┌─────────────────────────────────┐
│  GL-003 Instance 1              │
│  - Pods: 3-5 replicas           │
│  - DB: PostgreSQL Instance 1    │
│  - Meters: facility_a_*         │
└─────────────────────────────────┘

Shard 2: Facility B (Meters 501-1000)
┌─────────────────────────────────┐
│  GL-003 Instance 2              │
│  - Pods: 3-5 replicas           │
│  - DB: PostgreSQL Instance 2    │
│  - Meters: facility_b_*         │
└─────────────────────────────────┘

Shard 3: Facility C (Meters 1001-1500)
┌─────────────────────────────────┐
│  GL-003 Instance 3              │
│  - Pods: 3-5 replicas           │
│  - DB: PostgreSQL Instance 3    │
│  - Meters: facility_c_*         │
└─────────────────────────────────┘

Aggregation Layer
┌─────────────────────────────────┐
│  GL-003 Aggregator              │
│  - Collects data from shards    │
│  - Cross-facility analytics     │
│  - Unified reporting            │
└─────────────────────────────────┘
```

### Meter Assignment Strategy

```python
# meter_sharding.py
# Assign meters to shards based on facility

def assign_meter_shard(meter_id: str, facility_id: str) -> str:
    """
    Assign meter to appropriate shard.

    Strategy:
    - Facility-based sharding for geographic isolation
    - Round-robin within facility for load balancing
    """
    # Map facility to shard
    facility_shard_map = {
        "facility_a": "shard_1",
        "facility_b": "shard_2",
        "facility_c": "shard_3",
        # Add more facilities as needed
    }

    shard = facility_shard_map.get(facility_id)

    if not shard:
        # Dynamic shard assignment for new facilities
        # Based on current shard load
        shard = get_least_loaded_shard()

    return shard


def get_least_loaded_shard() -> str:
    """Get shard with lowest meter count."""
    shard_loads = {
        "shard_1": get_meter_count("shard_1"),
        "shard_2": get_meter_count("shard_2"),
        "shard_3": get_meter_count("shard_3"),
    }

    return min(shard_loads, key=shard_loads.get)


def get_meter_count(shard: str) -> int:
    """Get active meter count for shard."""
    # Query database for meter count
    pass


# Meter routing configuration
METER_ROUTING = {
    "facility_a": {
        "shard": "shard_1",
        "endpoint": "gl-003-shard-1.greenlang-agents.svc.cluster.local:8080",
        "database": "postgres-shard-1.greenlang-db.svc.cluster.local:5432",
    },
    "facility_b": {
        "shard": "shard_2",
        "endpoint": "gl-003-shard-2.greenlang-agents.svc.cluster.local:8080",
        "database": "postgres-shard-2.greenlang-db.svc.cluster.local:5432",
    },
    "facility_c": {
        "shard": "shard_3",
        "endpoint": "gl-003-shard-3.greenlang-agents.svc.cluster.local:8080",
        "database": "postgres-shard-3.greenlang-db.svc.cluster.local:5432",
    },
}
```

### Meter Scaling Procedure

```bash
#!/bin/bash
# scale_for_meters.sh

CURRENT_METERS="${1:-}"
TARGET_METERS="${2:-}"

echo "========================================="
echo "   GL-003 Meter Scaling Procedure"
echo "========================================="
echo "Current Meters: $CURRENT_METERS"
echo "Target Meters: $TARGET_METERS"
echo ""

# Calculate scaling requirements
if [ "$TARGET_METERS" -le 500 ]; then
  STRATEGY="single_instance"
  REPLICAS=3
  CPU="1000m"
  MEMORY="2Gi"
  DB_CONFIG="single"
elif [ "$TARGET_METERS" -le 1000 ]; then
  STRATEGY="scaled_instance"
  REPLICAS=5
  CPU="2000m"
  MEMORY="4Gi"
  DB_CONFIG="primary_replica"
elif [ "$TARGET_METERS" -le 2000 ]; then
  STRATEGY="large_instance"
  REPLICAS=7
  CPU="2000m"
  MEMORY="4Gi"
  DB_CONFIG="primary_2_replicas"
else
  STRATEGY="sharded"
  REPLICAS="3_per_shard"
  CPU="2000m"
  MEMORY="4Gi"
  DB_CONFIG="sharded_databases"
fi

echo "Scaling Strategy: $STRATEGY"
echo "Replicas: $REPLICAS"
echo "Resources: CPU=$CPU, Memory=$MEMORY"
echo "Database: $DB_CONFIG"
echo ""

case $STRATEGY in
  single_instance)
    echo "Scaling to single instance configuration..."
    ./horizontal_scale.sh $REPLICAS
    ./vertical_scale.sh $CPU "2000m" $MEMORY "4Gi"
    ;;

  scaled_instance)
    echo "Scaling to scaled instance configuration..."
    ./horizontal_scale.sh $REPLICAS
    ./vertical_scale.sh $CPU "4000m" $MEMORY "8Gi"
    ./add_read_replica.sh postgres-replica-1
    ;;

  large_instance)
    echo "Scaling to large instance configuration..."
    ./horizontal_scale.sh $REPLICAS
    ./vertical_scale.sh $CPU "4000m" $MEMORY "8Gi"
    ./add_read_replica.sh postgres-replica-1
    ./add_read_replica.sh postgres-replica-2
    ./deploy_pgbouncer.sh
    ;;

  sharded)
    echo "Scaling to sharded configuration..."
    echo "This requires manual setup of multiple GL-003 instances."
    echo "See: SHARDING_SETUP_GUIDE.md"
    SHARD_COUNT=$(echo "scale=0; $TARGET_METERS / 1500" | bc)
    echo "Recommended shards: $SHARD_COUNT"
    ;;
esac

echo ""
echo "========================================="
echo "Meter scaling procedure complete"
echo "========================================="
```

---

## High-Frequency Monitoring

### Scaling for High Sampling Rates (>10Hz)

**Challenges:**
- Increased data volume (10x-100x more data points)
- Higher CPU requirements for data processing
- Increased memory for buffering
- Database write pressure
- Network bandwidth consumption

**Solutions:**

**1. Data Buffering and Batching**

```python
# high_frequency_buffering.py
# Buffer high-frequency readings and batch database writes

from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import asyncio

class HighFrequencyBuffer:
    """Buffer for high-frequency steam readings."""

    def __init__(
        self,
        flush_interval: int = 10,  # Flush every 10 seconds
        max_buffer_size: int = 10000,  # Max readings before forced flush
    ):
        self.buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.flush_interval = flush_interval
        self.max_buffer_size = max_buffer_size
        self.last_flush = datetime.now()

    def add_reading(self, meter_id: str, reading: Dict):
        """Add reading to buffer."""
        self.buffer[meter_id].append(reading)

        # Check if flush needed
        total_readings = sum(len(readings) for readings in self.buffer.values())

        if total_readings >= self.max_buffer_size:
            # Force flush due to buffer size
            asyncio.create_task(self.flush())
        elif (datetime.now() - self.last_flush).seconds >= self.flush_interval:
            # Periodic flush
            asyncio.create_task(self.flush())

    async def flush(self):
        """Flush buffer to database."""
        if not self.buffer:
            return

        # Prepare batch insert
        readings_to_insert = []

        for meter_id, readings in self.buffer.items():
            for reading in readings:
                readings_to_insert.append({
                    "meter_id": meter_id,
                    "timestamp": reading["timestamp"],
                    "pressure_psi": reading["pressure_psi"],
                    "temperature_f": reading["temperature_f"],
                    "flow_rate_lbh": reading["flow_rate_lbh"],
                })

        # Batch insert to database
        await batch_insert_readings(readings_to_insert)

        # Clear buffer
        self.buffer.clear()
        self.last_flush = datetime.now()

        print(f"Flushed {len(readings_to_insert)} readings to database")


async def batch_insert_readings(readings: List[Dict]):
    """Batch insert readings to database."""
    # Use COPY for high-performance batch insert
    import psycopg2
    from io import StringIO

    conn = psycopg2.connect(
        host=os.getenv("DATABASE_HOST"),
        database=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD"),
    )

    cur = conn.cursor()

    # Prepare CSV buffer
    csv_buffer = StringIO()
    for reading in readings:
        csv_buffer.write(f"{reading['meter_id']},{reading['timestamp']},"
                        f"{reading['pressure_psi']},{reading['temperature_f']},"
                        f"{reading['flow_rate_lbh']}\n")

    csv_buffer.seek(0)

    # COPY data
    cur.copy_expert(
        """
        COPY steam_readings (meter_id, timestamp, pressure_psi, temperature_f, flow_rate_lbh)
        FROM STDIN WITH (FORMAT CSV)
        """,
        csv_buffer
    )

    conn.commit()
    cur.close()
    conn.close()
```

**2. Downsampling and Aggregation**

```python
# high_frequency_downsampling.py
# Downsample high-frequency data for storage optimization

import numpy as np
from typing import List, Dict

def downsample_readings(
    readings: List[Dict],
    target_frequency: int = 1,  # Downsample to 1Hz
    source_frequency: int = 100,  # From 100Hz
) -> List[Dict]:
    """
    Downsample high-frequency readings.

    Strategies:
    - Average: For stable metrics (pressure, temperature)
    - Min/Max: For detecting extremes
    - Last: For latest value
    """
    downsample_factor = source_frequency // target_frequency
    downsampled = []

    for i in range(0, len(readings), downsample_factor):
        chunk = readings[i:i + downsample_factor]

        if not chunk:
            continue

        # Calculate aggregates
        downsampled_reading = {
            "meter_id": chunk[0]["meter_id"],
            "timestamp": chunk[0]["timestamp"],  # Use first timestamp in window
            "pressure_psi_avg": np.mean([r["pressure_psi"] for r in chunk]),
            "pressure_psi_min": np.min([r["pressure_psi"] for r in chunk]),
            "pressure_psi_max": np.max([r["pressure_psi"] for r in chunk]),
            "temperature_f_avg": np.mean([r["temperature_f"] for r in chunk]),
            "flow_rate_lbh_avg": np.mean([r["flow_rate_lbh"] for r in chunk]),
            "sample_count": len(chunk),
        }

        downsampled.append(downsampled_reading)

    return downsampled


# Hybrid storage strategy: Store both raw and downsampled
def hybrid_storage_strategy(readings: List[Dict]):
    """
    Hybrid storage strategy for high-frequency data.

    - Raw data: Stored for 24 hours (for detailed analysis)
    - Downsampled: Stored indefinitely (for historical trends)
    """
    # Store raw data in hot storage (Redis or TimescaleDB)
    store_raw_readings(readings)

    # Downsample and store in cold storage
    downsampled = downsample_readings(readings, target_frequency=1, source_frequency=100)
    store_downsampled_readings(downsampled)


def store_raw_readings(readings: List[Dict]):
    """Store raw readings in hot storage with TTL."""
    # Store in Redis with 24-hour TTL for fast access
    pass


def store_downsampled_readings(readings: List[Dict]):
    """Store downsampled readings in TimescaleDB for long-term storage."""
    # Store in PostgreSQL/TimescaleDB
    pass
```

**3. High-Frequency Configuration**

```yaml
# configmap-high-frequency.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-003-config-highfreq
  namespace: greenlang-agents
data:
  # High-frequency monitoring configuration
  SAMPLING_FREQUENCY_HZ: "100"  # 100Hz sampling
  BUFFER_SIZE: "10000"  # Buffer 10,000 readings before flush
  FLUSH_INTERVAL_SECONDS: "10"  # Flush every 10 seconds
  DOWNSAMPLE_ENABLED: "true"
  DOWNSAMPLE_TARGET_HZ: "1"  # Downsample to 1Hz for storage
  RAW_DATA_RETENTION_HOURS: "24"  # Keep raw data for 24 hours

  # Database connection pool (increased for high write volume)
  DATABASE_POOL_MIN: "20"
  DATABASE_POOL_MAX: "100"
  DATABASE_POOL_TIMEOUT: "30"

  # Performance tuning
  BATCH_INSERT_SIZE: "1000"  # Insert 1000 readings per batch
  ASYNC_WORKERS: "10"  # 10 async workers for data processing

  # Memory settings
  JAVA_OPTS: "-Xms2g -Xmx6g -XX:+UseG1GC"
```

---

## Large Facility Support

### Scaling for 50+ Monitoring Zones

**Zone Architecture:**

```
Large Facility (50+ Zones)
│
├── Zone 1: Boiler Room
│   ├── Meters: 20
│   └── GL-003 Instance: Dedicated or shared
│
├── Zone 2: Production Area A
│   ├── Meters: 15
│   └── GL-003 Instance: Shared
│
├── Zone 3: Production Area B
│   ├── Meters: 15
│   └── GL-003 Instance: Shared
│
├── ... (47 more zones)
│
└── Zone 50: Maintenance Shop
    ├── Meters: 5
    └── GL-003 Instance: Shared

Aggregation and Cross-Zone Analytics
┌────────────────────────────────────┐
│  GL-003 Facility Aggregator         │
│  - Collects data from all zones    │
│  - Facility-wide analytics         │
│  - Energy optimization             │
│  - Unified reporting               │
└────────────────────────────────────┘
```

### Zone-Based Deployment

```yaml
# deployment-zone-based.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-003-facility-alpha
  namespace: greenlang-agents
  labels:
    facility: alpha
    zones: "1-50"
spec:
  replicas: 10  # Scale based on total meter count
  selector:
    matchLabels:
      app: gl-003-steam-analyzer
      facility: alpha
  template:
    metadata:
      labels:
        app: gl-003-steam-analyzer
        facility: alpha
    spec:
      containers:
      - name: steam-analyzer
        image: gcr.io/greenlang/gl-003-steam-analyzer:2.3.1
        env:
        - name: FACILITY_ID
          value: "facility_alpha"
        - name: ZONE_FILTER
          value: "zone_1,zone_2,zone_3,...,zone_50"  # All zones
        - name: DATABASE_HOST
          value: "postgres-facility-alpha.greenlang-db.svc.cluster.local"

        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi

---
# Service per facility
apiVersion: v1
kind: Service
metadata:
  name: gl-003-facility-alpha
  namespace: greenlang-agents
spec:
  selector:
    app: gl-003-steam-analyzer
    facility: alpha
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

### Cross-Zone Analytics

```python
# cross_zone_analytics.py
# Aggregate and analyze data across facility zones

from typing import List, Dict
import pandas as pd

class FacilityAnalyzer:
    """Analyze steam system performance across facility zones."""

    def __init__(self, facility_id: str):
        self.facility_id = facility_id

    def get_zone_energy_consumption(self, zone_id: str, time_range: str) -> Dict:
        """Get energy consumption for specific zone."""
        query = f"""
        SELECT
            zone_id,
            SUM(energy_mmbtu) AS total_energy,
            AVG(pressure_psi) AS avg_pressure,
            AVG(temperature_f) AS avg_temperature
        FROM steam_readings
        WHERE facility_id = '{self.facility_id}'
          AND zone_id = '{zone_id}'
          AND timestamp > NOW() - INTERVAL '{time_range}'
        GROUP BY zone_id
        """

        result = execute_query(query)
        return result

    def get_facility_energy_distribution(self, time_range: str = "1 day") -> pd.DataFrame:
        """Get energy distribution across all zones."""
        query = f"""
        SELECT
            zone_id,
            zone_name,
            COUNT(DISTINCT meter_id) AS meter_count,
            SUM(energy_mmbtu) AS total_energy,
            AVG(energy_mmbtu) AS avg_energy_per_reading,
            SUM(energy_mmbtu) * 100.0 / SUM(SUM(energy_mmbtu)) OVER () AS energy_percent
        FROM steam_readings
        JOIN zones USING (zone_id)
        WHERE facility_id = '{self.facility_id}'
          AND timestamp > NOW() - INTERVAL '{time_range}'
        GROUP BY zone_id, zone_name
        ORDER BY total_energy DESC
        """

        df = pd.read_sql(query, connection)
        return df

    def identify_optimization_opportunities(self) -> List[Dict]:
        """Identify energy optimization opportunities across zones."""
        opportunities = []

        # Find zones with high energy consumption
        high_consumption_zones = self.get_high_consumption_zones()

        for zone in high_consumption_zones:
            # Analyze for optimization
            opportunity = {
                "zone_id": zone["zone_id"],
                "zone_name": zone["zone_name"],
                "current_consumption": zone["total_energy"],
                "optimization_potential": self.calculate_optimization_potential(zone),
                "recommendations": self.get_recommendations(zone),
            }
            opportunities.append(opportunity)

        return opportunities

    def calculate_optimization_potential(self, zone: Dict) -> float:
        """Calculate potential energy savings for zone."""
        # Compare zone consumption to facility average
        facility_avg = self.get_facility_average_consumption()
        zone_consumption = zone["total_energy"]

        if zone_consumption > facility_avg * 1.2:  # 20% above average
            # Potential for 10-15% savings
            return zone_consumption * 0.125  # 12.5% savings potential

        return 0.0

    def get_recommendations(self, zone: Dict) -> List[str]:
        """Get recommendations for zone optimization."""
        recommendations = []

        # Check for pressure issues
        if zone["avg_pressure"] > 200:  # High pressure
            recommendations.append("Reduce steam pressure to optimal range (150-180 psi)")

        # Check for temperature issues
        if zone["avg_temperature"] > 400:  # High temperature
            recommendations.append("Review insulation and reduce heat loss")

        # Check for flow rate issues
        if zone["avg_flow_rate"] > zone["expected_flow_rate"] * 1.3:
            recommendations.append("Investigate potential steam leaks or overuse")

        return recommendations
```

---

## Distributed Steam Analysis

### Multi-Region Deployment

**Geographic Distribution Strategy:**

```
Region: US-EAST
┌────────────────────────────────┐
│  GL-003 US-EAST Cluster        │
│  - Facilities: East Coast      │
│  - Latency: <10ms              │
│  - Database: Regional primary  │
└────────────────────────────────┘
         │
         ├─── Replication ───┐
         │                   │
         ▼                   ▼
Region: US-WEST       Region: EU-WEST
┌──────────────────┐  ┌──────────────────┐
│ GL-003 US-WEST   │  │ GL-003 EU-WEST   │
│ Cluster          │  │ Cluster          │
│ - West Coast     │  │ - Europe         │
│ - Latency: <10ms │  │ - Latency: <10ms │
│ - DB: Replica    │  │ - DB: Replica    │
└──────────────────┘  └──────────────────┘

Global Analytics and Reporting
┌────────────────────────────────┐
│  GL-003 Global Aggregator      │
│  - Cross-region analytics      │
│  - Global reporting            │
│  - Centralized monitoring      │
└────────────────────────────────┘
```

### Multi-Region Configuration

```yaml
# deployment-multi-region.yaml

---
# US-EAST Region
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-003-us-east
  namespace: greenlang-agents
  labels:
    region: us-east
spec:
  replicas: 5
  selector:
    matchLabels:
      app: gl-003-steam-analyzer
      region: us-east
  template:
    metadata:
      labels:
        app: gl-003-steam-analyzer
        region: us-east
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values:
                - us-east-1

      containers:
      - name: steam-analyzer
        image: gcr.io/greenlang/gl-003-steam-analyzer:2.3.1
        env:
        - name: REGION
          value: "us-east"
        - name: DATABASE_HOST
          value: "postgres-us-east.greenlang-db.svc.cluster.local"

        resources:
          requests:
            cpu: 2000m
            memory: 4Gi

---
# US-WEST Region
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-003-us-west
  namespace: greenlang-agents
  labels:
    region: us-west
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gl-003-steam-analyzer
      region: us-west
  template:
    metadata:
      labels:
        app: gl-003-steam-analyzer
        region: us-west
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: topology.kubernetes.io/region
                operator: In
                values:
                - us-west-2

      containers:
      - name: steam-analyzer
        image: gcr.io/greenlang/gl-003-steam-analyzer:2.3.1
        env:
        - name: REGION
          value: "us-west"
        - name: DATABASE_HOST
          value: "postgres-us-west.greenlang-db.svc.cluster.local"

        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
```

---

## Performance Optimization

### Query Optimization

```sql
-- query_optimization.sql
-- Optimize frequently-used queries for scaled workloads

-- 1. Add indexes for common query patterns
CREATE INDEX CONCURRENTLY idx_steam_readings_meter_timestamp
ON steam_readings (meter_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_steam_readings_facility_zone_timestamp
ON steam_readings (facility_id, zone_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_steam_readings_timestamp_energy
ON steam_readings (timestamp DESC, energy_mmbtu);

-- 2. Optimize continuous aggregate queries
-- Use TimescaleDB's time_bucket for efficient aggregation
CREATE MATERIALIZED VIEW steam_readings_optimized_hourly
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 hour', timestamp) AS bucket,
  facility_id,
  zone_id,
  meter_id,
  AVG(pressure_psi) AS avg_pressure,
  MAX(pressure_psi) AS max_pressure,
  MIN(pressure_psi) AS min_pressure,
  STDDEV(pressure_psi) AS stddev_pressure,
  AVG(temperature_f) AS avg_temperature,
  AVG(flow_rate_lbh) AS avg_flow_rate,
  SUM(energy_mmbtu) AS total_energy,
  COUNT(*) AS reading_count
FROM steam_readings
GROUP BY bucket, facility_id, zone_id, meter_id
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('steam_readings_optimized_hourly',
  start_offset => INTERVAL '3 hours',
  end_offset => INTERVAL '1 hour',
  schedule_interval => INTERVAL '1 hour');

-- 3. Partition large tables (if not using TimescaleDB hypertables)
-- Note: TimescaleDB handles this automatically, but for standard PostgreSQL:
/*
CREATE TABLE steam_readings_partitioned (
  id BIGSERIAL,
  meter_id TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  ...
) PARTITION BY RANGE (timestamp);

CREATE TABLE steam_readings_2025_01 PARTITION OF steam_readings_partitioned
  FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
*/

-- 4. Optimize frequently-used analytical queries
-- Pre-aggregate common metrics
CREATE MATERIALIZED VIEW facility_daily_metrics AS
SELECT
  DATE_TRUNC('day', timestamp) AS day,
  facility_id,
  COUNT(DISTINCT meter_id) AS active_meters,
  SUM(energy_mmbtu) AS total_energy,
  AVG(pressure_psi) AS avg_pressure,
  AVG(temperature_f) AS avg_temperature,
  SUM(CASE WHEN pressure_psi > 200 THEN 1 ELSE 0 END) AS high_pressure_events
FROM steam_readings
GROUP BY day, facility_id;

-- Refresh daily
CREATE INDEX ON facility_daily_metrics (facility_id, day DESC);
```

### Application-Level Caching

```python
# caching_strategy.py
# Implement multi-tier caching for performance

import redis
from functools import wraps
from typing import Any, Callable
import json

class CacheLayer:
    """Multi-tier caching strategy."""

    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=6379,
            decode_responses=True
        )
        self.local_cache = {}  # In-memory cache

    def get(self, key: str, tier: str = "redis") -> Any:
        """Get value from cache."""
        if tier == "local":
            return self.local_cache.get(key)
        elif tier == "redis":
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        return None

    def set(self, key: str, value: Any, ttl: int = 300, tier: str = "redis"):
        """Set value in cache."""
        if tier == "local":
            self.local_cache[key] = value
        elif tier == "redis":
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )

    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)


def cached(ttl: int = 300, tier: str = "redis"):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Try to get from cache
            cache = CacheLayer()
            cached_result = cache.get(cache_key, tier=tier)

            if cached_result is not None:
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, ttl=ttl, tier=tier)

            return result

        return wrapper
    return decorator


# Usage examples
@cached(ttl=60, tier="local")  # Cache in local memory for 60 seconds
def get_meter_count():
    """Get active meter count (changes infrequently)."""
    return query_database("SELECT COUNT(*) FROM steam_meters WHERE status = 'active'")


@cached(ttl=300, tier="redis")  # Cache in Redis for 5 minutes
def get_facility_energy_consumption(facility_id: str, time_range: str):
    """Get facility energy consumption (moderate change frequency)."""
    query = f"""
    SELECT SUM(energy_mmbtu)
    FROM steam_readings
    WHERE facility_id = '{facility_id}'
      AND timestamp > NOW() - INTERVAL '{time_range}'
    """
    return query_database(query)
```

---

## Cost Optimization

### Resource Cost Analysis

```python
# cost_optimization.py
# Analyze and optimize infrastructure costs

from typing import Dict, List

class CostOptimizer:
    """Optimize infrastructure costs for scaled deployment."""

    # Cost per resource unit (example prices)
    COST_PER_CPU_HOUR = 0.05  # $0.05 per CPU core per hour
    COST_PER_GB_MEMORY_HOUR = 0.01  # $0.01 per GB memory per hour
    COST_PER_GB_STORAGE_MONTH = 0.10  # $0.10 per GB storage per month

    def calculate_compute_cost(
        self,
        replicas: int,
        cpu_per_replica: float,
        memory_per_replica_gb: float,
        hours_per_month: int = 730,
    ) -> Dict[str, float]:
        """Calculate monthly compute cost."""
        total_cpu = replicas * cpu_per_replica
        total_memory_gb = replicas * memory_per_replica_gb

        cpu_cost = total_cpu * self.COST_PER_CPU_HOUR * hours_per_month
        memory_cost = total_memory_gb * self.COST_PER_GB_MEMORY_HOUR * hours_per_month
        total_cost = cpu_cost + memory_cost

        return {
            "cpu_cost": cpu_cost,
            "memory_cost": memory_cost,
            "total_monthly_cost": total_cost,
            "cost_per_meter": total_cost / self.get_meter_count() if self.get_meter_count() > 0 else 0,
        }

    def calculate_storage_cost(
        self,
        database_size_gb: float,
        backup_size_gb: float,
        retention_days: int = 90,
    ) -> Dict[str, float]:
        """Calculate monthly storage cost."""
        total_storage_gb = database_size_gb + backup_size_gb
        storage_cost = total_storage_gb * self.COST_PER_GB_STORAGE_MONTH

        return {
            "database_cost": database_size_gb * self.COST_PER_GB_STORAGE_MONTH,
            "backup_cost": backup_size_gb * self.COST_PER_GB_STORAGE_MONTH,
            "total_monthly_cost": storage_cost,
        }

    def optimize_replica_count(
        self,
        current_replicas: int,
        average_cpu_utilization: float,
        average_memory_utilization: float,
    ) -> Dict[str, Any]:
        """Recommend optimal replica count based on utilization."""
        recommendations = []

        # Over-provisioned (low utilization)
        if average_cpu_utilization < 0.3 and average_memory_utilization < 0.3:
            optimal_replicas = max(3, int(current_replicas * 0.7))  # Reduce by 30%, min 3 for HA
            potential_savings = self.calculate_cost_savings(current_replicas, optimal_replicas)
            recommendations.append({
                "action": "reduce_replicas",
                "from": current_replicas,
                "to": optimal_replicas,
                "reason": "Low resource utilization (<30%)",
                "monthly_savings": potential_savings,
            })

        # Under-provisioned (high utilization)
        elif average_cpu_utilization > 0.8 or average_memory_utilization > 0.8:
            optimal_replicas = int(current_replicas * 1.4)  # Increase by 40%
            additional_cost = self.calculate_cost_savings(optimal_replicas, current_replicas)
            recommendations.append({
                "action": "increase_replicas",
                "from": current_replicas,
                "to": optimal_replicas,
                "reason": "High resource utilization (>80%)",
                "monthly_additional_cost": additional_cost,
            })

        # Optimal
        else:
            recommendations.append({
                "action": "no_change",
                "current": current_replicas,
                "reason": "Resource utilization is optimal (30-80%)",
            })

        return {
            "current_utilization": {
                "cpu": average_cpu_utilization,
                "memory": average_memory_utilization,
            },
            "recommendations": recommendations,
        }

    def calculate_cost_savings(self, current_replicas: int, new_replicas: int) -> float:
        """Calculate cost difference between replica counts."""
        current_cost = self.calculate_compute_cost(
            replicas=current_replicas,
            cpu_per_replica=2.0,
            memory_per_replica_gb=4.0,
        )["total_monthly_cost"]

        new_cost = self.calculate_compute_cost(
            replicas=new_replicas,
            cpu_per_replica=2.0,
            memory_per_replica_gb=4.0,
        )["total_monthly_cost"]

        return current_cost - new_cost

    def get_meter_count(self) -> int:
        """Get current meter count."""
        # Query database
        return 500  # Placeholder


# Cost optimization script
def run_cost_optimization():
    """Run cost optimization analysis."""
    optimizer = CostOptimizer()

    # Current configuration
    current_config = {
        "replicas": 7,
        "cpu_per_replica": 2.0,
        "memory_per_replica_gb": 4.0,
        "database_size_gb": 200,
        "backup_size_gb": 100,
    }

    # Calculate current costs
    compute_cost = optimizer.calculate_compute_cost(
        replicas=current_config["replicas"],
        cpu_per_replica=current_config["cpu_per_replica"],
        memory_per_replica_gb=current_config["memory_per_replica_gb"],
    )

    storage_cost = optimizer.calculate_storage_cost(
        database_size_gb=current_config["database_size_gb"],
        backup_size_gb=current_config["backup_size_gb"],
    )

    total_monthly_cost = compute_cost["total_monthly_cost"] + storage_cost["total_monthly_cost"]

    print("Current Monthly Costs:")
    print(f"  Compute: ${compute_cost['total_monthly_cost']:.2f}")
    print(f"  Storage: ${storage_cost['total_monthly_cost']:.2f}")
    print(f"  Total: ${total_monthly_cost:.2f}")
    print(f"  Cost per Meter: ${compute_cost['cost_per_meter']:.2f}")

    # Optimization recommendations
    optimization = optimizer.optimize_replica_count(
        current_replicas=7,
        average_cpu_utilization=0.45,
        average_memory_utilization=0.50,
    )

    print("\nOptimization Recommendations:")
    for rec in optimization["recommendations"]:
        print(f"  Action: {rec['action']}")
        print(f"  Reason: {rec['reason']}")
        if "monthly_savings" in rec:
            print(f"  Potential Savings: ${rec['monthly_savings']:.2f}/month")
```

### Spot Instance Strategy

For cost-sensitive workloads, use spot instances:

```yaml
# deployment-spot-instances.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-003-steam-analyzer-spot
  namespace: greenlang-agents
spec:
  replicas: 5
  selector:
    matchLabels:
      app: gl-003-steam-analyzer
      instance-type: spot
  template:
    metadata:
      labels:
        app: gl-003-steam-analyzer
        instance-type: spot
    spec:
      # Tolerate spot instance taints
      tolerations:
      - key: "node.kubernetes.io/instance-type"
        operator: "Equal"
        value: "spot"
        effect: "NoSchedule"

      # Prefer spot instances
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                - spot

      # Graceful shutdown handling for spot termination
      terminationGracePeriodSeconds: 120

      containers:
      - name: steam-analyzer
        image: gcr.io/greenlang/gl-003-steam-analyzer:2.3.1
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                # Gracefully drain connections before termination
                echo "Spot instance terminating - draining connections"
                curl -X POST http://localhost:8080/admin/drain
                sleep 60
```

---

## Capacity Planning

### Capacity Planning Formulas

```python
# capacity_planning.py
# Calculate infrastructure requirements

class CapacityPlanner:
    """Calculate infrastructure requirements for GL-003 scaling."""

    def calculate_pod_requirements(
        self,
        meter_count: int,
        sampling_frequency_hz: int = 1,
        safety_buffer: float = 1.3,  # 30% safety buffer
    ) -> Dict[str, Any]:
        """
        Calculate pod resource requirements.

        Returns:
            Dict with replica count, CPU, memory requirements
        """
        # Base calculations
        meters_per_pod = 150  # Baseline capacity
        base_cpu_m = 100
        cpu_per_meter_m = 2
        cpu_per_hz_m = 5

        base_memory_mi = 200
        memory_per_meter_mi = 5
        memory_per_hz_mi = 10
        memory_fixed_mi = 600

        # Calculate raw requirements
        raw_replicas = (meter_count / meters_per_pod)
        replicas = max(3, int(raw_replicas * safety_buffer))  # Min 3 for HA

        # CPU calculation
        cpu_per_pod_m = (
            base_cpu_m +
            (meters_per_pod * cpu_per_meter_m) +
            (sampling_frequency_hz * cpu_per_hz_m) +
            ((meters_per_pod / 100) * 50)  # Calculation overhead
        )
        cpu_request_m = int(cpu_per_pod_m)
        cpu_limit_m = cpu_request_m * 2

        # Memory calculation
        memory_per_pod_mi = (
            base_memory_mi +
            (meters_per_pod * memory_per_meter_mi) +
            (sampling_frequency_hz * memory_per_hz_mi) +
            memory_fixed_mi
        )
        memory_request_mi = int(memory_per_pod_mi)
        memory_limit_mi = memory_request_mi * 2

        # Total cluster requirements
        total_cpu_cores = (replicas * cpu_request_m) / 1000
        total_memory_gb = (replicas * memory_request_mi) / 1024

        return {
            "replicas": replicas,
            "cpu_request_m": cpu_request_m,
            "cpu_limit_m": cpu_limit_m,
            "memory_request_mi": memory_request_mi,
            "memory_limit_mi": memory_limit_mi,
            "total_cpu_cores": total_cpu_cores,
            "total_memory_gb": total_memory_gb,
            "meters_per_pod": int(meter_count / replicas),
        }

    def calculate_database_requirements(
        self,
        meter_count: int,
        sampling_frequency_hz: int = 1,
        retention_days: int = 90,
        compression_ratio: float = 0.2,  # 20% of original size after compression
    ) -> Dict[str, Any]:
        """
        Calculate database storage requirements.

        Assumes:
        - Each reading: ~100 bytes
        - Compression: 80% reduction (20% final size)
        - Indexes: 30% overhead
        - Continuous aggregates: 10% of raw data
        """
        readings_per_day = meter_count * sampling_frequency_hz * 86400  # 86400 seconds/day
        bytes_per_reading = 100

        # Raw data size
        raw_data_bytes = readings_per_day * retention_days * bytes_per_reading

        # After compression
        compressed_data_bytes = raw_data_bytes * compression_ratio

        # Indexes
        index_bytes = compressed_data_bytes * 0.3

        # Continuous aggregates
        aggregate_bytes = compressed_data_bytes * 0.1

        # Total
        total_bytes = compressed_data_bytes + index_bytes + aggregate_bytes
        total_gb = total_bytes / (1024 ** 3)

        # Add 50% buffer for growth
        provisioned_gb = total_gb * 1.5

        return {
            "readings_per_day": readings_per_day,
            "raw_data_gb": raw_data_bytes / (1024 ** 3),
            "compressed_data_gb": compressed_data_bytes / (1024 ** 3),
            "index_gb": index_bytes / (1024 ** 3),
            "aggregate_gb": aggregate_bytes / (1024 ** 3),
            "total_gb": total_gb,
            "provisioned_gb": provisioned_gb,
            "growth_rate_gb_per_day": (readings_per_day * bytes_per_reading * compression_ratio) / (1024 ** 3),
        }

    def calculate_network_bandwidth(
        self,
        meter_count: int,
        sampling_frequency_hz: int = 1,
    ) -> Dict[str, Any]:
        """Calculate network bandwidth requirements."""
        bytes_per_reading = 100  # Approximate payload size
        readings_per_second = meter_count * sampling_frequency_hz

        # Inbound (meter data)
        inbound_bytes_per_second = readings_per_second * bytes_per_reading
        inbound_mbps = (inbound_bytes_per_second * 8) / (1024 ** 2)

        # Outbound (API queries, dashboards, exports)
        # Assume 20% of inbound
        outbound_mbps = inbound_mbps * 0.2

        # Total
        total_mbps = inbound_mbps + outbound_mbps

        # Peak (3x average)
        peak_mbps = total_mbps * 3

        return {
            "readings_per_second": readings_per_second,
            "inbound_mbps": inbound_mbps,
            "outbound_mbps": outbound_mbps,
            "average_total_mbps": total_mbps,
            "peak_mbps": peak_mbps,
            "recommended_bandwidth_mbps": peak_mbps * 1.5,  # 50% buffer
        }


# Example usage
def generate_capacity_plan(meter_count: int, sampling_frequency_hz: int = 1):
    """Generate complete capacity plan."""
    planner = CapacityPlanner()

    print(f"Capacity Plan for {meter_count} meters at {sampling_frequency_hz}Hz")
    print("=" * 70)

    # Pod requirements
    pods = planner.calculate_pod_requirements(meter_count, sampling_frequency_hz)
    print("\nPod Requirements:")
    print(f"  Replicas: {pods['replicas']}")
    print(f"  CPU per pod: {pods['cpu_request_m']}m (request) / {pods['cpu_limit_m']}m (limit)")
    print(f"  Memory per pod: {pods['memory_request_mi']}Mi (request) / {pods['memory_limit_mi']}Mi (limit)")
    print(f"  Total cluster CPU: {pods['total_cpu_cores']:.1f} cores")
    print(f"  Total cluster memory: {pods['total_memory_gb']:.1f} GB")
    print(f"  Meters per pod: {pods['meters_per_pod']}")

    # Database requirements
    db = planner.calculate_database_requirements(meter_count, sampling_frequency_hz)
    print("\nDatabase Requirements (90-day retention):")
    print(f"  Readings per day: {db['readings_per_day']:,.0f}")
    print(f"  Raw data: {db['raw_data_gb']:.1f} GB")
    print(f"  Compressed data: {db['compressed_data_gb']:.1f} GB")
    print(f"  Indexes: {db['index_gb']:.1f} GB")
    print(f"  Aggregates: {db['aggregate_gb']:.1f} GB")
    print(f"  Total (with buffer): {db['provisioned_gb']:.1f} GB")
    print(f"  Growth rate: {db['growth_rate_gb_per_day']:.2f} GB/day")

    # Network requirements
    network = planner.calculate_network_bandwidth(meter_count, sampling_frequency_hz)
    print("\nNetwork Requirements:")
    print(f"  Readings/second: {network['readings_per_second']:,.0f}")
    print(f"  Inbound: {network['inbound_mbps']:.2f} Mbps")
    print(f"  Outbound: {network['outbound_mbps']:.2f} Mbps")
    print(f"  Average total: {network['average_total_mbps']:.2f} Mbps")
    print(f"  Peak: {network['peak_mbps']:.2f} Mbps")
    print(f"  Recommended: {network['recommended_bandwidth_mbps']:.2f} Mbps")


# Generate plans for different scales
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GL-003 SteamSystemAnalyzer Capacity Planning")
    print("=" * 70)

    # Small deployment
    generate_capacity_plan(meter_count=100, sampling_frequency_hz=1)

    # Medium deployment
    print("\n")
    generate_capacity_plan(meter_count=500, sampling_frequency_hz=1)

    # Large deployment
    print("\n")
    generate_capacity_plan(meter_count=2000, sampling_frequency_hz=1)

    # High-frequency deployment
    print("\n")
    generate_capacity_plan(meter_count=500, sampling_frequency_hz=10)
```

---

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
# hpa.yaml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-003-steam-analyzer-hpa
  namespace: greenlang-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-003-steam-analyzer

  minReplicas: 3
  maxReplicas: 10

  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale when CPU >70%

  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75  # Scale when memory >75%

  # Custom metric: Steam meters per pod
  - type: Pods
    pods:
      metric:
        name: gl003_meters_per_pod
      target:
        type: AverageValue
        averageValue: "150"  # Scale when >150 meters/pod

  # Custom metric: Request rate
  - type: Pods
    pods:
      metric:
        name: gl003_http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"  # Scale when >100 req/s per pod

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60  # Wait 60s before scaling up
      policies:
      - type: Percent
        value: 50  # Scale up by 50% of current replicas
        periodSeconds: 60
      - type: Pods
        value: 2  # Or add 2 pods
        periodSeconds: 60
      selectPolicy: Max  # Use the policy that adds more pods

    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5min before scaling down
      policies:
      - type: Percent
        value: 25  # Scale down by 25% of current replicas
        periodSeconds: 60
      - type: Pods
        value: 1  # Or remove 1 pod
        periodSeconds: 60
      selectPolicy: Min  # Use the policy that removes fewer pods
```

### Vertical Pod Autoscaler (VPA)

```yaml
# vpa.yaml

apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-003-steam-analyzer-vpa
  namespace: greenlang-agents
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-003-steam-analyzer

  updatePolicy:
    updateMode: "Auto"  # Automatically apply recommendations

  resourcePolicy:
    containerPolicies:
    - containerName: steam-analyzer
      minAllowed:
        cpu: 500m
        memory: 1Gi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources:
      - cpu
      - memory
```

### Cluster Autoscaler

```yaml
# cluster-autoscaler-config.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  autoscaler-config: |
    {
      "scaleDownUtilizationThreshold": 0.5,
      "scaleDownUnneededTime": "10m",
      "scaleDownDelayAfterAdd": "10m",
      "skipNodesWithSystemPods": false,
      "skipNodesWithLocalStorage": false,
      "balanceSimilarNodeGroups": true,
      "maxNodeProvisionTime": "15m"
    }
```

---

## Scaling Procedures

### Complete Scaling Workflow

```bash
#!/bin/bash
# complete_scaling_workflow.sh

TARGET_METERS="${1:-}"
TARGET_FREQUENCY_HZ="${2:-1}"

if [ -z "$TARGET_METERS" ]; then
  echo "ERROR: Target meter count required"
  echo "Usage: $0 <target_meters> [sampling_frequency_hz]"
  exit 1
fi

echo "========================================="
echo "   GL-003 Complete Scaling Workflow"
echo "========================================="
echo "Target Meters: $TARGET_METERS"
echo "Sampling Frequency: ${TARGET_FREQUENCY_HZ}Hz"
echo ""

# Step 1: Generate capacity plan
echo "[Step 1/8] Generating capacity plan..."
python3 capacity_planning.py $TARGET_METERS $TARGET_FREQUENCY_HZ > /tmp/capacity_plan.txt
cat /tmp/capacity_plan.txt

# Step 2: Assess current state
echo ""
echo "[Step 2/8] Assessing current state..."
./assess_horizontal_scaling.sh

# Step 3: Scale database first (if needed)
echo ""
echo "[Step 3/8] Scaling database..."
# Read provisioned_gb from capacity plan
PROVISIONED_GB=$(grep "provisioned_gb" /tmp/capacity_plan.txt | awk '{print $2}')
./scale_database_storage.sh $PROVISIONED_GB

# Step 4: Add database replicas (if needed)
echo ""
echo "[Step 4/8] Configuring database replication..."
if [ "$TARGET_METERS" -gt 500 ]; then
  ./add_read_replica.sh postgres-replica-1
fi

if [ "$TARGET_METERS" -gt 1000 ]; then
  ./add_read_replica.sh postgres-replica-2
  ./deploy_pgbouncer.sh
fi

# Step 5: Optimize TimescaleDB
echo ""
echo "[Step 5/8] Optimizing TimescaleDB..."
./optimize_timescaledb.sh

# Step 6: Scale application horizontally
echo ""
echo "[Step 6/8] Scaling application horizontally..."
# Read replicas from capacity plan
REPLICAS=$(grep "Replicas:" /tmp/capacity_plan.txt | awk '{print $2}')
./horizontal_scale.sh $REPLICAS

# Step 7: Scale application vertically
echo ""
echo "[Step 7/8] Scaling application vertically..."
# Read resources from capacity plan
CPU_REQUEST=$(grep "CPU per pod:" /tmp/capacity_plan.txt | awk '{print $4}')
MEMORY_REQUEST=$(grep "Memory per pod:" /tmp/capacity_plan.txt | awk '{print $4}')
./vertical_scale.sh $CPU_REQUEST "${CPU_REQUEST}000m" $MEMORY_REQUEST "${MEMORY_REQUEST}Gi"

# Step 8: Validate scaled deployment
echo ""
echo "[Step 8/8] Validating scaled deployment..."
./validate_scaled_deployment.sh

echo ""
echo "========================================="
echo "   Scaling Workflow Complete"
echo "========================================="
echo ""
echo "Capacity Plan: /tmp/capacity_plan.txt"
echo ""
echo "Post-Scaling Actions:"
echo "1. Monitor system for 1 hour"
echo "2. Run performance tests"
echo "3. Update capacity planning documentation"
echo "4. Update runbooks with new configuration"
echo ""
```

---

## Monitoring and Metrics

### Scaling Metrics Dashboard

```yaml
# grafana-dashboard-scaling.json
# Key metrics for monitoring scaled deployments

{
  "dashboard": {
    "title": "GL-003 Scaling Metrics",
    "panels": [
      {
        "title": "Meters per Pod",
        "targets": [
          {
            "expr": "sum(gl003_active_meters) / count(up{job='gl-003-steam-analyzer'})",
            "legendFormat": "Meters per Pod"
          }
        ]
      },
      {
        "title": "CPU Utilization",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{pod=~'gl-003.*'}[5m]) * 100",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Memory Utilization",
        "targets": [
          {
            "expr": "container_memory_working_set_bytes{pod=~'gl-003.*'} / container_spec_memory_limit_bytes{pod=~'gl-003.*'} * 100",
            "legendFormat": "{{pod}}"
          }
        ]
      },
      {
        "title": "Database Connections",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends{datname='gl003_steam'}",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Throughput (Readings/sec)",
        "targets": [
          {
            "expr": "rate(gl003_readings_ingested_total[5m])",
            "legendFormat": "Readings per Second"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(gl003_http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95 Response Time"
          }
        ]
      }
    ]
  }
}
```

---

## Troubleshooting

### Common Scaling Issues

**Issue 1: Pods not scaling up**

```bash
# Check HPA status
kubectl describe hpa gl-003-steam-analyzer-hpa -n greenlang-agents

# Check metrics server
kubectl top nodes
kubectl top pods -n greenlang-agents

# Check custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1/namespaces/greenlang-agents/pods/*/gl003_meters_per_pod
```

**Issue 2: Database connection pool exhausted**

```bash
# Check current connections
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam \
  -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Increase max_connections
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d postgres \
  -c "ALTER SYSTEM SET max_connections = 500;"

# Restart PostgreSQL
kubectl rollout restart statefulset/postgresql -n greenlang-db
```

**Issue 3: High memory usage / OOM kills**

```bash
# Check memory usage
kubectl top pods -n greenlang-agents -l app=gl-003-steam-analyzer

# Check for memory leaks
kubectl exec -n greenlang-agents \
  $(kubectl get pods -n greenlang-agents -l app=gl-003-steam-analyzer -o jsonpath='{.items[0].metadata.name}') \
  -- curl -s http://localhost:8080/metrics | grep memory

# Increase memory limits
./vertical_scale.sh 2000m 4000m 4Gi 8Gi
```

---

## Appendices

### Appendix A: Scaling Decision Matrix

| Meters | Frequency | Replicas | CPU/Pod | Memory/Pod | DB Config | Estimated Cost/Month |
|--------|-----------|----------|---------|------------|-----------|---------------------|
| 100 | 1Hz | 3 | 500m | 1Gi | Single | $250 |
| 500 | 1Hz | 4 | 1000m | 2Gi | Primary+Replica | $750 |
| 1000 | 1Hz | 5 | 2000m | 4Gi | Primary+2 Replicas | $1,500 |
| 2000 | 1Hz | 7 | 2000m | 4Gi | Primary+2 Replicas+PgBouncer | $2,500 |
| 5000 | 1Hz | 10 | 2000m | 4Gi | Sharded | $4,500 |
| 500 | 10Hz | 5 | 3000m | 6Gi | Primary+Replica | $2,000 |

### Appendix B: Performance Benchmarks

```bash
# run_performance_benchmarks.sh

echo "GL-003 Performance Benchmarks"
echo "=============================="

# Test 1: Ingest throughput
echo "Test 1: Ingest Throughput"
ab -n 10000 -c 100 -p test_reading.json -T application/json \
  http://gl-003-steam-analyzer:8080/api/v1/readings/ingest

# Test 2: Query performance
echo "Test 2: Query Performance"
for i in {1..100}; do
  curl -s http://gl-003-steam-analyzer:8080/api/v1/meters/meter-001/readings?limit=1000 > /dev/null
done

# Test 3: Database write performance
echo "Test 3: Database Write Performance"
pgbench -h $POSTGRES_HOST -U $POSTGRES_USER -d gl003_steam -c 10 -j 2 -t 1000
```

### Appendix C: Scaling Checklist

**Pre-Scaling:**
- [ ] Generate capacity plan
- [ ] Review current resource utilization
- [ ] Verify cluster has capacity
- [ ] Backup current configuration
- [ ] Notify stakeholders

**During Scaling:**
- [ ] Scale database first
- [ ] Add read replicas if needed
- [ ] Deploy connection pooling
- [ ] Scale application horizontally
- [ ] Scale application vertically
- [ ] Monitor pod startup

**Post-Scaling:**
- [ ] Validate all pods running
- [ ] Verify meter connectivity
- [ ] Test data ingestion
- [ ] Run performance benchmarks
- [ ] Monitor for 1 hour
- [ ] Update documentation

---

**END OF SCALING GUIDE**
