# GL-001 ProcessHeatOrchestrator - Scaling Guide

Comprehensive scaling procedures for GL-001 ProcessHeatOrchestrator managing multi-plant industrial process heat operations.

## Table of Contents

1. [When to Scale](#when-to-scale)
2. [Capacity Planning Calculator](#capacity-planning-calculator)
3. [Horizontal Scaling (HPA)](#horizontal-scaling-hpa)
4. [Vertical Scaling (Resources)](#vertical-scaling-resources)
5. [Sub-Agent Group Scaling](#sub-agent-group-scaling)
6. [Database Scaling (TimescaleDB)](#database-scaling-timescaledb)
7. [Message Bus Scaling](#message-bus-scaling)
8. [Multi-Region Deployment](#multi-region-deployment)
9. [Performance Testing](#performance-testing)
10. [Cost Optimization](#cost-optimization)

---

## When to Scale

### Scaling Triggers and Thresholds

| Metric | Current | Warning | Critical | Action |
|--------|---------|---------|----------|--------|
| **Master CPU Usage** | <70% | 70-80% | >80% | Horizontal or Vertical Scale |
| **Master Memory Usage** | <75% | 75-85% | >85% | Vertical Scale or Add Replicas |
| **Orchestration Latency (p95)** | <2s | 2-3s | >3s | Horizontal Scale + Optimize |
| **Agent Coordination Errors** | <1% | 1-5% | >5% | Scale Master + Message Bus |
| **Database Connections** | <70% | 70-85% | >85% | Increase Pool Size or Add Replicas |
| **Message Bus Lag** | <100ms | 100-500ms | >500ms | Scale Kafka/RabbitMQ |
| **Heat Optimization Time** | <2s | 2-5s | >5s | Vertical Scale or Optimize Algorithm |
| **Plant Count** | - | 8-10 plants | >10 plants | Proactive Horizontal Scale |
| **Sub-Agent Count** | - | 70-80 agents | >80 agents | Scale Master + Message Bus |

### Decision Tree: When to Scale vs Optimize

```
Performance Issue Detected
├─ CPU >80%?
│  ├─ Recent deployment? → Investigate for regression
│  ├─ Gradual increase? → Proactive horizontal scale
│  └─ Sudden spike? → Check for runaway process
│
├─ Memory >85%?
│  ├─ Growing over time (memory leak)? → Fix leak, then vertical scale
│  ├─ Stable but high? → Vertical scale
│  └─ Spikes then drops? → Optimize memory usage patterns
│
├─ Orchestration latency >3s?
│  ├─ Database slow? → Scale database (see Database Scaling)
│  ├─ Many agents coordinating? → Horizontal scale master
│  ├─ Large optimization problems? → Vertical scale or optimize algorithm
│  └─ Message bus lag? → Scale message bus
│
├─ Agent coordination errors >5%?
│  ├─ Network issues? → Investigate networking
│  ├─ Message bus lag? → Scale message bus
│  ├─ Master overloaded? → Horizontal scale
│  └─ Sub-agents failing? → Scale sub-agents (see Sub-Agent Scaling)
│
└─ Adding new plants (5+)?
   └─ Proactive scale: master, database, message bus
```

---

## Capacity Planning Calculator

### Formulas

**Master Orchestrator Replicas:**
```
Replicas = ceil(plant_count / 5) + ceil(active_subagent_count / 10)

Minimum: 3 (high availability)
Maximum: 20 (diminishing returns beyond this)

Examples:
- 5 plants, 25 agents: ceil(5/5) + ceil(25/10) = 1 + 3 = 4 replicas (use 4, meets minimum)
- 10 plants, 50 agents: ceil(10/5) + ceil(50/10) = 2 + 5 = 7 replicas
- 20 plants, 99 agents: ceil(20/5) + ceil(99/10) = 4 + 10 = 14 replicas
```

**Master CPU Requirements:**
```
CPU = 1000m (base) + (plant_count * 100m) + (subagent_count * 50m)

Examples:
- 5 plants, 25 agents: 1000m + 500m + 1250m = 2750m (~3 cores)
- 10 plants, 50 agents: 1000m + 1000m + 2500m = 4500m (~5 cores)
- 20 plants, 99 agents: 1000m + 2000m + 4950m = 7950m (~8 cores)
```

**Master Memory Requirements:**
```
Memory = 2Gi (base) + (plant_count * 200Mi) + (subagent_count * 100Mi)

Examples:
- 5 plants, 25 agents: 2Gi + 1Gi + 2.5Gi = 5.5Gi (~6Gi)
- 10 plants, 50 agents: 2Gi + 2Gi + 5Gi = 9Gi (~10Gi)
- 20 plants, 99 agents: 2Gi + 4Gi + 9.9Gi = 15.9Gi (~16Gi)
```

**Database Connection Pool Size:**
```
Pool Size = (plant_count * 5) + (master_replica_count * 10) + 20

Examples:
- 5 plants, 4 replicas: (5*5) + (4*10) + 20 = 25 + 40 + 20 = 85 connections
- 10 plants, 7 replicas: (10*5) + (7*10) + 20 = 50 + 70 + 20 = 140 connections
- 20 plants, 14 replicas: (20*5) + (14*10) + 20 = 100 + 140 + 20 = 260 connections
```

**Message Queue Workers:**
```
Workers = ceil(subagent_count / 20)

Minimum: 3
Maximum: 20

Examples:
- 25 agents: ceil(25/20) = 2 workers (use 3, minimum)
- 50 agents: ceil(50/20) = 3 workers
- 99 agents: ceil(99/20) = 5 workers
```

### Interactive Calculator

```bash
#!/bin/bash
# GL-001 Capacity Planning Calculator

read -p "Enter number of plants: " PLANT_COUNT
read -p "Enter number of active sub-agents: " AGENT_COUNT

# Calculate replicas
REPLICAS=$(echo "scale=0; (($PLANT_COUNT + 4) / 5) + (($AGENT_COUNT + 9) / 10)" | bc)
if [ $REPLICAS -lt 3 ]; then REPLICAS=3; fi
if [ $REPLICAS -gt 20 ]; then REPLICAS=20; fi

# Calculate CPU (in millicores)
CPU=$(echo "1000 + ($PLANT_COUNT * 100) + ($AGENT_COUNT * 50)" | bc)
CPU_CORES=$(echo "scale=1; $CPU / 1000" | bc)

# Calculate Memory (in MiB)
MEMORY=$(echo "2048 + ($PLANT_COUNT * 200) + ($AGENT_COUNT * 100)" | bc)
MEMORY_GB=$(echo "scale=1; $MEMORY / 1024" | bc)

# Calculate DB Pool
DB_POOL=$(echo "($PLANT_COUNT * 5) + ($REPLICAS * 10) + 20" | bc)

# Calculate Message Workers
MSG_WORKERS=$(echo "scale=0; (($AGENT_COUNT + 19) / 20)" | bc)
if [ $MSG_WORKERS -lt 3 ]; then MSG_WORKERS=3; fi

echo ""
echo "=== GL-001 Capacity Planning Results ==="
echo "Configuration: $PLANT_COUNT plants, $AGENT_COUNT sub-agents"
echo ""
echo "Master Orchestrator:"
echo "  - Replicas: $REPLICAS"
echo "  - CPU per replica: ${CPU}m (~${CPU_CORES} cores)"
echo "  - Memory per replica: ${MEMORY}Mi (~${MEMORY_GB}Gi)"
echo ""
echo "Database:"
echo "  - Connection Pool Size: $DB_POOL"
echo ""
echo "Message Bus:"
echo "  - Workers: $MSG_WORKERS"
echo ""
echo "Recommended kubectl commands:"
echo "kubectl scale deployment gl-001-process-heat-orchestrator --replicas=$REPLICAS -n greenlang"
echo "kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \\"
echo "  --limits=cpu=${CPU}m,memory=${MEMORY}Mi \\"
echo "  --requests=cpu=$(echo "$CPU * 0.6" | bc | cut -d. -f1)m,memory=$(echo "$MEMORY * 0.7" | bc | cut -d. -f1)Mi"
```

---

## Horizontal Scaling (HPA)

### Manual Horizontal Scaling

**For immediate capacity increase:**

```bash
# Calculate required replicas (use calculator above)
# Example: 10 plants, 50 sub-agents → 7 replicas

# Scale master orchestrator
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=7 -n greenlang

# Monitor scaling
kubectl get pods -n greenlang -l app=gl-001-process-heat-orchestrator -w

# Verify all pods ready
kubectl wait --for=condition=ready pod -l app=gl-001-process-heat-orchestrator \
  -n greenlang --timeout=5m

# Check load distribution
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/load-distribution | jq
```

### Automatic Horizontal Pod Autoscaler (HPA)

**Configure HPA for automatic scaling:**

```yaml
# hpa-gl-001.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-001-process-heat-orchestrator-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-001-process-heat-orchestrator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  # Custom metric: orchestration latency
  - type: Pods
    pods:
      metric:
        name: orchestration_latency_p95_seconds
      target:
        type: AverageValue
        averageValue: "2"
  # Custom metric: agent coordination rate
  - type: Pods
    pods:
      metric:
        name: agent_coordination_per_second
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
      - type: Pods
        value: 1
        periodSeconds: 60
      selectPolicy: Min
```

**Apply HPA:**
```bash
kubectl apply -f hpa-gl-001.yaml

# Verify HPA created
kubectl get hpa gl-001-process-heat-orchestrator-hpa -n greenlang

# Expected output:
# NAME                                      REFERENCE                                 TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
# gl-001-process-heat-orchestrator-hpa     Deployment/gl-001-process-heat-orchestrator   45%/70%   3         20        5          1m

# Watch HPA decisions
kubectl get hpa gl-001-process-heat-orchestrator-hpa -n greenlang -w

# Describe HPA for scaling events
kubectl describe hpa gl-001-process-heat-orchestrator-hpa -n greenlang
```

### Scaling for Large Facilities (10+ Plants)

**Pre-scale before adding plants:**

```bash
# Before adding Plant-011 through Plant-015 (total 15 plants)

# 1. Calculate new capacity requirements
# 15 plants, 70 agents (estimated) → 11 replicas

# 2. Pre-scale master orchestrator
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=11 -n greenlang

# 3. Increase resource limits
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=cpu=6000m,memory=12Gi \
  --requests=cpu=4000m,memory=8Gi

# 4. Increase database pool
kubectl exec -n greenlang postgresql-0 -- psql -U postgres -c \
  "ALTER SYSTEM SET max_connections = 300;"
kubectl rollout restart statefulset/postgresql -n greenlang

# 5. Update ConfigMap
kubectl edit configmap gl-001-config -n greenlang
# Update:
#   DB_POOL_SIZE: "200"
#   MAX_PLANTS: "15"

# 6. Restart master to apply config
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 7. Verify capacity
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/capacity | jq
```

---

## Vertical Scaling (Resources)

### When to Vertically Scale

**Vertical scaling is better when:**
- Complex heat optimization problems (>10 plants, many constraints)
- LP solver requires more CPU for faster solutions
- Single large plant with many sub-agents (>20)
- Memory-intensive operations (large datasets, caching)

### CPU Scaling

**Increase CPU for faster heat optimization:**

```bash
# Check current CPU usage
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator

# Expected output:
# NAME                                                CPU(cores)   MEMORY(bytes)
# gl-001-process-heat-orchestrator-5d8c7f-abc12      2100m        6500Mi
# gl-001-process-heat-orchestrator-5d8c7f-def34      1950m        6200Mi
# gl-001-process-heat-orchestrator-5d8c7f-ghi56      2200m        6800Mi

# If consistently >80% (>1600m of 2000m limit), increase CPU

# Increase CPU limits
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=cpu=4000m \
  --requests=cpu=2500m

# Monitor improvement
watch -n 10 'kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator'

# Check optimization time improvement
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | grep heat_optimization_time_p95
```

### Memory Scaling

**Increase memory for caching and large datasets:**

```bash
# Check current memory usage
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator

# If consistently >75% of limit, increase memory

# Increase memory limits
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=memory=16Gi \
  --requests=memory=12Gi

# Verify pods restarted with new limits
kubectl get pods -n greenlang -l app=gl-001-process-heat-orchestrator

# Check memory usage after restart
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator

# Verify no more OOMKilled events
kubectl get events -n greenlang --field-selector involvedObject.name=gl-001-process-heat-orchestrator | grep OOM
```

### Node Size Considerations

**For large facilities, use larger nodes:**

```bash
# Current node sizes for standard workers
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.capacity.cpu,MEMORY:.status.capacity.memory

# For 10+ plants, recommend:
# - CPU: 8+ cores per node
# - Memory: 32Gi+ per node
# - Allows 2-3 master pods per node with headroom

# Add large nodes (AWS EKS example)
eksctl create nodegroup \
  --cluster greenlang-cluster \
  --name large-workers \
  --node-type m5.2xlarge \
  --nodes 5 \
  --nodes-min 3 \
  --nodes-max 10 \
  --node-labels workload=gl-001-orchestrator

# Use node affinity to schedule on large nodes
kubectl patch deployment gl-001-process-heat-orchestrator -n greenlang \
  --type=json -p='[{
    "op": "add",
    "path": "/spec/template/spec/affinity",
    "value": {
      "nodeAffinity": {
        "preferredDuringSchedulingIgnoredDuringExecution": [{
          "weight": 100,
          "preference": {
            "matchExpressions": [{
              "key": "workload",
              "operator": "In",
              "values": ["gl-001-orchestrator"]
            }]
          }
        }]
      }
    }
  }]'
```

---

## Sub-Agent Group Scaling

### Scaling by Agent Group

**Scale sub-agents by functional group:**

```bash
# 1. Boiler and Steam Group (GL-002, GL-012, GL-016, GL-017, GL-022)
# Scale when plant count increases or boiler operations intensify

# Scale boiler efficiency agents (1 per boiler)
kubectl scale deployment gl-002-boiler-efficiency --replicas=12 -n greenlang

# Scale steam system agents (1 per plant)
kubectl scale deployment gl-012-steam-system --replicas=10 -n greenlang

# 2. Combustion and Emissions Group (GL-003, GL-004, GL-005)
# Scale for emissions compliance and optimization

kubectl scale deployment gl-003-combustion-optimizer --replicas=10 -n greenlang
kubectl scale deployment gl-004-emissions-control --replicas=10 -n greenlang

# 3. Heat Recovery Group (GL-006, GL-014, GL-020)
# Scale for efficiency optimization

kubectl scale deployment gl-006-waste-heat --replicas=8 -n greenlang

# Verify all sub-agents scaled
kubectl get deployments -n greenlang -l agent-group=process-heat-sub-agents
```

### Sub-Agent HPA Configuration

**Configure HPA for each sub-agent group:**

```yaml
# hpa-boiler-agents.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-002-boiler-efficiency-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-002-boiler-efficiency
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: boiler_calculation_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
```

```bash
kubectl apply -f hpa-boiler-agents.yaml

# Verify HPA
kubectl get hpa -n greenlang | grep gl-002
```

---

## Database Scaling (TimescaleDB)

### Connection Pool Scaling

**Increase connection pool for high load:**

```bash
# Check current pool usage
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | \
  jq '.connection_pool | {size, in_use, available, wait_count}'

# Example output:
# {
#   "size": 100,
#   "in_use": 92,
#   "available": 8,
#   "wait_count": 450
# }

# If in_use consistently >85% or wait_count increasing:

# 1. Update ConfigMap
kubectl edit configmap gl-001-config -n greenlang

# Update:
data:
  DB_POOL_SIZE: "200"  # Increase from 100
  DB_POOL_TIMEOUT: "60"
  DB_POOL_OVERFLOW: "50"

# 2. Increase PostgreSQL max_connections
kubectl exec -n greenlang postgresql-0 -- psql -U postgres -c \
  "ALTER SYSTEM SET max_connections = 500;"

kubectl exec -n greenlang postgresql-0 -- psql -U postgres -c \
  "ALTER SYSTEM SET shared_buffers = '8GB';"

kubectl rollout restart statefulset/postgresql -n greenlang

# 3. Restart master orchestrator
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 4. Verify increased pool
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | jq '.connection_pool.size'
```

### TimescaleDB Read Replicas

**Add read replicas for query load distribution:**

```bash
# 1. Create read replica (AWS RDS)
aws rds create-db-instance-read-replica \
  --db-instance-identifier gl-001-db-replica-1 \
  --source-db-instance-identifier gl-001-db-primary \
  --db-instance-class db.r5.2xlarge \
  --availability-zone us-east-1b

# Wait for replica to be available
aws rds wait db-instance-available --db-instance-identifier gl-001-db-replica-1

# 2. Update ConfigMap with read replica endpoint
kubectl edit configmap gl-001-config -n greenlang

data:
  DB_HOST: "gl-001-db-primary.xxxx.us-east-1.rds.amazonaws.com"
  DB_READ_REPLICAS: "gl-001-db-replica-1.xxxx.us-east-1.rds.amazonaws.com"
  DB_READ_REPLICA_COUNT: "1"

# 3. Restart deployment to use read replicas
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 4. Verify read traffic distributed
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | \
  jq '.read_replicas[] | {endpoint, query_count, avg_latency_ms}'
```

### TimescaleDB Hypertable Optimization

**Optimize time-series data storage:**

```bash
# Compress old data chunks (reduce storage, faster queries)
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT compress_chunk(i) FROM show_chunks('heat_measurements', older_than => INTERVAL '7 days') i;"

# Set compression policy (automatic)
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT add_compression_policy('heat_measurements', INTERVAL '7 days');"

# Add retention policy (delete old data)
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT add_retention_policy('heat_measurements', INTERVAL '90 days');"

# Verify chunk compression
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT chunk_name, compression_status FROM timescaledb_information.chunks WHERE hypertable_name = 'heat_measurements';"
```

---

## Message Bus Scaling

### Kafka Scaling

**Scale Kafka for increased agent coordination:**

```bash
# Check current Kafka lag
kubectl exec -n greenlang kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group gl-001-orchestrator

# If lag >1000 messages or increasing:

# 1. Scale Kafka brokers
kubectl scale statefulset kafka --replicas=5 -n greenlang

# Wait for new brokers to join
kubectl rollout status statefulset/kafka -n greenlang --timeout=5m

# 2. Increase topic partitions
kubectl exec -n greenlang kafka-0 -- kafka-topics.sh \
  --alter --topic agent.coordination \
  --partitions 12 \
  --bootstrap-server localhost:9092

# 3. Increase consumer threads in master
kubectl edit configmap gl-001-config -n greenlang

data:
  KAFKA_CONSUMER_THREADS: "8"  # Increase from 4
  KAFKA_BATCH_SIZE: "1000"
  KAFKA_MAX_POLL_RECORDS: "500"

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# 4. Verify lag reduced
watch -n 10 'kubectl exec -n greenlang kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group gl-001-orchestrator | grep LAG'
```

---

## Multi-Region Deployment

**For global facilities with plants in multiple regions:**

### Architecture

```
Region 1 (US-East):           Region 2 (EU-West):          Region 3 (APAC):
├─ Master (Primary)           ├─ Master (Replica)          ├─ Master (Replica)
├─ Plants 1-5                 ├─ Plants 6-10               ├─ Plants 11-15
├─ Sub-agents (25)            ├─ Sub-agents (25)           ├─ Sub-agents (25)
├─ TimescaleDB (Primary)      ├─ TimescaleDB (Read)        ├─ TimescaleDB (Read)
└─ Kafka (Cluster)            └─ Kafka (Mirror)            └─ Kafka (Mirror)
```

### Deploy Multi-Region

```bash
# Region 1 (Primary - US-East)
kubectl config use-context us-east-1

kubectl apply -f deployment/gl-001-deployment.yaml
kubectl apply -f deployment/timescaledb-primary.yaml
kubectl apply -f deployment/kafka-cluster.yaml

# Region 2 (EU-West)
kubectl config use-context eu-west-1

kubectl apply -f deployment/gl-001-deployment.yaml
kubectl apply -f deployment/timescaledb-replica.yaml
kubectl apply -f deployment/kafka-mirror.yaml

# Configure cross-region database replication
# (See TimescaleDB multi-region replication docs)

# Configure Kafka MirrorMaker for cross-region messaging
kubectl apply -f deployment/kafka-mirrormaker.yaml
```

---

## Performance Testing

### Load Testing Procedures

**Simulate high plant load:**

```bash
# 1. Create load test scenario
cat > load-test.yaml <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: gl-001-load-test
  namespace: greenlang
data:
  scenario.json: |
    {
      "test_name": "multi_plant_load_test",
      "duration_minutes": 30,
      "plants": 15,
      "sub_agents": 75,
      "heat_optimization_requests_per_minute": 100,
      "agent_coordination_events_per_second": 50,
      "ramp_up_minutes": 5
    }
EOF

kubectl apply -f load-test.yaml

# 2. Run load test
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/test/load-test \
    -H "Content-Type: application/json" \
    -d @/config/scenario.json

# 3. Monitor during load test
watch -n 5 'kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator'

# 4. Check metrics during test
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | \
  jq '{
    orchestration_latency_p95,
    cpu_percent,
    memory_percent,
    agent_coordination_success_rate,
    heat_optimization_time_p95
  }'

# 5. Generate load test report
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/test/load-test/report | jq
```

---

## Cost Optimization

### Resource Right-Sizing

```bash
# Analyze actual resource usage over 7 days
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator --sort-by=cpu

# Compare to limits
kubectl get deployment gl-001-process-heat-orchestrator -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'

# If actual usage <60% of limits, reduce limits
# If actual usage >80% of requests, increase requests
```

### Auto-Scaling for Cost Savings

```bash
# Configure aggressive scale-down for non-production hours
# (Only if heat operations allow reduced capacity)

kubectl patch hpa gl-001-process-heat-orchestrator-hpa -n greenlang \
  --type=json -p='[{
    "op": "replace",
    "path": "/spec/minReplicas",
    "value": 3
  }]'

# For weekend or night hours, scale down manually if safe
# (Ensure backup heat systems active)

kubectl scale deployment gl-001-process-heat-orchestrator --replicas=3 -n greenlang
```

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Maintained By**: GreenLang Platform Engineering & Process Heat Team
**Review Cycle**: Quarterly or before major facility expansions
