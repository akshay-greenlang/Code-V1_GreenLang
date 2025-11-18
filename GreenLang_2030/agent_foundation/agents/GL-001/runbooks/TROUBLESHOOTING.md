# GL-001 ProcessHeatOrchestrator - Troubleshooting Guide

Comprehensive troubleshooting procedures for GL-001 ProcessHeatOrchestrator managing multi-plant industrial process heat operations.

## Table of Contents

1. [Master Orchestrator Issues](#master-orchestrator-issues)
2. [Sub-Agent Coordination Issues](#sub-agent-coordination-issues)
3. [Multi-Plant Integration Issues](#multi-plant-integration-issues)
4. [Heat Distribution Optimization Issues](#heat-distribution-optimization-issues)
5. [Performance Issues](#performance-issues)
6. [Database Issues](#database-issues)
7. [Message Bus Issues](#message-bus-issues)
8. [SCADA Integration Issues](#scada-integration-issues)
9. [ERP Integration Issues](#erp-integration-issues)
10. [Calculation Issues](#calculation-issues)

---

## Master Orchestrator Issues

### Master Orchestrator Not Starting

**Symptoms**:
- All GL-001 pods in CrashLoopBackOff or Error state
- Pods restart continuously
- Cannot access health endpoint

**Diagnostic Steps**:

```bash
# 1. Check pod status
kubectl get pods -n greenlang | grep gl-001

# Expected output if failing:
# gl-001-process-heat-orchestrator-5d8c7f-abc12   0/1  CrashLoopBackOff  5  10m

# 2. Check pod events
kubectl describe pod -n greenlang <pod-name> | grep -A 20 Events

# 3. Check container logs
kubectl logs -n greenlang <pod-name> --tail=200

# 4. Check previous container logs (if pod restarted)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# 5. Check resource constraints
kubectl describe pod -n greenlang <pod-name> | grep -E '(Requests|Limits|QoS)'
```

**Common Causes & Solutions**:

#### Cause 1: Missing Environment Variables

**Symptoms in logs**:
```
ERROR: Required environment variable DB_HOST not set
ERROR: Required environment variable KAFKA_BROKERS not set
ERROR: Configuration validation failed
```

**Solution**:
```bash
# Check ConfigMap
kubectl get configmap gl-001-config -n greenlang -o yaml

# Verify all required variables present:
# - DB_HOST
# - DB_PORT
# - DB_NAME
# - DB_USER
# - KAFKA_BROKERS
# - REDIS_HOST
# - SCADA_ENDPOINTS
# - ERP_ENDPOINT

# If missing, edit ConfigMap
kubectl edit configmap gl-001-config -n greenlang

# Add missing variables:
data:
  DB_HOST: "postgresql.greenlang.svc.cluster.local"
  DB_PORT: "5432"
  KAFKA_BROKERS: "kafka-0.kafka:9092,kafka-1.kafka:9092,kafka-2.kafka:9092"

# Restart deployment to pick up changes
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Cause 2: Missing Secrets

**Symptoms in logs**:
```
ERROR: Database password not found
ERROR: Cannot connect to database: authentication failed
ERROR: Secret 'gl-001-db-credentials' not found
```

**Solution**:
```bash
# Check if secrets exist
kubectl get secrets -n greenlang | grep gl-001

# Should see:
# gl-001-db-credentials
# gl-001-kafka-credentials
# gl-001-scada-credentials
# gl-001-erp-credentials

# If missing, create secrets
kubectl create secret generic gl-001-db-credentials -n greenlang \
  --from-literal=username=gl001_user \
  --from-literal=password='<secure-password>'

kubectl create secret generic gl-001-kafka-credentials -n greenlang \
  --from-literal=username=gl001_kafka \
  --from-literal=password='<secure-password>'

# Restart deployment
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Cause 3: Database Connection Failure

**Symptoms in logs**:
```
ERROR: Could not connect to database at postgresql.greenlang.svc.cluster.local:5432
ERROR: Connection refused
ERROR: Database 'gl001_production' does not exist
```

**Solution**:
```bash
# Test database connectivity from pod
kubectl run -it --rm debug --image=postgres:15 -n greenlang -- \
  psql -h postgresql.greenlang.svc.cluster.local -U gl001_user -d gl001_production

# If connection fails, check database pod
kubectl get pods -n greenlang | grep postgres
kubectl logs -n greenlang <postgres-pod> --tail=100

# Check database service
kubectl get svc -n greenlang | grep postgres

# Verify database DNS resolution
kubectl run -it --rm debug --image=busybox -n greenlang -- \
  nslookup postgresql.greenlang.svc.cluster.local

# If database doesn't exist, create it
kubectl exec -it -n greenlang <postgres-pod> -- \
  psql -U postgres -c "CREATE DATABASE gl001_production OWNER gl001_user;"

# Run database migrations
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  python manage.py migrate
```

#### Cause 4: OOMKilled (Memory Exhaustion)

**Symptoms in pod events**:
```
Reason: OOMKilled
Exit Code: 137
Last State: Terminated (exit code 137)
```

**Solution**:
```bash
# Check current memory limits
kubectl get deployment gl-001-process-heat-orchestrator -n greenlang -o yaml | \
  grep -A 5 resources

# Increase memory limits (calculate based on plant count)
# Formula: Memory = 2Gi + (plant_count * 200Mi) + (subagent_count * 100Mi)
# Example for 10 plants, 50 sub-agents: 2Gi + 2Gi + 5Gi = 9Gi

kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=memory=10Gi \
  --requests=memory=6Gi

# Monitor pod startup
kubectl get pods -n greenlang -w | grep gl-001

# Check memory usage after startup
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator
```

#### Cause 5: Image Pull Failure

**Symptoms in pod events**:
```
Failed to pull image "gcr.io/greenlang/gl-001:latest"
ImagePullBackOff
ErrImagePull: unauthorized
```

**Solution**:
```bash
# Check image pull secrets
kubectl get secrets -n greenlang | grep docker

# Verify secret is attached to service account
kubectl get serviceaccount gl-001-orchestrator -n greenlang -o yaml

# Should have imagePullSecrets:
# - name: greenlang-registry

# If missing, patch service account
kubectl patch serviceaccount gl-001-orchestrator -n greenlang -p \
  '{"imagePullSecrets": [{"name": "greenlang-registry"}]}'

# Recreate image pull secret if needed
kubectl create secret docker-registry greenlang-registry -n greenlang \
  --docker-server=gcr.io \
  --docker-username=_json_key \
  --docker-password="$(cat /path/to/gcr-key.json)" \
  --docker-email=devops@greenlang.io

# Manually pull image to verify
docker pull gcr.io/greenlang/gl-001:latest

# Restart deployment
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Cause 6: Configuration Validation Failure

**Symptoms in logs**:
```
ERROR: Invalid configuration: plant_configurations must not be empty
ERROR: Invalid SCADA endpoint format for Plant-001
ERROR: Heat distribution constraints conflict detected
```

**Solution**:
```bash
# Check configuration validation
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  python -c "from config import ProcessHeatConfig; ProcessHeatConfig.load_and_validate()"

# Fix configuration in ConfigMap
kubectl edit configmap gl-001-config -n greenlang

# Common fixes:
# 1. Ensure plant_configurations is not empty
# 2. Verify SCADA endpoint URLs are valid
# 3. Check constraint values are within bounds
# 4. Ensure no circular dependencies in heat distribution

# Restart deployment
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

---

### Master Orchestrator High Error Rate

**Symptoms**:
- Error rate >5% on health metrics
- Frequent exceptions in logs
- Sub-agents reporting master communication failures

**Diagnostic Steps**:

```bash
# Check error rate
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | grep error_rate

# Check recent errors
kubectl logs -n greenlang deployment/gl-001-process-heat-orchestrator \
  --tail=500 | grep -E '(ERROR|CRITICAL|FATAL)' | tail -50

# Check error breakdown by type
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/errors | jq '.error_breakdown'
```

**Common Causes & Solutions**:

#### Cause 1: Database Connection Pool Exhaustion

**Symptoms**:
```
ERROR: Could not acquire database connection from pool
ERROR: Max pool size 20 reached
ERROR: Connection wait timeout after 30 seconds
```

**Solution**:
```bash
# Check current pool usage
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | \
  jq '.connection_pool'

# Increase pool size in ConfigMap
kubectl edit configmap gl-001-config -n greenlang

# Update DB_POOL_SIZE (calculate based on load)
# Formula: Pool Size = (plant_count * 5) + (replica_count * 10)
# Example: 10 plants, 3 replicas = 50 + 30 = 80
data:
  DB_POOL_SIZE: "100"
  DB_POOL_TIMEOUT: "60"
  DB_POOL_RECYCLE: "3600"

# Restart deployment
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# Verify pool size increased
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | jq '.pool_size'
```

---

## Sub-Agent Coordination Issues

### Sub-Agent Coordination Failures

**Symptoms**:
- Sub-agents not responding to master commands
- Task delegation failures
- Message timeouts
- Agents showing "disconnected" status

**Diagnostic Steps**:

```bash
# Check agent coordination status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents[] | select(.status != "connected")'

# Count disconnected agents
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq '.agents | map(select(.status != "connected")) | length'

# Check message bus lag
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/message-bus/lag | jq

# Check coordination queue depth
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/coordination/queue | \
  jq '.queue_depth'
```

**Common Causes & Solutions**:

#### Cause 1: Message Bus (Kafka) Connectivity Issues

**Symptoms**:
```
ERROR: Failed to publish message to Kafka topic 'agent.gl-002'
ERROR: Kafka broker kafka-0:9092 not reachable
ERROR: Message delivery timeout after 30s
```

**Solution**:
```bash
# Check Kafka pods
kubectl get pods -n greenlang | grep kafka

# Check Kafka broker health
kubectl exec -n greenlang kafka-0 -- kafka-broker-api-versions.sh \
  --bootstrap-server localhost:9092

# Check Kafka topics
kubectl exec -n greenlang kafka-0 -- kafka-topics.sh \
  --list --bootstrap-server localhost:9092 | grep agent

# Should see topics like:
# agent.gl-002
# agent.gl-003
# agent.gl-004
# ...

# If topics missing, create them
kubectl exec -n greenlang kafka-0 -- kafka-topics.sh \
  --create --topic agent.gl-002 \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 2

# Check consumer lag
kubectl exec -n greenlang kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group gl-001-orchestrator

# If Kafka unhealthy, restart brokers
kubectl rollout restart statefulset/kafka -n greenlang
kubectl rollout status statefulset/kafka -n greenlang --timeout=5m

# Verify master reconnected
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/message-bus/health
```

#### Cause 2: Sub-Agent Pods Not Running

**Symptoms**:
```
ERROR: Agent GL-002 not responding (timeout after 30s)
ERROR: Agent GL-003 status: disconnected
ERROR: Cannot delegate task to GL-004: agent unavailable
```

**Solution**:
```bash
# Check all sub-agent pods
kubectl get pods -n greenlang | grep -E '(gl-002|gl-003|gl-004|gl-005)'

# Count running vs total
kubectl get pods -n greenlang -l agent-group=process-heat-sub-agents \
  --no-headers | wc -l

kubectl get pods -n greenlang -l agent-group=process-heat-sub-agents \
  --field-selector=status.phase=Running --no-headers | wc -l

# Check failed pods
kubectl get pods -n greenlang -l agent-group=process-heat-sub-agents \
  --field-selector=status.phase!=Running

# Restart failed sub-agent
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# If multiple agents down, restart by group
# Boiler and Steam group
for agent in gl-002 gl-012 gl-016 gl-017; do
  kubectl rollout restart deployment/${agent}-* -n greenlang
done

# Verify agents reconnecting
watch -n 5 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | \
  jq ".agents | map(select(.status == \"connected\")) | length"'
```

#### Cause 3: Network Policy Blocking Agent Communication

**Symptoms**:
```
ERROR: Connection refused when connecting to agent GL-002 at 10.0.1.42:8002
ERROR: Network timeout to agent services
ERROR: Agent coordination failed: connection reset by peer
```

**Solution**:
```bash
# Check network policies
kubectl get networkpolicy -n greenlang

# Verify GL-001 can reach sub-agents
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -v http://gl-002-boiler-efficiency:8002/health --max-time 5

# If connection fails, check network policy
kubectl describe networkpolicy gl-001-orchestrator-policy -n greenlang

# Ensure policy allows egress to sub-agents
# Edit if needed:
kubectl edit networkpolicy gl-001-orchestrator-policy -n greenlang

# Add egress rule:
spec:
  egress:
  - to:
    - podSelector:
        matchLabels:
          agent-group: process-heat-sub-agents
    ports:
    - protocol: TCP
      port: 8002
    - protocol: TCP
      port: 8003
    # ... etc for all sub-agent ports

# Test connectivity again
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -v http://gl-002-boiler-efficiency:8002/health
```

#### Cause 4: Agent Registration Failures

**Symptoms**:
```
ERROR: Agent GL-002 failed to register with master
ERROR: Agent authentication failed
ERROR: Agent version mismatch (master: 1.5.0, agent: 1.4.0)
```

**Solution**:
```bash
# Check agent registration status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/registration | jq

# Check agent versions
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/versions | \
  jq '.agents[] | {agent_id, version, compatible}'

# If version mismatch, update sub-agents
kubectl set image deployment/gl-002-boiler-efficiency \
  gl-002-boiler-efficiency=gcr.io/greenlang/gl-002:1.5.0 \
  -n greenlang

# Verify registration secrets
kubectl get secret gl-001-agent-auth -n greenlang -o yaml

# If secret missing, create it
kubectl create secret generic gl-001-agent-auth -n greenlang \
  --from-literal=auth-token='<generated-token>' \
  --from-literal=encryption-key='<32-byte-key>'

# Restart sub-agents to re-register
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

---

### Task Delegation Failures

**Symptoms**:
- Tasks stuck in pending state
- Sub-agents not receiving tasks
- Task timeouts
- Orphaned tasks

**Diagnostic Steps**:

```bash
# Check task queue
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/coordination/tasks | \
  jq '.tasks[] | select(.status == "pending")'

# Count pending tasks
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/coordination/tasks | \
  jq '.tasks | map(select(.status == "pending")) | length'

# Check task assignment distribution
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/coordination/tasks/distribution | jq
```

**Solutions**:

```bash
# Option 1: Retry failed task delegations
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/coordination/tasks/retry-failed

# Option 2: Rebalance task distribution
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/coordination/tasks/rebalance

# Option 3: Clear orphaned tasks
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/coordination/tasks/clear-orphaned \
    -H "Content-Type: application/json" \
    -d '{"older_than_minutes": 60}'

# Option 4: Restart coordination workers
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/coordination/workers/restart
```

---

## Multi-Plant Integration Issues

### Multi-Plant Heat Balancing Errors

**Symptoms**:
- Heat imbalance across plants
- Optimization fails for multi-plant scenarios
- Plants not coordinating heat sharing

**Diagnostic Steps**:

```bash
# Check multi-plant heat balance
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-balance | jq

# Check inter-plant heat flows
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/inter-plant-flows | \
  jq '.flows[] | {from, to, heat_mw, status}'

# Check plant constraints
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/constraints | jq
```

**Solutions**:

#### Solution 1: Recalculate Heat Balance

```bash
# Force recalculation
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/plants/recalculate-balance \
    -H "Content-Type: application/json" \
    -d '{"plants": ["all"], "force": true}'

# Verify balance restored
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-balance | \
  jq '.plants[] | {plant_id, balance_status, imbalance_mw}'
```

#### Solution 2: Relax Multi-Plant Constraints

```bash
# Check violated constraints
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/violated-constraints | jq

# Temporarily relax constraints
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/plants/relax-constraints \
    -H "Content-Type: application/json" \
    -d '{
      "constraints": ["inter_plant_flow_limit", "min_efficiency"],
      "relaxation_factor": 0.9,
      "duration_minutes": 120
    }'
```

---

### SCADA Multi-Plant Connectivity Issues

**Symptoms**:
- Multiple plants showing SCADA connection loss
- Inconsistent data across plants
- SCADA feed timeouts

**Diagnostic Steps**:

```bash
# Check SCADA connectivity for all plants
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/status | \
  jq '.scada_feeds[] | {plant_id, status, last_update, latency_ms}'

# Count disconnected SCADA feeds
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/status | \
  jq '.scada_feeds | map(select(.status != "connected")) | length'

# Check SCADA data quality
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/data-quality | jq
```

**Solutions**:

#### Solution 1: Reconnect SCADA Feeds

```bash
# Reconnect all SCADA feeds
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/scada/reconnect-all

# Reconnect specific plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/scada/reconnect \
    -H "Content-Type: application/json" \
    -d '{"plant_id": "Plant-001"}'

# Verify reconnection
watch -n 10 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/status | \
  jq ".scada_feeds | map(select(.status == \"connected\")) | length"'
```

#### Solution 2: Switch to Backup SCADA

```bash
# Check backup SCADA availability
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/backup-status | jq

# Failover to backup SCADA for plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/scada/failover \
    -H "Content-Type: application/json" \
    -d '{
      "plant_id": "Plant-001",
      "backup_server": "scada-backup-01.plant001.local",
      "reason": "primary_scada_timeout"
    }'
```

#### Solution 3: Contact SCADA Vendor

```bash
# If multiple plants affected, contact vendor
# Siemens 24/7 Support: +1-800-XXX-XXXX
# Honeywell 24/7 Support: +1-800-YYY-YYYY

# Gather diagnostic data for vendor
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/scada/diagnostic-report > scada_diagnostics.json

# Email to vendor support with diagnostic report
```

---

## Heat Distribution Optimization Issues

### Heat Distribution LP Solver Failures

**Symptoms**:
- Optimization fails with "infeasible" or "unbounded"
- Heat distribution not optimal
- Long optimization times (>10 seconds)

**Diagnostic Steps**:

```bash
# Check optimization status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status | jq

# Check solver diagnostics
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/diagnostics | jq

# Check constraint violations
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/constraints | \
  jq '.violated_constraints'
```

**Solutions**:

#### Solution 1: Switch to Heuristic Algorithm

```bash
# Switch from LP solver to heuristic (faster, sub-optimal)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/heat-distribution/algorithm \
    -H "Content-Type: application/json" \
    -d '{
      "algorithm": "greedy_heuristic",
      "reason": "LP_solver_infeasible",
      "temporary": true,
      "duration_minutes": 60
    }'

# Verify optimization running
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status | \
  jq '.algorithm_in_use'
```

#### Solution 2: Relax Constraints (Infeasible Problem)

```bash
# Identify conflicting constraints
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/constraint-conflicts | jq

# Relax constraints temporarily
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/relax-constraints \
    -H "Content-Type: application/json" \
    -d '{
      "constraints": ["min_efficiency", "max_heat_loss"],
      "relaxation_factor": 0.95,
      "duration_minutes": 120,
      "reason": "infeasible_problem"
    }'

# Re-run optimization
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/optimization/heat-distribution/run
```

#### Solution 3: Check LP Solver License (Gurobi/CPLEX)

```bash
# Check solver license status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/license-status | jq

# If license expired:
# Expected output:
# {
#   "solver": "Gurobi",
#   "license_status": "expired",
#   "expiration_date": "2025-11-01",
#   "days_remaining": -15
# }

# Update license file
kubectl create secret generic gl-001-gurobi-license -n greenlang \
  --from-file=gurobi.lic=/path/to/new/gurobi.lic \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment to pick up new license
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Solution 4: Increase Solver Time Limit

```bash
# If optimization timing out
kubectl edit configmap gl-001-config -n greenlang

# Update solver time limit:
data:
  LP_SOLVER_TIME_LIMIT_SECONDS: "120"  # Increase from 60 to 120

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

---

## Performance Issues

### High Orchestration Latency

**Symptoms**:
- Orchestration latency >5 seconds (p95)
- Slow response times to plant requests
- Task delegation delays

**Diagnostic Steps**:

```bash
# Check latency metrics
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | \
  grep orchestration_latency

# Check latency breakdown
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/latency-breakdown | jq

# Check slow operations
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/slow-operations | jq
```

**Solutions**:

#### Solution 1: Scale Master Orchestrator (Horizontal)

```bash
# Calculate required replicas
# Formula: Replicas = ceil(plant_count / 5) + ceil(subagent_count / 10)
# Example: 15 plants, 60 agents = ceil(15/5) + ceil(60/10) = 3 + 6 = 9

# Scale up
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=9 -n greenlang

# Monitor latency improvement
watch -n 10 'kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | grep orchestration_latency_p95'
```

#### Solution 2: Increase CPU/Memory (Vertical Scaling)

```bash
# Check current resource usage
kubectl top pod -n greenlang -l app=gl-001-process-heat-orchestrator

# If CPU >80%, increase CPU
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=cpu=4000m --requests=cpu=2000m

# If memory >75%, increase memory
kubectl set resources deployment gl-001-process-heat-orchestrator -n greenlang \
  --limits=memory=16Gi --requests=memory=10Gi
```

#### Solution 3: Optimize Database Queries

```bash
# Check slow database queries
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database-slow-queries | jq

# Enable query caching
kubectl edit configmap gl-001-config -n greenlang

data:
  CACHE_QUERY_RESULTS: "true"
  CACHE_TTL_SECONDS: "300"

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

#### Solution 4: Add Redis Cache

```bash
# Check cache hit rate
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/cache | jq '.hit_rate'

# If cache hit rate <80%, increase Redis memory
kubectl set resources deployment redis -n greenlang \
  --limits=memory=8Gi --requests=memory=4Gi

# Scale Redis to cluster mode for high availability
kubectl scale deployment redis --replicas=3 -n greenlang
```

---

### Database Connection Pool Exhaustion

**Symptoms**:
```
ERROR: Could not acquire database connection from pool
WARNING: All database connections in use
ERROR: Connection wait timeout
```

**Solution**:

```bash
# Check current pool usage
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics/database | \
  jq '.connection_pool | {size, in_use, available, wait_count}'

# Calculate optimal pool size
# Formula: Pool Size = (plant_count * 5) + (replica_count * 10) + 20
# Example: 15 plants, 9 replicas = 75 + 90 + 20 = 185

kubectl edit configmap gl-001-config -n greenlang

data:
  DB_POOL_SIZE: "200"
  DB_POOL_TIMEOUT: "60"
  DB_POOL_OVERFLOW: "50"
  DB_POOL_RECYCLE: "3600"

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

---

### Message Bus Lag

**Symptoms**:
- Sub-agents receiving delayed commands
- Task delegation lag >2 seconds
- Kafka consumer lag increasing

**Diagnostic Steps**:

```bash
# Check message bus lag
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/message-bus/lag | jq

# Check Kafka consumer lag
kubectl exec -n greenlang kafka-0 -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 \
  --describe --group gl-001-orchestrator
```

**Solutions**:

```bash
# Scale Kafka partitions
kubectl exec -n greenlang kafka-0 -- kafka-topics.sh \
  --alter --topic agent.gl-002 \
  --partitions 6 \
  --bootstrap-server localhost:9092

# Scale Kafka brokers
kubectl scale statefulset kafka --replicas=5 -n greenlang

# Increase consumer threads
kubectl edit configmap gl-001-config -n greenlang

data:
  KAFKA_CONSUMER_THREADS: "8"
  KAFKA_BATCH_SIZE: "1000"

kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang
```

---

## Database Issues

### TimescaleDB Performance Degradation

**Symptoms**:
- Slow queries on time-series data
- High query latency (>5 seconds)
- Database CPU >90%

**Solutions**:

```bash
# Check TimescaleDB chunk status
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT show_chunks('heat_measurements');"

# Optimize chunks
kubectl exec -n greenlang postgresql-0 -- psql -U gl001_user -d gl001_production -c \
  "SELECT compress_chunk(i) FROM show_chunks('heat_measurements') i;"

# Add read replicas for query load
# See SCALING_GUIDE.md for TimescaleDB scaling procedures

# Tune autovacuum
kubectl exec -n greenlang postgresql-0 -- psql -U postgres -c \
  "ALTER SYSTEM SET autovacuum_max_workers = 4;"

kubectl rollout restart statefulset/postgresql -n greenlang
```

---

## Calculation Issues

### Thermal Efficiency Calculation Errors

**Symptoms**:
```
ERROR: Thermal efficiency calculation failed
ERROR: Invalid heat balance: input > output + losses + 10%
ERROR: Efficiency calculation returned NaN
```

**Solutions**:

```bash
# Check calculation validation
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/calculations/validate | jq

# Re-run calculation with debug mode
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -X POST http://localhost:8000/api/v1/calculations/thermal-efficiency \
    -H "Content-Type: application/json" \
    -d '{
      "plant_id": "Plant-001",
      "debug_mode": true,
      "validate_inputs": true
    }' | jq

# Check for data quality issues
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/data-quality/plant/Plant-001 | jq
```

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Maintained By**: GreenLang Platform Engineering & Process Heat Team
