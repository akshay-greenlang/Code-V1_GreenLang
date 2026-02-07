# High Execution Latency

## Alert

**Alert Name:** `OrchestratorHighLatency`

**Severity:** Warning

**Threshold:** `histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[10m])) by (le)) > 30` for 10 minutes

**Duration:** 10 minutes

---

## Description

This alert fires when the p99 node execution latency in the GreenLang Orchestrator (AGENT-FOUND-001) exceeds 30 seconds over a 10-minute window. High execution latency means DAG workflows are taking significantly longer than expected to complete, which delays compliance report generation, emissions calculations, and regulatory submissions.

### What Contributes to Execution Latency

DAG execution latency is the sum of several components:

1. **Scheduling overhead** (<5ms per node per NFR-002): Time for the orchestrator to select the next node, prepare inputs, and dispatch execution. This should be negligible.
2. **Agent execution time** (varies by agent type): The actual computation performed by each agent -- data loading, validation, calculation, aggregation, or report generation. This is typically the dominant factor.
3. **Checkpoint save time** (<50ms memory, <200ms file, varies for PostgreSQL per NFR-003): Time to persist the node's checkpoint after completion.
4. **Level synchronization** (depends on slowest node in level): Within each level, all nodes must complete before advancing to the next level. The slowest node in a level determines the level's latency.
5. **Retry delays** (configurable backoff): If nodes fail and retry, exponential backoff delays accumulate.
6. **Provenance calculation** (<1ms per node): SHA-256 hashing of inputs, outputs, and chain linking. Negligible.

High latency is typically caused by slow agents, resource contention, database performance degradation, or network latency between the orchestrator and agent services.

### Latency SLA Context

| Metric | Development | Staging | Production |
|--------|-------------|---------|------------|
| Node scheduling overhead | <10ms | <5ms | <5ms |
| Checkpoint save (PostgreSQL) | <500ms | <200ms | <200ms |
| DAG validation (100 nodes) | <20ms | <10ms | <10ms |
| Typical DAG execution (6 nodes) | <60s | <30s | <30s |
| Max DAG execution (500 nodes) | <600s | <300s | <300s |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Report generation and emissions calculations are delayed; users experience longer wait times |
| **Data Impact** | Low | No data loss; results are correct but slow |
| **SLA Impact** | Medium | Sustained high latency may cause regulatory submission deadlines to be at risk |
| **Revenue Impact** | Low-Medium | Customer experience degraded; compliance deliverables delayed |
| **Resource Impact** | Medium | Slow executions hold concurrency slots longer, reducing throughput for other workflows |
| **Cascading Impact** | Medium | Downstream DAGs waiting for results from slow executions are delayed |

---

## Symptoms

- `OrchestratorHighLatency` alert firing
- `gl_orchestrator_node_execution_duration_seconds` p99 exceeding 30s
- `gl_orchestrator_dag_execution_duration_seconds` showing upward trend
- Grafana DAG Execution dashboard shows extended execution bars
- Users reporting slow report generation or calculation results
- `gl_orchestrator_active_executions` gauge is elevated (executions accumulating because they take longer to complete)
- API responses for execution status show executions in "running" state for extended periods
- `gl_orchestrator_node_retries_total` may be elevated if slow operations are causing timeouts and retries

---

## Diagnostic Steps

### Step 1: Check Grafana Dashboard for Latency Breakdown

```bash
# Port-forward to check metrics directly
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080
```

```promql
# Overall DAG execution latency trend (p50, p95, p99)
histogram_quantile(0.50, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[10m])) by (le))
histogram_quantile(0.95, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[10m])) by (le))
histogram_quantile(0.99, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[10m])) by (le))

# Per-node execution latency (identify the slowest nodes)
histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[10m])) by (le, node_id))

# Node latency breakdown by DAG
histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[10m])) by (le, dag_id))
```

### Step 2: Identify Bottleneck Nodes

```bash
# Get execution metrics breakdown
curl -s "http://localhost:8080/api/v1/orchestrator/metrics" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Active executions: {data.get('active_executions', 'N/A')}\")
print(f\"Avg execution duration: {data.get('avg_execution_duration_ms', 'N/A')}ms\")
print(f\"p99 node duration: {data.get('p99_node_duration_ms', 'N/A')}ms\")
print()
if 'slowest_nodes' in data:
    print('Slowest nodes (last 10m):')
    for node in data['slowest_nodes'][:10]:
        print(f\"  {node['node_id']} ({node['dag_id']}): {node['avg_duration_ms']:.0f}ms avg, {node['p99_duration_ms']:.0f}ms p99\")
"
```

```promql
# Top 10 slowest nodes by average duration
topk(10,
  avg(rate(gl_orchestrator_node_execution_duration_seconds_sum[10m]))
  /
  avg(rate(gl_orchestrator_node_execution_duration_seconds_count[10m]))
  by (node_id, dag_id)
)
```

### Step 3: Check Agent Response Times

```bash
# Check if specific agents are slow
kubectl top pods -n greenlang -l app.kubernetes.io/component=agent

# Check agent-level metrics for the slow agent
kubectl logs -n greenlang -l agent-id=<slow_agent_id> --tail=200 \
  | grep -i "duration\|elapsed\|time\|slow\|latency"
```

```promql
# Check if agent pods are resource-constrained (CPU throttling)
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"<agent-pod-pattern>.*"}[5m]))

# Check agent memory pressure
container_memory_working_set_bytes{namespace="greenlang", pod=~"<agent-pod-pattern>.*"} /
container_spec_memory_limit_bytes{namespace="greenlang", pod=~"<agent-pod-pattern>.*"}
```

### Step 4: Check System Resources

```bash
# Check orchestrator pod resource usage
kubectl top pods -n greenlang -l app=orchestrator-service

# Check node (server) resource availability
kubectl top nodes

# Check for CPU throttling on orchestrator pods
kubectl get pods -n greenlang -l app=orchestrator-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.cpu}{"\t"}{.spec.containers[0].resources.limits.memory}{"\n"}{end}'
```

```promql
# Orchestrator CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"orchestrator.*"}[5m]))

# Orchestrator memory usage vs limit
container_memory_working_set_bytes{namespace="greenlang", pod=~"orchestrator.*"} /
container_spec_memory_limit_bytes{namespace="greenlang", pod=~"orchestrator.*"}
```

### Step 5: Check Database Performance

```bash
# Check for slow queries on orchestrator tables
kubectl run pg-slow --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    -- Check for slow queries involving orchestrator tables
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query
    FROM pg_stat_activity
    WHERE query LIKE '%dag_%' OR query LIKE '%node_traces%' OR query LIKE '%checkpoint%'
    ORDER BY duration DESC
    LIMIT 10;
  "

# Check checkpoint table size (large tables slow down queries)
kubectl run pg-size --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    SELECT
      'dag_checkpoints' as table_name,
      pg_size_pretty(pg_total_relation_size('dag_checkpoints')) as total_size,
      (SELECT count(*) FROM dag_checkpoints) as row_count
    UNION ALL
    SELECT
      'execution_provenance',
      pg_size_pretty(pg_total_relation_size('execution_provenance')),
      (SELECT count(*) FROM execution_provenance)
    UNION ALL
    SELECT
      'dag_executions',
      pg_size_pretty(pg_total_relation_size('dag_executions')),
      (SELECT count(*) FROM dag_executions);
  "

# Check connection pool utilization
kubectl run pg-pool --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    SELECT count(*) as total,
           count(*) FILTER (WHERE state = 'active') as active,
           count(*) FILTER (WHERE state = 'idle') as idle,
           count(*) FILTER (WHERE wait_event_type = 'Lock') as waiting_on_lock
    FROM pg_stat_activity
    WHERE application_name LIKE '%orchestrator%';
  "
```

### Step 6: Check Network Latency

```bash
# Test network latency between orchestrator and agents
kubectl exec -n greenlang <orchestrator-pod> -- \
  python3 -c "
import time, urllib.request
start = time.time()
urllib.request.urlopen('http://agent-factory.greenlang.svc.cluster.local:8080/health')
elapsed = (time.time() - start) * 1000
print(f'Agent factory round-trip: {elapsed:.1f}ms')
"

# Check for network policy issues
kubectl get networkpolicy -n greenlang
```

### Step 7: Check Concurrent Execution Load

```promql
# Current active execution count vs. configured maximum
gl_orchestrator_active_executions

# Parallel nodes per level (are levels overloaded?)
histogram_quantile(0.99, sum(rate(gl_orchestrator_parallel_nodes_per_level_bucket[10m])) by (le))

# Execution queue depth (executions waiting to start)
gl_orchestrator_queued_executions
```

### Step 8: Check for Retry Storms

```promql
# Node retry rate (high retries add latency from backoff delays)
sum(rate(gl_orchestrator_node_retries_total[10m])) by (dag_id, node_id)

# Node timeout rate (timeouts trigger retries which compound latency)
sum(rate(gl_orchestrator_node_timeouts_total[10m])) by (dag_id, node_id)

# Correlation: are retries concentrated on specific nodes?
topk(5, sum(increase(gl_orchestrator_node_retries_total[1h])) by (node_id))
```

---

## Resolution Steps

### Option 1: Reduce Parallelism to Decrease Contention

If too many concurrent nodes are competing for CPU, memory, database connections, or other shared resources:

```bash
# Reduce max_parallel_nodes (how many nodes execute simultaneously within a level)
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_MAX_PARALLEL_NODES=5

# Reduce max concurrent executions (how many DAGs run simultaneously)
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_MAX_CONCURRENT_EXECUTIONS=25

# Restart to apply
kubectl rollout restart deployment/orchestrator-service -n greenlang
```

### Option 2: Scale Up Orchestrator Replicas

If the orchestrator itself is the bottleneck (high CPU utilization, asyncio event loop contention):

```bash
# Scale up the orchestrator deployment
kubectl scale deployment/orchestrator-service -n greenlang --replicas=4

# Or adjust the HPA to allow more replicas
kubectl patch hpa orchestrator-service-hpa -n greenlang \
  -p '{"spec":{"maxReplicas": 8}}'

# Verify the new replicas are running
kubectl get pods -n greenlang -l app=orchestrator-service
```

### Option 3: Scale Up Slow Agent Pods

If the bottleneck is a specific agent that is resource-constrained:

```bash
# Increase resources for the slow agent
kubectl patch deployment <agent-deployment> -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "<agent-container>",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "4Gi"
              },
              "requests": {
                "cpu": "1",
                "memory": "2Gi"
              }
            }
          }
        ]
      }
    }
  }
}'

# Or scale out horizontally
kubectl scale deployment/<agent-deployment> -n greenlang --replicas=4
```

### Option 4: Increase Timeouts for Known-Slow Operations

If the high latency is expected for certain agent types (e.g., large dataset calculations):

```bash
# Update the DAG definition with increased timeouts for specific nodes
curl -X PUT "http://localhost:8080/api/v1/orchestrator/dags/<dag_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": {
      "<slow_node_id>": {
        "timeout_policy": {
          "timeout_seconds": 300,
          "on_timeout": "fail"
        }
      }
    }
  }' | python3 -m json.tool
```

**Note:** Increasing timeouts does not fix the root cause of slowness. It only prevents the orchestrator from marking the node as timed out. Combine this with root cause investigation.

### Option 5: Optimize Database Performance

If checkpoint saves or provenance writes are slow:

```bash
# Check and rebuild indexes on orchestrator tables
kubectl run pg-reindex --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    REINDEX TABLE dag_checkpoints;
    REINDEX TABLE execution_provenance;
    REINDEX TABLE node_traces;
  "

# Run ANALYZE to update query planner statistics
kubectl run pg-analyze --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    ANALYZE dag_checkpoints;
    ANALYZE execution_provenance;
    ANALYZE node_traces;
    ANALYZE dag_executions;
  "

# Clean up old checkpoints if the table is very large
kubectl run pg-cleanup --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    -- Delete checkpoints older than 30 days for completed executions
    DELETE FROM dag_checkpoints
    WHERE created_at < NOW() - INTERVAL '30 days'
    AND execution_id IN (
      SELECT execution_id FROM dag_executions WHERE status IN ('completed', 'failed', 'cancelled')
    );
  "

# Increase connection pool size if connections are exhausted
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_DB_POOL_SIZE=20 GL_ORCHESTRATOR_DB_POOL_MAX_OVERFLOW=10
```

### Option 6: Cache Frequently Used Calculation Results

If the same calculation inputs are being recomputed across multiple executions:

```bash
# Enable result caching in the orchestrator configuration
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_RESULT_CACHE_ENABLED=true \
  GL_ORCHESTRATOR_RESULT_CACHE_TTL_SECONDS=3600

# Restart to apply
kubectl rollout restart deployment/orchestrator-service -n greenlang
```

---

## Post-Resolution Verification

```promql
# 1. Verify p99 latency has dropped below threshold
histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[10m])) by (le)) < 30

# 2. Verify overall DAG execution duration is improving
histogram_quantile(0.95, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[10m])) by (le))

# 3. Verify active execution count is stable (not accumulating)
gl_orchestrator_active_executions

# 4. Verify retry rate has stabilized
sum(rate(gl_orchestrator_node_retries_total[10m]))

# 5. Verify checkpoint save latency is within bounds
histogram_quantile(0.99, sum(rate(gl_orchestrator_checkpoint_size_bytes_bucket[10m])) by (le))
```

```bash
# 6. Run a test execution and measure end-to-end latency
START_TIME=$(date +%s)
EXEC_RESPONSE=$(curl -s -X POST "http://localhost:8080/api/v1/orchestrator/dags/<test_dag_id>/execute" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"test": true}, "execution_options": {"checkpoint_enabled": true}}')

EXEC_ID=$(echo $EXEC_RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['execution_id'])")
echo "Execution ID: $EXEC_ID"

# Poll for completion
while true; do
  STATUS=$(curl -s "http://localhost:8080/api/v1/orchestrator/executions/$EXEC_ID" \
    -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  if [ "$STATUS" != "running" ] && [ "$STATUS" != "pending" ]; then
    break
  fi
  sleep 2
done

END_TIME=$(date +%s)
echo "Status: $STATUS"
echo "Total duration: $((END_TIME - START_TIME))s"
```

---

## Tuning Guide

### Optimal `max_parallel_nodes` by Environment

The `max_parallel_nodes` setting controls how many nodes within a single level can execute concurrently. Higher values increase throughput but also increase resource contention.

| Environment | Recommended Value | Rationale |
|-------------|------------------|-----------|
| Development | 2-4 | Limited resources; minimize contention for faster debugging |
| Staging | 5-8 | Mirror production proportionally; test concurrency behavior |
| Production | 8-12 | Balance throughput and resource usage; adjust based on load testing |
| Production (large cluster) | 12-20 | Only if cluster has significant spare capacity and agents are lightweight |

**How to determine the right value:**
1. Start with the recommended value for your environment
2. Run load tests with increasing `max_parallel_nodes` values
3. Plot execution duration vs. `max_parallel_nodes` -- look for the inflection point where latency starts increasing
4. Set the value 20% below the inflection point for safety margin

### Node Timeout Recommendations by Agent Type

| Agent Type | Default Timeout | Max Timeout | Notes |
|------------|----------------|-------------|-------|
| Intake agent (data loading) | 60s | 300s | Scales with data volume; set higher for files >100MB |
| Validation agent | 30s | 120s | Should be fast; high timeout indicates a bug |
| Scope 1 calculation agent | 120s | 300s | CPU-intensive; set based on facility count |
| Scope 2 calculation agent | 120s | 300s | CPU-intensive; depends on grid emission factors |
| Scope 3 calculation agent | 180s | 600s | Most complex; involves supplier data aggregation |
| Aggregation agent | 60s | 180s | Depends on number of input sources |
| Reporting agent (PDF/Excel) | 120s | 300s | I/O bound; large reports take longer |
| ERP connector agent | 180s | 600s | External API; unpredictable latency |
| Regulatory submission agent | 120s | 300s | External API; depends on regulatory portal |

### Checkpoint Frequency vs. Latency Tradeoff

Checkpointing after every node adds latency (the checkpoint save time per node) but provides maximum recoverability. The tradeoff:

| Strategy | Latency Overhead | Recoverability | When to Use |
|----------|-----------------|----------------|-------------|
| Every node (default) | +200ms per node | Maximum | Production; regulatory workflows |
| Every level | +200ms per level | Good | Staging; non-critical workflows |
| Disabled | None | None | Development; testing only |

For production workflows with regulatory compliance requirements, always use per-node checkpointing. The 200ms overhead per node is negligible compared to the cost of re-executing a multi-minute workflow from scratch.

### Monitoring Queries for Latency Analysis

Use these PromQL queries in Grafana to build a latency analysis dashboard:

```promql
# Overall execution latency by DAG (identify which DAGs are slow)
histogram_quantile(0.95, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[1h])) by (le, dag_id))

# Node-level latency heatmap (identify which nodes are slow within each DAG)
histogram_quantile(0.95, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[1h])) by (le, dag_id, node_id))

# Scheduling overhead (should be <5ms; if higher, orchestrator is the bottleneck)
# Compare node execution time to total level time
histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[10m])) by (le))

# Checkpoint save latency (should be <200ms for PostgreSQL)
histogram_quantile(0.99, sum(rate(gl_orchestrator_checkpoint_size_bytes_bucket[10m])) by (le))

# Retry-induced latency (retries add exponential backoff delays)
sum(increase(gl_orchestrator_node_retries_total[1h])) by (dag_id, node_id)

# Concurrent execution count vs. latency correlation
gl_orchestrator_active_executions
# Plot alongside:
histogram_quantile(0.99, sum(rate(gl_orchestrator_dag_execution_duration_seconds_bucket[10m])) by (le))

# Execution throughput (completed DAGs per minute)
sum(rate(gl_orchestrator_dag_executions_total{status="completed"}[10m])) * 60
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | p99 latency >30s for >10 minutes | On-call engineer | Within 30 minutes |
| L2 | p99 latency >60s for >30 minutes, multiple DAGs affected | Platform team lead + #orchestrator-oncall | Within 15 minutes |
| L3 | Sustained latency degradation causing regulatory deadline risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Latency caused by infrastructure failure (database, network) affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** DAG Execution Overview (`/d/dag-execution-overview`) -- latency panels
- **Dashboard:** Node Performance (`/d/node-performance`) -- per-node latency breakdown
- **Dashboard:** Orchestrator Service Health (`/d/orchestrator-health`) -- resource utilization
- **Alert:** `OrchestratorHighLatency` (this alert)
- **Key metrics to watch:**
  - `gl_orchestrator_dag_execution_duration_seconds` p95 and p99
  - `gl_orchestrator_node_execution_duration_seconds` p99 by node_id
  - `gl_orchestrator_active_executions` (trending upward = possible throughput issue)
  - `gl_orchestrator_node_retries_total` rate (retries compound latency)
  - `gl_orchestrator_node_timeouts_total` rate (timeouts indicate capacity issues)

### Capacity Planning

1. **Load test regularly** with production-representative DAG definitions and data volumes
2. **Monitor execution duration trends weekly** to detect gradual degradation before alerts fire
3. **Right-size agent pods** based on actual resource utilization, not just requests
4. **Plan for peak load** during quarterly reporting periods (CSRD, CBAM deadlines) when execution volume spikes
5. **Maintain 30% resource headroom** on the cluster for burst capacity

### Best Practices

1. **Keep DAGs shallow** when possible. Fewer levels means fewer synchronization points and lower end-to-end latency.
2. **Split large nodes** into smaller, parallelizable operations. A single 120s node is slower than two 60s nodes that can run in parallel.
3. **Use appropriate retry strategies.** Exponential backoff with jitter is the default. For idempotent operations with known transient failures, consider linear backoff with fewer retries.
4. **Profile slow agents independently** before adjusting orchestrator configuration. The orchestrator adds <5ms overhead per node; most latency is in the agents themselves.
5. **Enable result caching** for agents that are called with identical inputs across multiple executions.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any latency incident
- **Related alerts:** `OrchestratorServiceDown`, `OrchestratorExecutionTimeout`, `OrchestratorHighFailureRate`
- **Related dashboards:** DAG Execution Overview, Node Performance, Orchestrator Service Health
- **Related runbooks:** [Orchestrator Service Down](./orchestrator-service-down.md), [DAG Execution Stuck](./dag-execution-stuck.md), [Checkpoint Corruption](./checkpoint-corruption.md)
