# DAG Execution Stuck

## Alert

**Alert Name:** `OrchestratorExecutionTimeout`

**Severity:** Warning

**Threshold:** `gl_orchestrator_dag_execution_duration_seconds{status="running"} > gl_orchestrator_execution_timeout_seconds` for 10 minutes

**Duration:** 10 minutes

---

## Description

This alert fires when a DAG execution exceeds its configured timeout threshold while remaining in the "running" status. A stuck execution means one or more nodes in the DAG have not completed within the expected time window. This can be caused by an unresponsive agent, an external API call that is hanging, a resource deadlock, checkpoint save failures preventing progress, or resource contention from too many concurrent executions.

### How DAG Execution Works

The GreenLang Orchestrator (AGENT-FOUND-001) executes DAG workflows in levels determined by topological sort. Within each level, independent nodes execute concurrently up to the `max_parallel_nodes` limit. Execution advances to the next level only when all nodes in the current level have completed (or been skipped/failed per the on-failure strategy). If any node in a level hangs indefinitely, the entire execution stalls at that level boundary.

### Timeout Enforcement

Each node has its own `TimeoutPolicy`:
- `timeout_seconds`: Maximum wall-clock time for the node to complete
- `on_timeout`: What to do when timeout is reached -- `fail` (mark node as failed), `skip` (continue without result), or `compensate` (run compensation handler)

The DAG-level execution timeout is the overall timeout for the entire DAG run, independent of per-node timeouts. This alert fires when the DAG-level timeout is breached.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Specific workflow blocked; users waiting for compliance reports or calculation results |
| **Data Impact** | Low | Completed nodes are checkpointed; no data loss for finished nodes |
| **SLA Impact** | Medium | If the stuck DAG is on a critical path for a regulatory submission, deadlines may be at risk |
| **Revenue Impact** | Low-Medium | Single workflow affected; other DAG executions continue normally |
| **Downstream Impact** | Medium | Consumers waiting for this execution's results are blocked; dependent workflows cannot start |
| **Resource Impact** | Medium | Stuck execution holds concurrency slots, potentially reducing capacity for other executions |

---

## Symptoms

- `OrchestratorExecutionTimeout` alert firing for a specific `execution_id`
- `gl_orchestrator_dag_execution_duration_seconds{status="running"}` exceeds threshold
- `gl_orchestrator_active_executions` gauge includes the stuck execution
- Execution status remains "running" in the API response for an extended period
- Grafana DAG Execution dashboard shows one or more executions with abnormally long duration
- Node execution trace shows the last completed node, with subsequent nodes not started or stuck
- `gl_orchestrator_node_execution_duration_seconds` for a specific node shows extreme outlier
- Downstream applications report timeouts waiting for orchestrator results

---

## Diagnostic Steps

### Step 1: Identify the Stuck Execution

```bash
# List all currently running executions
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080

curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=running" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
from datetime import datetime, timezone
data = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for exec in data.get('items', []):
    started = datetime.fromisoformat(exec['started_at'].replace('Z', '+00:00'))
    duration = (now - started).total_seconds()
    print(f\"ID: {exec['execution_id']}\")
    print(f\"  DAG: {exec['dag_id']}\")
    print(f\"  Started: {exec['started_at']}\")
    print(f\"  Duration: {duration:.0f}s ({duration/60:.1f}m)\")
    print()
"
```

```promql
# Find executions exceeding their timeout in Prometheus
gl_orchestrator_dag_execution_duration_seconds{status="running"} > 300
```

### Step 2: Check the Execution Trace for Last Completed Node

```bash
# Get the execution trace to see which nodes completed and which are stuck
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/trace" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Execution: {data['execution_id']}\")
print(f\"Status: {data['status']}\")
print(f\"Topology levels: {data.get('topology_levels', [])}\")
print()
print('Node Status:')
for node_id, trace in data.get('node_traces', {}).items():
    status = trace.get('status', 'unknown')
    duration = trace.get('duration_ms', 0)
    attempts = trace.get('attempt_count', 0)
    error = trace.get('error', '')
    marker = '  ** STUCK **' if status == 'running' else ''
    print(f\"  {node_id}: {status} (duration={duration:.0f}ms, attempts={attempts}){marker}\")
    if error:
        print(f\"    Error: {error}\")
"
```

### Step 3: Identify the Bottleneck Node

```bash
# Get detailed information about the specific stuck node
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/trace" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for node_id, trace in data.get('node_traces', {}).items():
    if trace.get('status') == 'running':
        print(f\"STUCK NODE: {node_id}\")
        print(f\"  Agent ID: {trace.get('agent_id', 'N/A')}\")
        print(f\"  Started at: {trace.get('started_at', 'N/A')}\")
        print(f\"  Attempt count: {trace.get('attempt_count', 0)}\")
        print(f\"  Input hash: {trace.get('input_hash', 'N/A')}\")
"
```

### Step 4: Check if the Agent Is Responsive

```bash
# Check the agent pod status
kubectl get pods -n greenlang -l agent-id=<agent_id>

# Check agent logs for errors or hanging operations
kubectl logs -n greenlang -l agent-id=<agent_id> --tail=200 \
  | grep -i "error\|timeout\|hang\|stuck\|blocked\|waiting"

# Check if the agent is consuming excessive resources
kubectl top pods -n greenlang -l agent-id=<agent_id>
```

### Step 5: Check for Resource Contention

```bash
# Check how many executions are running concurrently
curl -s "http://localhost:8080/api/v1/orchestrator/metrics" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Active executions: {data.get('active_executions', 'N/A')}\")
print(f\"Max concurrent: {data.get('max_concurrent_executions', 'N/A')}\")
print(f\"Queued executions: {data.get('queued_executions', 'N/A')}\")
"
```

```promql
# Check active execution count vs. configured limit
gl_orchestrator_active_executions

# Check for database connection pool exhaustion
gl_db_pool_active_connections{service="orchestrator-service"}
gl_db_pool_pending_connections{service="orchestrator-service"}

# Check node execution duration percentiles for anomalies
histogram_quantile(0.99, sum(rate(gl_orchestrator_node_execution_duration_seconds_bucket[5m])) by (le, node_id))
```

### Step 6: Check for Distributed Lock Contention

```bash
# Check Redis for orchestrator locks
kubectl run redis-check --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:orchestrator:lock:*'

# Check lock TTLs (a stuck lock with no TTL could block other operations)
kubectl run redis-ttl --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 \
  TTL 'gl:orchestrator:lock:<execution_id>'
```

### Step 7: Check for External API Slowness

If the stuck node calls an external service (e.g., ERP connector, emissions factor database, regulatory API):

```bash
# Check the orchestrator logs for external call timing
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "<agent_id>\|external\|http\|api\|timeout"

# Check OTel traces for the stuck node's span
# Open Grafana Tempo: service.name = "orchestrator-service" AND execution_id = "<execution_id>"
```

### Step 8: Check Checkpoint Store Health

A stuck execution could be caused by checkpoint operations failing, preventing the execution from advancing past a completed node.

```bash
# Check checkpoint operations for errors
curl -s "http://localhost:8080/api/v1/orchestrator/checkpoints/<execution_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Checkpoints for execution: {len(data.get('checkpoints', []))}\")
for cp in data.get('checkpoints', []):
    print(f\"  Node: {cp['node_id']} | Status: {cp['status']} | Created: {cp['created_at']}\")
"
```

```promql
# Check for checkpoint save failures
increase(gl_orchestrator_checkpoint_operations_total{operation="save", status="error"}[15m])
```

---

## Resolution Steps

### Option 1: Cancel the Stuck Execution

If the execution cannot be salvaged or the stuck node is not critical, cancel the execution. Completed node checkpoints are preserved.

```bash
# Cancel the stuck execution
curl -X POST "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/cancel" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" | python3 -m json.tool
```

After cancellation, start a new execution. The new execution will start from scratch (a fresh DAG run). If you want to reuse completed work, use the resume option instead.

```bash
# Start a new execution of the same DAG
curl -X POST "http://localhost:8080/api/v1/orchestrator/dags/<dag_id>/execute" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": <original_input_data>,
    "execution_options": {
      "checkpoint_enabled": true,
      "deterministic_mode": true
    }
  }' | python3 -m json.tool
```

### Option 2: Resume from Checkpoint

If the stuck execution was interrupted but has valid checkpoints, resume from the last successful checkpoint. Completed nodes will be skipped.

```bash
# Resume the execution from the last checkpoint
curl -X POST "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/resume" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Execution ID: {data.get('execution_id')}\")
print(f\"Status: {data.get('status')}\")
print(f\"Nodes skipped (from checkpoint): {data.get('nodes_skipped', [])}\")
print(f\"Nodes to execute: {data.get('nodes_remaining', [])}\")
"
```

### Option 3: Restart the Stuck Agent

If the bottleneck is a specific agent that is unresponsive:

```bash
# Restart the agent pod
kubectl delete pod -n greenlang <agent-pod-name>

# Verify the agent pod is back
kubectl get pods -n greenlang -l agent-id=<agent_id>

# The orchestrator's retry policy should automatically retry the stuck node
# once the agent is available again. Monitor the execution trace:
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/trace" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -m json.tool
```

### Option 4: Reduce Concurrency to Relieve Resource Contention

If multiple executions are competing for resources and causing timeouts:

```bash
# Reduce max parallel nodes per execution
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_MAX_PARALLEL_NODES=5

# Reduce max concurrent executions
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_MAX_CONCURRENT_EXECUTIONS=25

# Restart to apply
kubectl rollout restart deployment/orchestrator-service -n greenlang
```

### Option 5: Release Stuck Distributed Locks

If a Redis distributed lock is preventing progress (e.g., the lock holder crashed before releasing):

```bash
# Delete the stuck lock (use with caution -- only if the lock holder is confirmed dead)
kubectl run redis-unlock --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 \
  DEL 'gl:orchestrator:lock:<execution_id>'
```

### Option 6: Increase Node Timeout for Known-Slow Operations

If the stuck node is performing a legitimately slow operation (e.g., large dataset calculation, external API with high latency):

```bash
# Update the DAG definition to increase the timeout for the specific node
curl -X PUT "http://localhost:8080/api/v1/orchestrator/dags/<dag_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": {
      "<node_id>": {
        "timeout_policy": {
          "timeout_seconds": 300,
          "on_timeout": "fail"
        }
      }
    }
  }' | python3 -m json.tool
```

---

## Post-Resolution Verification

After resolving the stuck execution:

```bash
# 1. Verify the execution completed or was cancelled cleanly
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Status: {data.get('status')}\")
print(f\"Completed at: {data.get('completed_at', 'N/A')}\")
print(f\"Error: {data.get('error', 'None')}\")
"

# 2. Verify no other executions are stuck
curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=running" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
count = len(data.get('items', []))
print(f\"Running executions: {count}\")
"

# 3. Verify the provenance chain is intact (if execution was resumed)
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/provenance" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Chain valid: {data.get('chain_valid', 'N/A')}\")
"
```

```promql
# 4. Verify active execution count has returned to normal
gl_orchestrator_active_executions

# 5. Verify node timeouts are not increasing
increase(gl_orchestrator_node_timeouts_total[15m])

# 6. Verify retry count is not abnormally high
increase(gl_orchestrator_node_retries_total[15m])
```

---

## Prevention

### Timeout Configuration Best Practices

Set appropriate timeouts per node based on the agent type and expected workload:

| Agent Type | Recommended Timeout | Rationale |
|------------|-------------------|-----------|
| Intake agents (data loading) | 60-120s | Depends on data volume; set higher for large files |
| Validation agents | 30-60s | Should be fast; increase only for complex validation rules |
| Calculation agents (Scope 1/2/3) | 120-300s | Computationally intensive; set based on dataset size |
| Aggregation agents | 60-120s | Depends on number of inputs to aggregate |
| Reporting agents (PDF/Excel) | 120-180s | Report generation can be I/O bound |
| External API agents (ERP, regulatory) | 180-300s | External APIs have unpredictable latency |

### Retry Configuration Best Practices

Configure retries to handle transient failures without causing cascading delays:

```yaml
# Recommended default retry policy
default_retry_policy:
  max_retries: 3
  strategy: exponential
  base_delay: 1.0
  max_delay: 30.0
  jitter: true
  retryable_exceptions:
    - ConnectionError
    - TimeoutError
    - ServiceUnavailable
```

### Monitoring and Alerting

- **Dashboard:** DAG Execution Overview (`/d/dag-execution-overview`) -- execution duration panels
- **Dashboard:** Node Performance (`/d/node-performance`) -- per-node latency breakdown
- **Key metrics to watch:**
  - `gl_orchestrator_dag_execution_duration_seconds` histogram (p50, p95, p99)
  - `gl_orchestrator_node_execution_duration_seconds` histogram by `node_id`
  - `gl_orchestrator_node_timeouts_total` rate (should be near 0)
  - `gl_orchestrator_node_retries_total` rate (low values are normal; spikes indicate problems)
  - `gl_orchestrator_active_executions` vs. max concurrent limit

### Capacity Planning

1. **Set `max_parallel_nodes` based on available resources:** Start with 10 and adjust based on load testing results
2. **Set `max_concurrent_executions` based on cluster capacity:** Start with 50 (NFR-004 target) and reduce if resource contention occurs
3. **Monitor execution duration trends weekly** to detect gradual performance degradation
4. **Size agent pods appropriately** for their workload; undersized agents are the most common cause of node timeouts

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any stuck execution incident
- **Related alerts:** `OrchestratorServiceDown`, `OrchestratorHighLatency`, `OrchestratorHighFailureRate`
- **Related dashboards:** DAG Execution Overview, Node Performance
- **Related runbooks:** [Orchestrator Service Down](./orchestrator-service-down.md), [Checkpoint Corruption](./checkpoint-corruption.md), [High Execution Latency](./high-execution-latency.md)
