# Orchestrator Service Down

## Alert

**Alert Name:** `OrchestratorServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="orchestrator-service"} == 1) or sum(up{job="orchestrator-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Orchestrator Service (AGENT-FOUND-001) are running. The Orchestrator Service is the DAG execution engine responsible for:

1. **Defining and validating DAG workflows** with explicit dependency edges, cycle detection, and structural validation
2. **Scheduling and executing DAG nodes** using topological sort with deterministic tie-breaking and level-based parallel execution
3. **Applying per-node retry and timeout policies** with configurable exponential backoff, jitter, and timeout enforcement
4. **Managing DAG-aware checkpoints** that enable node-level checkpoint/resume without re-executing completed nodes
5. **Tracking execution provenance** with SHA-256 hash chains linking all node inputs, outputs, and timing for regulatory audit
6. **Enforcing deterministic execution** with sorted scheduling, deterministic clocks, and content hashing for execution replay
7. **Exposing 20 REST API endpoints** for DAG CRUD, execution management, checkpoint management, provenance queries, and metrics
8. **Emitting 12 Prometheus metrics** for DAG execution monitoring, node-level performance tracking, and checkpoint operations

When the Orchestrator Service is down, all new DAG executions will fail to start, running executions may be interrupted at the current node boundary, checkpoint-resume operations will be unavailable, provenance chain verification cannot be performed, and the REST API will return errors. Every GreenLang application -- CSRD, CBAM, VCCI, SB253, EUDR, Taxonomy -- depends on the orchestrator for agent workflow coordination. A prolonged outage directly blocks all emissions calculations, compliance report generation, and regulatory submission pipelines.

**Note:** Executions that were checkpointed before the outage can be resumed once the service recovers. No completed node work is lost because checkpoint data persists in PostgreSQL independently of the service.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All DAG-based workflows (emissions calculations, report generation) are blocked |
| **Data Impact** | High | Running executions are interrupted; in-flight node results may not be checkpointed |
| **SLA Impact** | Critical | Regulatory submission deadlines may be missed if outage extends beyond checkpoint recovery window |
| **Revenue Impact** | High | Customer-facing compliance workflows are unavailable |
| **Compliance Impact** | Critical | CSRD, CBAM, and other regulatory reporting pipelines cannot execute; provenance chain verification unavailable |
| **Downstream Impact** | High | All GreenLang applications depend on the orchestrator; cascading failures across CSRD, CBAM, VCCI, SB253, EUDR, Taxonomy |

---

## Symptoms

- `up{job="orchestrator-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=orchestrator-service`
- `gl_orchestrator_active_executions` gauge drops to 0 unexpectedly
- `gl_orchestrator_dag_executions_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /api/v1/orchestrator/health` is unreachable
- Downstream applications report "orchestrator unavailable" errors in logs
- `gl_orchestrator_checkpoint_operations_total` counter stops incrementing
- Grafana Orchestrator dashboard shows "No Data" or stale timestamps

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List orchestrator service pods
kubectl get pods -n greenlang -l app=orchestrator-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=orchestrator-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to orchestrator
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=orchestrator-service | tail -30

# Check deployment status
kubectl describe deployment orchestrator-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment orchestrator-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=orchestrator-service

# Check for rollout issues
kubectl rollout status deployment/orchestrator-service -n greenlang

# Check HPA status (min 2, max 6)
kubectl get hpa -n greenlang -l app=orchestrator-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=orchestrator-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for DAG execution errors
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "dag\|execution\|checkpoint\|provenance\|topological"

# Look for database and Redis connection errors
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "database\|postgres\|redis\|connection\|pool"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of orchestrator pods
kubectl top pods -n greenlang -l app=orchestrator-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=orchestrator-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang
```

### Step 5: Check Database Connectivity (PostgreSQL)

The orchestrator stores DAG definitions, execution records, node traces, checkpoints, and provenance chains in PostgreSQL with TimescaleDB extensions.

```bash
# Verify PostgreSQL connectivity from within the namespace
kubectl run pg-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-postgresql.database.svc.cluster.local 5432'

# Check that the orchestrator tables exist (V021 migration)
kubectl run pg-check --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT table_name FROM information_schema.tables WHERE table_name IN ('dag_workflows', 'dag_executions', 'node_traces', 'dag_checkpoints', 'execution_provenance');"

# Check active database connections from the orchestrator
kubectl run pg-conns --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT count(*) as connections, state FROM pg_stat_activity WHERE application_name LIKE '%orchestrator%' GROUP BY state;"

# Check for connection pool exhaustion
kubectl run pg-pool --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT count(*) FROM pg_stat_activity;"
```

### Step 6: Check Redis Connectivity

The orchestrator uses Redis for distributed locking (concurrent execution limits) and caching frequently accessed DAG definitions.

```bash
# Verify Redis connectivity
kubectl run redis-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-redis.redis.svc.cluster.local 6379'

# Check Redis key count for orchestrator cache
kubectl run redis-check --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:orchestrator:*' | head -20

# Check Redis memory usage
kubectl run redis-mem --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO memory | head -10
```

### Step 7: Check ConfigMap and Secrets

```bash
# Verify the orchestrator ConfigMap exists and is valid
kubectl get configmap orchestrator-service-config -n greenlang
kubectl get configmap orchestrator-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (database URL, Redis URL, JWT signing key)
kubectl get secret orchestrator-service-secrets -n greenlang

# Check ESO sync status
kubectl get externalsecrets -n greenlang | grep orchestrator
```

### Step 8: Check Init Containers and Migration Status

The orchestrator pod may have an init container that verifies database migration V021 has been applied before starting.

```bash
# Check if init container is stuck
kubectl describe pod -n greenlang <pod-name> | grep -A20 "Init Containers"

# Check init container logs
kubectl logs -n greenlang <pod-name> -c init-db-check

# Verify migration status
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  info -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang \
  | grep V021
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137, `restartCount` is incrementing.

**Cause:** The orchestrator is consuming more memory than its configured limit, typically due to too many concurrent DAG executions, large DAG definitions loaded into memory, or a memory leak.

**Resolution:**

1. **Confirm the OOM cause:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. **Check the current memory limit and usage pattern:**

```bash
# Current memory limits
kubectl get deployment orchestrator-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check Prometheus for memory trend before OOM
# PromQL: container_memory_working_set_bytes{namespace="greenlang", pod=~"orchestrator.*"}
```

3. **Immediate mitigation -- increase memory limits:**

```bash
kubectl patch deployment orchestrator-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "orchestrator-service",
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
```

4. **If caused by excessive concurrent executions, reduce the concurrency limit:**

```bash
# Update the GL_ORCHESTRATOR_MAX_CONCURRENT_EXECUTIONS environment variable
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_MAX_CONCURRENT_EXECUTIONS=25
```

5. **Verify pods restart successfully:**

```bash
kubectl rollout status deployment/orchestrator-service -n greenlang
kubectl get pods -n greenlang -l app=orchestrator-service
```

### Scenario 2: CrashLoopBackOff (Application Error)

**Symptoms:** Pod status shows CrashLoopBackOff, container exits with non-zero code (not 137).

**Cause:** Application startup failure due to configuration error, missing secrets, database migration not applied, or code bug.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang <pod-name> --previous --tail=100
```

2. **If caused by missing database tables (V021 migration):**

```bash
# Check if V021 migration has been applied
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  info -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang

# Apply pending migrations if V021 is missing
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  migrate -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang
```

3. **If caused by missing secrets:**

```bash
# Verify secrets exist
kubectl get secret orchestrator-service-secrets -n greenlang

# Check ESO sync status
kubectl describe externalsecret orchestrator-service-secrets -n greenlang

# If secrets are missing, check SSM parameters
aws ssm get-parameters-by-path --path "/gl/prod/orchestrator-service/" --query "Parameters[*].Name"
```

4. **If caused by invalid configuration:**

```bash
# Check ConfigMap values
kubectl get configmap orchestrator-service-config -n greenlang -o yaml

# Look for config parsing errors in logs
kubectl logs -n greenlang <pod-name> --previous | grep -i "config\|env\|validation\|parse"
```

5. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/orchestrator-service -n greenlang
kubectl rollout status deployment/orchestrator-service -n greenlang
```

### Scenario 3: ImagePullBackOff

**Symptoms:** Pod status shows ImagePullBackOff or ErrImagePull.

**Cause:** Container image not found, registry authentication failure, or tag mismatch.

**Resolution:**

1. **Check the image and pull errors:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A5 "Events"
```

2. **Verify image exists in registry:**

```bash
# Check current image tag
kubectl get deployment orchestrator-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/orchestrator-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/orchestrator-service \
  orchestrator-service=greenlang/orchestrator-service:<correct-tag> -n greenlang
```

### Scenario 4: Liveness Probe Failure

**Symptoms:** Pod is running but not Ready, restarting due to liveness probe failure, health endpoint returning errors.

**Cause:** The `/api/v1/orchestrator/health` endpoint is not responding within the probe timeout. This can be caused by deadlocks, stuck async event loops, database connection pool exhaustion, or high CPU contention.

**Resolution:**

1. **Check liveness probe configuration and failure events:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Liveness"
kubectl get events -n greenlang --field-selector reason=Unhealthy --sort-by='.lastTimestamp'
```

2. **Test the health endpoint manually:**

```bash
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080
curl -s http://localhost:8080/api/v1/orchestrator/health | python3 -m json.tool
```

3. **Check for deadlocks in the async event loop:**

```bash
# Look for asyncio-related errors or warnings
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "deadlock\|asyncio\|event loop\|timeout\|blocked\|semaphore"
```

4. **Check database connection pool status:**

```bash
# High active connections with pending waiters indicates pool exhaustion
kubectl run pg-pool --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT count(*), state FROM pg_stat_activity WHERE application_name LIKE '%orchestrator%' GROUP BY state;"
```

5. **If pool is exhausted, increase pool size and restart:**

```bash
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_DB_POOL_SIZE=20 GL_ORCHESTRATOR_DB_POOL_MAX_OVERFLOW=10

kubectl rollout restart deployment/orchestrator-service -n greenlang
```

### Scenario 5: Database Connectivity Failure

**Symptoms:** Logs show "connection refused" to PostgreSQL, startup fails with "relation dag_workflows does not exist", or checkpoint operations fail.

**Cause:** PostgreSQL is down, network policy blocking traffic, connection string incorrect, or V021 migration has not been applied.

**Resolution:**

1. **Verify PostgreSQL is running:**

```bash
kubectl get pods -n database -l app=postgresql
kubectl get svc -n database | grep postgresql
```

2. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy orchestrator-service-egress -n greenlang
```

3. **Test connectivity from a debug pod:**

```bash
kubectl run debug --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" -c "SELECT 1"
```

4. **Check if orchestrator tables exist:**

```bash
kubectl run debug --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'dag_%' OR table_name IN ('node_traces', 'execution_provenance');"
```

If tables are missing, apply the V021 migration:

```bash
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  migrate -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang
```

5. **After PostgreSQL is restored or migration is applied, restart the orchestrator:**

```bash
kubectl rollout restart deployment/orchestrator-service -n greenlang
kubectl rollout status deployment/orchestrator-service -n greenlang
```

### Scenario 6: Redis Connectivity Failure

**Symptoms:** Logs show "connection refused" to Redis, distributed lock acquisition failing, execution concurrency limits not enforced.

**Cause:** Redis cluster is down or network policy is blocking traffic.

**Resolution:**

1. **Verify Redis is running:**

```bash
kubectl get pods -n redis -l app=redis
kubectl get svc -n redis | grep redis
```

2. **Test connectivity:**

```bash
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING
```

3. **Note:** Redis is used for distributed locking and DAG definition caching. The orchestrator should degrade gracefully if Redis is unavailable, falling back to in-process locking (single-replica only) and direct database reads. If the service is crashing instead of degrading, this indicates a bug. Restart and monitor:

```bash
kubectl rollout restart deployment/orchestrator-service -n greenlang
```

### Scenario 7: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns, no infrastructure issues found.

**Cause:** A code regression introduced in the latest deployment.

**Resolution:**

1. **Check recent deployment history:**

```bash
kubectl rollout history deployment/orchestrator-service -n greenlang
```

2. **Rollback to the previous version:**

```bash
kubectl rollout undo deployment/orchestrator-service -n greenlang
kubectl rollout status deployment/orchestrator-service -n greenlang
```

3. **Verify the rollback resolved the issue:**

```bash
kubectl get pods -n greenlang -l app=orchestrator-service
curl -s http://localhost:8080/api/v1/orchestrator/health | python3 -m json.tool
```

4. **Create a bug report** with the logs from the failed deployment for the development team.

---

## Post-Incident Steps

After the orchestrator service is restored, perform the following verification and cleanup steps.

### Step 1: Verify Service Health

```bash
# 1. Check all pods are running and ready
kubectl get pods -n greenlang -l app=orchestrator-service

# 2. Check the health endpoint
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080
curl -s http://localhost:8080/api/v1/orchestrator/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# 3. Verify the orchestrator is being scraped
up{job="orchestrator-service"} == 1

# 4. Verify execution metrics are incrementing (if new executions are starting)
increase(gl_orchestrator_dag_executions_total[5m])

# 5. Verify checkpoint operations are working
increase(gl_orchestrator_checkpoint_operations_total[5m])
```

### Step 3: Check for Interrupted Executions

```bash
# List executions that were running when the service went down
curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=running" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for exec in data.get('items', []):
    print(f\"Execution: {exec['execution_id']} | DAG: {exec['dag_id']} | Started: {exec['started_at']}\")
"
```

### Step 4: Resume Interrupted Executions from Checkpoints

```bash
# For each interrupted execution, attempt checkpoint resume
curl -X POST "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/resume" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" | python3 -m json.tool
```

### Step 5: Check for Orphaned Executions

Executions that were in the "running" state but whose nodes are no longer being executed are considered orphaned. They need to be either resumed or cancelled.

```bash
# Find executions stuck in running state for more than 1 hour
curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=running" | python3 -c "
import sys, json
from datetime import datetime, timezone, timedelta
data = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for exec in data.get('items', []):
    started = datetime.fromisoformat(exec['started_at'].replace('Z', '+00:00'))
    age = now - started
    if age > timedelta(hours=1):
        print(f\"ORPHANED: {exec['execution_id']} | DAG: {exec['dag_id']} | Age: {age}\")
"

# Cancel orphaned executions that cannot be resumed
curl -X POST "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/cancel" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" | python3 -m json.tool
```

### Step 6: Verify Provenance Chain Integrity

After resuming executions, verify that the provenance chain is intact for any execution that was interrupted and resumed.

```bash
# Get provenance chain for a resumed execution
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/provenance" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Chain hash: {data.get('provenance_chain_hash', 'N/A')}\")
print(f\"Nodes in chain: {len(data.get('nodes', []))}\")
print(f\"Chain valid: {data.get('chain_valid', 'N/A')}\")
"
```

### Step 7: Review Metrics for Root Cause

```promql
# Check for memory spikes before the outage
container_memory_working_set_bytes{namespace="greenlang", pod=~"orchestrator.*"}

# Check for CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"orchestrator.*"}[5m]))

# Check active execution count trend (were there too many concurrent executions?)
gl_orchestrator_active_executions

# Check checkpoint operation failures
increase(gl_orchestrator_checkpoint_operations_total{operation="save", status="error"}[1h])

# Check database connection pool metrics
gl_db_pool_active_connections{service="orchestrator-service"}
gl_db_pool_pending_connections{service="orchestrator-service"}
```

---

## Interim Mitigation

While the Orchestrator Service is being restored:

1. **Checkpoint data is safe.** All checkpoints are stored in PostgreSQL. No completed node work is lost. Executions can be resumed once the service recovers.

2. **No new executions can start.** Applications that depend on the orchestrator will receive errors. Downstream applications should implement retry logic and queue pending execution requests.

3. **Existing Prometheus alert rules continue firing.** Orchestrator-related alerts that were previously configured in Prometheus continue to evaluate. However, the `gl_orchestrator_*` metrics will become stale.

4. **Manual execution is not possible.** The orchestrator is the only execution path for DAG workflows. There is no manual fallback for DAG execution.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#compliance-ops` -- regulatory submission impact
   - `#platform-oncall` -- engineering response

6. **Monitor the outage duration.** If the orchestrator is down for more than 15 minutes, escalate per the escalation path below. Regulatory submission deadlines must be tracked manually during the outage.

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Orchestrator down, no running executions impacted | On-call engineer | Immediate (<5 min) |
| L2 | Orchestrator down > 15 minutes, running executions interrupted | Platform team lead + #orchestrator-oncall | 15 minutes |
| L3 | Orchestrator down > 30 minutes, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Orchestrator down due to database/infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Orchestrator Service Health (`/d/orchestrator-health`)
- **Dashboard:** DAG Execution Overview (`/d/dag-execution-overview`)
- **Alert:** `OrchestratorServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="orchestrator-service"}` (should always be >= 2)
  - `gl_orchestrator_active_executions` (normal baseline varies; sudden drop to 0 is suspicious)
  - `gl_orchestrator_dag_executions_total` rate (should be non-zero during business hours)
  - `gl_orchestrator_checkpoint_operations_total{status="error"}` rate (should be 0)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 6** based on active execution count and CPU utilization
4. **Resource requests** are sized for 50+ concurrent DAG executions (NFR-004)
5. **Database connection pool** is sized per replica (pool_size=10, max_overflow=5 per pod)

### Configuration Best Practices

- Set `GL_ORCHESTRATOR_MAX_CONCURRENT_EXECUTIONS` conservatively; start at 50 and increase based on load testing
- Set `GL_ORCHESTRATOR_DB_POOL_SIZE` to at least 10 per replica
- Use ESO for secrets rotation (database URL, Redis URL, JWT signing key)
- Test configuration changes in staging before production
- Apply database migrations (V021) in a separate maintenance window before deploying the orchestrator

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `OrchestratorServiceDown` | Critical | This alert -- no orchestrator pods running |
| `OrchestratorHighFailureRate` | Warning | >10% of DAG executions failing over 15 minutes |
| `OrchestratorHighLatency` | Warning | p99 node execution >30s over 10 minutes |
| `OrchestratorExecutionTimeout` | Warning | Execution exceeds configured timeout |
| `OrchestratorCheckpointFailure` | Critical | Checkpoint save/load operations failing |
| `OrchestratorProvenanceChainBroken` | Critical | Provenance hash chain verification failed |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any P1 orchestrator incident
- **Related alerts:** `OrchestratorHighFailureRate`, `OrchestratorHighLatency`, `OrchestratorExecutionTimeout`
- **Related dashboards:** Orchestrator Service Health, DAG Execution Overview
- **Related runbooks:** [DAG Execution Stuck](./dag-execution-stuck.md), [Checkpoint Corruption](./checkpoint-corruption.md), [High Execution Latency](./high-execution-latency.md)
