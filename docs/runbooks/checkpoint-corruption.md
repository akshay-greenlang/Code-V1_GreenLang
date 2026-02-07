# Checkpoint Corruption

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `OrchestratorCheckpointFailure` | Critical | Checkpoint save or load operations failing for 5 minutes |
| `OrchestratorProvenanceChainBroken` | Critical | Provenance chain hash verification failed |

**Thresholds:**

```promql
# OrchestratorCheckpointFailure
increase(gl_orchestrator_checkpoint_operations_total{status="error"}[5m]) > 0

# OrchestratorProvenanceChainBroken
gl_orchestrator_provenance_chain_valid == 0
```

---

## Description

These alerts fire when checkpoint data integrity or provenance chain integrity has been compromised. This is a critical situation because:

1. **Checkpoint corruption** means DAG executions cannot reliably resume from saved state. If a node completes and its checkpoint is corrupted, the orchestrator cannot verify that the node completed successfully on resume, potentially causing the node to re-execute (wasting resources) or skip (producing incorrect results).

2. **Provenance chain breakage** means the cryptographic hash chain linking all node executions has been violated. The provenance chain is a SHA-256 hash chain where each node's provenance record includes the hashes of its predecessor node provenances. A broken chain means the execution audit trail cannot be verified, which is a **regulatory compliance risk** for CSRD, CBAM, and other reporting frameworks that require auditable calculation provenance.

### How Checkpointing Works

After each node completes, the orchestrator saves a checkpoint record containing:
- `node_id`: The completed node
- `status`: Node completion status (completed, failed, skipped)
- `outputs`: The node's output data (JSONB)
- `output_hash`: SHA-256 hash of the outputs for integrity verification
- `attempt_count`: Number of retry attempts

On resume, the orchestrator loads all checkpoints for the execution, verifies each checkpoint's `output_hash` against the stored outputs, and skips nodes that have valid checkpoints.

### How Provenance Chains Work

Each node produces a `NodeProvenance` record containing:
- `input_hash`: SHA-256 of the node's input data
- `output_hash`: SHA-256 of the node's output data
- `parent_hashes`: List of provenance hashes from predecessor nodes
- `chain_hash`: SHA-256 of `(node_id + input_hash + output_hash + sorted(parent_hashes))`

The execution's `provenance_chain_hash` is the SHA-256 of all node chain hashes in topological order. Any modification to any node's inputs, outputs, or execution order will produce a different chain hash, making tampering detectable.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Affected executions cannot be resumed; users may need to re-run workflows from scratch |
| **Data Impact** | Critical | Checkpoint data is unreliable; provenance chain is broken; execution integrity cannot be verified |
| **SLA Impact** | High | Regulatory submissions that depend on provenance verification will be blocked |
| **Revenue Impact** | Medium | Re-execution costs time and compute resources; customer deliverables delayed |
| **Compliance Impact** | Critical | Broken provenance chain means the calculation audit trail is unverifiable; SOC 2 CC7.2, CSRD Article 29a, and CBAM Article 35 require auditable provenance |
| **Audit Impact** | Critical | External auditors cannot verify the execution path; affected reports may need to be regenerated with intact provenance |

---

## Symptoms

### Checkpoint Failure Symptoms

- `OrchestratorCheckpointFailure` alert firing
- `gl_orchestrator_checkpoint_operations_total{status="error"}` counter incrementing
- Execution resume fails with "checkpoint integrity verification failed" error
- Orchestrator logs show "hash mismatch" or "checkpoint corrupted" messages
- Resume operations return fewer completed nodes than expected
- `gl_orchestrator_checkpoint_size_bytes` histogram shows abnormally small values (truncated writes)

### Provenance Chain Broken Symptoms

- `OrchestratorProvenanceChainBroken` alert firing
- Provenance API returns `chain_valid: false`
- Execution provenance endpoint shows missing node hashes or hash mismatches
- `gl_orchestrator_provenance_chain_length` is shorter than expected for the DAG topology
- Audit export returns validation errors
- Logs show "provenance chain hash mismatch" or "parent hash not found"

---

## Diagnostic Steps

### Step 1: Identify Affected Executions

```bash
# Port-forward to the orchestrator service
kubectl port-forward -n greenlang svc/orchestrator-service 8080:8080

# Find executions with checkpoint errors
curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=failed" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for exec in data.get('items', []):
    error = exec.get('error', '')
    if 'checkpoint' in error.lower() or 'provenance' in error.lower() or 'hash' in error.lower():
        print(f\"AFFECTED: {exec['execution_id']} | DAG: {exec['dag_id']} | Error: {error[:120]}\")
"
```

### Step 2: Verify Checkpoint Store Health

```bash
# Check PostgreSQL checkpoint table for corruption indicators
kubectl run pg-check --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    -- Check for checkpoints with NULL output_hash (should never happen)
    SELECT execution_id, node_id, status, created_at
    FROM dag_checkpoints
    WHERE output_hash IS NULL AND status = 'completed'
    ORDER BY created_at DESC LIMIT 20;
  "

# Check for checkpoints where output_hash does not match outputs
# (This requires application-level verification; check orchestrator logs)
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "hash mismatch\|integrity\|corrupted\|checkpoint.*error"
```

### Step 3: Verify Provenance Chain Integrity

```bash
# Get the provenance chain for a specific execution and verify it
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<execution_id>/provenance" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Execution: {data.get('execution_id')}\")
print(f\"Chain hash: {data.get('provenance_chain_hash', 'MISSING')}\")
print(f\"Chain valid: {data.get('chain_valid', 'UNKNOWN')}\")
print(f\"Nodes in chain: {len(data.get('nodes', []))}\")
print()
for node in data.get('nodes', []):
    valid = node.get('valid', 'unknown')
    marker = '  ** BROKEN **' if valid == False else ''
    print(f\"  {node['node_id']}: chain_hash={node.get('chain_hash', 'MISSING')[:16]}... valid={valid}{marker}\")
    if not valid:
        print(f\"    Expected parent hashes: {node.get('expected_parent_hashes', [])}\")
        print(f\"    Actual parent hashes:   {node.get('actual_parent_hashes', [])}\")
"
```

### Step 4: Check Storage Backend Health

```bash
# Check PostgreSQL disk usage
kubectl run pg-disk --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    SELECT pg_size_pretty(pg_database_size('greenlang')) as db_size;
    SELECT pg_size_pretty(pg_total_relation_size('dag_checkpoints')) as checkpoint_table_size;
    SELECT pg_size_pretty(pg_total_relation_size('execution_provenance')) as provenance_table_size;
  "

# Check for PostgreSQL errors or disk issues
kubectl logs -n database -l app=postgresql --tail=200 \
  | grep -i "error\|full\|disk\|corrupt\|wal"

# Check TimescaleDB chunk health for node_traces
kubectl run pg-chunks --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT * FROM timescaledb_information.chunks WHERE hypertable_name = 'node_traces' ORDER BY range_start DESC LIMIT 10;"
```

### Step 5: Check for Concurrent Write Conflicts

```bash
# Check for deadlocks or serialization failures in PostgreSQL
kubectl run pg-locks --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    -- Check for active locks on checkpoint tables
    SELECT pid, relation::regclass, mode, granted
    FROM pg_locks
    WHERE relation IN ('dag_checkpoints'::regclass, 'execution_provenance'::regclass)
    ORDER BY granted, pid;
  "

# Check orchestrator logs for serialization errors
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "serialization\|deadlock\|conflict\|concurrent\|could not serialize"
```

### Step 6: Check for Aggressive Checkpoint Cleanup

```bash
# Check the checkpoint retention configuration
kubectl get configmap orchestrator-service-config -n greenlang -o yaml \
  | grep -i "retention\|cleanup\|ttl\|checkpoint"

# Check if checkpoint cleanup jobs are running too frequently
kubectl logs -n greenlang -l app=orchestrator-service --tail=500 \
  | grep -i "cleanup\|retention\|purge\|delete.*checkpoint"

# Check the number of checkpoints per execution (should match completed nodes)
kubectl run pg-cpcount --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    SELECT execution_id, count(*) as checkpoint_count
    FROM dag_checkpoints
    GROUP BY execution_id
    ORDER BY checkpoint_count DESC
    LIMIT 20;
  "
```

---

## Resolution Steps

### Step 1: Stop Accepting New Executions Temporarily

If checkpoint corruption is widespread and ongoing, temporarily prevent new executions from starting to avoid further corruption.

```bash
# Scale down to prevent new executions while investigating
# (Existing running executions will continue on current replicas)
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_ACCEPT_NEW_EXECUTIONS=false

# Alternatively, if the corruption is severe, scale down entirely
# WARNING: This will interrupt running executions
kubectl scale deployment/orchestrator-service -n greenlang --replicas=0
```

### Step 2: Delete Corrupted Checkpoints

For executions with corrupted checkpoints, delete the corrupted records so they do not interfere with re-execution.

```bash
# Identify corrupted checkpoints for a specific execution
kubectl run pg-corrupt --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    -- Delete checkpoints for a specific execution that has corruption
    DELETE FROM dag_checkpoints
    WHERE execution_id = '<execution_id>';
  "

# Also clean up the provenance chain for the affected execution
kubectl run pg-prov --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    DELETE FROM execution_provenance
    WHERE execution_id = '<execution_id>';
  "
```

### Step 3: Re-Execute Affected DAGs from Scratch

Since the checkpoints and provenance are corrupted, the safest approach is to start fresh executions. This ensures a clean provenance chain from the beginning.

```bash
# Mark the old execution as failed
kubectl run pg-fail --rm -it --image=postgres:14 -n greenlang --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "
    UPDATE dag_executions
    SET status = 'failed', error = 'Checkpoint corruption detected - re-execution required'
    WHERE execution_id = '<execution_id>';
  "

# Start a new execution
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

### Step 4: Verify Provenance Chain of New Execution

After the new execution completes, verify that the provenance chain is intact.

```bash
# Verify the new execution's provenance chain
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<new_execution_id>/provenance" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Execution: {data.get('execution_id')}\")
print(f\"Chain hash: {data.get('provenance_chain_hash')}\")
print(f\"Chain valid: {data.get('chain_valid')}\")
print(f\"Total nodes: {len(data.get('nodes', []))}\")
all_valid = all(n.get('valid', False) for n in data.get('nodes', []))
print(f\"All nodes valid: {all_valid}\")
"
```

### Step 5: Fix the Root Cause

Based on the diagnosis, apply the appropriate fix:

**If caused by concurrent write conflicts:**

```bash
# Increase PostgreSQL transaction isolation or add retry logic
# The orchestrator should use SERIALIZABLE isolation for checkpoint writes
# Check the configuration:
kubectl get configmap orchestrator-service-config -n greenlang -o yaml \
  | grep -i "isolation\|transaction"
```

**If caused by storage backend issues (disk full):**

```bash
# Check and expand PostgreSQL storage
kubectl get pvc -n database
kubectl describe pvc -n database <postgres-pvc>

# If PVC is full, expand it (if storage class supports it)
kubectl patch pvc <postgres-pvc> -n database -p '{"spec":{"resources":{"requests":{"storage":"100Gi"}}}}'
```

**If caused by aggressive cleanup:**

```bash
# Increase checkpoint retention period
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_CHECKPOINT_RETENTION_HOURS=168
```

**If caused by a code bug in hash calculation:**

```bash
# Rollback to a known-good version
kubectl rollout undo deployment/orchestrator-service -n greenlang
kubectl rollout status deployment/orchestrator-service -n greenlang
```

### Step 6: Re-Enable New Executions

```bash
# Re-enable accepting new executions
kubectl set env deployment/orchestrator-service -n greenlang \
  GL_ORCHESTRATOR_ACCEPT_NEW_EXECUTIONS=true

# Or scale back up if you scaled down
kubectl scale deployment/orchestrator-service -n greenlang --replicas=2

# Verify service health
kubectl get pods -n greenlang -l app=orchestrator-service
curl -s http://localhost:8080/api/v1/orchestrator/health | python3 -m json.tool
```

---

## Post-Incident Verification

```bash
# 1. Verify checkpoint operations are succeeding
kubectl logs -n greenlang -l app=orchestrator-service --tail=100 \
  | grep -i "checkpoint.*saved\|checkpoint.*loaded"
```

```promql
# 2. Verify checkpoint error rate has dropped to zero
increase(gl_orchestrator_checkpoint_operations_total{status="error"}[5m]) == 0

# 3. Verify checkpoint saves are succeeding
increase(gl_orchestrator_checkpoint_operations_total{operation="save", status="success"}[5m]) > 0

# 4. Verify provenance chains are valid for new executions
gl_orchestrator_provenance_chain_valid == 1
```

```bash
# 5. Run a test execution and verify end-to-end provenance
curl -X POST "http://localhost:8080/api/v1/orchestrator/dags/<test_dag_id>/execute" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"test": true}, "execution_options": {"checkpoint_enabled": true, "deterministic_mode": true}}' \
  | python3 -m json.tool

# Wait for completion, then verify provenance
curl -s "http://localhost:8080/api/v1/orchestrator/executions/<test_execution_id>/provenance" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Chain valid: {data.get(\"chain_valid\")}')
assert data.get('chain_valid') == True, 'PROVENANCE CHAIN STILL BROKEN'
print('Provenance chain verification PASSED')
"
```

---

## Prevention

### Storage Backend Recommendations

| Environment | Checkpoint Store | Provenance Store | Rationale |
|-------------|-----------------|------------------|-----------|
| Development | In-memory | In-memory | Fast iteration, no persistence needed |
| Staging | File-based | PostgreSQL | Test file I/O behavior; provenance always in DB |
| Production | PostgreSQL | PostgreSQL | ACID guarantees prevent corruption; WAL provides crash recovery |

**Important:** Never use in-memory or file-based checkpoint stores in production. PostgreSQL with ACID transactions is the only store that provides durability guarantees against corruption from concurrent writes, process crashes, and storage failures.

### Integrity Verification Best Practices

1. **Enable checkpoint hash verification on every resume** (default behavior; do not disable)
2. **Run periodic provenance chain integrity checks** as a scheduled job:

```bash
# Example CronJob query to verify all recent provenance chains
curl -s "http://localhost:8080/api/v1/orchestrator/executions?status=completed&since=24h" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | python3 -c "
import sys, json, requests
data = json.load(sys.stdin)
broken = []
for exec in data.get('items', []):
    prov = requests.get(
        f'http://localhost:8080/api/v1/orchestrator/executions/{exec[\"execution_id\"]}/provenance',
        headers={'Authorization': f'Bearer {sys.argv[1]}'}
    ).json()
    if not prov.get('chain_valid'):
        broken.append(exec['execution_id'])
if broken:
    print(f'BROKEN CHAINS: {broken}')
else:
    print(f'All {len(data.get(\"items\", []))} execution provenance chains are valid')
"
```

3. **Set appropriate checkpoint retention periods** -- at least 7 days for production, longer for regulatory compliance requirements
4. **Monitor checkpoint table size** and configure TimescaleDB compression for older node_traces

### Monitoring

- **Dashboard:** Orchestrator Service Health (`/d/orchestrator-health`) -- checkpoint and provenance panels
- **Alert:** `OrchestratorCheckpointFailure` (this alert)
- **Alert:** `OrchestratorProvenanceChainBroken` (this alert)
- **Key metrics to watch:**
  - `gl_orchestrator_checkpoint_operations_total{status="error"}` rate (should be 0)
  - `gl_orchestrator_checkpoint_operations_total{status="success"}` rate (should be non-zero during executions)
  - `gl_orchestrator_checkpoint_size_bytes` histogram (sudden changes indicate problems)
  - `gl_orchestrator_provenance_chain_length` histogram (should match DAG node counts)
  - PostgreSQL disk usage and WAL lag

### Regulatory Compliance Notes

- **CSRD Article 29a** requires companies to disclose the methodology used for sustainability calculations. The provenance chain provides cryptographic proof of the exact calculation path.
- **CBAM Article 35** requires verifiable emissions calculations for carbon border adjustment. A broken provenance chain means the calculation cannot be independently verified.
- **SOC 2 CC7.2** requires monitoring controls to be in place and functioning. Checkpoint and provenance verification failures must be documented in the audit trail.

If a provenance chain is broken for an execution that was used in a regulatory submission, the affected report must be regenerated with a clean execution that has an intact provenance chain. Notify the compliance team immediately.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any checkpoint/provenance incident
- **Related alerts:** `OrchestratorServiceDown`, `OrchestratorExecutionTimeout`
- **Related dashboards:** Orchestrator Service Health, DAG Execution Overview
- **Related runbooks:** [Orchestrator Service Down](./orchestrator-service-down.md), [DAG Execution Stuck](./dag-execution-stuck.md), [High Execution Latency](./high-execution-latency.md)
