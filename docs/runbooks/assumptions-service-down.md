# Assumptions Registry Service Down

## Alert

**Alert Name:** `AssumptionsServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="assumptions-service"} == 1) or sum(up{job="assumptions-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Assumptions Registry service (AGENT-FOUND-004) are running. The Assumptions Registry Service is the version-controlled, audit-ready registry for managing all assumptions used in zero-hallucination compliance calculations. It is responsible for:

1. **Assumption CRUD with full version history** -- Every assumption (emission factors, conversion rates, thresholds, regulatory parameters) is explicitly defined, versioned, and never inferred or hallucinated
2. **Scenario management** -- Maintaining baseline, optimistic, conservative, and custom scenarios with per-assumption override values for what-if analysis
3. **Validation engine** -- Enforcing constraints (min/max ranges, allowed values, regex patterns, custom validators) on every assumption value before it is persisted
4. **Dependency graph tracking** -- Mapping which calculations depend on which assumptions, enabling impact analysis when an assumption changes
5. **SHA-256 provenance hash chain** -- Every change to every assumption produces a cryptographic hash linking the new version to the previous, creating a tamper-evident audit trail
6. **Sensitivity analysis** -- Computing how changes in assumption values propagate through dependent calculations to quantify uncertainty
7. **Export/import with integrity verification** -- Producing auditor-ready packages with hash-verified assumption snapshots for regulatory submission
8. **Emitting Prometheus metrics** (12 metrics under the `gl_assumptions_*` prefix) for monitoring assumption operations, validation failures, scenario usage, audit completeness, and service health

When the Assumptions Registry is down, all upstream services that depend on assumption values will be unable to retrieve the parameters they need for compliance calculations. This includes every GreenLang application pipeline -- CSRD, CBAM, VCCI, SB253, EUDR, and Taxonomy. Calculation agents that query assumption values at runtime will receive errors or timeouts. Downstream compliance reports cannot be generated because the zero-hallucination guarantee requires every value to be traced back to an explicitly defined assumption.

**Note:** Assumption data and version history are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full assumption registry, version history, provenance chain, and scenario definitions will be immediately available from the database. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All compliance calculations blocked; users cannot create, update, or query assumptions |
| **Data Impact** | High | No new assumption values can be persisted; calculations using stale cached values may proceed temporarily but will lack audit trail linkage |
| **SLA Impact** | Critical | Assumption query latency SLA violated (P95 targets: <15ms single lookup, <100ms batch, <200ms sensitivity analysis) |
| **Revenue Impact** | High | Customer-facing compliance workflows cannot produce zero-hallucination calculations or auditable reports |
| **Compliance Impact** | Critical | Zero-hallucination guarantee broken -- calculations cannot verify their assumption sources; regulatory submission deadlines at risk; audit trail generation impossible |
| **Downstream Impact** | Critical | All 6 GreenLang application pipelines (CSRD, CBAM, VCCI, SB253, EUDR, Taxonomy) depend on the assumptions registry; cascading failures across calculation, reporting, and compliance agents |

---

## Symptoms

- `up{job="assumptions-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=assumptions-service`
- `gl_assumptions_operations_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Upstream calculation agents report "assumptions unavailable" or "assumption lookup timeout" errors in logs
- `gl_assumptions_active_count` gauge drops to 0 (cache lost on restart)
- Grafana Assumptions Registry dashboard shows "No Data" or stale timestamps
- Scenario query endpoint `GET /v1/assumptions/scenarios` returns errors
- Compliance export endpoint `POST /v1/assumptions/export` returns errors
- Downstream agents log "unable to resolve assumption" or "assumptions service connection refused"
- Provenance hash chain verification endpoint returns connection errors

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List assumptions service pods
kubectl get pods -n greenlang -l app=assumptions-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=assumptions-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to assumptions service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=assumptions-service | tail -30

# Check deployment status
kubectl describe deployment assumptions-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment assumptions-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=assumptions-service

# Check for rollout issues
kubectl rollout status deployment/assumptions-service -n greenlang

# Check HPA status (scales 2-6 replicas)
kubectl get hpa -n greenlang -l app=assumptions-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=assumptions-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for assumptions-specific errors
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "assumption\|scenario\|validation\|provenance\|hash\|dependency\|registry"

# Look for database connection errors
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of assumptions service pods
kubectl top pods -n greenlang -l app=assumptions-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=assumptions-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang
```

### Step 5: Check Database Connectivity

The assumptions service uses PostgreSQL with TimescaleDB for assumption storage, version history, provenance records, and change logs.

```bash
# Verify PostgreSQL connectivity from the pod network
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check database connection pool status in logs
kubectl logs -n greenlang -l app=assumptions-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres\|timescale"

# Check if the assumptions database tables exist
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='public' AND table_name LIKE 'assumptions_%'
   ORDER BY table_name;"

# Verify core tables have data
kubectl run pg-counts --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT 'assumptions' as table_name, count(*) FROM assumptions_registry
   UNION ALL SELECT 'versions', count(*) FROM assumptions_version_history
   UNION ALL SELECT 'scenarios', count(*) FROM assumptions_scenarios
   UNION ALL SELECT 'change_log', count(*) FROM assumptions_change_log
   UNION ALL SELECT 'dependencies', count(*) FROM assumptions_dependency_graph
   UNION ALL SELECT 'validation_rules', count(*) FROM assumptions_validation_rules;"
```

### Step 6: Check Init Container (Migration Runner)

The assumptions service runs database migrations as an init container before the main application starts.

```bash
# Check init container status
kubectl get pods -n greenlang -l app=assumptions-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.initContainerStatuses[*]}{.name}={.ready}{" "}{end}{"\n"}{end}'

# Check init container logs (migration runner)
kubectl logs -n greenlang <pod-name> -c migration-runner

# Check if migrations completed successfully
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, installed_on FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

### Step 7: Check ConfigMap and Secrets

```bash
# Verify the assumptions service ConfigMap exists and is valid
kubectl get configmap assumptions-service-config -n greenlang
kubectl get configmap assumptions-service-config -n greenlang -o yaml | head -50

# Verify secrets exist (DB credentials, signing keys)
kubectl get secret assumptions-service-secrets -n greenlang

# Check ESO sync status
kubectl get externalsecrets -n greenlang | grep assumptions

# Check environment variables are set correctly
kubectl get deployment assumptions-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

Required environment variables:
- `GL_ASSUMPTIONS_DATABASE_URL` -- PostgreSQL connection string (with TimescaleDB)
- `GL_ASSUMPTIONS_HASH_ALGORITHM` -- Provenance hash algorithm (SHA-256)
- `GL_ASSUMPTIONS_VALIDATION_MODE` -- Validation strictness (strict or lenient)
- `GL_ASSUMPTIONS_MAX_VERSIONS` -- Maximum version history depth per assumption
- `GL_ASSUMPTIONS_SCENARIO_DEFAULT` -- Default scenario name (baseline)
- `GL_ASSUMPTIONS_EXPORT_SIGNING_KEY` -- Key for signing audit exports
- `GL_ASSUMPTIONS_AUDIT_RETENTION_DAYS` -- Audit log retention period

### Step 8: Check Network Policies

```bash
# Check network policies affecting the assumptions service
kubectl get networkpolicy -n greenlang | grep assumptions

# Verify the assumptions service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the assumptions service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv assumptions-service.greenlang.svc.cluster.local 8080'
```

### Step 9: Check Health Endpoint Directly

```bash
# Port-forward to the assumptions service (if at least one pod exists)
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080

# Test the health endpoint
curl -s http://localhost:8080/health | python3 -m json.tool

# Test the readiness endpoint
curl -s http://localhost:8080/ready | python3 -m json.tool

# Test the metrics endpoint
curl -s http://localhost:8080/metrics | python3 -m json.tool
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137, `restartCount` is incrementing.

**Cause:** The assumptions service is consuming more memory than its configured limit. This typically occurs when loading large dependency graphs into memory for impact analysis, processing bulk assumption imports, or running sensitivity analysis across many assumptions and scenarios simultaneously.

**Resolution:**

1. **Confirm the OOM cause:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. **Check the current memory limit and usage pattern:**

```bash
# Current memory limits
kubectl get deployment assumptions-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].resources}'

# Check Prometheus for memory trend before OOM
# PromQL: container_memory_working_set_bytes{namespace="greenlang", pod=~"assumptions-service.*"}
```

3. **Immediate mitigation -- increase memory limits:**

```bash
kubectl patch deployment assumptions-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "assumptions-service",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "2Gi"
              },
              "requests": {
                "cpu": "500m",
                "memory": "1Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

4. **If caused by large dependency graph traversal, limit graph depth:**

```bash
kubectl set env deployment/assumptions-service -n greenlang \
  GL_ASSUMPTIONS_MAX_DEPENDENCY_DEPTH=10 \
  GL_ASSUMPTIONS_SENSITIVITY_BATCH_SIZE=100
```

5. **Verify pods restart successfully:**

```bash
kubectl rollout status deployment/assumptions-service -n greenlang
kubectl get pods -n greenlang -l app=assumptions-service
```

### Scenario 2: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show "migration failed" or "schema version mismatch".

**Cause:** The init container running Flyway database migrations cannot complete. This blocks the main application from starting because the assumptions service requires the correct schema to function.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang <pod-name> -c migration-runner --tail=100
```

2. **Verify database connectivity and schema state:**

```bash
# Check if PostgreSQL is reachable
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check current migration version
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. **If a migration failed mid-way, repair the schema history:**

```bash
# Identify the failed migration
kubectl run pg-failed --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   WHERE success = false;"

# Repair the Flyway schema history (mark failed migration for retry)
kubectl run flyway-repair --rm -it \
  --image=greenlang/flyway-migrations:latest -n greenlang --restart=Never -- \
  flyway repair
```

4. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/assumptions-service -n greenlang
kubectl rollout status deployment/assumptions-service -n greenlang
```

### Scenario 3: CrashLoopBackOff -- Configuration Error

**Symptoms:** Pod status shows CrashLoopBackOff, logs show "configuration error", "missing required environment variable", or "invalid hash algorithm".

**Cause:** Application startup failure due to missing or invalid configuration -- typically missing database URL, invalid hash algorithm, or misconfigured validation mode.

**Resolution:**

1. **Check configuration-related errors:**

```bash
kubectl logs -n greenlang <pod-name> --previous | grep -i "config\|env\|validation\|hash\|parse\|missing"
```

2. **Verify ConfigMap values:**

```bash
kubectl get configmap assumptions-service-config -n greenlang -o yaml
```

3. **Fix the configuration and restart:**

```bash
kubectl set env deployment/assumptions-service -n greenlang \
  GL_ASSUMPTIONS_HASH_ALGORITHM=SHA-256 \
  GL_ASSUMPTIONS_VALIDATION_MODE=strict \
  GL_ASSUMPTIONS_SCENARIO_DEFAULT=baseline \
  GL_ASSUMPTIONS_MAX_VERSIONS=1000 \
  GL_ASSUMPTIONS_AUDIT_RETENTION_DAYS=2555

kubectl rollout restart deployment/assumptions-service -n greenlang
```

### Scenario 4: Database Connection Pool Exhaustion

**Symptoms:** Service was running but health check fails, logs show "connection pool exhausted", "too many connections", or "timeout waiting for available connection".

**Cause:** All database connections in the pool are in use. This can happen during bulk assumption imports, large dependency graph queries, or when the database is slow and connections are held open longer than expected.

**Resolution:**

1. **Check database connection pool status:**

```bash
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "pool\|connection\|exhausted\|timeout\|waiting"
```

2. **Check active connections in PostgreSQL:**

```bash
kubectl run pg-conns --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT count(*), state, application_name FROM pg_stat_activity
   WHERE application_name LIKE '%assumptions%'
   GROUP BY state, application_name;"
```

3. **Increase the connection pool size:**

```bash
kubectl set env deployment/assumptions-service -n greenlang \
  GL_ASSUMPTIONS_DB_POOL_MIN=5 \
  GL_ASSUMPTIONS_DB_POOL_MAX=25 \
  GL_ASSUMPTIONS_DB_POOL_TIMEOUT=30

kubectl rollout restart deployment/assumptions-service -n greenlang
```

4. **If connections are stuck, restart the service:**

```bash
kubectl rollout restart deployment/assumptions-service -n greenlang
kubectl rollout status deployment/assumptions-service -n greenlang
```

### Scenario 5: ImagePullBackOff

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
kubectl get deployment assumptions-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/assumptions-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/assumptions-service \
  assumptions-service=greenlang/assumptions-service:<correct-tag> -n greenlang
```

### Scenario 6: Network Policy Blocking Traffic

**Symptoms:** Pods are running and healthy, but upstream services cannot reach the assumptions service. Logs show no incoming requests.

**Cause:** A misconfigured network policy is blocking ingress traffic to the assumptions service on port 8080.

**Resolution:**

1. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang -o yaml | grep -A20 "assumptions"
```

2. **Verify the service endpoint is correct:**

```bash
kubectl get svc assumptions-service -n greenlang
kubectl get endpoints assumptions-service -n greenlang
```

3. **Test connectivity from a debug pod:**

```bash
kubectl run debug --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  wget -q -O- http://assumptions-service.greenlang.svc.cluster.local:8080/health
```

4. **If the network policy is blocking, patch it:**

```bash
kubectl patch networkpolicy assumptions-service-ingress -n greenlang -p '
{
  "spec": {
    "ingress": [
      {
        "from": [
          {
            "namespaceSelector": {
              "matchLabels": {
                "name": "greenlang"
              }
            }
          }
        ],
        "ports": [
          {
            "port": 8080,
            "protocol": "TCP"
          }
        ]
      }
    ]
  }
}'
```

### Scenario 7: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns, no infrastructure issues found.

**Cause:** A code regression introduced in the latest deployment.

**Resolution:**

1. **Check recent deployment history:**

```bash
kubectl rollout history deployment/assumptions-service -n greenlang
```

2. **Rollback to the previous version:**

```bash
kubectl rollout undo deployment/assumptions-service -n greenlang
kubectl rollout status deployment/assumptions-service -n greenlang
```

3. **Verify the rollback resolved the issue:**

```bash
kubectl get pods -n greenlang -l app=assumptions-service
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

4. **Create a bug report** with the logs from the failed deployment for the development team.

---

## Post-Incident Steps

After the assumptions service is restored, perform the following verification and cleanup steps.

### Step 1: Verify Service Health

```bash
# 1. Check all pods are running and ready
kubectl get pods -n greenlang -l app=assumptions-service

# 2. Check the health endpoint
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# 3. Verify the assumptions service is being scraped
up{job="assumptions-service"} == 1

# 4. Verify assumption operation metrics are incrementing (if requests are coming in)
increase(gl_assumptions_operations_total[5m])

# 5. Verify active assumption count is populated
gl_assumptions_active_count > 0

# 6. Verify validation metrics are reporting
gl_assumptions_validation_failures_total
```

### Step 3: Verify Assumption Registry is Loaded

After a restart, verify the assumption registry is accessible and contains expected data.

```bash
# Check assumption counts via API
curl -s http://localhost:8080/v1/assumptions?limit=1 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Total assumptions: {data.get('pagination', {}).get('total_items', 0)}\")
print(f\"Active assumptions: {data.get('summary', {}).get('active_count', 0)}\")
"

# Check scenario availability
curl -s http://localhost:8080/v1/assumptions/scenarios \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for s in data.get('scenarios', []):
    print(f\"Scenario: {s['name']} (overrides: {s.get('override_count', 0)}, active: {s.get('is_active', False)})\")
"
```

### Step 4: Verify Provenance Hash Chain Integrity

```bash
# Run provenance chain verification
curl -s -X POST http://localhost:8080/v1/assumptions/admin/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Assumptions verified: {data.get('total_verified', 0)}\")
print(f\"Chain intact: {data.get('chain_intact', False)}\")
print(f\"Broken links: {data.get('broken_links', 0)}\")
if data.get('broken_links', 0) > 0:
    print('WARNING: Provenance chain has broken links. See assumptions-audit-compliance runbook.')
"
```

### Step 5: Verify Upstream Services Have Recovered

```bash
# Check calculation agents for assumptions service errors
kubectl logs -n greenlang -l app.kubernetes.io/component=calculation-agent --tail=100 \
  | grep -i "assumption\|registry\|unavailable"

# Check compliance reporting agents for recovery
kubectl logs -n greenlang -l app.kubernetes.io/component=reporting-agent --tail=100 \
  | grep -i "assumption\|registry\|lookup"
```

### Step 6: Run a Smoke Test

```bash
# Test creating and reading an assumption
curl -X POST http://localhost:8080/v1/assumptions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{
    "key": "smoke_test_assumption",
    "value": 42.0,
    "unit": "test",
    "category": "smoke_test",
    "source": "Operational smoke test",
    "effective_date": "2026-01-01",
    "change_reason": "Post-incident smoke test"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Created: key={data.get(\"key\")}, version={data.get(\"version\")}')
print(f'Provenance hash: {data.get(\"provenance_hash\", \"N/A\")[:16]}...')
"

# Clean up smoke test assumption
curl -X DELETE http://localhost:8080/v1/assumptions/smoke_test_assumption \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Change-Reason: Post-incident cleanup"
```

### Step 7: Review Metrics for Root Cause

```promql
# Check for memory spikes before the outage
container_memory_working_set_bytes{namespace="greenlang", pod=~"assumptions-service.*"}

# Check for CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"assumptions-service.*"}[5m]))

# Check assumption operation throughput pattern (was there a spike?)
rate(gl_assumptions_operations_total[5m])

# Check active assumption count trend (drop to 0 indicates restart)
gl_assumptions_active_count

# Check database connection pool utilization
gl_assumptions_db_pool_active / gl_assumptions_db_pool_max
```

---

## Interim Mitigation

While the Assumptions Registry Service is being restored:

1. **Assumption data is safe.** All assumptions, version history, provenance chains, and scenario definitions are stored in PostgreSQL with TimescaleDB. The database persists independently and all data is intact.

2. **No new assumption operations can complete.** Applications that depend on the assumptions service for runtime assumption lookups will receive errors. Upstream applications should implement retry logic with exponential backoff.

3. **Cached assumptions may be available.** Some downstream services may have locally cached assumption values from recent lookups. These cached values can be used temporarily, but new calculations relying on cached values will lack provenance linkage to the registry and may not satisfy audit requirements.

4. **Provenance chain is preserved.** All previously generated provenance hashes are stored in the database and are unaffected by the outage. New provenance records will resume once the service recovers.

5. **Manual assumption override is not recommended.** While hard-coding assumption values is technically possible, it bypasses the version history, validation engine, and provenance tracking system. This creates audit trail gaps and violates the zero-hallucination guarantee. Only use manual overrides as a last resort for critical regulatory deadlines.

6. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#data-ops` -- data pipeline impact
   - `#platform-oncall` -- engineering response
   - `#compliance-ops` -- regulatory deadline impact

7. **Monitor the outage duration.** If the assumptions service is down for more than 15 minutes, escalate per the escalation path below. Compliance calculations and regulatory reports cannot proceed without verified assumption values.

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Assumptions service down, no active compliance pipelines impacted | On-call engineer | Immediate (<5 min) |
| L2 | Assumptions service down > 15 minutes, compliance pipelines blocked | Platform team lead + #assumptions-oncall | 15 minutes |
| L3 | Assumptions service down > 30 minutes, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Assumptions service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Assumptions Registry Health (`/d/assumptions-service-health`)
- **Dashboard:** Assumptions Operations Overview (`/d/assumptions-operations-overview`)
- **Alert:** `AssumptionsServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="assumptions-service"}` (should always be >= 2)
  - `gl_assumptions_operations_total` rate (should be non-zero during business hours)
  - `gl_assumptions_operations_failed_total` rate (should be low relative to total)
  - `gl_assumptions_active_count` (should be stable; drop to 0 indicates restart)
  - `gl_assumptions_validation_failures_total` rate (should be low)
  - `gl_assumptions_provenance_verifications_total` (should show passing verifications)
  - `gl_assumptions_scenario_queries_total` (should be non-zero if scenarios are in use)
  - `gl_assumptions_export_total` (tracks audit export operations)
  - `gl_assumptions_db_pool_active / gl_assumptions_db_pool_max` (should stay below 80%)
  - `gl_assumptions_query_duration_seconds` P95 (should stay below SLA targets)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 6 replicas** based on CPU utilization and request throughput
4. **Database connection pool** is sized for expected concurrency (default: min 5, max 25)
5. **Dependency graph queries** are bounded by `GL_ASSUMPTIONS_MAX_DEPENDENCY_DEPTH` (default: 10)
6. **Sensitivity analysis** is bounded by `GL_ASSUMPTIONS_SENSITIVITY_BATCH_SIZE` (default: 100)
7. **Export operations** stream results to avoid memory accumulation for large assumption sets

### Configuration Best Practices

- Set `GL_ASSUMPTIONS_DB_POOL_MAX` based on expected concurrency (2x expected peak concurrent requests)
- Set `GL_ASSUMPTIONS_MAX_DEPENDENCY_DEPTH` conservatively to prevent runaway graph traversal
- Set `GL_ASSUMPTIONS_VALIDATION_MODE=strict` in production (never `lenient`)
- Set `GL_ASSUMPTIONS_HASH_ALGORITHM=SHA-256` explicitly (do not rely on defaults)
- Set `GL_ASSUMPTIONS_AUDIT_RETENTION_DAYS` to at least 2555 (7 years) for SOC 2 compliance
- Use ESO for secrets rotation (DB credentials, export signing keys)
- Test configuration changes in staging before production
- Ensure TimescaleDB extension is enabled on the PostgreSQL instance

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `AssumptionsServiceDown` | Critical | This alert -- no assumptions service pods running |
| `AssumptionsValidationFailureSpike` | Warning | Sudden increase in validation failures |
| `AssumptionsScenarioDrift` | Warning | Scenario override values diverging significantly from baseline |
| `AssumptionsAuditGap` | Critical | Missing audit entries or provenance hash chain breaks |
| `AssumptionsHighLatency` | Warning | P95 assumption query latency exceeds SLA targets |
| `AssumptionsHighErrorRate` | Warning | >10% assumption operation errors over 5 minutes |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any P1 assumptions service incident
- **Related alerts:** `AssumptionsValidationFailureSpike`, `AssumptionsScenarioDrift`, `AssumptionsAuditGap`
- **Related dashboards:** Assumptions Registry Health, Assumptions Operations Overview
- **Related runbooks:** [Assumption Validation Failures](./assumption-validation-failures.md), [Scenario Drift Detection](./scenario-drift-detection.md), [Assumptions Audit Compliance](./assumptions-audit-compliance.md)
