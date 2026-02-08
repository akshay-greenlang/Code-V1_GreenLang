# Reproducibility Agent Service Down

## Alert

**Alert Name:** `ReproducibilityServiceDown`

**Severity:** Critical

**Threshold:** `up{job="reproducibility-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Reproducibility Agent (AGENT-FOUND-008) are running. The Reproducibility Agent is the determinism verification and drift detection service for all GreenLang Climate OS calculations. It is responsible for:

1. **Artifact hashing** -- Computing SHA-256/SHA-512/BLAKE2b hashes of input and output data artifacts with optional floating-point normalization to detect changes between executions
2. **Determinism verification** -- Comparing input/output hashes, environment fingerprints, seed configurations, and version manifests to verify that an execution produces identical results when re-run
3. **Drift detection** -- Comparing execution outputs against established baselines to detect gradual drift in calculation results, with severity classification (none, minor, moderate, critical)
4. **Replay mode** -- Re-executing a prior computation with identical inputs, environment, seeds, and versions to verify end-to-end reproducibility
5. **Environment fingerprinting** -- Capturing and comparing the full execution environment (Python version, platform, dependency versions, hardware) to detect environment-caused differences
6. **Seed management** -- Configuring and verifying random number generator state across Python, NumPy, PyTorch, and custom frameworks to ensure deterministic randomness
7. **Version pinning** -- Tracking version manifests of all agents, models, emission factors, data sources, configurations, and schemas active during an execution
8. **Non-determinism source detection** -- Identifying sources of non-determinism including floating-point accumulation, unseeded RNGs, parallel execution ordering, system time, external APIs, file system ordering, network calls, memory allocation, GPU computation, and hash ordering
9. **Provenance hash chains** -- Maintaining SHA-256 hash chains across verification records and audit events for tamper detection and compliance
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_reproducibility_*` prefix) for monitoring verification rates, hash computations, drift detections, replay results, cache performance, and service health

When the Reproducibility Agent is down:
- **Calculation results cannot be verified** for determinism, and new executions proceed without reproducibility guarantees
- **Drift detection stops** and gradual changes in calculation outputs will not be detected
- **Replay mode is unavailable** and historical executions cannot be re-verified
- **Audit trail has a gap** and compliance requirements for traceable, reproducible climate calculations are violated

**Note:** All verification runs, artifact hashes, environment fingerprints, seed configurations, version manifests, drift baselines, drift detections, replay sessions, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full state will be immediately available. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Calculations proceed but without reproducibility verification |
| **Data Impact** | Medium | No new verification records; audit trail gap during outage |
| **SLA Impact** | High | Reproducibility verification SLA violated (all verifications fail) |
| **Revenue Impact** | Medium | Compliance-sensitive customers require reproducibility guarantees |
| **Compliance Impact** | High | CSRD, CBAM, and SOC 2 require traceable, reproducible calculations |
| **Downstream Impact** | Medium | Orchestrator (GL-FOUND-X-001) may skip reproducibility checks or fail depending on configuration |

---

## Symptoms

- `up{job="reproducibility-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=reproducibility-service`
- `gl_reproducibility_verifications_total` counter stops incrementing
- `gl_reproducibility_hashes_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Orchestrator logs show "reproducibility service unavailable" or "verification skipped" warnings
- Grafana Reproducibility Agent dashboard shows "No Data" or stale timestamps
- Drift detection alerts stop firing even when baselines should be triggering

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List reproducibility service pods
kubectl get pods -n greenlang -l app=reproducibility-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=reproducibility-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to reproducibility service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=reproducibility-service | tail -30

# Check deployment status
kubectl describe deployment reproducibility-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment reproducibility-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=reproducibility-service

# Check for rollout issues
kubectl rollout status deployment/reproducibility-service -n greenlang

# Check HPA status (scales 2-10 replicas)
kubectl get hpa -n greenlang -l app=reproducibility-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=reproducibility-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=reproducibility-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for reproducibility-specific errors
kubectl logs -n greenlang -l app=reproducibility-service --tail=500 \
  | grep -i "verification\|hash\|drift\|replay\|fingerprint\|seed\|determinism"

# Look for database connection errors
kubectl logs -n greenlang -l app=reproducibility-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of reproducibility service pods
kubectl top pods -n greenlang -l app=reproducibility-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=reproducibility-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check node resource availability
kubectl top nodes
```

### Step 5: Check Database Connectivity

```bash
# Verify PostgreSQL connectivity
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check database connection pool status in logs
kubectl logs -n greenlang -l app=reproducibility-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the reproducibility_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='reproducibility_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the reproducibility service ConfigMap exists and is valid
kubectl get configmap reproducibility-service-config -n greenlang
kubectl get configmap reproducibility-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret reproducibility-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment reproducibility-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the reproducibility service
kubectl get networkpolicy -n greenlang | grep reproducibility

# Verify the reproducibility service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the reproducibility service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv reproducibility-service.greenlang.svc.cluster.local 8080'
```

---

## Resolution Steps

### Scenario 1: OOMKilled (Out of Memory)

**Symptoms:** Pod status shows OOMKilled, container exits with code 137.

**Resolution:**

1. Confirm the OOM cause:
```bash
kubectl describe pod -n greenlang <pod-name> | grep -A10 "Last State"
kubectl get events -n greenlang --field-selector reason=OOMKilling --sort-by='.lastTimestamp'
```

2. Increase memory limits:
```bash
kubectl patch deployment reproducibility-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "reproducibility-service",
            "resources": {
              "limits": {
                "cpu": "1",
                "memory": "1Gi"
              },
              "requests": {
                "cpu": "250m",
                "memory": "512Mi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

3. Verify pods restart successfully:
```bash
kubectl rollout status deployment/reproducibility-service -n greenlang
kubectl get pods -n greenlang -l app=reproducibility-service
```

### Scenario 2: CrashLoopBackOff -- Database Migration Failure

**Symptoms:** Pod status shows CrashLoopBackOff, init container logs show migration errors.

**Resolution:**

1. Check init container logs:
```bash
kubectl logs -n greenlang <pod-name> -c check-db-migration --tail=100
```

2. Verify database schema:
```bash
kubectl run pg-migration --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT version, description, success FROM flyway_schema_history
   ORDER BY installed_rank DESC LIMIT 5;"
```

3. Restart the deployment after fixing:
```bash
kubectl rollout restart deployment/reproducibility-service -n greenlang
kubectl rollout status deployment/reproducibility-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/reproducibility-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/reproducibility-service -n greenlang
kubectl rollout status deployment/reproducibility-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=reproducibility-service
kubectl port-forward -n greenlang svc/reproducibility-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=reproducibility-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/reproducibility-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the reproducibility service is being scraped
up{job="reproducibility-service"} == 1

# Verify verification count metric is populated
gl_reproducibility_verifications_total > 0

# Verify hash computation metrics are incrementing
increase(gl_reproducibility_hashes_total[5m])
```

### Step 3: Verify Verification Operations Work

```bash
# Check recent verification runs via API
curl -s http://localhost:8080/v1/verifications?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check drift baselines are loaded
curl -s http://localhost:8080/v1/drift/baselines?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Drift Detection Works

```bash
# Check recent drift detections
curl -s http://localhost:8080/v1/drift/detections?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Reproducibility Agent is being restored:

1. **Verification data is safe.** All verification runs, artifact hashes, environment fingerprints, seed configurations, version manifests, drift baselines, detections, replay sessions, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Calculations continue without verification.** The orchestrator and other agents will continue to execute pipelines, but results will not be verified for reproducibility. This is acceptable for short outages but must be resolved promptly for compliance.

3. **Drift detection is suspended.** Output drift will not be monitored during the outage. Any baselines that would have triggered alerts will not fire.

4. **Replay mode is unavailable.** Historical executions cannot be replayed for verification during the outage.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#compliance-ops` -- compliance impact notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Reproducibility service down, calculations proceeding without verification | On-call engineer | Immediate (<5 min) |
| L2 | Reproducibility service down > 15 minutes, audit gap growing | Platform team lead + #platform-foundation | 15 minutes |
| L3 | Reproducibility service down > 30 minutes, compliance impact, drift unmonitored | Platform team + compliance team + CTO notification | Immediate |
| L4 | Reproducibility service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Reproducibility Agent Health (`/d/reproducibility-service`)
- **Alert:** `ReproducibilityServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="reproducibility-service"}` (should always be >= 2)
  - `gl_reproducibility_verifications_total` rate (should be non-zero during business hours)
  - `gl_reproducibility_hashes_total` rate (should be non-zero)
  - `gl_reproducibility_drift_detections_total{severity="critical"}` (should be 0)
  - `gl_reproducibility_hash_mismatches_total` (should be 0 or near-zero)
  - `gl_reproducibility_verification_duration_seconds` p99 (should stay below 2s)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 80%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 10 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ReproducibilityServiceDown` | Critical | This alert -- no reproducibility service pods running |
| `ReproducibilityHighFailureRateCritical` | Critical | >25% of verifications are failing |
| `ReproducibilityHashMismatch` | Critical | Any artifact hash mismatch detected |
| `ReproducibilityDriftCritical` | Critical | Critical severity drift detected |
| `ReproducibilityAuditGap` | Critical | Verification operations without audit entries |
| `ReproducibilityDatabaseConnectionFailure` | Critical | Database connection errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 reproducibility service incident
- **Related runbooks:** [Reproducibility Drift Detection](./reproducibility-drift-detection.md)
