# Citations & Evidence Agent Service Down

## Alert

**Alert Name:** `CitationsServiceDown`

**Severity:** Critical

**Threshold:** `up{job="citations-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Citations & Evidence Agent service (AGENT-FOUND-005) are running. The Citations & Evidence Agent Service is the authoritative citation registry and evidence package system for zero-hallucination compliance calculations. It is responsible for:

1. **Citation CRUD with full version history** -- Every citation (emission factors, regulatory documents, methodology standards, scientific publications, databases) is explicitly registered, versioned, and verified
2. **Verification tracking** -- Citations pass through a verification pipeline (manual review, DOI lookup, URL validation, cross-reference, registry check) with a full audit trail of verification events
3. **Evidence package assembly** -- Bundling citations and supporting data into hash-verified evidence packages for audit submission with finalization (immutability) support
4. **Methodology reference management** -- Linking GHG Protocol, ISO 14064, and other calculation methodologies to their authoritative citations with scope applicability tracking
5. **Regulatory mapping** -- Mapping CSRD, CBAM, EU Taxonomy, SEC Climate, and other regulatory requirements to their authoritative citations with compliance status and deadline tracking
6. **Data source attribution** -- Tracking which data values were extracted from which source datasets (DEFRA, EPA eGRID, ecoinvent, IPCC) with validity dates
7. **SHA-256 provenance hash chain** -- Every change to every citation produces a cryptographic hash linking the new version to the previous, creating a tamper-evident audit trail
8. **Supersession management** -- Tracking when newer versions of sources replace older ones (e.g., DEFRA 2024 supersedes DEFRA 2023) and ensuring downstream calculations use current citations
9. **Emitting Prometheus metrics** (12 metrics under the `gl_citations_*` prefix) for monitoring citation operations, verification status, evidence packages, cache performance, and service health

When the Citations & Evidence Agent is down, all upstream services that depend on citation provenance will be unable to retrieve the references they need for compliance calculations. This includes every GreenLang application pipeline -- CSRD, CBAM, VCCI, SB253, EUDR, and Taxonomy. The zero-hallucination guarantee requires every calculated value to be traceable to an explicitly cited source, so calculations cannot proceed without the citations service.

**Note:** Citation data, version history, evidence packages, and verification records are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full citation registry will be immediately available. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | All compliance calculations blocked; users cannot create, verify, or query citations; evidence packages cannot be assembled or finalized |
| **Data Impact** | High | No new citations can be registered; verification pipeline is paused; evidence packages cannot be finalized |
| **SLA Impact** | Critical | Citation query latency SLA violated (P95 targets: <15ms single lookup, <100ms batch, <500ms evidence package assembly) |
| **Revenue Impact** | High | Customer-facing compliance workflows cannot produce zero-hallucination calculations or auditable evidence packages |
| **Compliance Impact** | Critical | Zero-hallucination guarantee broken -- calculations cannot verify their citation sources; regulatory submission deadlines at risk; audit trail generation impossible |
| **Downstream Impact** | Critical | All 6 GreenLang application pipelines depend on the citations registry; assumptions service cannot resolve citation links; cascading failures across calculation, reporting, and compliance agents |

---

## Symptoms

- `up{job="citations-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=citations-service`
- `gl_citations_operations_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Upstream calculation agents report "citation unavailable" or "citation lookup timeout" errors in logs
- Grafana Citations dashboard shows "No Data" or stale timestamps
- Evidence package finalization endpoint returns errors
- Verification pipeline stops processing pending verifications
- Assumptions service reports "unable to resolve citation reference" errors

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List citations service pods
kubectl get pods -n greenlang -l app=citations-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=citations-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to citations service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=citations-service | tail -30

# Check deployment status
kubectl describe deployment citations-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment citations-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=citations-service

# Check for rollout issues
kubectl rollout status deployment/citations-service -n greenlang

# Check HPA status (scales 2-8 replicas)
kubectl get hpa -n greenlang -l app=citations-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=citations-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=citations-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for citations-specific errors
kubectl logs -n greenlang -l app=citations-service --tail=500 \
  | grep -i "citation\|evidence\|verification\|provenance\|hash\|supersed"

# Look for database connection errors
kubectl logs -n greenlang -l app=citations-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of citations service pods
kubectl top pods -n greenlang -l app=citations-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=citations-service \
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
kubectl logs -n greenlang -l app=citations-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the citations_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='citations_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the citations service ConfigMap exists and is valid
kubectl get configmap citations-service-config -n greenlang
kubectl get configmap citations-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret citations-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment citations-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the citations service
kubectl get networkpolicy -n greenlang | grep citations

# Verify the citations service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the citations service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv citations-service.greenlang.svc.cluster.local 8080'
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
kubectl patch deployment citations-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "citations-service",
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
kubectl rollout status deployment/citations-service -n greenlang
kubectl get pods -n greenlang -l app=citations-service
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
kubectl rollout restart deployment/citations-service -n greenlang
kubectl rollout status deployment/citations-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/citations-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/citations-service -n greenlang
kubectl rollout status deployment/citations-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=citations-service
kubectl port-forward -n greenlang svc/citations-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=citations-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/citations-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the citations service is being scraped
up{job="citations-service"} == 1

# Verify citation operation metrics are incrementing
increase(gl_citations_operations_total[5m])

# Verify registered citation count is populated
gl_citations_registered_total > 0
```

### Step 3: Verify Citation Registry is Loaded

```bash
# Check citation counts via API
curl -s http://localhost:8080/v1/citations?limit=1 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check verification pipeline status
curl -s http://localhost:8080/v1/citations/verifications/stats \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Provenance Hash Chain Integrity

```bash
# Run provenance chain verification
curl -s -X POST http://localhost:8080/v1/citations/admin/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Citations & Evidence Agent Service is being restored:

1. **Citation data is safe.** All citations, version history, evidence packages, and verification records are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **No new citation operations can complete.** Applications that depend on the citations service for runtime lookups will receive errors.

3. **Cached citations may be available.** Some downstream services may have locally cached citation data from recent lookups.

4. **Provenance chain is preserved.** All previously generated provenance hashes are stored in the database and are unaffected by the outage.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#data-ops` -- data pipeline impact
   - `#platform-oncall` -- engineering response
   - `#compliance-ops` -- regulatory deadline impact

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Citations service down, no active compliance pipelines impacted | On-call engineer | Immediate (<5 min) |
| L2 | Citations service down > 15 minutes, compliance pipelines blocked | Platform team lead + #citations-oncall | 15 minutes |
| L3 | Citations service down > 30 minutes, regulatory deadline at risk | Platform team + compliance team + CTO notification | Immediate |
| L4 | Citations service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Citations & Evidence Agent Health (`/d/citations-service`)
- **Alert:** `CitationsServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="citations-service"}` (should always be >= 2)
  - `gl_citations_operations_total` rate (should be non-zero during business hours)
  - `gl_citations_registered_total` (should be stable; drop to 0 indicates restart)
  - `gl_citations_verifications_total` rate (should show verification activity)
  - `gl_citations_evidence_packages_total` (should be non-zero)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 85%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 8 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `CitationsServiceDown` | Critical | This alert -- no citations service pods running |
| `CitationsHighErrorRate` | Warning | >5% citation operation errors over 5 minutes |
| `CitationsHighErrorRateCritical` | Critical | >15% citation operation errors over 5 minutes |
| `CitationsHashIntegrityFailure` | Critical | Content hash integrity failure detected |
| `CitationsProvenanceChainBroken` | Critical | Provenance hash chain break detected |
| `CitationsAuditGap` | Critical | Missing version entries for citation changes |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any P1 citations service incident
- **Related runbooks:** [Citations Audit Compliance](./citations-audit-compliance.md)
