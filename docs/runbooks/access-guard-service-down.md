# Access & Policy Guard Agent Service Down

## Alert

**Alert Name:** `AccessGuardServiceDown`

**Severity:** Critical

**Threshold:** `up{job="access-guard-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang Access & Policy Guard Agent service (AGENT-FOUND-006) are running. The Access & Policy Guard Agent Service is the centralized access control enforcement point for all GreenLang Climate OS services. It is responsible for:

1. **Policy-based access control** -- Every access request is evaluated against a set of policies and rules with a deny-wins evaluation model, producing allow/deny/rate_limited decisions with SHA-256 decision hashes for tamper-evident audit
2. **Tenant isolation enforcement** -- Cross-tenant access attempts are detected and blocked, ensuring each tenant's data (emissions, calculations, reports, assumptions, citations) is completely isolated
3. **Data classification enforcement** -- Resources are classified (public, internal, confidential, restricted, top_secret) and access is restricted based on principal clearance level
4. **Rate limiting** -- Per-role, per-tenant request quotas are enforced (viewer: 60/min, analyst: 120/min, admin: 300/min, service: 500/min, agent: 200/min)
5. **OPA Rego evaluation** -- Complex policy logic is evaluated using Open Policy Agent Rego modules for tenant isolation, classification access, and rate limiting
6. **Access decision audit trail** -- Every access decision is recorded in a TimescaleDB hypertable with request details, matching rules, deny reasons, evaluation time, and decision hash
7. **Comprehensive audit events** -- All access control operations (policy changes, rule changes, classification changes, rate limit hits, tenant violations) are logged with source IP, user agent, and details
8. **Compliance reporting** -- Weekly compliance reports summarize access activity, denial rates, policy coverage, and classification-based access patterns for SOC 2, ISO 27001, and CSRD
9. **Emitting Prometheus metrics** (12+ metrics under the `gl_access_guard_*` prefix) for monitoring decisions, denials, latency, rate limits, tenant violations, classification breaches, cache performance, and service health

When the Access & Policy Guard Agent is down, the behavior depends on the downstream service configuration:
- **Fail-closed services** will deny all requests, causing a complete service outage for users
- **Fail-open services** will allow all requests without access control, creating a security gap

**Note:** Access decision data, audit events, policies, rules, and classifications are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full policy set and audit trail will be immediately available. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | Fail-closed: all user requests blocked; Fail-open: no access control enforcement |
| **Data Impact** | High | No new access decisions recorded; audit trail gap during outage |
| **SLA Impact** | Critical | Access decision latency SLA violated (P99 targets: <50ms single decision) |
| **Revenue Impact** | High | Customer-facing workflows either blocked entirely (fail-closed) or running without access control (fail-open) |
| **Compliance Impact** | Critical | SOC 2, ISO 27001, CSRD audit trail requirements violated; tenant isolation not enforced during fail-open; classification enforcement suspended |
| **Downstream Impact** | Critical | All GreenLang services depend on access guard for authorization; cascading failures across calculation, reporting, compliance, and agent pipelines |

---

## Symptoms

- `up{job="access-guard-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=access-guard-service`
- `gl_access_guard_decisions_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- Upstream services report "access guard unavailable" or "authorization timeout" errors in logs
- Grafana Access Guard dashboard shows "No Data" or stale timestamps
- Rate limiting not enforced; traffic spikes pass through unchecked
- Tenant isolation not enforced; cross-tenant access may succeed (fail-open mode)

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List access guard service pods
kubectl get pods -n greenlang -l app=access-guard-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=access-guard-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to access guard service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=access-guard-service | tail -30

# Check deployment status
kubectl describe deployment access-guard-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment access-guard-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=access-guard-service

# Check for rollout issues
kubectl rollout status deployment/access-guard-service -n greenlang

# Check HPA status (scales 2-10 replicas)
kubectl get hpa -n greenlang -l app=access-guard-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=access-guard-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=access-guard-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for access-guard-specific errors
kubectl logs -n greenlang -l app=access-guard-service --tail=500 \
  | grep -i "policy\|decision\|tenant\|classification\|rego\|rate.limit\|opa"

# Look for database connection errors
kubectl logs -n greenlang -l app=access-guard-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of access guard service pods
kubectl top pods -n greenlang -l app=access-guard-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=access-guard-service \
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
kubectl logs -n greenlang -l app=access-guard-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the access_guard_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='access_guard_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the access guard service ConfigMap exists and is valid
kubectl get configmap access-guard-service-config -n greenlang
kubectl get configmap access-guard-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret access-guard-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment access-guard-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the access guard service
kubectl get networkpolicy -n greenlang | grep access-guard

# Verify the access guard service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the access guard service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv access-guard-service.greenlang.svc.cluster.local 8080'
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
kubectl patch deployment access-guard-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "access-guard-service",
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
kubectl rollout status deployment/access-guard-service -n greenlang
kubectl get pods -n greenlang -l app=access-guard-service
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
kubectl rollout restart deployment/access-guard-service -n greenlang
kubectl rollout status deployment/access-guard-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/access-guard-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/access-guard-service -n greenlang
kubectl rollout status deployment/access-guard-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=access-guard-service
kubectl port-forward -n greenlang svc/access-guard-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=access-guard-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/access-guard-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the access guard service is being scraped
up{job="access-guard-service"} == 1

# Verify access decision metrics are incrementing
increase(gl_access_guard_decisions_total[5m])

# Verify policy count is populated
gl_access_guard_policies_total > 0
```

### Step 3: Verify Policy Engine is Loaded

```bash
# Check policy count via API
curl -s http://localhost:8080/v1/policies?limit=1 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check OPA Rego module status
curl -s http://localhost:8080/v1/rego/status \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Tenant Isolation is Active

```bash
# Run tenant isolation verification
curl -s -X POST http://localhost:8080/v1/admin/verify-tenant-isolation \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the Access & Policy Guard Agent Service is being restored:

1. **Access decision data is safe.** All decisions, audit events, policies, and classifications are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Downstream behavior depends on configuration.** Fail-closed services will block all requests (secure but disruptive). Fail-open services will allow all requests (available but insecure).

3. **Rate limiting is not enforced.** Traffic spikes may reach downstream services unchecked.

4. **Tenant isolation is not enforced in fail-open mode.** Cross-tenant access may succeed during the outage.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#security-ops` -- security posture impact
   - `#platform-oncall` -- engineering response
   - `#compliance-ops` -- audit trail gap notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Access guard service down, fail-closed mode active (users blocked) | On-call engineer | Immediate (<5 min) |
| L2 | Access guard service down > 15 minutes, fail-open mode (security gap) | Platform team lead + #security-oncall | 15 minutes |
| L3 | Access guard service down > 30 minutes, tenant isolation breached or audit gap | Platform team + security team + CTO notification | Immediate |
| L4 | Access guard service down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Access & Policy Guard Agent Health (`/d/access-guard-service`)
- **Alert:** `AccessGuardServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="access-guard-service"}` (should always be >= 2)
  - `gl_access_guard_decisions_total` rate (should be non-zero during business hours)
  - `gl_access_guard_policies_total` (should be stable; drop to 0 indicates restart)
  - `gl_access_guard_tenant_violations_total` (should be 0; any increase is a security event)
  - `gl_access_guard_evaluation_duration_seconds` p99 (should stay below 50ms)
  - Pod restart count (should be 0)
  - Container memory usage vs limit (should stay below 85%)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales from 2 to 10 replicas** based on CPU and memory utilization
4. **Database connection pool** is sized for expected concurrency (default: min 2, max 10)

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `AccessGuardServiceDown` | Critical | This alert -- no access guard service pods running |
| `AccessGuardHighDenialRate` | Warning | >20% access denial rate over 5 minutes |
| `AccessGuardHighDenialRateCritical` | Critical | >50% access denial rate over 5 minutes |
| `AccessGuardTenantViolation` | Critical | Any cross-tenant access attempt detected |
| `AccessGuardClassificationBreach` | Critical | Unauthorized access to restricted/top_secret resource |
| `AccessGuardAuditGap` | Critical | Missing audit entries for access decisions |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Security Team
- **Review cadence:** Quarterly or after any P1 access guard service incident
- **Related runbooks:** [Access Guard Policy Audit](./access-guard-policy-audit.md)
