# QA Test Harness Service Down

## Alert

**Alert Name:** `QATestHarnessServiceDown`

**Severity:** Critical

**Threshold:** `up{job="qa-test-harness-service"} == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang QA Test Harness (AGENT-FOUND-009) are running. The QA Test Harness is the comprehensive testing framework for all GreenLang Climate OS agents. It is responsible for:

1. **Zero-hallucination verification** -- Verifying that agent outputs contain zero hallucinated data by checking all output values trace to input data or known reference sources
2. **Determinism testing** -- Executing the same agent with identical inputs N times and verifying outputs are bit-for-bit identical
3. **Lineage completeness checks** -- Verifying that agent outputs include complete provenance chains tracing every value to its source
4. **Golden file/snapshot testing** -- Comparing agent outputs against approved golden file snapshots with configurable diff tolerance
5. **Regression detection** -- Detecting output regressions by comparing current output hashes against established baselines
6. **Performance benchmarking** -- Benchmarking agent execution time and memory usage with statistical analysis (min, max, mean, median, p95, p99, std_dev)
7. **Coverage tracking** -- Tracking test coverage percentage for agent methods and identifying uncovered code paths
8. **Multi-format report generation** -- Generating test reports in JSON, HTML, Markdown, and JUnit XML formats with detailed results and trend analysis
9. **Provenance hash chains** -- Maintaining SHA-256 hash chains across test records and audit events for tamper detection and compliance
10. **Emitting Prometheus metrics** (12+ metrics under the `gl_qa_test_harness_*` prefix) for monitoring test execution rates, pass/fail ratios, regression detections, golden file mismatches, performance breaches, cache performance, and service health

When the QA Test Harness is down:
- **Agent quality cannot be verified** and new deployments proceed without test verification
- **Regression detection stops** and output changes in agents will not be caught
- **Golden file comparisons are unavailable** and snapshot testing cannot be performed
- **Performance benchmarking is suspended** and performance regressions will not be detected
- **Coverage tracking stops** and test coverage trends cannot be updated
- **Audit trail has a gap** and compliance requirements for traceable quality verification are violated

**Note:** All test suites, test cases, test runs, assertions, golden files, performance baselines, coverage snapshots, regression baselines, and audit events are stored in PostgreSQL with TimescaleDB and are not affected by a service outage. Once the service recovers, the full state will be immediately available. No data is lost during an outage.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Agents continue operating but without quality verification |
| **Data Impact** | Medium | No new test results; audit trail gap during outage |
| **SLA Impact** | High | QA verification SLA violated (all test executions fail) |
| **Revenue Impact** | Medium | Compliance-sensitive customers require quality verification guarantees |
| **Compliance Impact** | High | CSRD, CBAM, and SOC 2 require traceable, verified agent outputs |
| **Downstream Impact** | Medium | CI/CD pipelines that depend on QA verification will be blocked |

---

## Symptoms

- `up{job="qa-test-harness-service"}` metric returns 0 or is absent
- No pods running in the `greenlang` namespace with label `app=qa-test-harness-service`
- `gl_qa_test_harness_test_runs_total` counter stops incrementing
- `gl_qa_test_harness_assertions_total` counter stops incrementing
- REST API returns 503 Service Unavailable or connection refused
- Health endpoint `GET /health` is unreachable
- CI/CD pipelines show "qa-test-harness service unavailable" or "test verification skipped" warnings
- Grafana QA Test Harness dashboard shows "No Data" or stale timestamps
- Regression detection alerts stop firing even when baselines should be triggering

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List QA test harness service pods
kubectl get pods -n greenlang -l app=qa-test-harness-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang -l app=qa-test-harness-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace related to QA test harness service
kubectl get events -n greenlang --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=qa-test-harness-service | tail -30

# Check deployment status
kubectl describe deployment qa-test-harness-service -n greenlang
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment qa-test-harness-service -n greenlang

# Check ReplicaSet status
kubectl get replicaset -n greenlang -l app=qa-test-harness-service

# Check for rollout issues
kubectl rollout status deployment/qa-test-harness-service -n greenlang

# Check HPA status (scales 2-10 replicas)
kubectl get hpa -n greenlang -l app=qa-test-harness-service
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang -l app=qa-test-harness-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang -l app=qa-test-harness-service --tail=500 \
  | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for QA-specific errors
kubectl logs -n greenlang -l app=qa-test-harness-service --tail=500 \
  | grep -i "test\|assertion\|golden\|regression\|benchmark\|coverage\|determinism\|hallucination"

# Look for database connection errors
kubectl logs -n greenlang -l app=qa-test-harness-service --tail=500 \
  | grep -i "database\|postgres\|timescale\|connection\|pool\|migration"
```

### Step 4: Check Resource Usage

```bash
# Check current CPU and memory usage of QA test harness service pods
kubectl top pods -n greenlang -l app=qa-test-harness-service

# Check if pods were OOMKilled
kubectl get pods -n greenlang -l app=qa-test-harness-service \
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
kubectl logs -n greenlang -l app=qa-test-harness-service --tail=200 \
  | grep -i "pool\|connection\|database\|postgres"

# Check if the qa_test_harness_service schema exists
kubectl run pg-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='qa_test_harness_service'
   ORDER BY table_name;"
```

### Step 6: Check ConfigMap and Secrets

```bash
# Verify the QA test harness service ConfigMap exists and is valid
kubectl get configmap qa-test-harness-service-config -n greenlang
kubectl get configmap qa-test-harness-service-config -n greenlang -o yaml | head -50

# Verify secrets exist
kubectl get secret qa-test-harness-service-secrets -n greenlang

# Check environment variables are set correctly
kubectl get deployment qa-test-harness-service -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].env[*].name}' | tr ' ' '\n' | sort
```

### Step 7: Check Network Policies

```bash
# Check network policies affecting the QA test harness service
kubectl get networkpolicy -n greenlang | grep qa-test-harness

# Verify the QA test harness service can reach PostgreSQL
kubectl run net-test --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv greenlang-db.postgres.svc.cluster.local 5432'

# Verify upstream services can reach the QA test harness service
kubectl run net-test-2 --rm -it --image=busybox:1.36 -n greenlang --restart=Never -- \
  sh -c 'nc -zv qa-test-harness-service.greenlang.svc.cluster.local 8080'
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
kubectl patch deployment qa-test-harness-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "qa-test-harness-service",
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
kubectl rollout status deployment/qa-test-harness-service -n greenlang
kubectl get pods -n greenlang -l app=qa-test-harness-service
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
kubectl rollout restart deployment/qa-test-harness-service -n greenlang
kubectl rollout status deployment/qa-test-harness-service -n greenlang
```

### Scenario 3: Code Bug -- Rollback Deployment

**Symptoms:** Service was working before a recent deployment, logs show new error patterns.

**Resolution:**

1. Check recent deployment history:
```bash
kubectl rollout history deployment/qa-test-harness-service -n greenlang
```

2. Rollback to the previous version:
```bash
kubectl rollout undo deployment/qa-test-harness-service -n greenlang
kubectl rollout status deployment/qa-test-harness-service -n greenlang
```

3. Verify the rollback resolved the issue:
```bash
kubectl get pods -n greenlang -l app=qa-test-harness-service
kubectl port-forward -n greenlang svc/qa-test-harness-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

---

## Post-Incident Steps

### Step 1: Verify Service Health

```bash
# Check all pods are running and ready
kubectl get pods -n greenlang -l app=qa-test-harness-service

# Check the health endpoint
kubectl port-forward -n greenlang svc/qa-test-harness-service 8080:8080
curl -s http://localhost:8080/health | python3 -m json.tool
```

### Step 2: Verify Prometheus Metrics Are Flowing

```promql
# Verify the QA test harness service is being scraped
up{job="qa-test-harness-service"} == 1

# Verify test run count metric is populated
gl_qa_test_harness_test_runs_total > 0

# Verify assertion metrics are incrementing
increase(gl_qa_test_harness_assertions_total[5m])
```

### Step 3: Verify Test Operations Work

```bash
# Check recent test runs via API
curl -s http://localhost:8080/v1/runs?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check golden files are loaded
curl -s http://localhost:8080/v1/golden-files?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

### Step 4: Verify Regression Detection Works

```bash
# Check recent regression baselines
curl -s http://localhost:8080/v1/regression/baselines?limit=5 \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

---

## Interim Mitigation

While the QA Test Harness is being restored:

1. **Test data is safe.** All test suites, test cases, test runs, assertions, golden files, performance baselines, coverage snapshots, regression baselines, and audit events are stored in PostgreSQL with TimescaleDB. The database persists independently.

2. **Agents continue operating without verification.** All GreenLang agents will continue to execute, but their outputs will not be verified for quality. This is acceptable for short outages but must be resolved promptly for compliance.

3. **Regression detection is suspended.** Output changes in agents will not be caught during the outage. Any regressions introduced will not trigger alerts.

4. **CI/CD pipelines may be blocked.** Pipelines that gate on QA verification results will fail or skip the verification step.

5. **Communicate to affected teams.** Notify the following channels:
   - `#greenlang-incidents` -- incident status
   - `#platform-foundation` -- engineering response
   - `#platform-oncall` -- on-call engineer
   - `#compliance-ops` -- compliance impact notification

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | QA test harness down, agents proceeding without verification | On-call engineer | Immediate (<5 min) |
| L2 | QA test harness down > 15 minutes, audit gap growing | Platform team lead + #platform-foundation | 15 minutes |
| L3 | QA test harness down > 30 minutes, compliance impact, CI/CD blocked | Platform team + compliance team + CTO notification | Immediate |
| L4 | QA test harness down due to infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** QA Test Harness Health (`/d/qa-test-harness-service`)
- **Alert:** `QATestHarnessServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="qa-test-harness-service"}` (should always be >= 2)
  - `gl_qa_test_harness_test_runs_total` rate (should be non-zero during business hours)
  - `gl_qa_test_harness_assertions_total` rate (should be non-zero)
  - `gl_qa_test_harness_regressions_detected_total` (should be 0 or near-zero)
  - `gl_qa_test_harness_golden_file_mismatches_total` (should be 0 or near-zero)
  - `gl_qa_test_harness_test_duration_seconds` p99 (should stay below 5s)
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
| `QATestHarnessServiceDown` | Critical | This alert -- no QA test harness pods running |
| `QAHighTestFailureRateCritical` | Critical | >30% of test runs are failing |
| `QARegressionDetected` | Critical | Any regression detected in agent output |
| `QAZeroHallucinationFailure` | Critical | Zero-hallucination test failure |
| `QADeterminismFailure` | Critical | Determinism test failure |
| `QAAuditGap` | Critical | Test operations without audit entries |
| `QADatabaseConnectionFailure` | Critical | Database connection errors |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any P1 QA test harness incident
- **Related runbooks:** [QA Test Harness Regression Detected](./qa-test-harness-regression-detected.md)
