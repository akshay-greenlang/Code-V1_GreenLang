# SLO Service Down

## Alert

**Alert Name:** `SLOServiceDown`

**Severity:** Critical

**Threshold:** `absent(up{job="slo-service"} == 1) or sum(up{job="slo-service"}) == 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when no instances of the GreenLang SLO/SLI Management Service (OBS-005) are running. The SLO Service is responsible for:

1. **Evaluating SLO compliance** against defined targets for all GreenLang services
2. **Tracking error budgets** in real time with consumption rate and forecasting
3. **Computing burn rates** across fast (1h/5m), medium (6h/30m), and slow (3d/6h) windows
4. **Generating Prometheus recording rules** for SLI ratio computation at the Prometheus level
5. **Generating Prometheus alert rules** for multi-window burn rate alerting (Google SRE Book methodology)
6. **Provisioning Grafana dashboards** for SLO overview, per-service detail, and error budget deep dive
7. **Producing compliance reports** (weekly, monthly, quarterly) with trend analysis
8. **Bridging to OBS-004 Alerting** for SLO violation notifications through all 6 channels

When the SLO Service is down, SLO evaluations stop, error budget tracking becomes stale, burn rate alerts will not update, SLO dashboards will display outdated data, and compliance reports will not be generated. The service itself does NOT prevent existing Prometheus alert rules from firing -- those are evaluated directly by Prometheus. However, if recording rules need to be regenerated (e.g., after an SLO definition change), that operation will be blocked until the service recovers.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | SLO dashboards show stale data; engineers lose real-time reliability visibility |
| **Data Impact** | High | Error budget snapshots stop being written to TimescaleDB; gap in historical data |
| **SLA Impact** | Medium | Existing burn rate alerts continue firing via Prometheus; but budget policies (deployment freeze) cannot be enforced |
| **Revenue Impact** | Medium | Reduced reliability visibility may delay incident detection for SLO-monitored services |
| **Compliance Impact** | Medium | SOC 2 CC7.2 (monitoring) controls require continuous SLO tracking; gap may be flagged in audit |

---

## Symptoms

- `up{job="slo-service"}` metric returns 0 or is absent
- No pods running in the `greenlang-slo` namespace with label `app=slo-service`
- SLO Overview Grafana dashboard (`/d/slo-overview`) shows "No Data" or stale timestamps
- `gl_slo_evaluations_total` counter has stopped incrementing
- `gl_slo_error_budget_remaining_percent` gauge is not being updated (timestamp is stale)
- `gl_slo_budget_snapshots_total` counter has stopped incrementing
- Error budget cache in Redis (`gl:slo:*`) returns stale values (check TTL)
- SLO compliance report CronJob shows failures or pending status
- Grafana SLO dashboard annotations show no recent SLO evaluation events

---

## Diagnostic Steps

### Step 1: Check Pod Status

```bash
# List SLO service pods
kubectl get pods -n greenlang-slo -l app=slo-service

# Check for failed pods, CrashLoopBackOff, or ImagePullBackOff
kubectl get pods -n greenlang-slo -l app=slo-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].state}{"\n"}{end}'

# Check events for the namespace
kubectl get events -n greenlang-slo --sort-by='.lastTimestamp' | tail -30

# Check deployment status
kubectl describe deployment slo-service -n greenlang-slo
```

### Step 2: Check Deployment and ReplicaSet

```bash
# Verify deployment desired vs available replicas (expect 2 minimum)
kubectl get deployment slo-service -n greenlang-slo

# Check ReplicaSet status
kubectl get replicaset -n greenlang-slo -l app=slo-service

# Check for rollout issues
kubectl rollout status deployment/slo-service -n greenlang-slo

# Check HPA status (min 2, max 4)
kubectl get hpa -n greenlang-slo
```

### Step 3: Check Container Logs

```bash
# Get logs from the most recent pod (even if crashed)
kubectl logs -n greenlang-slo -l app=slo-service --tail=200

# Get logs from a specific crashed pod (previous container)
kubectl logs -n greenlang-slo <pod-name> --previous --tail=200

# Look for specific error patterns
kubectl logs -n greenlang-slo -l app=slo-service --tail=500 | grep -i "error\|fatal\|panic\|fail\|connection refused"

# Look for Prometheus connectivity issues
kubectl logs -n greenlang-slo -l app=slo-service --tail=500 | grep -i "prometheus\|promql\|query\|timeout"

# Look for database connection errors
kubectl logs -n greenlang-slo -l app=slo-service --tail=500 | grep -i "database\|postgres\|timescale\|redis\|connection"
```

### Step 4: Check Prometheus Connectivity

The SLO service queries Prometheus for SLI values and burn rate calculations. If Prometheus is unreachable, the service may crash or hang.

```bash
# Verify Prometheus is running
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus

# Test Prometheus connectivity from within the SLO service namespace
kubectl run prom-test --rm -it --image=busybox:1.36 -n greenlang-slo --restart=Never -- \
  sh -c 'wget -q -O- http://prometheus-server.monitoring.svc.cluster.local:9090/api/v1/status/runtimeinfo | head -c 200'

# If using Thanos Query, test that endpoint instead
kubectl run thanos-test --rm -it --image=busybox:1.36 -n greenlang-slo --restart=Never -- \
  sh -c 'wget -q -O- http://thanos-query.monitoring.svc.cluster.local:9090/api/v1/status/runtimeinfo | head -c 200'
```

### Step 5: Check Database Connectivity (PostgreSQL/TimescaleDB)

The SLO service stores SLO definitions, error budget snapshots, evaluation logs, and compliance reports in TimescaleDB.

```bash
# Verify PostgreSQL connectivity from within the namespace
kubectl run pg-test --rm -it --image=busybox:1.36 -n greenlang-slo --restart=Never -- \
  sh -c 'nc -zv greenlang-postgresql.database.svc.cluster.local 5432'

# Check that the slo schema exists and is accessible
kubectl run pg-check --rm -it --image=postgres:14 -n greenlang-slo --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT COUNT(*) FROM slo.definitions WHERE is_active = true;"
```

### Step 6: Check Redis Connectivity

The SLO service caches error budget state in Redis for fast lookups (key prefix `gl:slo:`).

```bash
# Verify Redis connectivity
kubectl run redis-test --rm -it --image=busybox:1.36 -n greenlang-slo --restart=Never -- \
  sh -c 'nc -zv greenlang-redis.redis.svc.cluster.local 6379'

# Check Redis key count for SLO cache
kubectl run redis-check --rm -it --image=redis:7 -n greenlang-slo --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 KEYS 'gl:slo:*' | head -20
```

### Step 7: Check Resource Limits and Node Capacity

```bash
# Check if pods are being evicted due to resource pressure
kubectl get events -n greenlang-slo --field-selector reason=Evicted

# Check node resource availability
kubectl top nodes

# Check if PDB is preventing scheduling
kubectl get pdb -n greenlang-slo

# Check if resource quota is exhausted
kubectl describe resourcequota -n greenlang-slo
```

### Step 8: Check ConfigMap and Secrets

```bash
# Verify the SLO definitions ConfigMap exists and is valid
kubectl get configmap slo-service-config -n greenlang-slo
kubectl get configmap slo-service-config -n greenlang-slo -o yaml | head -50

# Verify secrets exist (database URL, Redis URL, Grafana API key)
kubectl get secret slo-service-secrets -n greenlang-slo

# Check ESO sync status
kubectl get externalsecrets -n greenlang-slo
```

---

## Resolution Steps

### Scenario 1: CrashLoopBackOff (Application Error)

**Symptoms:** Pod status shows CrashLoopBackOff, container exits with non-zero code

**Cause:** Application startup failure due to configuration error, missing secrets, or code bug.

**Resolution:**

1. **Check the crash reason:**

```bash
kubectl describe pod -n greenlang-slo <pod-name> | grep -A10 "Last State"
kubectl logs -n greenlang-slo <pod-name> --previous --tail=100
```

2. **If caused by missing secrets:**

```bash
# Verify secrets exist
kubectl get secret slo-service-secrets -n greenlang-slo

# Check ESO sync status
kubectl get externalsecrets -n greenlang-slo
kubectl describe externalsecret slo-service-secrets -n greenlang-slo

# If secrets are missing, check SSM parameters
aws ssm get-parameters-by-path --path "/gl/prod/slo-service/" --query "Parameters[*].Name"
```

3. **If caused by invalid SLO definitions YAML:**

```bash
# Validate the SLO definitions ConfigMap
kubectl get configmap slo-service-config -n greenlang-slo -o yaml

# Check for YAML syntax errors in the logs
kubectl logs -n greenlang-slo <pod-name> --previous | grep -i "yaml\|parse\|validation\|config\|definition"

# Validate the source SLO definitions file
python3 -c "import yaml; yaml.safe_load(open('deployment/infrastructure/monitoring/slos/slo_definitions.yaml'))"
```

4. **Restart the deployment after fixing:**

```bash
kubectl rollout restart deployment/slo-service -n greenlang-slo
kubectl rollout status deployment/slo-service -n greenlang-slo
```

### Scenario 2: ImagePullBackOff

**Symptoms:** Pod status shows ImagePullBackOff or ErrImagePull

**Cause:** Container image not found, registry authentication failure, or tag mismatch.

**Resolution:**

1. **Check the image and pull errors:**

```bash
kubectl describe pod -n greenlang-slo <pod-name> | grep -A5 "Events"
```

2. **Verify image exists in registry:**

```bash
# Check current image tag
kubectl get deployment slo-service -n greenlang-slo -o jsonpath='{.spec.template.spec.containers[0].image}'

# Verify the image exists
aws ecr describe-images --repository-name greenlang/slo-service --image-ids imageTag=latest
```

3. **Fix image tag if needed:**

```bash
kubectl set image deployment/slo-service slo-service=greenlang/slo-service:<correct-tag> -n greenlang-slo
```

### Scenario 3: Database Connectivity Failure

**Symptoms:** Logs show "connection refused" to PostgreSQL, init container stuck, or "relation slo.definitions does not exist"

**Cause:** PostgreSQL is down, network policy blocking traffic, connection string incorrect, or database migration V020 has not been applied.

**Resolution:**

1. **Verify PostgreSQL is running:**

```bash
kubectl get pods -n database -l app=postgresql
kubectl get svc -n database | grep postgresql
```

2. **Check network policies:**

```bash
kubectl get networkpolicy -n greenlang-slo
kubectl describe networkpolicy slo-service-egress -n greenlang-slo
```

3. **Test connectivity from a debug pod:**

```bash
kubectl run debug --rm -it --image=postgres:14 -n greenlang-slo --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" -c "SELECT 1"
```

4. **Check if the slo schema and tables exist:**

```bash
kubectl run debug --rm -it --image=postgres:14 -n greenlang-slo --restart=Never -- \
  psql "postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang" \
  -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'slo';"
```

If tables are missing, apply the V020 migration:

```bash
# Check Flyway migration status
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  info -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang

# Apply pending migrations
kubectl run flyway --rm -it --image=flyway/flyway:10 -n database --restart=Never -- \
  migrate -url=jdbc:postgresql://greenlang-postgresql.database.svc.cluster.local:5432/greenlang
```

5. **After PostgreSQL is restored or migration is applied, restart the SLO service:**

```bash
kubectl rollout restart deployment/slo-service -n greenlang-slo
kubectl rollout status deployment/slo-service -n greenlang-slo
```

### Scenario 4: Redis Connectivity Failure

**Symptoms:** Logs show "connection refused" to Redis, error budget cache lookups failing

**Cause:** Redis cluster is down or network policy is blocking traffic.

**Resolution:**

1. **Verify Redis is running:**

```bash
kubectl get pods -n redis -l app=redis
kubectl get svc -n redis | grep redis
```

2. **Test connectivity:**

```bash
kubectl run redis-test --rm -it --image=redis:7 -n greenlang-slo --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING
```

3. **Note:** Redis is used for caching error budgets (TTL 60s). The SLO service should degrade gracefully if Redis is unavailable, falling back to direct Prometheus queries. If the service is crashing instead of degrading, this indicates a bug. Restart and monitor:

```bash
kubectl rollout restart deployment/slo-service -n greenlang-slo
```

### Scenario 5: Prometheus Unreachable

**Symptoms:** Logs show Prometheus connection errors, SLI queries timing out, evaluation duration increasing

**Cause:** Prometheus is down, network policy blocking traffic, or Prometheus URL misconfigured.

**Resolution:**

1. **Verify Prometheus is running:**

```bash
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus
```

2. **Check the configured Prometheus URL:**

```bash
kubectl get configmap slo-service-config -n greenlang-slo -o yaml | grep prometheus_url
```

3. **Test connectivity:**

```bash
kubectl run prom-test --rm -it --image=busybox:1.36 -n greenlang-slo --restart=Never -- \
  sh -c 'wget -q -O- http://prometheus-server.monitoring.svc.cluster.local:9090/api/v1/query?query=up | head -c 200'
```

4. **If Prometheus is down, follow the Prometheus runbooks first, then restart SLO service:**

```bash
kubectl rollout restart deployment/slo-service -n greenlang-slo
```

---

## Interim Mitigation

While the SLO Service is being restored, the following still function independently:

1. **Existing Prometheus recording rules continue evaluating.** Recording rules that were previously generated and loaded by Prometheus will continue to compute SLI ratios. These are not affected by the SLO service being down.

2. **Existing Prometheus alert rules continue firing.** Multi-window burn rate alerts that were previously generated will continue to fire through Alertmanager and OBS-004. No action is needed.

3. **Check current SLO status directly via Prometheus:**

```promql
# Check SLI ratios via recording rules (if they exist)
greenlang:slo:api_gateway_availability_30d:ratio

# Check error budget remaining
greenlang:slo:api_gateway_error_budget_30d:remaining

# Check burn rates
greenlang:slo:api_gateway_burn_rate_fast:ratio
greenlang:slo:api_gateway_burn_rate_medium:ratio
greenlang:slo:api_gateway_burn_rate_slow:ratio
```

4. **Manual error budget check (if recording rules are not available):**

```promql
# Example: API Gateway availability SLI over 30 days
1 - (
  sum(rate(http_requests_total{job="api-gateway", code=~"5.."}[30d]))
  /
  sum(rate(http_requests_total{job="api-gateway"}[30d]))
)
```

5. **Monitor that the gap does not extend beyond 30 minutes.** If the SLO service is down for more than 30 minutes, create an incident per the escalation path below, as error budget snapshot data will have a significant gap.

---

## Verification Steps

After restoring the SLO service, verify full recovery:

```bash
# 1. Check all pods are running and ready
kubectl get pods -n greenlang-slo -l app=slo-service

# 2. Check the health endpoint
kubectl port-forward -n greenlang-slo svc/slo-service 8080:8080
curl -s http://localhost:8080/api/v1/slos/health | python3 -m json.tool

# 3. Verify SLO evaluations are resuming
kubectl logs -n greenlang-slo -l app=slo-service --tail=50 | grep -i "evaluation"
```

```promql
# 4. Verify evaluations are incrementing
increase(gl_slo_evaluations_total[5m]) > 0

# 5. Verify error budget snapshots are being recorded
increase(gl_slo_budget_snapshots_total[5m]) > 0

# 6. Verify the SLO service's own health metric
up{job="slo-service"} == 1
```

```bash
# 7. Verify the SLO overview API returns data
curl -s http://localhost:8080/api/v1/slos/overview | python3 -m json.tool

# 8. Check Grafana SLO Overview dashboard is refreshing
# Open https://grafana.greenlang.io/d/slo-overview
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | SLO service down, existing alerts still firing | On-call engineer | Immediate (<5 min) |
| L2 | SLO service down > 30 minutes, budget data gap | Platform team lead + #observability | 15 minutes |
| L3 | SLO service and Prometheus both down (no SLO monitoring at all) | Platform team + SRE lead + CTO notification | Immediate |
| L4 | SLO service down due to database/infrastructure failure affecting multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** SLO Service Health (`/d/slo-service-health`)
- **Alert:** `SLOServiceDown` (this alert)
- **Key metrics to watch:**
  - `up{job="slo-service"}` (should always be >= 2)
  - `gl_slo_evaluations_total` rate (should be non-zero)
  - `gl_slo_evaluation_duration_seconds` P99 (should be < 5s)
  - `gl_slo_budget_snapshots_total` rate (should be non-zero)
  - Pod restart count (should be 0)

### Capacity Planning

1. **Maintain minimum 2 replicas** across different availability zones
2. **PDB ensures at least 1 pod** is available during disruptions
3. **HPA scales to 4** for high evaluation load
4. **Resource requests** are sized for evaluating 50+ SLOs every 60 seconds

### Configuration Best Practices

- Always validate SLO definitions YAML before applying ConfigMap changes
- Use ESO for secrets rotation (database URL, Grafana API key)
- Test configuration changes in staging before production
- Validate generated recording rules with `promtool check rules` before applying

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 SLO service incident
- **Related alerts:** `SLOEvaluationFailing`, `RecordingRuleGenerationFailed`, `BudgetSnapshotStale`
- **Related dashboards:** SLO Overview, SLO Service Health
- **Related runbooks:** [Error Budget Exhausted](./error-budget-exhausted.md), [High Burn Rate](./high-burn-rate.md), [SLO Compliance Degraded](./slo-compliance-degraded.md)
