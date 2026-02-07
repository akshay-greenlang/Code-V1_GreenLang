# High Burn Rate

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `HighBurnRateFast` | Critical | Fast burn rate (1h/5m windows) exceeds 14.4x threshold for 5 minutes |
| `HighBurnRateMedium` | Warning | Medium burn rate (6h/30m windows) exceeds 6.0x threshold for 30 minutes |
| `HighBurnRateSlow` | Info | Slow burn rate (3d/6h windows) exceeds 1.0x threshold for 6 hours |

---

## Description

These alerts fire when the error budget for an SLO is being consumed faster than allowed. The alerts use the **Google SRE Book multi-window, multi-burn-rate** methodology, which provides high-precision alerting with minimal false positives while detecting issues across different time horizons.

### What is Burn Rate?

Burn rate is a unitless multiplier that indicates how fast the error budget is being consumed relative to the allowed rate:

```
Burn Rate = Actual Error Rate / Allowed Error Rate
```

A burn rate of 1.0 means the error budget is being consumed exactly at the rate that would exhaust it at the end of the window (e.g., 30 days). A burn rate of 14.4 means the budget is being consumed 14.4x faster than allowed, which would exhaust a 30-day budget in approximately 2 hours.

### Multi-Window Methodology

Each burn rate alert uses **two windows** to fire: a long window and a short window. The alert fires only when BOTH windows exceed the threshold simultaneously. This dual-window approach has two important properties:

- The **long window** ensures the alert is significant (not just a brief blip)
- The **short window** ensures the issue is currently ongoing (not just historical)

### Burn Rate Windows and Thresholds

| Tier | Long Window | Short Window | Threshold | Budget Exhaustion Time | Severity | Response Time |
|------|-------------|--------------|-----------|------------------------|----------|---------------|
| **Fast** | 1 hour | 5 minutes | 14.4x | ~2 hours | Critical | Immediate (<5 min) |
| **Medium** | 6 hours | 30 minutes | 6.0x | ~1 day | Warning | Within 30 min |
| **Slow** | 3 days | 6 hours | 1.0x | ~30 days (full window) | Info | Next business day |

### How the Thresholds are Derived

The thresholds are calculated based on how quickly each burn rate would exhaust a 30-day error budget:

```
Fast:   30d / 2h  = 720h / 50h  = 14.4x    (budget gone in 2 hours)
Medium: 30d / 5d  = 720h / 120h = 6.0x     (budget gone in 5 days)
Slow:   30d / 30d = 720h / 720h = 1.0x     (budget gone in 30 days)
```

The fast threshold (14.4x) also accounts for alerting on 5% budget consumption within a 1-hour detection window.

### Alert PromQL Expressions

```promql
# HighBurnRateFast (Critical)
# Fires when BOTH the 1-hour AND 5-minute error rates exceed 14.4x the allowed rate
(
  greenlang:slo:{service}_burn_rate_1h:ratio > 14.4
  and
  greenlang:slo:{service}_burn_rate_5m:ratio > 14.4
)

# HighBurnRateMedium (Warning)
# Fires when BOTH the 6-hour AND 30-minute error rates exceed 6.0x the allowed rate
(
  greenlang:slo:{service}_burn_rate_6h:ratio > 6.0
  and
  greenlang:slo:{service}_burn_rate_30m:ratio > 6.0
)

# HighBurnRateSlow (Info)
# Fires when BOTH the 3-day AND 6-hour error rates exceed 1.0x the allowed rate
(
  greenlang:slo:{service}_burn_rate_3d:ratio > 1.0
  and
  greenlang:slo:{service}_burn_rate_6h_slow:ratio > 1.0
)
```

---

## Impact Assessment

### HighBurnRateFast (Critical)

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | Service is actively failing NOW; users are experiencing errors or high latency |
| **Data Impact** | High | Error budget will exhaust in ~2 hours at current rate |
| **SLA Impact** | Critical | SLA breach imminent if not resolved within minutes |
| **Revenue Impact** | High | Active user-facing degradation |

### HighBurnRateMedium (Warning)

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Sustained degradation over hours; users are impacted but possibly intermittently |
| **Data Impact** | Medium | Error budget will exhaust in ~1 day at current rate |
| **SLA Impact** | High | SLA breach likely within 24 hours if trend continues |
| **Revenue Impact** | Medium | Ongoing reliability issues eroding user experience |

### HighBurnRateSlow (Info)

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Gradual erosion; users may notice increased error rates over days/weeks |
| **Data Impact** | Low | Error budget is being slowly consumed; may exhaust before window resets |
| **SLA Impact** | Medium | SLA breach possible by end of measurement window |
| **Revenue Impact** | Low | Subtle degradation that accumulates over time |

---

## Symptoms

### HighBurnRateFast

- `HighBurnRateFast` alert firing for a specific SLO
- `gl_slo_burn_rate{slo_name="...", window="fast"}` is above 14.4
- Grafana Burn Rate dashboard shows red on the fast window panel
- Error budget remaining percentage is dropping visibly in real-time (multiple percent per minute)
- Spike visible in error rate or latency metrics for the affected service
- Recent deployment or infrastructure change correlating with the spike
- User-facing errors or timeouts actively reported

### HighBurnRateMedium

- `HighBurnRateMedium` alert firing for a specific SLO
- `gl_slo_burn_rate{slo_name="...", window="medium"}` is above 6.0
- Sustained elevated error rate over the past 6 hours
- Error budget has dropped noticeably over the past few hours
- May correlate with a deployment from earlier in the day

### HighBurnRateSlow

- `HighBurnRateSlow` alert firing for a specific SLO
- `gl_slo_burn_rate{slo_name="...", window="slow"}` is above 1.0
- Error budget consumption trend shows steady decline over days
- No single incident visible, but cumulative errors are adding up
- May correlate with infrastructure drift, increased traffic, or subtle bugs

---

## Diagnostic Steps

### Step 1: Identify the Affected SLO and Service

```promql
# Find all high burn rate alerts currently firing
ALERTS{alertname=~"HighBurnRate.*"}

# Get the specific SLO and service
gl_slo_burn_rate > 1.0
```

```bash
# Query the SLO Service API for burn rate details
kubectl port-forward -n greenlang-slo svc/slo-service 8080:8080

curl -s http://localhost:8080/api/v1/slos/<slo-id>/burn-rate | python3 -m json.tool
```

### Step 2: Assess Current Error Budget Status

```promql
# Current error budget remaining
gl_slo_error_budget_remaining_percent{slo_name="<slo-name>"}

# Error budget consumption trend (last 6 hours)
gl_slo_error_budget_remaining_percent{slo_name="<slo-name>"}[6h]
```

```bash
# Get the error budget from the API
curl -s http://localhost:8080/api/v1/slos/<slo-id>/budget | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Budget remaining: {data.get('remaining_percent', 'N/A'):.2f}%\")
print(f\"Status: {data.get('status', 'N/A')}\")
print(f\"Burn rate (1h): {data.get('burn_rate_1h', 'N/A'):.2f}x\")
print(f\"Burn rate (6h): {data.get('burn_rate_6h', 'N/A'):.2f}x\")
print(f\"Burn rate (3d): {data.get('burn_rate_3d', 'N/A'):.2f}x\")
print(f\"Forecast exhaustion: {data.get('forecast_exhaustion_date', 'N/A')}\")
"
```

### Step 3: Investigate the Underlying SLI

The burn rate is driven by the SLI. Investigate what is causing the SLI to degrade.

**For Availability SLIs (most common):**

```promql
# Current error rate for the service
sum(rate(http_requests_total{job="<service>", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="<service>"}[5m]))

# Error rate broken down by endpoint
sum by (handler) (rate(http_requests_total{job="<service>", code=~"5.."}[5m])) /
sum by (handler) (rate(http_requests_total{job="<service>"}[5m]))

# Error rate broken down by status code
sum by (code) (rate(http_requests_total{job="<service>", code=~"[45].."}[5m]))
```

**For Latency SLIs:**

```promql
# P95/P99 latency
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="<service>"}[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{job="<service>"}[5m])) by (le))

# Latency broken down by endpoint
histogram_quantile(0.95, sum by (handler, le) (rate(http_request_duration_seconds_bucket{job="<service>"}[5m])))

# Percentage of requests above the SLO threshold
1 - (
  sum(rate(http_request_duration_seconds_bucket{job="<service>", le="<threshold>"}[5m]))
  /
  sum(rate(http_request_duration_seconds_count{job="<service>"}[5m]))
)
```

### Step 4: Correlate with Other Signals

**Check traces (OBS-003):**

```bash
# Search for error traces in the affected service
# Open Grafana Tempo: https://grafana.greenlang.io/explore -> Tempo data source
# Search: service.name = "<service>" AND status = ERROR
# Time range: last 1h (for fast burn) or last 6h (for medium burn)
```

**Check logs (OBS-009 Loki):**

```bash
# Search for error logs from the affected service
# Open Grafana Explore -> Loki data source
# LogQL: {namespace="<namespace>", app="<service>"} |= "error" | logfmt
```

**Check recent deployments:**

```bash
kubectl rollout history deployment/<service> -n <namespace>
kubectl describe deployment/<service> -n <namespace> | grep -A5 "NewReplicaSet"
```

**Check infrastructure metrics:**

```promql
# CPU throttling
sum(rate(container_cpu_cfs_throttled_seconds_total{namespace="<namespace>", pod=~"<service>.*"}[5m]))

# Memory pressure
container_memory_working_set_bytes{namespace="<namespace>", pod=~"<service>.*"} /
container_spec_memory_limit_bytes{namespace="<namespace>", pod=~"<service>.*"}

# Pod restarts
sum(increase(kube_pod_container_status_restarts_total{namespace="<namespace>", pod=~"<service>.*"}[1h]))

# Database connection pool
sum(gl_db_pool_active_connections{service="<service>"})
sum(gl_db_pool_pending_connections{service="<service>"})
```

**Check downstream dependencies:**

```promql
# Error rate of downstream services
sum(rate(http_requests_total{job="<downstream-service>", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="<downstream-service>"}[5m]))
```

### Step 5: Determine the Root Cause Category

Common root causes by burn rate tier:

**Fast burn (service is failing NOW):**
- Bad deployment (most common) -- check recent rollouts
- Infrastructure failure (database down, Redis down, node failure)
- External dependency outage (third-party API)
- Traffic spike beyond capacity
- Certificate expiry
- Configuration change (bad ConfigMap, expired secret)

**Medium burn (sustained degradation):**
- Partial infrastructure failure (one replica unhealthy, connection pool exhaustion)
- Resource constraints (CPU throttling, memory pressure)
- Slow memory leak leading to GC pauses
- Gradual increase in error responses from a dependency
- Deployment with a subtle bug affecting a subset of requests

**Slow burn (gradual erosion):**
- Slowly increasing latency due to data growth (database table bloat, cache eviction)
- Intermittent errors from flaky dependencies
- Growing traffic without proportional scaling
- Disk space slowly filling
- Certificate approaching expiry causing intermittent TLS failures

---

## Resolution Steps

### HighBurnRateFast (Critical) -- Immediate Response

**Time constraint:** Budget exhausts in ~2 hours. Act immediately.

1. **Acknowledge the alert** in PagerDuty/Opsgenie

2. **If caused by a recent deployment, rollback immediately:**

```bash
# Check when the last deployment occurred
kubectl rollout history deployment/<service> -n <namespace>

# Rollback to previous version
kubectl rollout undo deployment/<service> -n <namespace>

# Monitor rollback progress
kubectl rollout status deployment/<service> -n <namespace>
```

3. **If not deployment-related, check infrastructure:**

```bash
# Check pod health
kubectl get pods -n <namespace> -l app=<service>

# Check for OOMKilled or CrashLoopBackOff
kubectl get pods -n <namespace> -l app=<service> -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Restart unhealthy pods
kubectl rollout restart deployment/<service> -n <namespace>
```

4. **If caused by traffic spike, scale up:**

```bash
# Scale the deployment
kubectl scale deployment/<service> -n <namespace> --replicas=<higher-count>

# Or adjust the HPA
kubectl patch hpa <service>-hpa -n <namespace> -p '{"spec":{"maxReplicas": <higher-max>}}'
```

5. **If caused by downstream dependency failure, enable circuit breaker or fallback:**

```bash
# Check logs for dependency errors
kubectl logs -n <namespace> -l app=<service> --tail=200 | grep -i "timeout\|connection refused\|circuit breaker"
```

6. **Verify burn rate is decreasing after mitigation:**

```promql
# Watch the 5-minute burn rate (should start dropping immediately)
gl_slo_burn_rate{slo_name="<slo-name>", window="fast"}
```

### HighBurnRateMedium (Warning) -- Sustained Investigation

**Time constraint:** Budget exhausts in ~1 day. Investigate within 30 minutes.

1. **Acknowledge the alert** in PagerDuty/Opsgenie

2. **Identify the specific failure pattern:**

```promql
# Is the error rate constant or intermittent?
sum(rate(http_requests_total{job="<service>", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="<service>"}[5m]))

# Is it concentrated on specific endpoints?
topk(5, sum by (handler) (rate(http_requests_total{job="<service>", code=~"5.."}[30m])))
```

3. **Check resource utilization:**

```bash
kubectl top pods -n <namespace> -l app=<service>

# Check if any pod is unhealthy
kubectl get pods -n <namespace> -l app=<service> -o wide
```

4. **If partial failure, restart unhealthy pods:**

```bash
# Delete the specific unhealthy pod (it will be recreated by the ReplicaSet)
kubectl delete pod <unhealthy-pod-name> -n <namespace>
```

5. **If resource-related, adjust limits:**

```bash
kubectl patch deployment <service> -n <namespace> -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "<container>",
            "resources": {
              "limits": {
                "cpu": "2",
                "memory": "2Gi"
              },
              "requests": {
                "cpu": "1",
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

6. **Verify the medium burn rate is decreasing:**

```promql
gl_slo_burn_rate{slo_name="<slo-name>", window="medium"}
```

### HighBurnRateSlow (Info) -- Planned Investigation

**Time constraint:** Budget may exhaust by end of window. Investigate within next business day.

1. **Acknowledge the alert**

2. **Create a tracking ticket** for the reliability investigation

3. **Analyze the trend over the past 7 days:**

```promql
# SLI trend over 7 days
avg_over_time(greenlang:slo:{service}_{sli_type}_{window}:ratio[7d])

# Error budget consumption trend
gl_slo_error_budget_remaining_percent{slo_name="<slo-name>"}[7d]
```

4. **Common investigations for slow burns:**

```bash
# Check database performance (slow queries, table bloat)
kubectl exec -n database <postgres-pod> -- psql -c "
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 10;
"

# Check cache hit rates
kubectl exec -n redis <redis-pod> -- redis-cli INFO stats | grep -E "keyspace_hits|keyspace_misses"

# Check disk space
kubectl exec -n <namespace> <pod> -- df -h
```

5. **Plan remediation** (may require a sprint item):
   - Database query optimization or index addition
   - Cache tuning (TTL, eviction policy)
   - Horizontal scaling
   - Code optimization for hot paths
   - Dependency upgrade or replacement

---

## Post-Incident: Should the SLO Target Be Adjusted?

After resolving a high burn rate incident, consider whether the SLO target itself needs adjustment. This is NOT about lowering standards -- it is about setting targets that are achievable, meaningful, and aligned with user expectations.

### Signs the SLO Target is Too Tight

- The SLO is consistently breached despite good engineering practices
- The burn rate slow alert fires frequently during normal operations
- The error budget is exhausted every measurement window
- The team is constantly in "firefighting mode" due to SLO alerts
- User satisfaction surveys do not correlate with SLO breaches (users are happy but SLO says otherwise)

### Signs the SLO Target is Too Loose

- The error budget is never consumed below 80%
- Burn rate alerts never fire, even during known incidents
- Users complain about reliability despite the SLO showing "healthy"
- The SLO does not drive meaningful reliability improvements

### How to Adjust

```bash
# View the current SLO definition
curl -s http://localhost:8080/api/v1/slos/<slo-id> | python3 -m json.tool

# Update the SLO target (creates a new version in the history)
curl -X PUT http://localhost:8080/api/v1/slos/<slo-id> \
  -H "Content-Type: application/json" \
  -d '{
    "target": 99.5,
    "change_description": "Adjusted from 99.9% to 99.5% based on Q1 2026 review. Current infrastructure supports 99.7% reliably."
  }'

# The SLO service will automatically regenerate recording rules and alert rules
```

### Adjustment Process

1. Propose the change in the quarterly SLO review (see [SLO Compliance Degraded](./slo-compliance-degraded.md))
2. Gather data: actual SLI values over the past 90 days, user feedback, business requirements
3. Get approval from the SLO owner (team lead) and the SRE team
4. Apply the change and document the rationale in the SLO version history
5. Communicate the change to stakeholders

---

## Prevention

### Monitoring

- **Dashboard:** SLO Overview (`/d/slo-overview`) -- burn rate panels
- **Dashboard:** Burn Rate Dashboard (`/d/slo-burn-rate`) -- detailed burn rate visualization
- **Alerts:** `HighBurnRateFast`, `HighBurnRateMedium`, `HighBurnRateSlow` (these alerts)
- **Key metrics:**
  - `gl_slo_burn_rate{window="fast"}` (should be < 14.4)
  - `gl_slo_burn_rate{window="medium"}` (should be < 6.0)
  - `gl_slo_burn_rate{window="slow"}` (should be < 1.0)
  - `gl_slo_error_budget_remaining_percent` (should be > 20%)

### Best Practices

1. **Act on fast burn alerts within 5 minutes.** These represent active service failures. Every minute of delay costs error budget.
2. **Correlate burn rate spikes with deployments.** Most fast burn events are caused by bad deploys. Implement automated canary analysis to catch these before full rollout.
3. **Use medium burn alerts as a 24-hour warning.** These give you time to investigate and fix without panic, but do not ignore them.
4. **Treat slow burn alerts as reliability debt.** They indicate systemic issues that need planned engineering work.
5. **Review burn rate history in quarterly SLO reviews.** Frequent burn rate alerts indicate a reliability gap that needs architectural attention.
6. **Test your alerting chain.** Ensure fast burn alerts actually page on-call engineers via PagerDuty/Opsgenie.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 SLO incident
- **Related alerts:** `ErrorBudgetExhausted`, `ErrorBudgetCritical`, `ErrorBudgetWarning`, `SLOServiceDown`
- **Related dashboards:** SLO Overview, Burn Rate Dashboard, Error Budget Deep Dive
- **Related runbooks:** [SLO Service Down](./slo-service-down.md), [Error Budget Exhausted](./error-budget-exhausted.md), [SLO Compliance Degraded](./slo-compliance-degraded.md)

---

## References

- [Google SRE Book - Chapter 5: Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/)
- [Google SRE Workbook - Multi-Window Multi-Burn-Rate Alerts](https://sre.google/workbook/alerting-on-slos/#6-multiwindow-multi-burn-rate-alerts)
- [GreenLang Performance SLOs](../../deployment/monitoring/slo/PERFORMANCE_SLOS.md)
- [OBS-005 PRD: SLO/SLI Definitions & Error Budget Management](../../GreenLang%20Development/05-Documentation/PRD-OBS-005-SLO-SLI-Definitions.md)
