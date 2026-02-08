# Observability Agent SLO Breach

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `ObsAgentSLOBurning` | Warning | SLO compliance below 99.9% target for 15 minutes |
| `ObsAgentErrorBudgetLow` | Warning | Error budget remaining below 10% |
| `ObsAgentErrorBudgetExhausted` | Critical | Error budget remaining below 0% (exhausted) |

**Thresholds:**

```promql
# ObsAgentSLOBurning
# SLO compliance ratio drops below 99.9% target
gl_obs_slo_compliance_ratio < 0.999
# sustained for 15 minutes

# ObsAgentErrorBudgetLow
# Error budget remaining drops below 10%
gl_obs_error_budget_remaining < 0.10 and gl_obs_error_budget_remaining >= 0
# sustained for 15 minutes

# ObsAgentErrorBudgetExhausted
# Error budget is fully exhausted (negative remaining)
gl_obs_error_budget_remaining < 0
# sustained for 5 minutes
```

---

## Description

These alerts fire when the Observability & Telemetry Agent (AGENT-FOUND-010) SLO targets are being breached or the error budget is being consumed at an unsustainable rate. The Observability Agent is the centralized telemetry layer for all 47+ GreenLang Climate OS agents, and its reliability directly impacts the platform's ability to monitor, alert, and maintain compliance.

### How SLO Compliance Works

The Observability Agent maintains SLO compliance tracking using the Google SRE Book burn rate model:

1. **SLI Types** -- The agent tracks multiple Service Level Indicators:
   - **Availability SLI**: Percentage of successful health checks (`gl_obs_health_checks_total{status="healthy"}` / `gl_obs_health_checks_total`)
   - **Latency SLI**: Percentage of operations completing within the latency target (`gl_obs_operation_duration_seconds_bucket{le="1.0"}` / `gl_obs_operation_duration_seconds_count`)
   - **Error Rate SLI**: Percentage of operations completing without errors (1 - error_rate)
   - **Throughput SLI**: Minimum metrics recording rate is maintained

2. **SLO Targets** -- Each SLI has a defined target:
   - Availability: 99.9% (43.8 minutes of downtime per 30-day window)
   - Latency (p99): 99.0% of requests under 1 second
   - Error Rate: < 0.1% of operations result in errors

3. **Error Budget** -- The remaining allowance for failures within the SLO window:
   - For 99.9% availability over 30 days: 43.2 minutes of total allowed downtime
   - Budget is consumed by any period where the SLI is below target
   - When budget reaches 0%, the SLO is breached

4. **Burn Rate** -- How fast the error budget is being consumed:
   - **Fast burn (14.4x)**: Budget exhausted in 2 hours -- triggers immediate page
   - **Medium burn (6x)**: Budget exhausted in 5 hours -- triggers warning
   - **Slow burn (1x)**: Budget exhausted at window end -- triggers ticket

### Why SLO Breaches Matter

The Observability Agent SLO breach impacts the entire GreenLang platform:

- **Alert coverage gap**: If the agent is degraded, alert evaluations may fail, causing missed notifications for critical conditions in other services
- **SLO tracking disruption**: If the agent cannot track its own SLOs, it also cannot accurately track SLOs for other services, creating a cascading visibility gap
- **Compliance risk**: SOC 2 Type II requires continuous monitoring and evidence of operational controls; SLO breaches must be documented and remediated
- **Deployment freeze**: When error budget is exhausted, non-essential deployments to the observability pipeline should be frozen until reliability is restored

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium-High | Platform operators may experience delayed or missing alerts |
| **Data Impact** | Low | Telemetry data continues flowing to backends; SLO calculations may be stale |
| **SLA Impact** | High | Internal SLA for observability coverage violated |
| **Revenue Impact** | Medium | Enterprise customers with SLA requirements expect continuous monitoring |
| **Compliance Impact** | High | SOC 2 requires documented SLO tracking and breach remediation |
| **Downstream Impact** | High | SLO compliance tracking for all GreenLang agents may be affected |

---

## Symptoms

- `gl_obs_slo_compliance_ratio` is below the 99.9% target
- `gl_obs_error_budget_remaining` is below 10% or negative
- Grafana SLO dashboard shows declining compliance trend
- Error budget burn rate chart shows accelerated consumption
- `gl_obs_alerts_evaluated_total{result="error"}` rate is elevated
- `gl_obs_operation_duration_seconds` p99 is above 1 second
- `gl_obs_health_status` is intermittently below 1 (degraded)
- Multiple downstream service SLO dashboards show stale data

---

## Diagnostic Steps

### Step 1: Identify Which SLOs Are Breaching

```bash
# Check current SLO compliance via API
kubectl port-forward -n greenlang svc/observability-agent-service 8080:8080
curl -s http://localhost:8080/v1/slo/compliance \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool

# Check error budget status for all SLOs
curl -s http://localhost:8080/v1/slo/error-budget \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  | python3 -m json.tool
```

```promql
# Check SLO compliance for each SLI
gl_obs_slo_compliance_ratio

# Check error budget remaining
gl_obs_error_budget_remaining

# Check burn rate (how fast budget is being consumed)
# Fast burn window (1h)
1 - (
  sum(rate(gl_obs_operation_duration_seconds_count{status="success"}[1h]))
  / sum(rate(gl_obs_operation_duration_seconds_count[1h]))
) / (1 - 0.999)

# Medium burn window (6h)
1 - (
  sum(rate(gl_obs_operation_duration_seconds_count{status="success"}[6h]))
  / sum(rate(gl_obs_operation_duration_seconds_count[6h]))
) / (1 - 0.999)
```

### Step 2: Analyze the Burn Rate

```promql
# Determine the burn rate multiplier
# If burn_rate > 14.4 => budget exhausted in < 2 hours (critical)
# If burn_rate > 6    => budget exhausted in < 5 hours (warning)
# If burn_rate > 3    => budget exhausted in < 10 hours (warning)
# If burn_rate > 1    => budget will be exhausted before window ends (ticket)

# Fast burn rate (1h window, 5m short window)
(
  sum(rate(gl_obs_operation_duration_seconds_count{status!="success"}[1h]))
  / sum(rate(gl_obs_operation_duration_seconds_count[1h]))
) / (1 - 0.999)

# Medium burn rate (6h window, 30m short window)
(
  sum(rate(gl_obs_operation_duration_seconds_count{status!="success"}[6h]))
  / sum(rate(gl_obs_operation_duration_seconds_count[6h]))
) / (1 - 0.999)

# Slow burn rate (3d window, 6h short window)
(
  sum(rate(gl_obs_operation_duration_seconds_count{status!="success"}[3d]))
  / sum(rate(gl_obs_operation_duration_seconds_count[3d]))
) / (1 - 0.999)
```

### Step 3: Identify the Root Cause

```bash
# Check error breakdown by operation type
curl -s http://localhost:8080/metrics | grep gl_obs_operation_duration_seconds_count

# Check which operations are failing
curl -s http://localhost:8080/metrics | grep 'status="error"'

# Check health status of subsystems
curl -s http://localhost:8080/health \
  | python3 -m json.tool
```

```promql
# Error rate by operation type (identify the failing operation)
sum(rate(gl_obs_operation_duration_seconds_count{status="error"}[5m])) by (operation)
/ sum(rate(gl_obs_operation_duration_seconds_count[5m])) by (operation)

# Latency by operation type (identify the slow operation)
histogram_quantile(0.99,
  sum(rate(gl_obs_operation_duration_seconds_bucket[5m])) by (le, operation)
)

# Health check failure rate
sum(rate(gl_obs_health_checks_total{status!="healthy"}[5m]))
/ sum(rate(gl_obs_health_checks_total[5m]))

# Alert evaluation failure rate
sum(rate(gl_obs_alerts_evaluated_total{result="error"}[5m]))
/ sum(rate(gl_obs_alerts_evaluated_total[5m]))
```

### Step 4: Check Underlying Service Issues

```bash
# Check Prometheus health
kubectl get pods -n monitoring -l app=prometheus
kubectl logs -n monitoring -l app=prometheus --tail=50 | grep -i error

# Check Tempo (tracing) health
kubectl get pods -n monitoring -l app=tempo
kubectl logs -n monitoring -l app=tempo --tail=50 | grep -i error

# Check Loki (logging) health
kubectl get pods -n monitoring -l app=loki
kubectl logs -n monitoring -l app=loki --tail=50 | grep -i error

# Check PostgreSQL health
kubectl run pg-test --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  pg_isready -h greenlang-db.postgres.svc.cluster.local -p 5432

# Check Redis health
kubectl run redis-test --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 PING

# Check observability agent resource usage
kubectl top pods -n greenlang -l app=observability-agent-service
```

### Step 5: Check Recent Changes

```bash
# Check recent deployments
kubectl rollout history deployment/observability-agent-service -n greenlang

# Check recent events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -30

# Check if other services deployed recently that may be generating excessive telemetry
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep -i "deploy\|image\|rolling" | tail -20
```

---

## Resolution Steps

### Scenario 1: Availability SLO Breach -- Service Restarts

**Symptoms:** SLO compliance is dropping due to frequent service restarts or health check failures.

**Resolution:**

1. Check pod restart count and reasons:
```bash
kubectl get pods -n greenlang -l app=observability-agent-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'
```

2. If OOMKilled, increase memory limits (see [Service Down runbook](./observability-agent-service-down.md), Scenario 1).

3. If CrashLoopBackOff, check logs and resolve the root cause.

4. Verify stability:
```bash
kubectl rollout status deployment/observability-agent-service -n greenlang
# Monitor for 15+ minutes to confirm no further restarts
```

### Scenario 2: Latency SLO Breach -- Slow Operations

**Symptoms:** p99 latency exceeds 1 second, SLO compliance dropping.

**Resolution:**

1. Identify the slow operation:
```promql
histogram_quantile(0.99,
  sum(rate(gl_obs_operation_duration_seconds_bucket[5m])) by (le, operation)
)
```

2. Common causes and fixes:
   - **Prometheus remote write backpressure**: Check Prometheus WAL size and remote write queue. Increase `max_shards` or `capacity` in remote write config.
   - **Database query latency**: Check PostgreSQL slow query log. Add indexes or optimize queries.
   - **Loki ingestion backlog**: Check Loki ingester ring health. Scale up ingesters if needed.
   - **High metric cardinality**: Identify high-cardinality metrics via `count by (__name__)({__name__=~".+"})`. Drop or aggregate excessive label combinations.

3. If caused by resource saturation, scale up:
```bash
kubectl scale deployment observability-agent-service -n greenlang --replicas=4
```

### Scenario 3: Error Rate SLO Breach -- Backend Failures

**Symptoms:** Error rate exceeds 0.1%, operations failing against telemetry backends.

**Resolution:**

1. Identify which backend is failing:
```bash
curl -s http://localhost:8080/health | python3 -m json.tool
# Look at the "dependencies" section for unhealthy backends
```

2. Restore the failing backend:
```bash
# Example: Restart Prometheus
kubectl rollout restart deployment/prometheus-server -n monitoring

# Example: Restart Loki
kubectl rollout restart statefulset/loki -n monitoring

# Example: Restart Tempo
kubectl rollout restart statefulset/tempo -n monitoring
```

3. Verify error rate returns to normal:
```promql
sum(rate(gl_obs_operation_duration_seconds_count{status="error"}[5m]))
/ sum(rate(gl_obs_operation_duration_seconds_count[5m]))
```

### Scenario 4: Error Budget Exhausted -- Deployment Freeze

**Symptoms:** Error budget is at or below 0%, `ObsAgentErrorBudgetExhausted` is firing.

**Resolution:**

1. **Immediately freeze non-essential deployments** to the observability pipeline.

2. Identify and fix the root cause using Steps 1-5 from the diagnostic section above.

3. Document the SLO breach for compliance:
```bash
# Record the breach in the audit log
curl -s -X POST http://localhost:8080/v1/slo/breach-report \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "slo_name": "<breached_slo>",
    "breach_start": "<ISO8601_timestamp>",
    "root_cause": "<description>",
    "remediation": "<actions_taken>"
  }' | python3 -m json.tool
```

4. Only resume deployments after the error budget has recovered above 10%.

---

## Communication Procedures

### When to Communicate

| Trigger | Audience | Channel | Template |
|---------|----------|---------|----------|
| `ObsAgentSLOBurning` fires | Platform team | `#platform-foundation` | SLO Warning Template |
| `ObsAgentErrorBudgetLow` fires | Platform team + management | `#platform-foundation`, `#platform-oncall` | Budget Warning Template |
| `ObsAgentErrorBudgetExhausted` fires | All engineering + compliance | `#greenlang-incidents`, `#compliance-ops` | Budget Exhausted Template |

### SLO Warning Template

```
[SLO WARNING] Observability Agent SLO compliance below target

SLO: <slo_name>
Current compliance: <compliance_percentage>%
Target: 99.9%
Error budget remaining: <budget_percentage>%
Burn rate: <burn_rate>x

Investigation: <link_to_dashboard>
Runbook: https://docs.greenlang.ai/runbooks/observability-agent/slo-breach
```

### Budget Exhausted Template

```
[SLO BREACH] Observability Agent error budget exhausted

SLO: <slo_name>
Error budget: EXHAUSTED (<remaining>% remaining)
Breach started: <timestamp>
Impact: Alert evaluation and SLO tracking for all GreenLang services may be affected

ACTION REQUIRED:
1. Non-essential deployments to observability pipeline are FROZEN
2. Investigate root cause immediately
3. Compliance team notified (SOC 2 documentation required)

Dashboard: /d/obs-agent-svc
Runbook: https://docs.greenlang.ai/runbooks/observability-agent/slo-breach
On-call: <oncall_engineer>
```

---

## Post-Incident Review Template

After any SLO breach that exhausts the error budget, a post-incident review is required within 5 business days.

### Review Checklist

- [ ] **Timeline**: Document the full incident timeline from first alert to resolution
- [ ] **Root cause**: Identify the primary and contributing causes
- [ ] **Impact**: Quantify the impact (duration, error budget consumed, downstream effects)
- [ ] **Detection**: How was the issue detected? Was the detection time acceptable?
- [ ] **Response**: How quickly was the response initiated? Were the right people paged?
- [ ] **Resolution**: What steps resolved the issue? Were they documented in this runbook?
- [ ] **Prevention**: What changes will prevent recurrence?
- [ ] **Action items**: List specific follow-up tasks with owners and due dates

### Review Document Structure

```markdown
# Post-Incident Review: Observability Agent SLO Breach

**Date:** YYYY-MM-DD
**Incident ID:** INC-XXXX
**Severity:** P1 / P2
**Duration:** X hours Y minutes
**Error budget consumed:** X%

## Summary
<One paragraph summary of what happened>

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | First alert fired |
| HH:MM | On-call engineer acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Fix deployed |
| HH:MM | SLO compliance restored |

## Root Cause
<Description of the root cause>

## Impact
- Error budget consumed: X%
- Downstream services affected: <list>
- Alerts missed during degradation: <count>
- Compliance documentation required: Yes/No

## What Went Well
- <bullet points>

## What Could Be Improved
- <bullet points>

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| <action> | <owner> | YYYY-MM-DD | Open |

## Attendees
- <list of review participants>
```

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | SLO compliance below target, error budget above 50% | On-call engineer | 30 minutes |
| L2 | Error budget below 25%, burn rate > 6x | Platform team lead + #platform-foundation | 15 minutes |
| L3 | Error budget below 10%, burn rate > 14.4x | Platform team + compliance team | Immediate (<5 min) |
| L4 | Error budget exhausted, multiple service SLO tracking affected | All-hands engineering + incident commander + CTO notification | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Observability Agent SLO (`/d/obs-agent-svc`)
- **Alerts:** `ObsAgentSLOBurning`, `ObsAgentErrorBudgetLow`, `ObsAgentErrorBudgetExhausted`
- **Key metrics to watch:**
  - `gl_obs_slo_compliance_ratio` (should be >= 0.999)
  - `gl_obs_error_budget_remaining` (should be > 0.25 in steady state)
  - `gl_obs_operation_duration_seconds` p99 (should be < 1s)
  - `gl_obs_health_status` (should be 1)
  - `gl_obs_alerts_evaluated_total{result="error"}` rate (should be near 0)
  - `gl_obs_metrics_recorded_total` rate (should be stable and non-zero)
  - Error budget burn rate (should be < 1x in steady state)

### Proactive Measures

1. **Weekly SLO review**: Review SLO compliance trends weekly in the team standup
2. **Error budget policy**: Document and enforce the deployment freeze threshold (10% budget)
3. **Capacity planning**: Ensure the observability agent has headroom for metric cardinality growth
4. **Canary deployments**: Use canary releases for observability pipeline changes
5. **Chaos testing**: Periodically inject failures to validate alerting and SLO measurement
6. **Dependency health**: Monitor telemetry backend health (Prometheus, Tempo, Loki) independently
7. **Cardinality management**: Regularly review and prune high-cardinality metrics

### Related Alerts

| Alert | Severity | When It Fires |
|-------|----------|---------------|
| `ObsAgentSLOBurning` | Warning | SLO compliance below 99.9% target |
| `ObsAgentErrorBudgetLow` | Warning | Error budget remaining below 10% |
| `ObsAgentErrorBudgetExhausted` | Critical | Error budget fully exhausted |
| `ObsAgentHighErrorRate` | Warning | >5% error rate |
| `ObsAgentHighLatency` | Warning | p99 latency above 1s |
| `ObsAgentDown` | Critical | No pods running |
| `ObsAgentHealthDegraded` | Warning | Health status degraded |
| `ObsAgentHealthUnhealthy` | Critical | Health status unhealthy |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Foundation Team
- **Review cadence:** Quarterly or after any error budget exhaustion incident
- **Related runbooks:** [Observability Agent Service Down](./observability-agent-service-down.md)
