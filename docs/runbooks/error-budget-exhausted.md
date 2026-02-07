# Error Budget Exhausted

## Alert

**Alert Name:** `ErrorBudgetExhausted`

**Severity:** Critical

**Threshold:** `gl_slo_error_budget_remaining_percent{} <= 0` for 5 minutes

**Duration:** 5 minutes

---

## Description

This alert fires when the error budget for any Service Level Objective (SLO) has been completely consumed (0% remaining). Error budget exhaustion means the service has exceeded its allowed error rate for the entire measurement window (typically 30 days).

### What is an Error Budget?

An error budget represents the maximum amount of unreliability a service is allowed before violating its SLO. It is calculated as:

```
Error Budget = (1 - SLO Target) x Window Duration
```

**Examples:**

| SLO Target | Window | Error Budget (minutes) | Error Budget (human-readable) |
|------------|--------|------------------------|-------------------------------|
| 99.99% | 30 days | 4.32 minutes | ~4 minutes of downtime |
| 99.9% | 30 days | 43.2 minutes | ~43 minutes of downtime |
| 99.5% | 30 days | 216 minutes | ~3.6 hours of downtime |
| 99.0% | 30 days | 432 minutes | ~7.2 hours of downtime |

When the error budget reaches 0%, the service has consumed all allowed unreliability for the current window. The budget will only recover as older errors age out of the rolling window.

### What Happens When the Budget is Exhausted?

The SLO Service (OBS-005) enforces the following **Error Budget Exhaustion Policy** (configurable per SLO):

1. **Deployment freeze** (default for critical-tier SLOs): No new feature deployments are permitted. Only reliability fixes and rollbacks are allowed.
2. **Alert-only** (default for standard-tier SLOs): Critical alert fires but no automated actions are taken.
3. **None** (best-effort tier): Logged only.

The deployment freeze is enforced via the CI/CD pipeline (`slo-service-ci.yml`) which queries the SLO Service API before allowing deployments to proceed.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Critical | SLO is breached; users are experiencing more errors/latency than promised |
| **Data Impact** | Medium | Error budget exhaustion is recorded in TimescaleDB for compliance reporting |
| **SLA Impact** | Critical | If SLA is tied to this SLO, the organization may owe contractual penalties or credits |
| **Revenue Impact** | High | Sustained poor reliability erodes user trust and may trigger SLA credits |
| **Compliance Impact** | High | SOC 2 CC7.2 (monitoring) and availability controls may be triggered; auditors will review this period |
| **Development Impact** | High | Deployment freeze may be active, blocking feature releases until budget recovers |

---

## Symptoms

- `ErrorBudgetExhausted` alert firing for one or more SLOs
- `gl_slo_error_budget_remaining_percent{slo_name="..."}` is at or below 0
- SLO Overview Grafana dashboard shows red/exhausted status for the affected SLO
- CI/CD pipeline rejecting deployments with "error budget exhausted" message (if deployment freeze is active)
- Burn rate gauges showing elevated values (the cause of the exhaustion)
- `gl_slo_alerts_fired_total{type="budget_exhausted"}` counter incrementing
- SLO compliance report showing the affected SLO as "breached"

---

## Diagnostic Steps

### Step 1: Identify Which SLO is Exhausted

```promql
# Find all SLOs with exhausted budgets
gl_slo_error_budget_remaining_percent <= 0

# Get the SLO name and service
gl_slo_error_budget_remaining_percent{} <= 0
```

```bash
# Query the SLO Service API for exhausted budgets
kubectl port-forward -n greenlang-slo svc/slo-service 8080:8080

curl -s http://localhost:8080/api/v1/slos/budgets | python3 -c "
import sys, json
data = json.load(sys.stdin)
for slo in data.get('budgets', []):
    if slo.get('remaining_percent', 100) <= 0:
        print(f\"EXHAUSTED: {slo['slo_name']} ({slo['service']}) - Budget: {slo['remaining_percent']:.2f}%\")
"
```

### Step 2: Understand the SLO and Its SLI

```bash
# Get full SLO details
curl -s http://localhost:8080/api/v1/slos/<slo-id> | python3 -m json.tool
```

Key fields to note:
- **target**: What percentage was promised (e.g., 99.9%)
- **sli.sli_type**: What is being measured (availability, latency, correctness, throughput, freshness)
- **window**: The measurement window (e.g., 30d rolling)
- **sli.good_query / sli.total_query**: The underlying PromQL queries

### Step 3: Determine When the Budget Was Exhausted

```promql
# Error budget consumption over time (look for the moment it crossed 0%)
gl_slo_error_budget_remaining_percent{slo_name="<slo-name>"}

# Check the burn rate that caused the exhaustion
gl_slo_burn_rate{slo_name="<slo-name>", window="fast"}
gl_slo_burn_rate{slo_name="<slo-name>", window="medium"}
gl_slo_burn_rate{slo_name="<slo-name>", window="slow"}
```

```bash
# Query error budget history from TimescaleDB via the API
curl -s "http://localhost:8080/api/v1/slos/<slo-id>/budget/history?days=7" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for snapshot in data.get('snapshots', [])[-10:]:
    print(f\"{snapshot['snapshot_time']}: {snapshot['remaining_percent']:.2f}% remaining, burn_1h={snapshot.get('burn_rate_1h', 'N/A')}\")
"
```

### Step 4: Identify the Root Cause

The error budget was exhausted because the SLI dropped below the target for a sustained period. Investigate what caused the SLI degradation:

**For Availability SLIs:**

```promql
# Error rate over the last 24 hours
sum(rate(http_requests_total{job="<service>", code=~"5.."}[1h])) /
sum(rate(http_requests_total{job="<service>"}[1h]))

# When did the error rate spike?
sum(rate(http_requests_total{job="<service>", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="<service>"}[5m]))
```

**For Latency SLIs:**

```promql
# P95 latency over time
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="<service>"}[5m])) by (le))

# Percentage of requests exceeding the threshold
1 - (
  sum(rate(http_request_duration_seconds_bucket{job="<service>", le="<threshold>"}[5m]))
  /
  sum(rate(http_request_duration_seconds_count{job="<service>"}[5m]))
)
```

**For Correctness SLIs:**

```promql
# Correctness rate
sum(rate(gl_processing_results_total{job="<service>", result="correct"}[5m])) /
sum(rate(gl_processing_results_total{job="<service>"}[5m]))
```

### Step 5: Correlate with Recent Changes and Incidents

```bash
# Check recent deployments to the affected service
kubectl rollout history deployment/<service> -n <namespace>

# Check recent incidents
# Look at OBS-004 alerts for the affected service
curl -s http://localhost:8080/api/v1/slos/<slo-id>/compliance | python3 -m json.tool

# Check git log for recent changes
git log --oneline --since="7 days ago" -- <service-path>
```

```promql
# Check if there was a specific incident that consumed most of the budget
# Look for sudden spikes in error rate
sum(rate(http_requests_total{job="<service>", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="<service>"}[5m]))
```

---

## Response Actions

### Immediate Actions (First 15 Minutes)

1. **Acknowledge the alert** in PagerDuty/Opsgenie

2. **Confirm the deployment freeze is active** (for critical-tier SLOs):

```bash
# Check if the SLO service has enforced deployment freeze
curl -s http://localhost:8080/api/v1/slos/<slo-id>/budget | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Status: {data.get('status')}\")
print(f\"Policy Action: {data.get('policy_action', 'none')}\")
print(f\"Deployment Freeze: {data.get('deployment_freeze', False)}\")
"
```

3. **Communicate to stakeholders** using the communication template below

4. **Create an incident** in your incident management system:
   - Severity: P2 (or P1 if SLA-backed)
   - Summary: "SLO breached: [SLO Name] error budget exhausted"
   - Service: [Affected service]
   - Impact: [Description of user impact]

### Short-Term Actions (First Hour)

5. **Identify and mitigate the root cause.** If the service is actively degraded:

```bash
# Check if the service is currently healthy
kubectl get pods -n <namespace> -l app=<service>

# Rollback the most recent deployment if it correlates with budget exhaustion
kubectl rollout undo deployment/<service> -n <namespace>

# Check resource constraints
kubectl top pods -n <namespace> -l app=<service>
```

6. **If the issue is resolved but the budget is already exhausted,** the budget will recover naturally as older errors age out of the rolling window. No manual budget reset is possible (by design -- this ensures SLO integrity).

### Recovery Timeline

The error budget recovers as the rolling window moves forward and old errors fall out:

| SLO Target | Time to Recover 10% Budget | Time to Recover 50% Budget |
|------------|---------------------------|---------------------------|
| 99.99% | ~3 days of perfect uptime | ~15 days of perfect uptime |
| 99.9% | ~3 days of perfect uptime | ~15 days of perfect uptime |
| 99.5% | ~3 days of perfect uptime | ~15 days of perfect uptime |

The key insight is: the budget is calculated over a **rolling window**. As long as the service is now healthy, the errors that exhausted the budget will gradually roll out of the window.

```promql
# Monitor budget recovery
gl_slo_error_budget_remaining_percent{slo_name="<slo-name>"}
```

### Post-Incident: Root Cause Analysis

7. **Conduct a post-incident review** within 5 business days. Address:

   - What specific events consumed the error budget?
   - Was there a single large incident or gradual degradation?
   - Could the degradation have been detected earlier?
   - Were burn rate alerts (fast/medium/slow) firing before the budget exhausted?
   - Were those alerts actioned promptly?
   - What preventive measures can be put in place?

8. **Determine if the SLO target needs adjustment:**

   - If the SLO is consistently being breached, the target may be unrealistically tight
   - If the SLO is trivially easy to meet, the target may be too loose
   - Follow the SLO Review Process in the [SLO Compliance Degraded](./slo-compliance-degraded.md) runbook

---

## Stakeholder Communication Template

Use the following template when communicating about an exhausted error budget:

```
Subject: [SLO BREACH] {Service Name} - Error Budget Exhausted

Team,

The error budget for the following SLO has been exhausted:

  SLO Name:      {SLO Name}
  Service:       {Service Name}
  SLO Target:    {Target}% {SLI Type}
  Current SLI:   {Current Value}%
  Window:        {Window} rolling
  Budget Status: EXHAUSTED (0% remaining)

IMPACT:
  - {Description of user impact}
  - Deployment freeze is {ACTIVE / NOT ACTIVE} for this service

ROOT CAUSE:
  - {Brief root cause or "Under investigation"}
  - Primary incident: {Incident ID if applicable}

ACTIONS TAKEN:
  - {Action 1}
  - {Action 2}

RECOVERY TIMELINE:
  - Error budget will begin recovering as errors age out of the {Window} window
  - Estimated return to healthy budget: {Date/time estimate}
  - Deployment freeze will be lifted when budget reaches {threshold}%

NEXT STEPS:
  - Post-incident review scheduled for {Date}
  - Reliability improvements: {Brief list}

Please direct questions to #{incident-channel}.

-- {On-call engineer name}
```

---

## Deployment Freeze Procedures

When a deployment freeze is active due to budget exhaustion:

### What IS Allowed

- Bug fixes that directly improve the SLI (e.g., fixing the root cause of errors)
- Rollbacks to a known-good version
- Infrastructure changes that improve reliability (scaling, resource increases)
- Configuration changes that fix the root cause
- Security patches

### What is NOT Allowed

- New feature deployments
- Refactoring that does not directly improve the SLI
- Dependency upgrades that are not security-related
- Non-essential configuration changes

### Lifting the Freeze

The deployment freeze is lifted when the error budget recovers above the threshold:

```bash
# Check current budget status
curl -s http://localhost:8080/api/v1/slos/<slo-id>/budget | python3 -c "
import sys, json
data = json.load(sys.stdin)
remaining = data.get('remaining_percent', 0)
status = data.get('status', 'unknown')
print(f'Budget remaining: {remaining:.2f}%')
print(f'Status: {status}')
print(f'Freeze active: {remaining <= 0}')
print(f'Freeze lifts when: remaining > 0% (budget must recover)')
"
```

The CTO may override the deployment freeze with explicit approval. Document the override reason in the incident timeline.

---

## Prevention

### Monitoring

- **Dashboard:** SLO Overview (`/d/slo-overview`)
- **Dashboard:** Error Budget Deep Dive (`/d/slo-error-budget`)
- **Alert:** `ErrorBudgetExhausted` (this alert)
- **Related alerts:**
  - `ErrorBudgetCritical` (budget < 50%) -- fires hours/days before exhaustion
  - `ErrorBudgetWarning` (budget < 80%) -- fires days/weeks before exhaustion
  - `HighBurnRateFast` -- fires when budget is being consumed at 14.4x the allowed rate
  - `HighBurnRateMedium` -- fires when budget is being consumed at 6x the allowed rate

### Key Metrics to Watch

```promql
# Error budget remaining (catch before it hits 0)
gl_slo_error_budget_remaining_percent

# Burn rate (early warning of budget consumption)
gl_slo_burn_rate{window="fast"}
gl_slo_burn_rate{window="medium"}
gl_slo_burn_rate{window="slow"}

# SLI value trend
greenlang:slo:{service}_{sli_type}_{window}:ratio
```

### Best Practices

1. **Act on burn rate alerts BEFORE the budget exhausts.** Fast burn (critical) means the budget will exhaust in ~2 hours. Medium burn (warning) means ~1 day. Slow burn (info) means gradual erosion over 30 days.
2. **Set up error budget burn-down alerts** at 80% and 50% consumed to provide early warning.
3. **Conduct quarterly SLO reviews** to ensure targets are realistic and aligned with business needs.
4. **Include SLO impact in deployment risk assessments** -- high-risk deployments should consider remaining error budget.
5. **Automate rollbacks** for deployments that cause SLI degradation within the first 15 minutes.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any error budget exhaustion event
- **Related alerts:** `ErrorBudgetCritical`, `ErrorBudgetWarning`, `HighBurnRateFast`, `HighBurnRateMedium`
- **Related dashboards:** SLO Overview, Error Budget Deep Dive
- **Related runbooks:** [SLO Service Down](./slo-service-down.md), [High Burn Rate](./high-burn-rate.md), [SLO Compliance Degraded](./slo-compliance-degraded.md)
