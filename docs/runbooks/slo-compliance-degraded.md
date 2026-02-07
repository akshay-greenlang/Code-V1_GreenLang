# SLO Compliance Degraded

## Alert

**Alert Name:** `SLOComplianceBelow95`

**Severity:** Warning

**Threshold:** `gl_slo_compliance_percent < 95` for 1 hour

**Duration:** 1 hour

---

## Description

This alert fires when the overall SLO compliance percentage drops below 95%. SLO compliance is defined as the percentage of all active SLO definitions that are currently meeting their targets. For example, if 16 SLOs are defined and 2 are breached, compliance is 87.5%.

This alert indicates a **systemic reliability issue** across the GreenLang platform. Unlike individual SLO alerts (HighBurnRateFast, ErrorBudgetExhausted) which focus on a single service, this alert fires when multiple services simultaneously fail to meet their reliability objectives.

### How SLO Compliance is Calculated

```
SLO Compliance % = (SLOs Meeting Target / Total Active SLOs) x 100
```

The SLO Service (OBS-005) evaluates all active SLO definitions at regular intervals (every 60 seconds). For each SLO, it queries Prometheus for the current SLI value and compares it against the SLO target. The compliance percentage is exposed as the `gl_slo_compliance_percent` gauge metric.

### Compliance Tiers

| Compliance | Status | Interpretation |
|------------|--------|----------------|
| 100% | Excellent | All SLOs are being met |
| 95-99% | Good | Minor degradation; 1-2 SLOs may be close to or slightly below target |
| 90-94% | Warning | Multiple SLOs are failing; reliability attention needed |
| 80-89% | Critical | Widespread reliability issues; multiple services affected |
| < 80% | Emergency | Platform-wide reliability failure; incident commander required |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | High | Multiple services are degraded; users may experience errors across the platform |
| **Data Impact** | Medium | SLO compliance data is recorded in TimescaleDB for audit and reporting |
| **SLA Impact** | High | Multiple SLA-backed services may be at risk of breach |
| **Revenue Impact** | High | Widespread reliability issues affect user trust and may trigger multiple SLA credits |
| **Compliance Impact** | High | SOC 2 CC7.2 (monitoring) requires demonstrating continuous SLO tracking and compliance |

---

## Symptoms

- `SLOComplianceBelow95` alert firing
- `gl_slo_compliance_percent` gauge is below 95
- SLO Overview Grafana dashboard shows multiple SLOs in warning/critical/exhausted status
- Multiple `ErrorBudgetWarning` or `ErrorBudgetCritical` alerts firing simultaneously
- Multiple services showing elevated error rates or latency on their respective dashboards
- Compliance reports showing declining trend over weeks
- Team retrospectives frequently referencing reliability issues

---

## Diagnostic Steps

### Step 1: Get the Current Compliance Overview

```promql
# Current compliance percentage
gl_slo_compliance_percent

# How many SLOs are meeting their target?
count(gl_slo_error_budget_remaining_percent > 0)

# How many SLOs have exhausted their budget?
count(gl_slo_error_budget_remaining_percent <= 0)
```

```bash
# Query the SLO Service API for a full overview
kubectl port-forward -n greenlang-slo svc/slo-service 8080:8080

curl -s http://localhost:8080/api/v1/slos/overview | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Overall Compliance: {data.get('compliance_percent', 'N/A'):.1f}%\")
print(f\"Total SLOs: {data.get('total_slos', 'N/A')}\")
print(f\"Meeting Target: {data.get('meeting_target', 'N/A')}\")
print(f\"Not Meeting Target: {data.get('not_meeting_target', 'N/A')}\")
print()
print('FAILING SLOs:')
for slo in data.get('slos', []):
    if not slo.get('met', True):
        print(f\"  - {slo['slo_name']} ({slo['service']}): SLI={slo.get('current_sli', 'N/A'):.3f}% target={slo.get('target', 'N/A')}% budget={slo.get('budget_remaining_percent', 'N/A'):.1f}%\")
"
```

### Step 2: Identify Which SLOs are Failing

```promql
# All SLOs currently below target (error budget at or below 0%)
gl_slo_error_budget_remaining_percent <= 0

# All SLOs in warning or critical status
gl_slo_error_budget_remaining_percent < 50

# SLOs sorted by remaining budget (worst first)
sort_desc(gl_slo_error_budget_remaining_percent)
```

```bash
# Get detailed breakdown from the API
curl -s http://localhost:8080/api/v1/slos/budgets | python3 -c "
import sys, json
data = json.load(sys.stdin)
budgets = sorted(data.get('budgets', []), key=lambda x: x.get('remaining_percent', 100))
for b in budgets:
    status_icon = 'OK' if b.get('remaining_percent', 0) > 50 else 'WARN' if b.get('remaining_percent', 0) > 0 else 'FAIL'
    print(f\"[{status_icon}] {b['slo_name']:40s} budget={b.get('remaining_percent', 0):6.1f}%  service={b.get('service', 'N/A')}\")
"
```

### Step 3: Categorize the Failures

Group the failing SLOs by category to identify if there is a common root cause:

**By Service:**

```promql
# Failing SLOs grouped by service
count by (service) (gl_slo_error_budget_remaining_percent <= 0)
```

If multiple SLOs for the same service are failing, the issue is likely service-specific (bad deployment, resource exhaustion, dependency failure).

**By SLI Type:**

```promql
# Failing SLOs grouped by SLI type
count by (sli_type) (gl_slo_error_budget_remaining_percent <= 0)
```

If all latency SLOs are failing across services, the issue may be infrastructure-wide (database slowdown, network latency, shared dependency).

**By Tier:**

```promql
# Check if critical-tier SLOs are failing
gl_slo_error_budget_remaining_percent{tier="critical"} <= 0
```

Critical-tier SLO failures take priority over standard-tier failures.

### Step 4: Analyze the Compliance Trend

```promql
# Compliance trend over the past 7 days
gl_slo_compliance_percent[7d]

# Compliance trend over the past 30 days
gl_slo_compliance_percent[30d]
```

```bash
# Get compliance history from the API
curl -s "http://localhost:8080/api/v1/slos/compliance/report?type=weekly&count=4" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for report in data.get('reports', []):
    print(f\"{report.get('period_start', 'N/A')} to {report.get('period_end', 'N/A')}: {report.get('overall_compliance_percent', 'N/A'):.1f}% ({report.get('meeting_target', 'N/A')}/{report.get('total_slos', 'N/A')} SLOs met)\")
"
```

Determine if the compliance drop is:
- **Sudden**: Likely caused by a specific incident or deployment
- **Gradual**: Indicates systemic reliability degradation that needs engineering attention
- **Recurring**: Suggests the SLO targets may not be aligned with the platform's actual capabilities

### Step 5: Check for Common Root Causes

**Infrastructure-wide issues:**

```promql
# Database latency (affects all services using PostgreSQL)
histogram_quantile(0.99, sum(rate(gl_db_query_duration_seconds_bucket[5m])) by (le))

# Redis latency
histogram_quantile(0.99, sum(rate(gl_redis_command_duration_seconds_bucket[5m])) by (le))

# Node resource pressure
sum by (node) (rate(node_cpu_seconds_total{mode="idle"}[5m]))
sum by (node) (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)
```

**Shared dependency failures:**

```promql
# LLM service errors (affects multiple agents)
sum(rate(http_requests_total{job="llm-service", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="llm-service"}[5m]))

# Factor broker errors
sum(rate(http_requests_total{job="factor-broker", code=~"5.."}[5m])) /
sum(rate(http_requests_total{job="factor-broker"}[5m]))
```

---

## Actions

### Priority 1: Address Critical-Tier SLO Failures

If any critical-tier SLOs are failing:

1. Follow the [Error Budget Exhausted](./error-budget-exhausted.md) runbook for each critical SLO
2. Follow the [High Burn Rate](./high-burn-rate.md) runbook if burn rate alerts are also firing
3. These take precedence over standard-tier failures

### Priority 2: Address Infrastructure-Wide Issues

If the compliance drop is caused by a shared infrastructure problem:

```bash
# Check key infrastructure components
kubectl get pods -n database -l app=postgresql
kubectl get pods -n redis -l app=redis
kubectl get pods -n monitoring -l app.kubernetes.io/name=prometheus
kubectl top nodes
```

Follow the relevant infrastructure runbook:
- Database: Prometheus Target Down for PostgreSQL
- Redis: Redis cluster runbooks
- Nodes: Node resource pressure runbooks

### Priority 3: Address Service-Specific Failures

For each failing service:

1. Check recent deployments: `kubectl rollout history deployment/<service> -n <namespace>`
2. Check pod health: `kubectl get pods -n <namespace> -l app=<service>`
3. Check logs: `kubectl logs -n <namespace> -l app=<service> --tail=100`
4. Rollback if a recent deployment correlates: `kubectl rollout undo deployment/<service> -n <namespace>`

### Priority 4: Plan Reliability Improvements

If the compliance drop is gradual and not caused by a single incident, create reliability improvement items:

1. **Identify the top 3 worst-performing SLOs** and create tracking tickets
2. **Analyze error budget consumption patterns** over the past 90 days
3. **Propose architectural changes** if infrastructure limitations are the cause
4. **Add chaos engineering tests** to proactively find reliability gaps
5. **Review resource allocations** and scaling configurations

---

## SLO Review Process

When compliance is consistently below target, conduct a formal SLO review. This process should happen quarterly at minimum, or immediately after a period of sustained low compliance.

### When to Tighten SLO Targets

Consider tightening (making more strict) an SLO target when:

- The SLO has been consistently met with > 50% error budget remaining for 3+ months
- Users or stakeholders expect higher reliability than the current target provides
- Infrastructure improvements have increased the platform's reliability baseline
- Competitive pressure requires higher reliability commitments
- The SLO is backing an SLA that needs to be more aggressive

**Process:**
1. Tighten gradually (e.g., 99.5% to 99.7%, not 99.5% to 99.99%)
2. Monitor for 2 weeks in staging before applying to production
3. Communicate the tighter target to the owning team

### When to Loosen SLO Targets

Consider loosening (making less strict) an SLO target when:

- The SLO is consistently breached despite genuine engineering effort
- The cost of meeting the current target exceeds the business value
- The target was set aspirationally without data-driven basis
- User satisfaction does not correlate with the SLO (users are satisfied even when the SLO says "breached")
- The team is in constant firefighting mode due to unrealistic targets

**Process:**
1. Document why the current target is unrealistic (include data)
2. Propose a new target based on actual SLI performance (e.g., set target at the 90th percentile of historical SLI values)
3. Get approval from the SLO owner and SRE team
4. Apply the change and regenerate recording rules and alerts

### When to Retire an SLO

Remove an SLO when:

- The service it monitors has been decommissioned
- The SLI is no longer meaningful (e.g., the metric was replaced)
- The SLO is redundant with another SLO that provides better coverage
- The SLO was experimental and has served its purpose

---

## Quarterly SLO Review Template

Use the following template for quarterly SLO reviews. The SLO Service generates compliance reports that feed into this review.

```bash
# Generate the quarterly compliance report
curl -s "http://localhost:8080/api/v1/slos/compliance/report?type=quarterly&quarter=1&year=2026" | python3 -m json.tool > quarterly_slo_review_q1_2026.json
```

### Review Agenda

**1. Compliance Summary (10 minutes)**

| Metric | Q Value | Target |
|--------|---------|--------|
| Overall SLO Compliance | ___% | > 95% |
| SLOs Meeting Target | ___ / ___ | All |
| SLOs Breached | ___ | 0 |
| Average Error Budget Remaining | ___% | > 50% |
| Fast Burn Alerts Fired | ___ | < 5 |
| Error Budget Exhaustion Events | ___ | 0 |

**2. Per-SLO Review (20 minutes)**

For each SLO, review:

| SLO Name | Service | Target | Achieved | Budget Remaining | Trend | Action |
|----------|---------|--------|----------|------------------|-------|--------|
| | | | | | | |

Trend values: improving, stable, degrading

Action values:
- KEEP: No changes needed
- TIGHTEN: Target can be made stricter
- LOOSEN: Target should be relaxed
- INVESTIGATE: Reliability improvements needed
- RETIRE: SLO should be removed

**3. Worst Performers Deep Dive (15 minutes)**

For the 3 worst-performing SLOs:
- What caused budget consumption?
- Were incidents related to these SLOs?
- What reliability improvements were made?
- What improvements are still needed?
- Should the target be adjusted?

**4. New SLO Proposals (10 minutes)**

Are there services or user journeys that should have SLOs but currently do not?

| Proposed SLO | Service | Proposed Target | SLI Type | Justification |
|--------------|---------|-----------------|----------|---------------|
| | | | | |

**5. Action Items (5 minutes)**

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| | | | |

### Review Participants

| Role | Attendee |
|------|----------|
| SLO Review Lead | SRE Team Lead |
| Engineering Leads | Backend, Frontend, Infrastructure |
| Product Management | Product Lead |
| Approval | VP Engineering / CTO |

### Review Cadence

- **Weekly**: Quick compliance check (automated report, Slack notification)
- **Monthly**: SLO health review with team leads (30 minutes)
- **Quarterly**: Full SLO review with all stakeholders (60 minutes)
- **Ad-hoc**: After any error budget exhaustion event or platform-wide incident

---

## Generating Compliance Reports

The SLO Service provides automated compliance reporting. Reports are stored in TimescaleDB and can be retrieved via the API.

### Weekly Report

```bash
# Generate weekly report
curl -s "http://localhost:8080/api/v1/slos/compliance/report?type=weekly" | python3 -m json.tool

# The weekly report CronJob runs automatically every Monday at 09:00 UTC
kubectl get cronjob slo-weekly-report -n greenlang-slo
```

### Monthly Report

```bash
# Generate monthly report for a specific month
curl -s "http://localhost:8080/api/v1/slos/compliance/report?type=monthly&month=1&year=2026" | python3 -m json.tool
```

### Quarterly Report

```bash
# Generate quarterly report
curl -s "http://localhost:8080/api/v1/slos/compliance/report?type=quarterly&quarter=4&year=2025" | python3 -m json.tool
```

### Report Fields

Each compliance report includes:

| Field | Description |
|-------|-------------|
| `report_id` | Unique report identifier |
| `report_type` | weekly, monthly, or quarterly |
| `period_start` / `period_end` | Reporting period |
| `overall_compliance_percent` | Percentage of SLOs meeting target |
| `total_slos` | Total active SLOs |
| `meeting_target` | SLOs meeting their target |
| `breached` | SLOs that breached their target |
| `entries[]` | Per-SLO details (target, achieved, budget, trend) |

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Compliance 90-95%, no critical SLOs affected | On-call engineer | Within 30 minutes |
| L2 | Compliance 80-90%, or any critical-tier SLO breached | Platform team lead + #observability | Within 15 minutes |
| L3 | Compliance < 80%, multiple critical SLOs breached | Platform team + SRE lead + VP Engineering | Immediate |
| L4 | Compliance < 80% for > 24 hours, SLA credits at risk | All engineering + CTO + customer success | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** SLO Overview (`/d/slo-overview`)
- **Dashboard:** Error Budget Deep Dive (`/d/slo-error-budget`)
- **Alert:** `SLOComplianceBelow95` (this alert)
- **Key metrics:**
  - `gl_slo_compliance_percent` (should be > 95%)
  - `gl_slo_definitions_total` (total active SLOs)
  - `gl_slo_error_budget_remaining_percent` (per SLO, should be > 20%)

### Best Practices

1. **Monitor the compliance trend, not just the current value.** A gradual decline from 98% to 96% over 4 weeks is a strong signal, even if the alert has not yet fired.
2. **Act on ErrorBudgetWarning (80% consumed) alerts proactively.** These fire days or weeks before compliance drops below 95%.
3. **Conduct quarterly SLO reviews** using the template above. SLO targets should evolve with the platform.
4. **Use SLO compliance as a deployment gate.** If compliance is below 95%, consider deferring non-critical deployments until reliability is restored.
5. **Include SLO compliance in team OKRs.** Teams that own services with SLOs should have reliability as a measurable objective.
6. **Invest in reliability engineering proactively.** Allocate 10-20% of engineering capacity to reliability improvements based on SLO compliance data.
7. **Automate canary deployments.** Most compliance drops are caused by bad deployments. Canary analysis prevents widespread impact.

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly (aligned with SLO reviews)
- **Related alerts:** `ErrorBudgetExhausted`, `ErrorBudgetCritical`, `ErrorBudgetWarning`, `HighBurnRateFast`, `HighBurnRateMedium`, `HighBurnRateSlow`
- **Related dashboards:** SLO Overview, Error Budget Deep Dive, Per-Service SLO Detail
- **Related runbooks:** [SLO Service Down](./slo-service-down.md), [Error Budget Exhausted](./error-budget-exhausted.md), [High Burn Rate](./high-burn-rate.md)
