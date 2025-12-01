# GL-011 FUELCRAFT - Prometheus Alert Rules Documentation

## Overview

This directory contains comprehensive Prometheus alert rules for GL-011 FUELCRAFT, the intelligent fuel procurement and optimization agent. The alerting system provides 55+ production-grade alert rules across four severity levels to ensure reliable, efficient, and compliant fuel management operations.

## Directory Structure

```
monitoring/alerts/
├── prometheus_rules.yaml      # Main Prometheus alert rules (55+ alerts)
├── runbook_mapping.yaml       # Alert-to-runbook mapping and escalation procedures
└── README.md                  # This file - documentation and testing guide
```

## Alert Severity Levels

### CRITICAL (15 alerts)
**Response Time:** Immediate to 10 minutes
**Resolution Time:** 30 minutes to 4 hours

Indicates immediate service impact or regulatory violation requiring urgent response:
- Service outages (AgentDown, DatabaseConnectionLoss)
- Data integrity failures (DataCorruption, ProvenanceGap)
- Regulatory violations (EmissionsViolation, ContractBreach)
- Resource exhaustion (MemoryExhaustion, DiskFull, DeadlockDetected)
- Security incidents (SecurityViolation, CertificateExpiringSoon)
- Critical operational failures (FuelShortage, FuelQualityFailure, OptimizationFailure)

### HIGH (20 alerts)
**Response Time:** 15 minutes to 1 hour
**Resolution Time:** 4 to 24 hours

Indicates service degradation or significant risk requiring urgent attention:
- Performance issues (HighLatency, HighCPU, HighMemory, OptimizationSlow)
- Integration problems (IntegrationLatency, ApiRateLimitApproaching)
- Operational concerns (LowInventory, SupplierUnreliable, QueueBacklog)
- Financial risks (CostOverrun, PriceSpike)
- Compliance risks (ComplianceReportLate, CarbonTargetMiss)
- Data quality issues (StaleData, CalculatorError, ProvenanceGap)

### MEDIUM (15 alerts)
**Response Time:** 4 to 24 hours
**Resolution Time:** 24 hours to 7 days

Indicates potential issues requiring investigation but no immediate impact:
- Moderate performance degradation (ModerateLatency, OptimizationIterationsHigh)
- Early warning indicators (InventoryReorderPoint, ApiQuotaUsage)
- Efficiency concerns (FuelEfficiencyDrop, CostSavingsBelowTarget)
- Operational optimization (ContractUtilizationLow, SupplierDiversityLow)
- System health (CacheExpired, LogVolumeHigh, TestFailure)
- Configuration management (ConfigDrift, UnusualActivity)

### LOW (5 alerts)
**Response Time:** Not applicable
**Resolution Time:** Not applicable (informational)

Informational alerts for awareness and proactive maintenance:
- Routine notifications (InfoLog, ScheduledMaintenanceReminder)
- Advance warnings (CertificateExpiringNext30Days)
- Maintenance opportunities (DependencyUpdateAvailable, PerformanceTuningOpportunity)

## Alert Categories

### 1. Service Health (8 alerts)
Monitors overall service availability and health:
- `AgentDown` - Service unavailability
- `HighErrorRate` - Excessive error rates
- `HighLatency` - Response time degradation
- `ModerateLatency` - Minor latency issues
- `TestFailure` - Integration test failures
- `AgentVersion` - Version tracking
- `HealthCheckFailure` - Health endpoint failures
- `ReadinessProbeFailure` - Readiness check failures

### 2. Optimization Engine (7 alerts)
Monitors optimization performance and quality:
- `OptimizationFailure` - Solver failures
- `OptimizationSlow` - Long optimization times
- `OptimizationIterationsHigh` - Excessive solver iterations
- `CostSavingsBelowTarget` - Suboptimal results
- `PerformanceTuningOpportunity` - Optimization opportunities
- `ConstraintViolation` - Infeasible solutions
- `SolutionQualityDegraded` - Poor solution quality

### 3. Inventory Management (4 alerts)
Monitors fuel inventory levels:
- `FuelShortage` - Critical low inventory (<24 hours)
- `LowInventory` - Low inventory warning (<48 hours)
- `InventoryReorderPoint` - Reorder trigger (<72 hours)
- `InventoryDiscrepancy` - Inventory data mismatch

### 4. Integration & External Systems (6 alerts)
Monitors external integrations and APIs:
- `CriticalIntegrationDown` - Critical system unavailability
- `IntegrationLatency` - Slow integration responses
- `IntegrationWarning` - Integration error rate elevated
- `StaleData` - Outdated market data
- `ApiRateLimitApproaching` - Near API quota limit
- `ApiQuotaUsage` - Elevated API usage

### 5. Financial & Compliance (8 alerts)
Monitors costs, contracts, and compliance:
- `ContractBreach` - Contract limit exceeded
- `ContractUtilizationHigh` - Near contract capacity
- `ContractUtilizationLow` - Underutilized contracts
- `CostOverrun` - Budget exceeded
- `PriceSpike` - Significant price increases
- `PriceVolatility` - Unusual price volatility
- `EmissionsViolation` - Regulatory emissions violation
- `CarbonTargetMiss` - Missing carbon targets

### 6. Supplier Management (3 alerts)
Monitors supplier performance and diversity:
- `SupplierUnreliable` - High supplier failure rate
- `SupplierDiversityLow` - Supplier concentration risk
- `FuelQualityFailure` - Quality specification violations

### 7. Data Integrity & Provenance (4 alerts)
Monitors data quality and audit trail:
- `DataCorruption` - Provenance hash mismatches
- `ProvenanceGap` - Missing provenance records
- `CalculatorError` - Calculation failures
- `DataValidationFailure` - Invalid input data

### 8. System Resources (7 alerts)
Monitors infrastructure and resource utilization:
- `HighCPU` - CPU utilization >80%
- `HighMemory` - Memory utilization >80%
- `MemoryExhaustion` - Memory critical (>95%)
- `DiskFull` - Disk space critical (>95%)
- `ThreadPoolSaturated` - Thread pool exhausted
- `DeadlockDetected` - Thread deadlocks
- `DatabaseConnectionLoss` - DB connection pool exhausted

### 9. Security (5 alerts)
Monitors security and certificates:
- `SecurityViolation` - Unauthorized access detected
- `CertificateExpiringSoon` - Certificate expires <7 days
- `CertificateExpiringNext30Days` - Certificate expires <30 days
- `DependencyVulnerability` - CVE in dependencies
- `UnusualActivity` - Anomalous request patterns

### 10. Performance & Caching (3 alerts)
Monitors cache efficiency and performance:
- `LowCacheHitRate` - Cache hit rate <50%
- `CacheExpired` - Frequent cache invalidations
- `FuelEfficiencyDrop` - Operational efficiency decrease

## Alert Rule Files

### prometheus_rules.yaml

Main Prometheus alert rules file containing all 55+ alerts organized into rule groups:

```yaml
groups:
  - name: fuelcraft_critical   # 15 critical alerts
    interval: 30s
    rules: [...]

  - name: fuelcraft_high       # 20 high severity alerts
    interval: 60s
    rules: [...]

  - name: fuelcraft_medium     # 15 medium severity alerts
    interval: 120s
    rules: [...]

  - name: fuelcraft_low        # 5 low severity alerts
    interval: 300s
    rules: [...]
```

**Key Features:**
- PromQL expressions for all conditions
- Appropriate `for` durations to reduce flapping
- Comprehensive labels (severity, component, team, agent, runbook)
- Detailed annotations (summary, description, action, impact)
- Runbook URLs for troubleshooting guidance

### runbook_mapping.yaml

Comprehensive mapping of alerts to runbooks and escalation procedures:

```yaml
critical_alerts:
  AgentDown:
    severity: critical
    component: agent
    runbook_path: runbooks/TROUBLESHOOTING.md
    runbook_section: "#agent-down"
    escalation_path: [...]
    sla_response_time: "5 minutes"
    sla_resolution_time: "30 minutes"
    troubleshooting_steps: [...]
    related_alerts: [...]
```

**Key Features:**
- Detailed escalation paths (Level 1, 2, 3)
- SLA response and resolution times
- Troubleshooting step checklists
- Related alert correlations
- Automated action definitions
- Contact methods and response times

## Deployment

### 1. Prometheus Configuration

Add the alert rules to your Prometheus configuration:

```yaml
# prometheus.yml
rule_files:
  - /etc/prometheus/rules/fuelcraft/prometheus_rules.yaml
```

### 2. Validate Alert Rules

Before deployment, validate the alert rules syntax:

```bash
# Using promtool
promtool check rules monitoring/alerts/prometheus_rules.yaml

# Expected output:
# Checking monitoring/alerts/prometheus_rules.yaml
#   SUCCESS: 55 rules found
```

### 3. Reload Prometheus

Reload Prometheus configuration to apply new rules:

```bash
# Send SIGHUP to Prometheus process
kill -HUP <prometheus-pid>

# OR use HTTP API
curl -X POST http://localhost:9090/-/reload
```

### 4. Verify Alert Rules Loaded

Check Prometheus UI to verify rules are loaded:

```
Navigate to: http://localhost:9090/rules
Verify: All rule groups appear (fuelcraft_critical, fuelcraft_high, etc.)
```

## Alert Testing

### Manual Alert Testing

#### Test 1: Simulate AgentDown Alert

```bash
# Stop the FUELCRAFT service
kubectl scale deployment/fuelcraft --replicas=0 -n greenlang

# Wait 5 minutes for alert to fire
# Check Prometheus alerts page

# Verify alert appears with:
# - Severity: critical
# - Labels: component=agent, team=greenlang
# - Annotations with runbook URL

# Restore service
kubectl scale deployment/fuelcraft --replicas=3 -n greenlang
```

#### Test 2: Simulate HighErrorRate Alert

```bash
# Inject errors using chaos engineering
# OR temporarily modify code to return errors

# Monitor error rate metric:
watch -n 5 'curl -s http://localhost:9090/api/v1/query?query=rate(fuelcraft_errors_total[10m])'

# Wait for error rate to exceed 5% for 10 minutes
# Verify alert fires with correct annotations
```

#### Test 3: Simulate MemoryExhaustion Alert

```bash
# Reduce memory limit temporarily
kubectl set resources deployment/fuelcraft --limits=memory=256Mi -n greenlang

# Trigger memory-intensive operations
# Monitor memory usage
kubectl top pod -n greenlang | grep fuelcraft

# Verify alert fires when memory > 95%
# Restore original limits
kubectl set resources deployment/fuelcraft --limits=memory=2Gi -n greenlang
```

### Automated Alert Testing

Create alert test suite using `promtool`:

```bash
# alert_test.yaml
rule_files:
  - prometheus_rules.yaml

evaluation_interval: 1m

tests:
  # Test AgentDown alert
  - interval: 1m
    input_series:
      - series: 'up{job="fuelcraft", instance="fuelcraft-0"}'
        values: '1 1 1 0 0 0 0 0'
    alert_rule_test:
      - eval_time: 7m
        alertname: AgentDown
        exp_alerts:
          - exp_labels:
              severity: critical
              component: agent
              instance: fuelcraft-0
            exp_annotations:
              summary: "GL-011 FUELCRAFT agent is down"

  # Test HighErrorRate alert
  - interval: 1m
    input_series:
      - series: 'fuelcraft_errors_total{instance="fuelcraft-0"}'
        values: '0+10x20'
      - series: 'fuelcraft_requests_total{instance="fuelcraft-0"}'
        values: '0+100x20'
    alert_rule_test:
      - eval_time: 15m
        alertname: HighErrorRate
        exp_alerts:
          - exp_labels:
              severity: critical
              component: optimization

# Run tests
promtool test rules alert_test.yaml
```

## Alert Manager Integration

### Configure Alert Routing

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'severity', 'component']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'default'

  routes:
    # Critical alerts -> PagerDuty + Slack
    - match:
        severity: critical
        agent: GL-011
      receiver: 'fuelcraft-critical'
      group_wait: 0s
      repeat_interval: 5m

    # High alerts -> Slack + Email
    - match:
        severity: high
        agent: GL-011
      receiver: 'fuelcraft-high'
      group_interval: 10m
      repeat_interval: 1h

    # Medium alerts -> Slack
    - match:
        severity: medium
        agent: GL-011
      receiver: 'fuelcraft-medium'
      group_interval: 30m
      repeat_interval: 4h

    # Low alerts -> Email digest
    - match:
        severity: low
        agent: GL-011
      receiver: 'fuelcraft-low'
      group_interval: 1h
      repeat_interval: 24h

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#alerts'

  - name: 'fuelcraft-critical'
    pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
        severity: 'critical'
        description: '{{ .CommonAnnotations.summary }}'
    slack_configs:
      - channel: '#fuelcraft-critical'
        color: 'danger'
        title: 'CRITICAL: {{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'fuelcraft-high'
    slack_configs:
      - channel: '#fuelcraft-alerts'
        color: 'warning'
        title: 'HIGH: {{ .CommonAnnotations.summary }}'
    email_configs:
      - to: 'fuelcraft-oncall@greenlang.io'

  - name: 'fuelcraft-medium'
    slack_configs:
      - channel: '#fuelcraft-alerts'
        color: '#0099CC'
        title: 'MEDIUM: {{ .CommonAnnotations.summary }}'

  - name: 'fuelcraft-low'
    email_configs:
      - to: 'fuelcraft-team@greenlang.io'
        send_resolved: false
```

### Alert Notification Templates

```yaml
# slack_template.tmpl
{{ define "slack.fuelcraft.title" }}
[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .CommonLabels.alertname }}
{{ end }}

{{ define "slack.fuelcraft.text" }}
{{ range .Alerts }}
*Alert:* {{ .Labels.alertname }}
*Severity:* {{ .Labels.severity }}
*Component:* {{ .Labels.component }}
*Summary:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}
*Action:* {{ .Annotations.action }}
*Runbook:* {{ .Annotations.runbook_url }}
{{ end }}
{{ end }}
```

## Monitoring Dashboard

Create a Grafana dashboard to visualize alert status:

### Alert Status Panel

```json
{
  "title": "FUELCRAFT Alert Status",
  "targets": [
    {
      "expr": "ALERTS{agent=\"GL-011\"}",
      "legendFormat": "{{ alertname }} - {{ severity }}"
    }
  ],
  "type": "stat",
  "options": {
    "colorMode": "background",
    "graphMode": "none",
    "thresholds": {
      "mode": "absolute",
      "steps": [
        { "value": 0, "color": "green" },
        { "value": 1, "color": "red" }
      ]
    }
  }
}
```

### Alert Firing Rate

```json
{
  "title": "Alert Firing Rate",
  "targets": [
    {
      "expr": "rate(ALERTS{agent=\"GL-011\",alertstate=\"firing\"}[1h])",
      "legendFormat": "{{ severity }}"
    }
  ],
  "type": "graph"
}
```

### Time to Resolution

```json
{
  "title": "Alert Resolution Time",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, sum(rate(alert_resolution_duration_seconds_bucket{agent=\"GL-011\"}[1h])) by (severity, le))",
      "legendFormat": "P95 - {{ severity }}"
    }
  ],
  "type": "graph"
}
```

## Alert Metrics

Track alert effectiveness with these metrics:

```promql
# Alert firing frequency
rate(ALERTS{agent="GL-011", alertstate="firing"}[24h])

# Mean time to acknowledge
avg(alert_acknowledged_timestamp - alert_fired_timestamp)

# Mean time to resolution
avg(alert_resolved_timestamp - alert_fired_timestamp)

# Alert accuracy (true positives vs false positives)
sum(alert_true_positive_total) / sum(alert_fired_total)

# SLA compliance
sum(alert_resolved_within_sla_total) / sum(alert_fired_total)
```

## Runbook Integration

Each alert references a specific runbook section:

```yaml
annotations:
  runbook_url: "https://docs.greenlang.io/runbooks/troubleshooting#agent-down"
```

### Runbook Structure

```markdown
# FUELCRAFT Troubleshooting Runbook

## Agent Down

**Alert:** AgentDown
**Severity:** Critical
**SLA:** 5 min response, 30 min resolution

### Symptoms
- FUELCRAFT service unavailable
- Health check endpoint returning 503
- No optimization requests being processed

### Diagnosis
1. Check pod status: `kubectl get pods -n greenlang`
2. Check pod logs: `kubectl logs -n greenlang <pod-name>`
3. Check events: `kubectl describe pod -n greenlang <pod-name>`
4. Check resource usage: `kubectl top pod -n greenlang`

### Resolution
1. If OOMKilled: Increase memory limits
2. If CrashLoopBackOff: Check application logs for errors
3. If ImagePullBackOff: Verify image tag and registry access
4. If pending: Check resource availability and node capacity

### Escalation
- L1 (0-10 min): On-call DevOps Engineer
- L2 (10-20 min): Engineering Manager
- L3 (20+ min): VP Engineering
```

## Alert Tuning Guide

### Reducing False Positives

1. **Adjust `for` Duration**
   ```yaml
   # Too sensitive (fires on brief spikes)
   for: 1m

   # Better (waits for sustained issue)
   for: 5m
   ```

2. **Refine Thresholds**
   ```yaml
   # Too sensitive
   expr: cpu_usage > 0.7

   # Better (allows for normal spikes)
   expr: cpu_usage > 0.8
   for: 15m
   ```

3. **Add Context to Conditions**
   ```yaml
   # Simple (may fire during maintenance)
   expr: up{job="fuelcraft"} == 0

   # Better (excludes maintenance windows)
   expr: up{job="fuelcraft"} == 0 unless on() maintenance_mode == 1
   ```

### Preventing Alert Fatigue

1. **Use Alert Grouping**
   ```yaml
   # Group related alerts together
   route:
     group_by: ['component', 'severity']
     group_interval: 5m
   ```

2. **Set Appropriate Repeat Intervals**
   ```yaml
   # Critical: Repeat every 5 minutes until resolved
   repeat_interval: 5m

   # High: Repeat every hour
   repeat_interval: 1h

   # Medium: Repeat every 4 hours
   repeat_interval: 4h
   ```

3. **Implement Alert Dependencies**
   ```yaml
   # Don't alert on high latency if service is down
   - alert: HighLatency
     expr: latency > 5 and on() up{job="fuelcraft"} == 1
   ```

## Best Practices

### 1. Alert Design
- ✅ Alert on symptoms, not causes
- ✅ Make alerts actionable
- ✅ Include runbook links
- ✅ Use meaningful alert names
- ✅ Set appropriate severity levels
- ❌ Don't alert on information that doesn't require action
- ❌ Don't create alerts that are too noisy

### 2. Alert Configuration
- ✅ Use `for` duration to reduce flapping
- ✅ Set sensible thresholds based on historical data
- ✅ Include rich labels and annotations
- ✅ Test alerts before deploying
- ❌ Don't set thresholds too aggressively
- ❌ Don't create overlapping alerts

### 3. Alert Response
- ✅ Acknowledge alerts promptly
- ✅ Follow runbook procedures
- ✅ Document resolution steps
- ✅ Update runbooks based on learnings
- ❌ Don't ignore or silence alerts without investigation
- ❌ Don't let alert fatigue reduce response quality

### 4. Alert Maintenance
- ✅ Regularly review alert effectiveness
- ✅ Tune thresholds based on performance
- ✅ Remove or deprecate obsolete alerts
- ✅ Update runbooks as system evolves
- ❌ Don't let alert rules become stale
- ❌ Don't accumulate technical debt in alerting

## Troubleshooting Alerts

### Alert Not Firing

```bash
# Check if rule is loaded
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="fuelcraft_critical")'

# Check if metrics exist
curl http://localhost:9090/api/v1/query?query=up{job="fuelcraft"}

# Evaluate rule manually
curl 'http://localhost:9090/api/v1/query?query=up{job="fuelcraft"}==0'

# Check Prometheus logs
kubectl logs -n monitoring prometheus-0 | grep -i error
```

### Alert Firing Too Frequently

```bash
# Check alert history
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname=="HighLatency")'

# Analyze metric patterns
# (Open Prometheus UI and graph the alert expression)

# Adjust 'for' duration or threshold
# Edit prometheus_rules.yaml and reload
```

### Alert Not Routing Correctly

```bash
# Check AlertManager configuration
curl http://localhost:9093/api/v1/status | jq .

# Test route matching
amtool config routes test --config.file=/etc/alertmanager/alertmanager.yml \
  severity=critical component=agent

# Check AlertManager logs
kubectl logs -n monitoring alertmanager-0
```

## Support and Contacts

### Development Team
- **Email:** fuelcraft-team@greenlang.io
- **Slack:** #fuelcraft-dev
- **On-Call:** PagerDuty rotation

### Operations Team
- **Email:** ops@greenlang.io
- **Slack:** #operations
- **On-Call:** See PagerDuty schedule

### Security Team
- **Email:** security@greenlang.io
- **Slack:** #security
- **On-Call:** security-oncall@greenlang.io

## Additional Resources

- **Prometheus Documentation:** https://prometheus.io/docs/
- **Alert Manager Documentation:** https://prometheus.io/docs/alerting/latest/alertmanager/
- **PromQL Guide:** https://prometheus.io/docs/prometheus/latest/querying/basics/
- **GreenLang Runbooks:** https://docs.greenlang.io/runbooks/
- **FUELCRAFT Architecture:** ../../../docs/ARCHITECTURE.md
- **FUELCRAFT Monitoring:** ../README.md

## Changelog

### v1.0.0 (2025-12-01)
- Initial release with 55+ alert rules
- Complete runbook mapping
- Alert testing framework
- Comprehensive documentation

---

**Document Version:** 1.0.0
**Last Updated:** 2025-12-01
**Maintained By:** GL-DevOpsEngineer
**Review Cycle:** Quarterly
