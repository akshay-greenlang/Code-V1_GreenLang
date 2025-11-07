# GreenLang Alerting Guide

## Overview

GreenLang uses Prometheus Alertmanager for alert management, providing flexible routing, grouping, and notification capabilities.

## Alert Rules

### Viewing Alerts

```bash
# Check alert status in Prometheus
http://localhost:9090/alerts

# View in Alertmanager
http://localhost:9093
```

### Alert States

- **Inactive**: Condition not met
- **Pending**: Condition met, waiting for `for` duration
- **Firing**: Alert active and notifications sent

## Built-in Alerts

### Application Alerts

#### High Error Rate
- **Severity**: Warning
- **Threshold**: > 5% error rate
- **Duration**: 5 minutes
- **Action**: Investigate recent code changes, check logs

#### Critical Error Rate
- **Severity**: Critical
- **Threshold**: > 10% error rate
- **Duration**: 2 minutes
- **Action**: Page on-call engineer, rollback if needed

#### Slow Pipeline Execution
- **Severity**: Warning
- **Threshold**: P95 latency > 60s
- **Duration**: 10 minutes
- **Action**: Check database performance, review pipeline code

### Resource Alerts

#### High CPU Usage
- **Severity**: Warning
- **Threshold**: > 80%
- **Duration**: 5 minutes
- **Action**: Check for resource-intensive operations, scale horizontally

#### Critical CPU Usage
- **Severity**: Critical
- **Threshold**: > 95%
- **Duration**: 2 minutes
- **Action**: Immediate scaling required, investigate CPU-bound processes

#### Low Disk Space
- **Severity**: Warning
- **Threshold**: > 85% usage
- **Duration**: 5 minutes
- **Action**: Clean up logs, increase disk capacity

### Health Alerts

#### Service Down
- **Severity**: Critical
- **Threshold**: Service unreachable
- **Duration**: 1 minute
- **Action**: Check service status, restart if necessary, check infrastructure

## Custom Alert Rules

### Creating Alerts

Edit `observability/alerting-rules.yml`:

```yaml
groups:
  - name: custom_alerts
    interval: 30s
    rules:
      - alert: CustomMetricHigh
        expr: my_custom_metric > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: 'Custom metric is high'
          description: 'Value is {{ $value }}'
```

### Alert Expression Examples

```yaml
# Rate-based alert
- alert: HighRequestRate
  expr: rate(http_requests_total[5m]) > 1000

# Comparison alert
- alert: MemoryUsageHigh
  expr: (memory_used / memory_total) > 0.90

# Absence alert
- alert: ServiceMissing
  expr: absent(up{job="my_service"})

# Multiple conditions
- alert: PerformanceDegraded
  expr: |
    (rate(request_errors[5m]) > 0.01)
    and
    (histogram_quantile(0.95, rate(request_duration[5m])) > 1)
```

## Alert Routing

### Route Configuration

Alerts are routed based on labels in `alertmanager-config.yml`:

```yaml
route:
  receiver: 'default'
  group_by: ['alertname', 'cluster']
  routes:
    # Critical alerts → immediate notification
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 0s
      repeat_interval: 1h

    # Warning alerts → grouped notifications
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 30s
      repeat_interval: 4h

    # Component-specific routing
    - match:
        component: database
      receiver: 'database-team'
```

### Route Parameters

- **group_by**: Group alerts by labels
- **group_wait**: Wait time before sending first notification
- **group_interval**: Wait time between notifications for grouped alerts
- **repeat_interval**: Time between repeat notifications

## Notification Channels

### Slack

```yaml
receivers:
  - name: 'slack-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Severity:* {{ .Labels.severity }}
          {{ end }}
        send_resolved: true
```

### Email

```yaml
receivers:
  - name: 'email-alerts'
    email_configs:
      - to: 'team@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'alertmanager'
        auth_password: 'password'
        headers:
          Subject: '[{{ .Status }}] {{ .GroupLabels.alertname }}'
```

### PagerDuty

```yaml
receivers:
  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
        description: '{{ .GroupLabels.alertname }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'
```

### Webhook

```yaml
receivers:
  - name: 'webhook-alerts'
    webhook_configs:
      - url: 'http://example.com/webhook'
        send_resolved: true
        http_config:
          basic_auth:
            username: 'webhook_user'
            password: 'webhook_pass'
```

## Alert Inhibition

Suppress less important alerts when more critical ones are firing:

```yaml
inhibit_rules:
  # Don't alert on warnings if critical alert is active
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']

  # Don't alert on individual components if service is down
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: '.*'
    equal: ['service']
```

## Alert Templates

### Custom Templates

Create `/etc/alertmanager/templates/custom.tmpl`:

```tmpl
{{ define "slack.default.title" }}
[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}]
{{ .GroupLabels.SortedPairs.Values | join " " }}
{{ end }}

{{ define "slack.default.text" }}
{{ range .Alerts }}
*Alert:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}
*Severity:* {{ .Labels.severity }}
*Started:* {{ .StartsAt.Format "2006-01-02 15:04:05" }}
{{ if .EndsAt }}*Ended:* {{ .EndsAt.Format "2006-01-02 15:04:05" }}{{ end }}
{{ end }}
{{ end }}
```

## Alert Best Practices

### 1. Set Appropriate Thresholds

```yaml
# Good: Actionable thresholds
- alert: HighErrorRate
  expr: (rate(errors[5m]) / rate(requests[5m])) > 0.01
  for: 5m

# Bad: Too sensitive
- alert: AnyError
  expr: errors > 0
```

### 2. Include Runbooks

```yaml
annotations:
  summary: 'High error rate detected'
  description: 'Error rate is {{ $value | humanizePercentage }}'
  runbook_url: 'https://wiki.example.com/runbooks/high-error-rate'
```

### 3. Use Meaningful Labels

```yaml
labels:
  severity: critical
  component: api
  team: platform
  environment: production
```

### 4. Group Related Alerts

```yaml
route:
  group_by: ['alertname', 'service']
  # Groups alerts to reduce notification noise
```

## Testing Alerts

### Manual Testing

```bash
# Trigger alert by exposing test metric
curl -X POST http://localhost:9091/metrics/job/test \
  -d 'test_metric 200'

# Check alert in Prometheus
http://localhost:9090/alerts

# Verify notification received
```

### Alert Simulation

```yaml
# Create test alert rule
- alert: TestAlert
  expr: vector(1)  # Always true
  labels:
    severity: info
  annotations:
    summary: 'Test alert'
```

## Silencing Alerts

### Via Alertmanager UI

1. Open Alertmanager UI (http://localhost:9093)
2. Click "Silence" next to alert
3. Set silence duration and comment
4. Submit

### Via API

```bash
# Create silence
curl -X POST http://localhost:9093/api/v2/silences \
  -H 'Content-Type: application/json' \
  -d '{
    "matchers": [
      {
        "name": "alertname",
        "value": "HighErrorRate",
        "isRegex": false
      }
    ],
    "startsAt": "2025-01-15T10:00:00Z",
    "endsAt": "2025-01-15T12:00:00Z",
    "createdBy": "ops-team",
    "comment": "Maintenance window"
  }'
```

## Monitoring Alert Health

### Alert Metrics

```promql
# Firing alerts
ALERTS{alertstate="firing"}

# Alert evaluation failures
prometheus_rule_evaluation_failures_total

# Alert notification failures
alertmanager_notifications_failed_total
```

### Dashboard

Create Grafana dashboard to monitor:
- Active alerts count
- Alert firing rate
- Notification success rate
- Silenced alerts

## Troubleshooting

### Alerts Not Firing

**Check:**
1. Alert expression in Prometheus
2. Alert state (pending vs firing)
3. For duration not met
4. Inhibition rules

### Notifications Not Received

**Check:**
1. Alertmanager configuration
2. Receiver configuration
3. Route matching
4. Network connectivity
5. Notification logs

### Too Many Alerts

**Solutions:**
1. Adjust thresholds
2. Increase `for` duration
3. Add inhibition rules
4. Group related alerts
5. Review alert necessity

## Response Procedures

### Critical Alert Response

1. **Acknowledge** alert in PagerDuty
2. **Assess** impact and scope
3. **Mitigate** immediate issue
4. **Communicate** to stakeholders
5. **Investigate** root cause
6. **Document** incident
7. **Post-mortem** and prevention

### Warning Alert Response

1. **Review** alert details
2. **Check** trends and patterns
3. **Determine** if action needed
4. **Schedule** fix if non-urgent
5. **Monitor** for escalation

## Integration with CI/CD

### Pre-deployment Testing

```yaml
# .github/workflows/deploy.yml
- name: Validate Alert Rules
  run: |
    promtool check rules observability/alerting-rules.yml

- name: Test Alertmanager Config
  run: |
    amtool check-config observability/alertmanager-config.yml
```

### Deployment Silencing

```bash
# Silence alerts during deployment
./scripts/silence-alerts.sh "Deployment in progress" 30m
```
