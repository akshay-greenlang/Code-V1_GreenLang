# Prometheus Stack Operational Runbooks

This directory contains operational runbooks for the GreenLang Prometheus observability stack. Each runbook provides step-by-step guidance for diagnosing and resolving alerts related to metrics collection, storage, and alerting.

---

## Quick Navigation

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Prometheus Memory](./prometheus-high-memory.md) | Memory management | PrometheusHighMemoryUsage, OOMKill |
| [Prometheus Targets](./prometheus-target-down.md) | Target health | PrometheusTargetMissing, ServiceDown |
| [Prometheus Queries](./prometheus-slow-queries.md) | Query performance | PrometheusSlowQueries, QueryTimeout |
| [Thanos Compactor](./thanos-compactor-halted.md) | Long-term storage | ThanosCompactorHalted, BlockOverlap |
| [Thanos Store Gateway](./thanos-store-gateway-issues.md) | Historical queries | ThanosStoreGatewayBucketOperationsFailed |
| [Alertmanager](./alertmanager-notifications-failing.md) | Notifications | AlertmanagerNotificationsFailing |
| [Batch Jobs](./batch-job-metrics-stale.md) | PushGateway jobs | BatchJobStale, PushgatewayDown |

---

## Alert Severity Reference

| Severity | Response Time | Notification | Example Alerts |
|----------|---------------|--------------|----------------|
| **Critical** | Immediate (<5 min) | PagerDuty page + Slack | PrometheusTargetMissing, ThanosCompactorHalted |
| **Warning** | Within 30 min | Slack notification | PrometheusHighMemoryUsage, BatchJobStale |
| **Info** | Next business day | Email digest | PrometheusConfigReloaded |

---

## Alert Summary by Component

### Prometheus Server Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [PrometheusHighMemoryUsage](./prometheus-high-memory.md) | Warning | >80% memory | Check cardinality, add recording rules |
| [PrometheusTargetMissing](./prometheus-target-down.md) | Critical | up == 0 for 5m | Check pod, network policy, ServiceMonitor |
| [PrometheusSlowQueries](./prometheus-slow-queries.md) | Warning | P99 >30s | Add recording rules, increase resources |
| PrometheusConfigReloadFailed | Critical | reload != 1 | Check config syntax, restart |
| PrometheusStorageAlmostFull | Warning | >80% storage | Reduce retention, expand PVC |
| PrometheusTSDBCompactionsFailed | Warning | failures > 0 | Check disk I/O, restart |

### Thanos Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [ThanosCompactorHalted](./thanos-compactor-halted.md) | Critical | halted == 1 | Check overlaps, S3 access, disk |
| [ThanosStoreGatewayBucketOperationsFailed](./thanos-store-gateway-issues.md) | Warning | failures > 0 | Check IRSA, S3 permissions |
| ThanosQueryHighDNSFailures | Warning | failures > 0.5/s | Check service discovery |
| ThanosSidecarPrometheusDown | Critical | prometheus_up != 1 | Check Prometheus, sidecar |
| ThanosCompactorMultipleRunning | Critical | compactors > 1 | Scale down to 1 |

### Alertmanager Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [AlertmanagerNotificationsFailing](./alertmanager-notifications-failing.md) | Critical | failures > 0 | Check webhooks, credentials |
| AlertmanagerConfigInconsistent | Warning | config mismatch | Sync config, restart |
| AlertmanagerClusterDown | Critical | members < expected | Check mesh, network |
| AlertmanagerMembersInconsistent | Warning | peer count varies | Check gossip ports |

### PushGateway Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [BatchJobStale](./batch-job-metrics-stale.md) | Warning | >1h since success | Check CronJob, push failures |
| PushGatewayDown | Critical | up == 0 | Restart deployment |
| BatchJobDurationHigh | Warning | duration > threshold | Investigate job performance |
| BatchJobErrorsHigh | Warning | errors increasing | Check job logs |

---

## On-Call Quick Reference

### First Response Checklist

1. **Acknowledge** the alert in PagerDuty
2. **Identify** the affected component and runbook
3. **Diagnose** using the runbook diagnostic steps
4. **Mitigate** with quick actions
5. **Communicate** status in #greenlang-incidents
6. **Resolve** or escalate per runbook
7. **Document** actions in incident timeline

### Common Commands

```bash
# Check Prometheus stack health
kubectl get pods -n monitoring

# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090
# Open http://localhost:9090/targets

# Check Alertmanager
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
# Open http://localhost:9093

# Check Thanos Query
kubectl port-forward -n monitoring svc/thanos-query 9091:9090
# Open http://localhost:9091

# Restart components
kubectl rollout restart statefulset -n monitoring prometheus-server
kubectl rollout restart statefulset -n monitoring alertmanager
kubectl rollout restart deployment -n monitoring thanos-query

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus --tail=100
kubectl logs -n monitoring -l app.kubernetes.io/name=alertmanager --tail=100
kubectl logs -n monitoring -l app.kubernetes.io/name=thanos-compactor --tail=100
```

### Key Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| Prometheus Health | https://grafana.greenlang.io/d/prometheus-health | Prometheus server metrics |
| Thanos Overview | https://grafana.greenlang.io/d/thanos-overview | Thanos component health |
| Alertmanager | https://grafana.greenlang.io/d/alertmanager | Alert delivery status |
| Targets | https://grafana.greenlang.io/d/prometheus-targets | Scrape target status |
| Cardinality | https://grafana.greenlang.io/d/cardinality-explorer | Metric cardinality analysis |

### Escalation Contacts

| Team | Slack Channel | PagerDuty Schedule |
|------|---------------|-------------------|
| Platform | #platform-oncall | platform-oncall |
| Observability | #observability | observability-oncall |
| Infrastructure | #infrastructure-oncall | infrastructure-oncall |

---

## Component Architecture

```
                                    +------------------+
                                    |    Grafana       |
                                    +--------+---------+
                                             |
                            +----------------+----------------+
                            |                                 |
                   +--------v--------+              +---------v---------+
                   |  Thanos Query   |              |   Alertmanager    |
                   +--------+--------+              +---------+---------+
                            |                                 |
        +-------------------+-------------------+             |
        |                   |                   |             |
+-------v-------+  +--------v--------+  +-------v-------+    |
| Thanos Sidecar|  | Thanos Store GW |  | Thanos Ruler  +----+
+-------+-------+  +--------+--------+  +---------------+
        |                   |
+-------v-------+           |
|  Prometheus   |           |
+-------+-------+  +--------v--------+
        |          |   S3 Bucket     |
        |          +--------+--------+
        |                   |
+-------v-------+  +--------v--------+
|  PushGateway  |  | Thanos Compactor|
+---------------+  +-----------------+
```

---

## Incident Severity Levels

| Level | Description | Response | Example |
|-------|-------------|----------|---------|
| P1 | Complete monitoring outage | Immediate, all hands | Prometheus down, no alerts |
| P2 | Major degradation | Within 15 minutes | All notifications failing |
| P3 | Minor issue | Within 1 hour | Single target down |
| P4 | Low priority | Next business day | Slow historical queries |

---

## Maintenance Windows

Regular maintenance windows for the Prometheus stack:

| Day | Time (UTC) | Duration | Purpose |
|-----|------------|----------|---------|
| Sunday | 02:00-04:00 | 2 hours | Helm upgrades, PVC expansion |
| Daily | 03:00-03:30 | 30 min | Thanos compaction (automated) |

During maintenance windows:
- Silence non-critical alerts
- Notify #greenlang-ops before starting
- Have rollback plan ready

---

## Contributing to Runbooks

### When to Update

- After resolving an incident not covered by existing runbooks
- When discovering new diagnostic techniques
- When resolution steps change due to architecture changes
- After post-incident reviews with new learnings

### Runbook Structure

Each runbook section should include:

1. **Alert Details** - Name, severity, threshold, PromQL
2. **Description** - What the alert means
3. **Impact Assessment** - User, data, SLA, revenue impact
4. **Diagnostic Steps** - How to identify the problem
5. **Resolution Steps** - Multiple scenarios with commands
6. **Emergency Actions** - Immediate mitigation
7. **Escalation Path** - When and who to escalate to
8. **Prevention** - How to avoid in the future
9. **Related Dashboards** - Links to monitoring
10. **Related Alerts** - Other relevant alerts

### Style Guidelines

- Use clear, actionable language
- Include actual commands that can be copy-pasted
- Explain what each command does
- Provide multiple resolution scenarios
- Keep commands idempotent where possible
- Test all commands before documenting

---

## Alert Configuration

Alert rules are defined in:

```
deployment/kubernetes/monitoring/prometheus-rules/
  - prometheus-health-alerts.yaml
  - thanos-health-alerts.yaml
  - alertmanager-health-alerts.yaml
  - pushgateway-alerts.yaml
```

To modify alert thresholds or add new alerts:

```bash
# Edit alert rules
kubectl edit prometheusrule -n monitoring prometheus-health-alerts

# Or apply from file
kubectl apply -f deployment/kubernetes/monitoring/prometheus-rules/
```

---

## Related Documentation

- [Prometheus Stack Architecture](../architecture/prometheus-stack.md)
- [Prometheus Operations Guide](../operations/prometheus-operations.md)
- [Metrics Developer Guide](../development/metrics-guide.md)
- [GreenLang Monitoring Setup](../monitoring/README.md)
- [Incident Response Procedure](../operations/incident-response.md)

---

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Thanos Documentation](https://thanos.io/tip/thanos/getting-started.md/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
