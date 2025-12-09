# GreenLang Runbooks

This directory contains operational runbooks for responding to alerts in the GreenLang platform. Each runbook provides step-by-step guidance for diagnosing and resolving incidents.

---

## Quick Navigation

| Category | Runbook | Alerts Covered |
|----------|---------|----------------|
| [Service Availability](./service-availability.md) | Service health and uptime | ServiceDown, APIEndpointUnhealthy, HighPodRestartRate |
| [Error Rates](./error-rates.md) | Application errors and failures | HighErrorRate, CalculationFailureSpike, HighHTTP5xxRate, EFLookupFailures |
| [Latency](./latency.md) | Response time and performance | HighP99Latency, HighP95Latency, APIResponseTimeSlow, EFLookupSlow |
| [Resources](./resources.md) | CPU, memory, and disk | HighCPUUsage, CriticalCPUUsage, HighMemoryUsage, CriticalMemoryUsage, DiskSpaceLow |
| [Database](./database.md) | PostgreSQL and Redis | PostgreSQLConnectionPoolHigh, PostgreSQLSlowQueries, RedisMemoryHigh, RedisConnectionsHigh |
| [Business](./business.md) | Business logic and agents | LowCalculationThroughput, AgentRegistryEmpty, NoActiveAgents, EFCacheMissRateHigh |
| [SLA](./sla.md) | Service level agreements | SLAAvailabilityViolation, SLALatencyViolation, ErrorBudgetBurning |

---

## Alert Severity Reference

| Severity | Response Time | Notification | Example |
|----------|---------------|--------------|---------|
| **Critical** | Immediate (<5 min) | PagerDuty page + Slack | ServiceDown, SLAAvailabilityViolation |
| **Warning** | Within 30 min | Slack notification | HighCPUUsage, HighP95Latency |
| **Info** | Next business day | Email digest | (Informational alerts) |

---

## Alert Summary by Runbook

### Service Availability Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [ServiceDown](./service-availability.md#servicedown) | Critical | `up == 0` for 1m | Check pods, restart service |
| [APIEndpointUnhealthy](./service-availability.md#apiendpointunhealthy) | Critical | Probe fails for 2m | Check ingress, TLS certs |
| [HighPodRestartRate](./service-availability.md#highpodrestartrate) | Warning | >5 restarts/hour | Check OOM, probe config |

### Error Rate Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [HighErrorRate](./error-rates.md#higherrorrate) | Critical | >1% errors | Rollback, check dependencies |
| [CalculationFailureSpike](./error-rates.md#calculationfailurespike) | Warning | >50 failures/10m | Check batch jobs, data quality |
| [HighHTTP5xxRate](./error-rates.md#highhttp5xxrate) | Critical | >1% 5xx | Check logs, rollback |
| [EFLookupFailures](./error-rates.md#eflookupfailures) | Warning | >20 failures/15m | Check EF service, enable cache |

### Latency Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [HighP99Latency](./latency.md#highp99latency) | Critical | >1s P99 | Scale, check DB queries |
| [HighP95Latency](./latency.md#highp95latency) | Warning | >500ms P95 | Monitor, proactive scaling |
| [APIResponseTimeSlow](./latency.md#apiresponsetime slow) | Warning | >2s P95 | Add caching, optimize queries |
| [EFLookupSlow](./latency.md#eflookup slow) | Warning | >500ms P95 | Warm cache, check DB indexes |

### Resource Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [HighCPUUsage](./resources.md#highcpuusage) | Warning | >80% | Scale, check for CPU hogs |
| [CriticalCPUUsage](./resources.md#criticalcpuusage) | Critical | >95% | Kill processes, scale urgently |
| [HighMemoryUsage](./resources.md#highmemoryusage) | Warning | >85% | Restart memory hogs |
| [CriticalMemoryUsage](./resources.md#criticalmemoryusage) | Critical | >95% | Drain node, add capacity |
| [DiskSpaceLow](./resources.md#diskspacelow) | Warning | >80% | Clean logs, prune images |

### Database Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [PostgreSQLConnectionPoolHigh](./database.md#postgresqlconnectionpoolhigh) | Warning | >80% connections | Kill idle, check leaks |
| [PostgreSQLSlowQueries](./database.md#postgresqlslowqueries) | Warning | >1s mean | Add indexes, analyze |
| [RedisMemoryHigh](./database.md#redismemoryhigh) | Warning | >80% memory | Set eviction, clean keys |
| [RedisConnectionsHigh](./database.md#redisconnectionshigh) | Warning | >500 clients | Kill idle, fix leaks |

### Business Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [LowCalculationThroughput](./business.md#lowcalculationthroughput) | Warning | <1 calc/sec | Check workers, queue |
| [AgentRegistryEmpty](./business.md#agentregistryempty) | Critical | 0 agents | Restart registry, agents |
| [NoActiveAgents](./business.md#noactiveagents) | Critical | 0 active | Check health, restart |
| [EFCacheMissRateHigh](./business.md#efcachemissratehigh) | Warning | >50% miss | Warm cache, increase TTL |

### SLA Alerts

| Alert | Severity | Threshold | Quick Action |
|-------|----------|-----------|--------------|
| [SLAAvailabilityViolation](./sla.md#slaavailabilityviolation) | Critical | <99.9% | P1 incident, all hands |
| [SLALatencyViolation](./sla.md#slalatencyviolation) | Critical | P99 >1s | Scale, optimize |
| [ErrorBudgetBurning](./sla.md#errorbudgetburning) | Warning | >0.5% error | Fix errors, freeze deploys |

---

## On-Call Quick Reference

### First Response Checklist

1. **Acknowledge** the alert in PagerDuty
2. **Assess** severity and user impact
3. **Diagnose** using runbook steps
4. **Mitigate** with quick actions
5. **Communicate** status to stakeholders
6. **Resolve** or escalate
7. **Document** actions taken

### Common Commands

```bash
# Check pod status
kubectl get pods -n greenlang

# Get pod logs
kubectl logs -n greenlang -l app=<service> --tail=100

# Restart deployment
kubectl rollout restart deployment -n greenlang <deployment>

# Rollback deployment
kubectl rollout undo deployment -n greenlang <deployment>

# Scale deployment
kubectl scale deployment -n greenlang <deployment> --replicas=<count>

# Check resource usage
kubectl top pods -n greenlang

# Get events
kubectl get events -n greenlang --sort-by='.lastTimestamp'
```

### Key Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| System Health | https://grafana.greenlang.io/d/system-health | Overall platform health |
| Agent Performance | https://grafana.greenlang.io/d/agent-performance | Agent-specific metrics |
| SLA Compliance | https://grafana.greenlang.io/d/sla-compliance | SLA tracking |
| Database Health | https://grafana.greenlang.io/d/database | PostgreSQL and Redis |
| Kubernetes | https://grafana.greenlang.io/d/k8s-cluster | Cluster resources |

### Escalation Contacts

| Team | Slack Channel | PagerDuty Schedule |
|------|---------------|-------------------|
| Platform | #platform-oncall | platform-oncall |
| Backend | #backend-oncall | backend-oncall |
| Database | #database-oncall | database-oncall |
| Data | #data-oncall | data-oncall |

### Incident Severity Levels

| Level | Description | Response | Example |
|-------|-------------|----------|---------|
| P1 | Service outage | Immediate, all hands | Complete API failure |
| P2 | Major degradation | Within 15 minutes | High error rate, SLA risk |
| P3 | Minor issue | Within 1 hour | Warning alerts, no SLA impact |
| P4 | Low priority | Next business day | Informational, optimization |

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
5. **Resolution Steps** - How to fix it (multiple scenarios)
6. **Escalation Path** - When and who to escalate to
7. **Related Dashboards** - Links to monitoring

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
monitoring/prometheus/alert_rules.yml
```

To modify alert thresholds or add new alerts, update this file and apply:

```bash
kubectl apply -f monitoring/prometheus/alert_rules.yml
```

---

## Related Documentation

- [Monitoring Setup Guide](../monitoring/README.md)
- [Incident Response Procedure](../operations/incident-response.md)
- [SLA Documentation](../sla/README.md)
- [Architecture Overview](../architecture/README.md)
