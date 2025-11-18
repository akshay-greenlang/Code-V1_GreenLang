# GL-001 Monitoring Quick Reference

**Fast lookup guide for operators and DevOps team**

---

## Critical Metrics at a Glance

### Orchestrator Health
```promql
# Is the orchestrator healthy?
gl_001_orchestrator_health_status
# 1 = healthy, 0 = unhealthy

# How long has it been running?
gl_001_orchestrator_uptime_seconds / 3600
# Result in hours

# Current state
gl_001_orchestrator_state
# 0=INIT, 1=READY, 2=EXECUTING, 3=ERROR, 4=RECOVERING, 5=TERMINATED
```

### Fleet Status
```promql
# How many plants are active?
gl_001_active_plants_count

# How many sub-agents are active (out of 99)?
gl_001_active_subagents_count

# Fleet efficiency
gl_001_aggregate_thermal_efficiency_percent
# Target: > 85%
```

### Performance
```promql
# Request rate (requests/sec)
rate(gl_001_orchestration_requests_total[5m])

# Error rate (percentage)
100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m]))
/ sum(rate(gl_001_orchestration_requests_total[5m]))
# Target: < 1%

# P95 latency (seconds)
histogram_quantile(0.95, rate(gl_001_orchestration_duration_seconds_bucket[5m]))
# Target: < 5s
```

---

## Top 10 Troubleshooting Queries

### 1. Which plants are unhealthy?
```promql
gl_001_plant_health_status{} == 0
```

### 2. Which sub-agents are down?
```promql
gl_001_subagent_health_status{} == 0
```

### 3. Which plants have low efficiency?
```promql
gl_001_plant_thermal_efficiency_percent < 75
```

### 4. What's the current error rate?
```promql
100 * sum(rate(gl_001_orchestration_requests_total{status="failure"}[5m]))
/ sum(rate(gl_001_orchestration_requests_total[5m]))
```

### 5. Which agents have the highest queue depth?
```promql
topk(10, gl_001_subagent_message_queue_depth)
```

### 6. Which SCADA connections are down?
```promql
gl_001_scada_connection_status == 0
```

### 7. What's the total heat loss across all plants?
```promql
sum(gl_001_plant_heat_losses_mw)
```

### 8. Which plants are in emissions violation?
```promql
gl_001_emissions_compliance_status == 0
```

### 9. What's the cache hit rate?
```promql
100 * sum(rate(gl_001_calculation_cache_hits_total[5m]))
/ (sum(rate(gl_001_calculation_cache_hits_total[5m])) + sum(rate(gl_001_calculation_cache_misses_total[5m])))
```

### 10. How many tasks are pending?
```promql
sum(gl_001_task_queue_depth) by (priority)
```

---

## Critical Alert Quick Response

### GL001MasterOrchestratorDown
**Impact**: Total system outage
**Response**:
1. Check pod: `kubectl get pods -n greenlang -l app=gl-001`
2. View logs: `kubectl logs -n greenlang <pod> --tail=100`
3. Restart: `kubectl rollout restart deployment/gl-001 -n greenlang`
4. Escalate if not resolved in 5 minutes

### GL001AllSubAgentsFailed
**Impact**: Zero operational capacity
**Response**:
1. Check message bus: `kubectl get svc message-bus -n greenlang`
2. Check network: `kubectl exec -it <gl-001-pod> -- ping <subagent-service>`
3. Review deployment: `kubectl get deployments -n greenlang | grep gl-`
4. Check for recent changes

### GL001MultiPlantHeatLoss
**Impact**: Massive energy waste (> 100 MW)
**Response**:
1. Identify source: Query `gl_001_plant_heat_losses_mw`
2. Check for distribution issues, leaks, or equipment failures
3. Alert plant operations team
4. Initiate emergency optimization

### GL001EmissionsComplianceViolation
**Impact**: Regulatory violation - potential fines
**Response**:
1. Identify plant: Query `gl_001_emissions_compliance_status == 0`
2. Check pollutant levels: `gl_001_emissions_co2_tons_hr`, `gl_001_emissions_nox_kg_hr`
3. Alert compliance team immediately
4. Initiate emissions reduction protocol
5. Document for regulatory reporting

---

## Dashboard Quick Links

| Dashboard | URL | Use Case |
|-----------|-----|----------|
| Master Orchestrator | `/d/gl-001-master` | DevOps operations |
| Multi-Plant | `/d/gl-001-multi-plant` | Plant performance |
| Sub-Agents | `/d/gl-001-subagents` | Agent health |
| Thermal Efficiency | `/d/gl-001-thermal` | Energy optimization |
| Operations | `/d/gl-001-operations` | 24/7 NOC |

---

## Common kubectl Commands

```bash
# Check GL-001 pod status
kubectl get pods -n greenlang -l app=gl-001-orchestrator

# View logs (last 100 lines)
kubectl logs -n greenlang <pod-name> --tail=100

# Follow logs in real-time
kubectl logs -n greenlang <pod-name> -f

# Check metrics endpoint
kubectl port-forward -n greenlang svc/gl-001-orchestrator 8000:8000
curl http://localhost:8000/metrics | grep gl_001

# Restart orchestrator
kubectl rollout restart deployment/gl-001-orchestrator -n greenlang

# Scale horizontally
kubectl scale deployment gl-001-orchestrator --replicas=3 -n greenlang

# Check resource usage
kubectl top pod -n greenlang -l app=gl-001-orchestrator
```

---

## Metrics Endpoint

**URL**: `http://gl-001-orchestrator:8000/metrics`

**Sample Output**:
```
# HELP gl_001_orchestrator_health_status Master orchestrator health
# TYPE gl_001_orchestrator_health_status gauge
gl_001_orchestrator_health_status 1.0

# HELP gl_001_active_plants_count Number of active plants
# TYPE gl_001_active_plants_count gauge
gl_001_active_plants_count 12.0

# HELP gl_001_active_subagents_count Number of active sub-agents
# TYPE gl_001_active_subagents_count gauge
gl_001_active_subagents_count{agent_category="boiler"} 15.0
gl_001_active_subagents_count{agent_category="steam"} 10.0
```

---

## Alert Severity & Response Times

| Severity | Response Time | Examples | Action |
|----------|---------------|----------|--------|
| CRITICAL | Immediate (< 5 min) | Orchestrator down, All agents failed | Page on-call, war room |
| WARNING | 15 minutes | Low efficiency, Agent degradation | Investigate, ticket |
| INFO | 1 hour | Low savings, Business metrics | Review, plan |

---

## Key Performance Thresholds

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Orchestrator Availability | 99.9% | < 99.9% | < 99.5% |
| P95 Latency | < 2s | > 2s | > 5s |
| Error Rate | < 0.1% | > 0.1% | > 1% |
| Fleet Efficiency | > 85% | < 85% | < 75% |
| Sub-Agent Availability | > 95% | < 95% | < 80% |
| Cache Hit Rate | > 80% | < 80% | < 60% |

---

## Contact & Escalation

| Issue Type | First Contact | Escalation | Emergency |
|------------|---------------|------------|-----------|
| Orchestrator Down | DevOps On-Call | Engineering Lead | CTO |
| Plant Failure | Operations Manager | Plant Director | VP Operations |
| Emissions Violation | Compliance Officer | Legal Team | Chief Compliance Officer |
| Data Integration | Integration Team | Platform Lead | VP Engineering |
| Performance Issues | DevOps Team | SRE Lead | VP Engineering |

### Contact Methods
- **PagerDuty**: GL-001 Escalation Policy
- **Slack**: #greenlang-gl-001 (primary), #greenlang-critical (critical)
- **Email**: ops@greenlang.io
- **Phone**: +1-XXX-XXX-XXXX (emergency hotline)

---

## Runbook Index

| Scenario | Runbook Link |
|----------|--------------|
| Orchestrator Down | https://runbooks.greenlang.io/gl-001-orchestrator-down |
| All Agents Failed | https://runbooks.greenlang.io/gl-001-all-agents-failed |
| High Memory Usage | https://runbooks.greenlang.io/gl-001-high-memory |
| SCADA Connection Lost | https://runbooks.greenlang.io/gl-001-scada-lost |
| ERP Integration Failure | https://runbooks.greenlang.io/gl-001-erp-failure |
| Emissions Violation | https://runbooks.greenlang.io/gl-001-emissions-violation |
| Performance Degradation | https://runbooks.greenlang.io/gl-001-perf-degradation |

---

## Quick Health Check Script

```bash
#!/bin/bash
# GL-001 Health Check Script

echo "=== GL-001 Health Check ==="

# 1. Pod status
echo -e "\n[1] Pod Status:"
kubectl get pods -n greenlang -l app=gl-001-orchestrator

# 2. Orchestrator health
echo -e "\n[2] Orchestrator Health:"
curl -s http://gl-001-orchestrator:8000/metrics | grep gl_001_orchestrator_health_status

# 3. Active plants
echo -e "\n[3] Active Plants:"
curl -s http://gl-001-orchestrator:8000/metrics | grep gl_001_active_plants_count

# 4. Active sub-agents
echo -e "\n[4] Active Sub-Agents:"
curl -s http://gl-001-orchestrator:8000/metrics | grep gl_001_active_subagents_count

# 5. Recent errors
echo -e "\n[5] Recent Errors:"
kubectl logs -n greenlang -l app=gl-001-orchestrator --tail=10 | grep ERROR

# 6. Current alerts
echo -e "\n[6] Active Alerts:"
curl -s http://alertmanager:9093/api/v2/alerts | jq '.[] | select(.labels.job=="gl-001-orchestrator")'

echo -e "\n=== Health Check Complete ==="
```

---

**Keep this guide handy for quick reference during incidents!**

**Last Updated**: 2025-11-17
**Version**: 1.0.0
