# GL-002 Monitoring - Quick Reference Card

## Dashboard URLs

| Dashboard | URL | Use Case |
|-----------|-----|----------|
| **Executive** | https://grafana.greenlang.io/d/gl-002-executive | KPIs, ROI, Carbon Impact |
| **Operations** | https://grafana.greenlang.io/d/gl-002-operations | Health, Performance, Errors |
| **Agent** | https://grafana.greenlang.io/d/gl-002-agent | Tools, Cache Performance |
| **Quality** | https://grafana.greenlang.io/d/gl-002-quality | Determinism, Accuracy |

## Key Metrics

### Success Metrics
```promql
# Error Rate (Target: <0.1%)
100 * sum(rate(gl_002_http_requests_total{status="error"}[5m]))
/ sum(rate(gl_002_http_requests_total[5m]))

# P95 Latency (Target: <2s)
histogram_quantile(0.95, sum(rate(gl_002_http_request_duration_seconds_bucket[5m])) by (le))

# Cache Hit Rate (Target: >85%)
100 * sum(rate(gl_002_cache_hits_total[5m]))
/ (sum(rate(gl_002_cache_hits_total[5m])) + sum(rate(gl_002_cache_misses_total[5m])))
```

### Business Metrics
```promql
# Annual Cost Savings
sum(gl_002_optimization_annual_savings_usd)

# Annual CO2 Reduction (tons)
sum(gl_002_optimization_annual_emissions_reduction_tons)

# Average Efficiency
avg(gl_002_boiler_efficiency_percent)
```

## Critical Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| **AgentUnavailable** | Down >1min | Restart pod, check logs |
| **HighErrorRate** | >5% | Check application logs, rollback if needed |
| **DeterminismFailure** | Failure >1% | Review calculation logic, check data quality |
| **ComplianceViolation** | Any violation | Notify compliance team, regulatory reporting |
| **DatabaseFailure** | Pool = 0 | Check DB connectivity, scale pool |
| **HighMemory** | >4GB | Check for leaks, restart pod, scale horizontally |
| **OptimizationTimeout** | p95 >10s | Investigate slow calculations, optimize algorithms |

## Quick Commands

### Check Agent Status
```bash
# Pod status
kubectl get pods -n greenlang -l app=gl-002-boiler-optimizer

# Logs
kubectl logs -n greenlang -l app=gl-002-boiler-optimizer --tail=100

# Metrics endpoint
kubectl port-forward -n greenlang svc/gl-002-metrics 8000:8000
curl http://localhost:8000/metrics | grep gl_002
```

### Check Monitoring Stack
```bash
# Prometheus targets
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Visit: http://localhost:9090/targets

# Alert status
# Visit: http://localhost:9090/alerts

# Grafana
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Visit: http://localhost:3000
```

### Troubleshooting
```bash
# Check ServiceMonitor
kubectl get servicemonitor -n greenlang gl-002-boiler-optimizer

# Check PrometheusRule
kubectl get prometheusrule -n greenlang gl-002-alerts

# Test metric collection
kubectl exec -it -n greenlang <pod-name> -- curl localhost:8000/metrics

# Check AlertManager
kubectl logs -n monitoring -l app=alertmanager --tail=50
```

## Alert Severity Response

| Severity | Response Time | Notification | Escalation |
|----------|--------------|--------------|------------|
| **CRITICAL** | Immediate | PagerDuty | After 5 min |
| **WARNING** | 15 minutes | Slack | After 1 hour |
| **INFO** | 1 hour | Email | None |

## Common Issues

### Metrics Not Showing
1. Check `/metrics` endpoint is accessible
2. Verify ServiceMonitor matches service labels
3. Check Prometheus targets page
4. Review Prometheus logs

### Dashboard No Data
1. Test query in Prometheus directly
2. Verify datasource configuration in Grafana
3. Check time range matches data availability
4. Verify metric names are correct

### Alerts Not Firing
1. Check PrometheusRule is loaded
2. Verify alert expression returns data in Prometheus
3. Check AlertManager routing configuration
4. Review AlertManager logs

## Runbook Links

- **Agent Unavailable:** https://runbooks.greenlang.io/gl-002/agent-unavailable
- **High Error Rate:** https://runbooks.greenlang.io/gl-002/high-error-rate
- **Performance Degradation:** https://runbooks.greenlang.io/gl-002/performance-degradation
- **Low Cache Hit:** https://runbooks.greenlang.io/gl-002/low-cache-hit
- **Compliance Violation:** https://runbooks.greenlang.io/gl-002/compliance-violation

## Contact

- **Slack:** #greenlang-gl-002
- **Email:** ops@greenlang.io
- **PagerDuty:** GL-002 Escalation Policy
- **Documentation:** https://docs.greenlang.io/gl-002/monitoring

## SLOs

| Metric | Target | Current |
|--------|--------|---------|
| Availability | 99.9% | [Dashboard] |
| P95 Latency | <2s | [Dashboard] |
| Error Rate | <0.1% | [Dashboard] |
| MTTR | <15min | [Incidents] |

---

**Print this card and keep it handy!**
