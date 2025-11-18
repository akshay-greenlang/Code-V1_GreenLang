# GL-003 Monitoring - Quick Reference

## Common Prometheus Queries

### Request Metrics

```promql
# Request rate
sum(rate(gl_003_http_requests_total[5m])) by (endpoint)

# Error rate percentage
100 * sum(rate(gl_003_http_requests_total{status="error"}[5m]))
/ sum(rate(gl_003_http_requests_total[5m]))

# P95 latency
histogram_quantile(0.95, sum(rate(gl_003_http_request_duration_seconds_bucket[5m])) by (le, endpoint))

# P99 latency
histogram_quantile(0.99, sum(rate(gl_003_http_request_duration_seconds_bucket[5m])) by (le))
```

### Steam System Metrics

```promql
# Average steam pressure
avg(gl_003_steam_pressure_bar) by (system_id)

# Steam flow rate over time
gl_003_steam_flow_rate_kg_hr{system_id="STEAM-001"}

# Condensate return efficiency
gl_003_condensate_return_percent{system_id="STEAM-001"}

# Distribution efficiency by system
avg(gl_003_distribution_efficiency_percent) by (system_id)

# Total heat loss
sum(gl_003_pipe_heat_loss_kw) by (system_id)
```

### Leak Detection

```promql
# Active critical leaks
sum(gl_003_active_leaks_count{severity="critical"}) by (system_id)

# Total leak cost impact
sum(gl_003_leak_cost_impact_usd_hr) by (system_id)

# Leak detection confidence
avg(gl_003_leak_detection_confidence_percent) by (system_id)

# Leaks detected in last 24h
increase(gl_003_steam_leaks_detected_total[24h])
```

### Steam Trap Performance

```promql
# Failed steam traps
sum(gl_003_steam_trap_failed_count) by (system_id, failure_mode)

# Trap failure rate
100 * sum(gl_003_steam_trap_failed_count) by (system_id)
/ (sum(gl_003_steam_trap_operational_count) by (system_id) + sum(gl_003_steam_trap_failed_count) by (system_id))

# Trap performance score
avg(gl_003_steam_trap_performance_score_percent) by (system_id)
```

### Analysis Performance

```promql
# Analysis success rate
100 * sum(rate(gl_003_analysis_requests_total{status="success"}[5m]))
/ sum(rate(gl_003_analysis_requests_total[5m]))

# Analysis duration by type
histogram_quantile(0.95, sum(rate(gl_003_analysis_duration_seconds_bucket[5m])) by (le, analysis_type))

# Efficiency improvement distribution
histogram_quantile(0.50, sum(rate(gl_003_analysis_efficiency_improvement_percent_bucket[1h])) by (le))
```

### Cache Performance

```promql
# Cache hit rate
100 * sum(rate(gl_003_cache_hits_total[5m]))
/ (sum(rate(gl_003_cache_hits_total[5m])) + sum(rate(gl_003_cache_misses_total[5m])))

# Cache hit rate by pattern
100 * sum(rate(gl_003_cache_hits_total[5m])) by (cache_key_pattern)
/ (sum(rate(gl_003_cache_hits_total[5m])) by (cache_key_pattern) + sum(rate(gl_003_cache_misses_total[5m])) by (cache_key_pattern))

# Cache eviction rate
sum(rate(gl_003_cache_evictions_total[5m]))
```

### Business Metrics

```promql
# Total annual savings projection
sum(gl_003_analysis_annual_savings_usd)

# Annual energy savings
sum(gl_003_analysis_annual_energy_savings_mwh)

# Average cost savings per analysis
avg_over_time(gl_003_analysis_cost_savings_usd_sum[24h])

# Steam cost per ton
avg(gl_003_steam_cost_per_ton_usd) by (system_id)
```

### Determinism Metrics

```promql
# Determinism score
gl_003_determinism_score_percent{component="orchestrator"}

# Determinism violations
sum(rate(gl_003_determinism_verification_failures_total[1h])) by (violation_type)

# Provenance hash verification success rate
100 * sum(rate(gl_003_provenance_hash_verifications_total{status="success"}[1h]))
/ sum(rate(gl_003_provenance_hash_verifications_total[1h]))
```

## Common kubectl Commands

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-003-steam-analyzer

# View logs
kubectl logs -n greenlang -l app=gl-003-steam-analyzer --tail=100 -f

# Check metrics endpoint
kubectl port-forward -n greenlang svc/gl-003-metrics 8000:8000
curl http://localhost:8000/metrics | grep gl_003

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-analyzer -n greenlang

# Scale deployment
kubectl scale deployment gl-003-steam-analyzer --replicas=3 -n greenlang

# Check resource usage
kubectl top pod -n greenlang -l app=gl-003-steam-analyzer

# Describe pod (troubleshooting)
kubectl describe pod <pod-name> -n greenlang
```

## Alert Triage

### Critical Leak Alert

```bash
# 1. Check active leaks
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=gl_003_active_leaks_count{severity="critical"}' | jq

# 2. Get leak details
kubectl logs -n greenlang -l app=gl-003-steam-analyzer | grep "CRITICAL_LEAK"

# 3. Check leak cost impact
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=gl_003_leak_cost_impact_usd_hr{system_id="STEAM-001"}' | jq
```

### Low Distribution Efficiency

```bash
# 1. Check current efficiency
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=gl_003_distribution_efficiency_percent' | jq

# 2. Check heat losses
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(gl_003_pipe_heat_loss_kw) by (system_id, pipe_segment)' | jq

# 3. Check steam trap failures
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(gl_003_steam_trap_failed_count) by (system_id, trap_type)' | jq
```

### High Error Rate

```bash
# 1. Identify error endpoints
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=sum(rate(gl_003_http_requests_total{status="error"}[5m])) by (endpoint)' | jq

# 2. Check application logs
kubectl logs -n greenlang -l app=gl-003-steam-analyzer --tail=500 | grep ERROR

# 3. Check recent deployments
kubectl rollout history deployment/gl-003-steam-analyzer -n greenlang
```

## Dashboard URLs

- Executive: https://grafana.greenlang.io/d/gl-003-executive
- Operations: https://grafana.greenlang.io/d/gl-003-operations
- Agent: https://grafana.greenlang.io/d/gl-003-agent
- Quality: https://grafana.greenlang.io/d/gl-003-quality
- Determinism: https://grafana.greenlang.io/d/gl-003-determinism
- Feedback: https://grafana.greenlang.io/d/gl-003-feedback

## Runbook Links

- Agent Unavailable: https://docs.greenlang.io/runbooks/gl-003-unavailable
- High Error Rate: https://docs.greenlang.io/runbooks/gl-003-high-error-rate
- Critical Leak: https://docs.greenlang.io/runbooks/gl-003-critical-leak
- Low Efficiency: https://docs.greenlang.io/runbooks/gl-003-low-efficiency
- Trap Failure: https://docs.greenlang.io/runbooks/gl-003-trap-failure

## Contact

- **On-Call**: PagerDuty - GL-003 Escalation Policy
- **Slack**: #greenlang-gl-003
- **Email**: ops@greenlang.io
