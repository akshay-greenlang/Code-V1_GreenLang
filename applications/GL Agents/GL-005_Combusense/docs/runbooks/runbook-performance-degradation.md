# Runbook: Performance Degradation

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-005 |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | MEDIUM - Operational |
| Review Cycle | Quarterly |

---

## Overview

This runbook provides procedures for detecting, diagnosing, and resolving performance degradation in the GL-005 CombustionControlAgent system. Performance issues can lead to control instability, missed safety responses, and operational inefficiency.

**Performance Requirements:**
- Control loop cycle time: <100ms
- DCS read latency: <50ms
- DCS write latency: <100ms
- Flame detection response: <50ms
- Safety interlock response: <200ms

---

## Slow Response Detection

### Performance Metrics

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Control cycle time | <100ms | >100ms | >200ms |
| DCS read latency | <50ms | >80ms | >150ms |
| DCS write latency | <100ms | >150ms | >300ms |
| Calculation time | <20ms | >40ms | >80ms |
| Memory usage | <70% | >80% | >90% |
| CPU usage | <60% | >75% | >90% |
| Database query time | <30ms | >50ms | >100ms |

### Prometheus Alerts

```yaml
# Control cycle too slow
- alert: ControlCycleSlowWarning
  expr: histogram_quantile(0.95, gl005_control_cycle_duration_seconds_bucket) > 0.1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Control cycle P95 latency exceeds 100ms"

- alert: ControlCycleCritical
  expr: histogram_quantile(0.99, gl005_control_cycle_duration_seconds_bucket) > 0.2
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Control cycle P99 latency exceeds 200ms - control stability at risk"

# DCS communication slow
- alert: DCSLatencyHigh
  expr: histogram_quantile(0.95, gl005_dcs_read_latency_seconds_bucket) > 0.08
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "DCS read latency P95 exceeds 80ms"

# Resource exhaustion approaching
- alert: MemoryUsageHigh
  expr: gl005_memory_usage_percent > 85
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Memory usage at {{ $value }}%"

- alert: CPUUsageHigh
  expr: gl005_cpu_usage_percent > 80
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "CPU usage at {{ $value }}%"
```

### Symptoms

**Control System Symptoms:**
- Heat output oscillating more than normal
- Setpoints not being tracked smoothly
- PID controller hunting
- Delayed response to load changes

**System Symptoms:**
- Pod restart events
- OOMKilled events in logs
- Prometheus scrape timeouts
- SCADA data updates delayed

**Log Patterns:**
```
WARNING - CombustionControlOrchestrator - Control cycle time 156ms exceeds 100ms target
WARNING - DCSConnector - Read latency 92ms exceeds threshold
ERROR - CalculationEngine - Calculation timeout after 50ms
WARNING - Kubernetes - Memory usage approaching limit
INFO - HPA - Scaling up from 3 to 5 replicas due to CPU
```

---

## Diagnostic Procedures

### Step 1: Quick Health Check

```bash
# Overall system health
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/health | jq '.'

# Performance metrics summary
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/metrics/performance | jq '.'

# Pod resource usage
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Recent events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | tail -20
```

### Step 2: Detailed Latency Analysis

```bash
# Control cycle breakdown
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "gl005_control_cycle"

# DCS latency histogram
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "gl005_dcs_read_latency"

# Identify slowest components
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/metrics/latency-breakdown | jq '.'

# Example response:
# {
#   "total_cycle_ms": 145,
#   "breakdown": {
#     "dcs_read": 65,
#     "calculations": 35,
#     "safety_validation": 25,
#     "dcs_write": 15,
#     "logging": 5
#   },
#   "bottleneck": "dcs_read"
# }
```

### Step 3: Resource Analysis

```bash
# Detailed pod resources
kubectl describe pod -n greenlang -l app=gl-005-combustion-control | \
  grep -A 20 "Containers:"

# Memory profiling
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
import tracemalloc
tracemalloc.start()
# Run control cycle
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
"

# CPU profiling (sample)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
import cProfile
import pstats
# Profile control cycle
profiler = cProfile.Profile()
profiler.enable()
# Run control cycle here
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
"
```

### Step 4: Database Analysis

```bash
# Check database connection pool
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/database/pool-status | jq '.'

# Identify slow queries
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/database/slow-queries | jq '.'

# Check database latency
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "database_query_duration"
```

### Step 5: Integration Analysis

```bash
# Check all integration latencies
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/latency-report | jq '.'

# Network latency to DCS
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 10 <DCS_HOST> | tail -3

# Check for packet loss
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 100 -i 0.1 <DCS_HOST> | grep "packet loss"
```

---

## Resource Exhaustion Handling

### Memory Exhaustion

**Symptoms:**
- OOMKilled events
- Pod restarts
- Gradual memory increase over time
- Swap usage (if enabled)

**Immediate Actions:**
```bash
# Check current memory usage
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Check for memory leak indicators
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "process_resident_memory"

# Check pod restart count
kubectl get pods -n greenlang -l app=gl-005-combustion-control \
  -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}'
```

**Resolution:**
```bash
# Increase memory limit
kubectl patch deployment gl-005-combustion-control -n greenlang -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"gl-005","resources":{"limits":{"memory":"3Gi"},"requests":{"memory":"2Gi"}}}]}}}}'

# If memory leak suspected, restart with fresh memory
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang

# Monitor memory after restart
watch -n 5 "kubectl top pod -n greenlang -l app=gl-005-combustion-control"
```

**Long-term Fix:**
- Profile memory usage to identify leak
- Review buffer sizes and cache configurations
- Check for circular references in code
- Update to latest version (may contain fix)

### CPU Exhaustion

**Symptoms:**
- High CPU metrics
- Control cycle timeouts
- HPA scaling up
- Thermal throttling (if bare metal)

**Immediate Actions:**
```bash
# Check CPU usage
kubectl top pod -n greenlang -l app=gl-005-combustion-control

# Check HPA status
kubectl get hpa gl-005-hpa -n greenlang

# Check for CPU-intensive processes
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  top -b -n 1 | head -20
```

**Resolution:**
```bash
# Scale up replicas immediately
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=5

# Or patch HPA minimums
kubectl patch hpa gl-005-hpa -n greenlang -p '{"spec":{"minReplicas":5}}'

# Increase CPU limit
kubectl patch deployment gl-005-combustion-control -n greenlang -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"gl-005","resources":{"limits":{"cpu":"2"},"requests":{"cpu":"1"}}}]}}}}'
```

**Long-term Fix:**
- Profile CPU usage to identify hotspots
- Optimize calculation algorithms
- Add caching for repeated calculations
- Consider async processing for non-critical paths

### Connection Pool Exhaustion

**Symptoms:**
- "Connection pool exhausted" errors
- Increasing query latency
- Timeout errors on database operations

**Immediate Actions:**
```bash
# Check pool status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/database/pool-status | jq '.'

# Check for connection leaks
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=100 | \
  grep -i "connection\|pool\|timeout"
```

**Resolution:**
```bash
# Increase connection pool size
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DB_POOL_SIZE=30 DB_POOL_MAX_OVERFLOW=10

# Restart to apply new pool settings
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang
```

---

## Scaling Procedures

### Horizontal Scaling (Add Replicas)

**When to Scale:**
- CPU usage >75% sustained
- Control cycle time >100ms sustained
- Upcoming load increase expected

**Procedure:**
```bash
# Check current replica count
kubectl get deployment gl-005-combustion-control -n greenlang

# Manual scale up
kubectl scale deployment gl-005-combustion-control -n greenlang --replicas=<NEW_COUNT>

# Verify new pods are ready
kubectl get pods -n greenlang -l app=gl-005-combustion-control -w

# Monitor after scaling
kubectl top pod -n greenlang -l app=gl-005-combustion-control
```

**HPA Configuration:**
```yaml
# Check current HPA settings
kubectl get hpa gl-005-hpa -n greenlang -o yaml

# Modify HPA parameters
kubectl patch hpa gl-005-hpa -n greenlang -p '{
  "spec": {
    "minReplicas": 3,
    "maxReplicas": 10,
    "metrics": [
      {
        "type": "Resource",
        "resource": {
          "name": "cpu",
          "target": {
            "type": "Utilization",
            "averageUtilization": 70
          }
        }
      }
    ]
  }
}'
```

### Vertical Scaling (Increase Resources)

**When to Scale:**
- Memory usage >85%
- Single-threaded bottleneck identified
- Database connection limits reached

**Procedure:**
```bash
# Check current resource allocation
kubectl get deployment gl-005-combustion-control -n greenlang -o yaml | \
  grep -A 10 "resources:"

# Apply new resource limits
kubectl patch deployment gl-005-combustion-control -n greenlang -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "gl-005",
          "resources": {
            "requests": {
              "cpu": "1500m",
              "memory": "2Gi"
            },
            "limits": {
              "cpu": "3",
              "memory": "4Gi"
            }
          }
        }]
      }
    }
  }
}'

# Monitor rollout
kubectl rollout status deployment/gl-005-combustion-control -n greenlang
```

### Scaling Best Practices

| Scenario | Recommended Approach |
|----------|---------------------|
| Sustained high CPU | Horizontal scaling (more replicas) |
| Memory pressure | Vertical scaling (more RAM) |
| Database bottleneck | Connection pool + read replicas |
| Network latency | Closer placement / dedicated network |
| Calculation intensive | Algorithm optimization + caching |

---

## Rollback Procedures

### When to Rollback

- New deployment introduced performance regression
- Configuration change caused issues
- Memory leak in new version
- Control instability after update

### Quick Rollback

```bash
# Check rollout history
kubectl rollout history deployment/gl-005-combustion-control -n greenlang

# Rollback to previous version
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-005-combustion-control -n greenlang \
  --to-revision=<REVISION_NUMBER>

# Verify rollback
kubectl rollout status deployment/gl-005-combustion-control -n greenlang

# Check which version is running
kubectl get deployment gl-005-combustion-control -n greenlang \
  -o jsonpath='{.spec.template.spec.containers[0].image}'
```

### Configuration Rollback

```bash
# Check ConfigMap history (if using version control)
kubectl get configmap gl-005-config -n greenlang -o yaml

# Restore previous ConfigMap
kubectl apply -f <PREVIOUS_CONFIG_FILE>

# Force pod restart to pick up config
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang
```

### Database Rollback

```bash
# If database migration caused issues
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python manage.py migrate <APP_NAME> <PREVIOUS_MIGRATION>

# Verify database state
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python manage.py showmigrations
```

### Rollback Verification

```bash
# Verify performance restored
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/metrics/performance | jq '.'

# Compare with baseline
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "gl005_control_cycle_duration"

# Monitor for stability (5-10 minutes)
watch -n 10 "kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/health | jq '.performance'"
```

---

## Prevention Measures

### Proactive Monitoring

```yaml
# Performance SLO alerts
- alert: PerformanceSLOBreach
  expr: |
    (
      histogram_quantile(0.95, gl005_control_cycle_duration_seconds_bucket) > 0.08
    ) /
    (
      histogram_quantile(0.95, gl005_control_cycle_duration_seconds_bucket) < 0.1
    ) * 100 < 95
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "Performance SLO at risk - P95 latency approaching threshold"

# Trend-based alerts
- alert: MemoryTrendIncreasing
  expr: predict_linear(gl005_memory_usage_percent[1h], 3600) > 95
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Memory usage predicted to exceed 95% in 1 hour"
```

### Capacity Planning

| Time Horizon | Review Frequency | Actions |
|--------------|-----------------|---------|
| Daily | Automated | Alert on threshold breach |
| Weekly | Manual | Review trends, adjust HPA |
| Monthly | Meeting | Capacity planning review |
| Quarterly | Strategic | Infrastructure planning |

### Performance Testing

| Test Type | Frequency | Target |
|-----------|-----------|--------|
| Load test | Weekly (automated) | 150% normal load |
| Stress test | Monthly | Find breaking point |
| Soak test | Quarterly | 48-hour sustained load |
| Chaos test | Quarterly | Random failure injection |

### Performance Baselines

Maintain documented baselines:

```yaml
# Performance baseline (update quarterly)
baseline:
  date: "2025-12-22"
  version: "1.0.0"
  conditions: "Normal production load"
  metrics:
    control_cycle_p50_ms: 45
    control_cycle_p95_ms: 78
    control_cycle_p99_ms: 95
    dcs_read_latency_p95_ms: 35
    memory_usage_percent: 55
    cpu_usage_percent: 40
    pod_count: 3
```

---

## Appendix

### A. Performance Tuning Parameters

| Parameter | Default | Tuning Range | Impact |
|-----------|---------|--------------|--------|
| DB_POOL_SIZE | 10 | 5-50 | DB connection availability |
| CONTROL_CYCLE_INTERVAL_MS | 100 | 50-200 | Control responsiveness |
| DCS_READ_TIMEOUT_MS | 500 | 200-1000 | Integration reliability |
| CALCULATION_CACHE_SIZE | 1000 | 100-10000 | Memory vs speed |
| LOG_LEVEL | INFO | DEBUG-ERROR | I/O overhead |

### B. Grafana Dashboard

Key panels to include:
- Control cycle latency histogram
- DCS latency time series
- CPU/Memory usage
- Pod count and HPA status
- Error rate
- Database connection pool
- Integration health

### C. Related Documentation

- [SCALING_GUIDE.md](../../runbooks/SCALING_GUIDE.md) - Detailed scaling procedures
- [TROUBLESHOOTING.md](../../runbooks/TROUBLESHOOTING.md) - General troubleshooting
- [runbook-communication-loss.md](./runbook-communication-loss.md) - DCS latency issues

### D. Emergency Contacts

| Role | Contact |
|------|---------|
| Platform Engineering | @platform-oncall |
| Database Admin | @dba-oncall |
| Network Operations | @network-oncall |

### E. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |
