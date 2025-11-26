# GL-006 HeatRecoveryMaximizer Troubleshooting Guide

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-006 |
| Codename | HEATRECLAIM |
| Version | 1.0.0 |
| Last Updated | 2024-11-26 |

---

## 1. Quick Diagnostics

### 1.1 Health Check Commands

```bash
# Check pod status
kubectl get pods -n greenlang -l app=gl-006-heatreclaim

# Check service endpoints
kubectl get endpoints gl-006-heatreclaim -n greenlang

# Verify health endpoint
kubectl exec -it <pod-name> -n greenlang -- curl -s localhost:8000/health

# Check readiness
kubectl exec -it <pod-name> -n greenlang -- curl -s localhost:8000/ready
```

### 1.2 Quick Status Dashboard

```bash
# All-in-one status check
echo "=== PODS ===" && \
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide && \
echo "=== HPA ===" && \
kubectl get hpa gl-006-heatreclaim-hpa -n greenlang && \
echo "=== EVENTS ===" && \
kubectl get events -n greenlang --field-selector involvedObject.name=gl-006-heatreclaim --sort-by='.lastTimestamp' | tail -10
```

---

## 2. Common Issues and Solutions

### 2.1 Pod Issues

#### 2.1.1 Pod in CrashLoopBackOff

**Symptoms:**
- Pod status shows `CrashLoopBackOff`
- Repeated restarts

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n greenlang

# Check previous logs
kubectl logs <pod-name> -n greenlang --previous

# Check container exit code
kubectl get pod <pod-name> -n greenlang -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
```

**Common Causes and Solutions:**

| Exit Code | Cause | Solution |
|-----------|-------|----------|
| 1 | Application error | Check logs for exception |
| 137 | OOM killed | Increase memory limits |
| 143 | SIGTERM | Check for graceful shutdown issues |

**Solution:**
```bash
# If OOM, increase memory
kubectl patch deployment gl-006-heatreclaim -n greenlang -p '{"spec":{"template":{"spec":{"containers":[{"name":"gl-006-heatreclaim","resources":{"limits":{"memory":"1Gi"}}}]}}}}'
```

#### 2.1.2 Pod in Pending State

**Symptoms:**
- Pod stuck in `Pending` status
- No container started

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n greenlang | grep -A 10 "Events:"
```

**Common Causes:**

| Cause | Event Message | Solution |
|-------|--------------|----------|
| Insufficient CPU | `Insufficient cpu` | Scale down or add nodes |
| Insufficient Memory | `Insufficient memory` | Scale down or add nodes |
| PVC not bound | `persistentvolumeclaim not found` | Create/bind PVC |
| Node selector | `node(s) didn't match` | Update node selector |

#### 2.1.3 Pod in ImagePullBackOff

**Symptoms:**
- Pod status shows `ImagePullBackOff` or `ErrImagePull`

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n greenlang | grep -A 5 "Warning"
```

**Solutions:**
```bash
# Check image name
kubectl get deployment gl-006-heatreclaim -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# Check image pull secret
kubectl get secret -n greenlang | grep registry

# Verify image exists
docker pull <image-name>
```

---

### 2.2 Application Issues

#### 2.2.1 High Error Rate

**Symptoms:**
- Error rate > 1%
- Frequent 500 responses

**Diagnosis:**
```bash
# Check error metrics
curl -s localhost:9090/metrics | grep gl006_errors_total

# Check recent errors in logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=10m | grep -i error | tail -20
```

**Common Causes:**

| Error Type | Possible Cause | Solution |
|------------|---------------|----------|
| Database connection | DB unreachable | Check PostgreSQL pod |
| Redis connection | Cache unavailable | Check Redis pod |
| Validation error | Invalid input data | Check input validation |
| Calculation error | Algorithm issue | Check calculation logs |

#### 2.2.2 Calculation Failures

**Symptoms:**
- Pinch analysis fails
- Network synthesis errors
- ROI calculation errors

**Diagnosis:**
```bash
# Check calculation metrics
curl -s localhost:9090/metrics | grep gl006_calculation

# Check validation errors
curl -s localhost:9090/metrics | grep gl006_validation_failures

# Check specific calculation logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim | grep -E "(pinch|network|roi)" | tail -20
```

**Solutions:**

1. **Thermodynamic validation failures:**
   - Check input temperatures are above absolute zero
   - Verify temperature approach >= minimum (10C default)
   - Ensure energy balance is maintained

2. **Convergence issues:**
   - Increase max iterations in config
   - Adjust convergence tolerance
   - Check for conflicting constraints

#### 2.2.3 Slow Response Times

**Symptoms:**
- P99 latency > 5 seconds
- Timeouts on requests

**Diagnosis:**
```bash
# Check latency metrics
curl -s localhost:9090/metrics | grep gl006_http_request_duration

# Check active calculations
curl -s localhost:9090/metrics | grep gl006_active_calculations

# Check resource usage
kubectl top pods -n greenlang -l app=gl-006-heatreclaim
```

**Solutions:**
```bash
# Scale horizontally
kubectl scale deployment gl-006-heatreclaim --replicas=5 -n greenlang

# Increase resources
kubectl patch deployment gl-006-heatreclaim -n greenlang -p '{"spec":{"template":{"spec":{"containers":[{"name":"gl-006-heatreclaim","resources":{"limits":{"cpu":"1000m","memory":"1Gi"}}}]}}}}'
```

---

### 2.3 Database Issues

#### 2.3.1 Database Connection Errors

**Symptoms:**
- `Connection refused` errors
- `Too many connections` errors

**Diagnosis:**
```bash
# Check PostgreSQL pod
kubectl get pods -n greenlang -l app=postgresql

# Check connection count
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -c "SELECT count(*) FROM pg_stat_activity;"

# Check database logs
kubectl logs postgresql-0 -n greenlang --tail=50
```

**Solutions:**
```bash
# Restart database if needed
kubectl rollout restart statefulset postgresql -n greenlang

# Increase connection pool (in ConfigMap)
kubectl edit configmap gl-006-heatreclaim-config -n greenlang
# Update DATABASE_POOL_SIZE
```

#### 2.3.2 Slow Queries

**Diagnosis:**
```bash
# Check slow queries
kubectl exec -it postgresql-0 -n greenlang -- psql -U greenlang -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 5;"
```

**Solutions:**
- Add missing indexes
- Optimize queries
- Increase work_mem for complex calculations

---

### 2.4 Cache Issues

#### 2.4.1 Redis Connection Errors

**Symptoms:**
- Cache misses
- Connection timeout to Redis

**Diagnosis:**
```bash
# Check Redis pod
kubectl get pods -n greenlang -l app=redis

# Check Redis connectivity
kubectl exec -it <app-pod> -n greenlang -- nc -zv redis 6379

# Check Redis info
kubectl exec -it redis-0 -n greenlang -- redis-cli INFO
```

**Solutions:**
```bash
# Restart Redis
kubectl rollout restart statefulset redis -n greenlang

# Clear cache if corrupted
kubectl exec -it redis-0 -n greenlang -- redis-cli FLUSHDB
```

---

### 2.5 Integration Issues

#### 2.5.1 SCADA Connection Failures

**Symptoms:**
- `scada_connection_status = 0`
- OPC UA connection errors

**Diagnosis:**
```bash
# Check integration metrics
curl -s localhost:9090/metrics | grep gl006_scada

# Check connectivity
kubectl exec -it <pod-name> -n greenlang -- nc -zv scada-server 4840

# Check logs for OPC UA errors
kubectl logs -n greenlang -l app=gl-006-heatreclaim | grep -i "opc\|scada" | tail -20
```

**Solutions:**
- Verify SCADA endpoint is correct
- Check firewall rules
- Verify OPC UA server is running
- Check authentication credentials

#### 2.5.2 Historian Data Missing

**Symptoms:**
- Empty data returned from historian
- Historical queries failing

**Diagnosis:**
```bash
# Check historian connection
curl -s localhost:9090/metrics | grep gl006_historian

# Check data points collected
curl -s localhost:9090/metrics | grep gl006_data_points_collected
```

**Solutions:**
- Verify historian endpoint
- Check time range in queries
- Verify data exists in historian
- Check query permissions

---

## 3. Performance Tuning

### 3.1 CPU Optimization

```bash
# Check CPU usage
kubectl top pods -n greenlang -l app=gl-006-heatreclaim

# If CPU is throttled, increase limits
kubectl patch deployment gl-006-heatreclaim -n greenlang -p '{"spec":{"template":{"spec":{"containers":[{"name":"gl-006-heatreclaim","resources":{"limits":{"cpu":"2000m"},"requests":{"cpu":"500m"}}}]}}}}'
```

### 3.2 Memory Optimization

```bash
# Check memory usage
kubectl top pods -n greenlang -l app=gl-006-heatreclaim

# If OOM issues, increase limits
kubectl patch deployment gl-006-heatreclaim -n greenlang -p '{"spec":{"template":{"spec":{"containers":[{"name":"gl-006-heatreclaim","resources":{"limits":{"memory":"2Gi"},"requests":{"memory":"512Mi"}}}]}}}}'
```

### 3.3 Calculation Performance

Optimize calculation performance by adjusting config:

```bash
# Edit ConfigMap
kubectl edit configmap gl-006-heatreclaim-config -n greenlang

# Key parameters:
# MAX_ITERATIONS: 1000 (increase for better accuracy)
# CONVERGENCE_TOLERANCE: 1e-6 (decrease for faster convergence)
# ENABLE_PARALLEL_PROCESSING: true
# MAX_PARALLEL_WORKERS: 4 (adjust based on CPU cores)
```

---

## 4. Log Analysis

### 4.1 Log Locations

```bash
# Application logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim

# Previous container logs
kubectl logs -n greenlang <pod-name> --previous

# All container logs
kubectl logs -n greenlang -l app=gl-006-heatreclaim --all-containers
```

### 4.2 Log Filtering

```bash
# Filter by log level
kubectl logs -n greenlang -l app=gl-006-heatreclaim | jq 'select(.level=="ERROR")'

# Filter by component
kubectl logs -n greenlang -l app=gl-006-heatreclaim | grep "pinch_analysis"

# Filter by time range
kubectl logs -n greenlang -l app=gl-006-heatreclaim --since=1h
```

### 4.3 Common Log Patterns

| Pattern | Meaning | Action |
|---------|---------|--------|
| `"level":"ERROR"` | Application error | Investigate root cause |
| `"timeout":true` | Request timeout | Check performance |
| `"validation_failed":true` | Input validation failed | Check input data |
| `"connection_error"` | Integration failure | Check connectivity |

---

## 5. Diagnostic Commands Reference

```bash
# === POD DIAGNOSTICS ===
kubectl get pods -n greenlang -l app=gl-006-heatreclaim -o wide
kubectl describe pod <pod-name> -n greenlang
kubectl logs <pod-name> -n greenlang
kubectl logs <pod-name> -n greenlang --previous
kubectl exec -it <pod-name> -n greenlang -- /bin/bash

# === DEPLOYMENT DIAGNOSTICS ===
kubectl get deployment gl-006-heatreclaim -n greenlang -o yaml
kubectl describe deployment gl-006-heatreclaim -n greenlang
kubectl rollout history deployment gl-006-heatreclaim -n greenlang

# === RESOURCE DIAGNOSTICS ===
kubectl top pods -n greenlang -l app=gl-006-heatreclaim
kubectl top nodes

# === NETWORK DIAGNOSTICS ===
kubectl get svc gl-006-heatreclaim -n greenlang
kubectl get endpoints gl-006-heatreclaim -n greenlang
kubectl exec -it <pod-name> -n greenlang -- curl localhost:8000/health

# === METRICS ===
curl -s localhost:9090/metrics | grep gl006

# === EVENTS ===
kubectl get events -n greenlang --sort-by='.lastTimestamp'
```

---

## 6. Related Documents

- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md)
- [ROLLBACK_PROCEDURE.md](./ROLLBACK_PROCEDURE.md)
- [SCALING_GUIDE.md](./SCALING_GUIDE.md)
- [MAINTENANCE.md](./MAINTENANCE.md)

---

*This guide is maintained by the Platform Team. For updates, contact platform-team@greenlang.io*
