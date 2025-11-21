# GL-002 BoilerEfficiencyOptimizer - Troubleshooting Guide

Production troubleshooting guide for GL-002 BoilerEfficiencyOptimizer. This runbook provides step-by-step solutions for common issues encountered in production environments.

## Table of Contents

1. [Agent Not Starting](#agent-not-starting)
2. [High Error Rates](#high-error-rates)
3. [Performance Degradation](#performance-degradation)
4. [Determinism Failures](#determinism-failures)
5. [Integration Failures](#integration-failures)
6. [Database Connection Issues](#database-connection-issues)
7. [Redis Cache Issues](#redis-cache-issues)
8. [Memory Leaks](#memory-leaks)
9. [CPU Throttling](#cpu-throttling)
10. [Network Connectivity Issues](#network-connectivity-issues)

---

## Agent Not Starting

### Symptom

Agent pod is in `CrashLoopBackOff` state or fails to start.

### Diagnosis Steps

```bash
# 1. Check pod status
kubectl get pods -n greenlang | grep gl-002

# 2. View recent logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=100

# 3. Check previous logs (if pod restarted)
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --previous --tail=100

# 4. Describe pod for events
kubectl describe pod -n greenlang <pod-name>

# 5. Check pod events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-002
```

### Common Causes and Solutions

#### 1. Configuration Errors

**Error Message:**
```
ValueError: Configuration validation failed: max_steam_capacity_kg_hr must be >= min_steam_capacity_kg_hr
```

**Solution:**
```bash
# Check ConfigMap
kubectl get configmap gl-002-config -n greenlang -o yaml

# Fix configuration
kubectl edit configmap gl-002-config -n greenlang

# Verify fix
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

**Validation:**
```bash
# Watch pod startup
kubectl logs -f -n greenlang deployment/gl-002-boiler-efficiency

# Verify health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -f http://localhost:8000/api/v1/health
```

#### 2. Missing Environment Variables

**Error Message:**
```
KeyError: 'DATABASE_URL'
EnvironmentError: Required environment variable DATABASE_URL not found
```

**Solution:**
```bash
# Check existing secrets
kubectl get secret gl-002-secrets -n greenlang -o jsonpath='{.data}' | jq 'keys'

# Add missing secret
kubectl create secret generic gl-002-secrets \
  --from-literal=DATABASE_URL='postgresql://user:pass@db-host:5432/boiler' \
  --from-literal=REDIS_URL='redis://redis-host:6379/0' \
  --from-literal=API_KEY='your-api-key' \
  --from-literal=JWT_SECRET='your-jwt-secret' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Restart deployment to pick up new secrets
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

**Validation:**
```bash
# Verify secrets are mounted
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- env | grep -E '(DATABASE_URL|REDIS_URL|API_KEY)'
```

#### 3. Dependency Issues (Database/Redis Unreachable)

**Error Message:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution:**
```bash
# Test database connectivity from pod
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql 'postgresql://user:pass@db-host:5432/boiler'

# Test Redis connectivity from pod
kubectl run -it --rm debug --image=redis:7 --restart=Never -n greenlang -- \
  redis-cli -h redis-host -p 6379 ping

# Check network policies
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy gl-002-network-policy -n greenlang

# Verify DNS resolution
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- nslookup db-host
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- nslookup redis-host
```

**Fix Network Policy (if blocking):**
```bash
kubectl edit networkpolicy gl-002-network-policy -n greenlang

# Ensure egress rules allow database and Redis
# Add:
#   - to:
#     - podSelector:
#         matchLabels:
#           app: postgresql
#     ports:
#     - protocol: TCP
#       port: 5432
#   - to:
#     - podSelector:
#         matchLabels:
#           app: redis
#     ports:
#     - protocol: TCP
#       port: 6379
```

#### 4. Image Pull Errors

**Error Message:**
```
Failed to pull image "ghcr.io/greenlang/gl-002:latest": rpc error: code = Unknown desc = Error response from daemon: unauthorized
```

**Solution:**
```bash
# Create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=your-username \
  --docker-password=your-token \
  --docker-email=your-email@example.com \
  -n greenlang

# Update deployment to use the secret
kubectl patch deployment gl-002-boiler-efficiency -n greenlang \
  -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-secret"}]}}}}'

# Verify image pull
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

#### 5. Resource Constraints

**Error Message:**
```
0/5 nodes are available: 2 Insufficient cpu, 3 Insufficient memory
```

**Solution:**
```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Option 1: Reduce resource requests
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --requests=cpu=250m,memory=256Mi \
  --limits=cpu=1000m,memory=512Mi

# Option 2: Add more nodes (cloud provider specific)
# AWS EKS:
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=6 --name=standard-workers

# Option 3: Evict low-priority pods
kubectl delete pod <low-priority-pod> -n <namespace>
```

---

## High Error Rates

### Symptom

Agent is running but showing high error rates (>5% of requests failing).

### Diagnosis Steps

```bash
# 1. Check error rate in metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error|fail)'

# 2. Check recent logs for errors
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=500 | grep -i error

# 3. Check Prometheus metrics (if available)
curl -s "http://prometheus:9090/api/v1/query?query=rate(gl_002_http_requests_total{status=~'5..'}[5m])"

# 4. Check Grafana dashboard
# Open: https://grafana.greenlang.io/d/gl-002/boiler-efficiency-optimizer
```

### Common Causes and Solutions

#### 1. Integration Failures (SCADA/ERP Connectivity)

**Error Message:**
```
OPC UA Connection Error: Failed to connect to opc.tcp://scada-server:4840
Modbus TCP Error: Connection refused to 192.168.1.100:502
```

**Diagnosis:**
```bash
# Check integration health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/integrations/health | jq

# Test connectivity
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  telnet scada-server 4840

kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  nc -zv 192.168.1.100 502
```

**Solution:**
```bash
# Check firewall rules
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -v telnet://scada-server:4840

# Verify credentials
kubectl get secret gl-002-secrets -n greenlang -o jsonpath='{.data.SCADA_USERNAME}' | base64 -d
kubectl get secret gl-002-secrets -n greenlang -o jsonpath='{.data.SCADA_PASSWORD}' | base64 -d

# Update credentials if needed
kubectl create secret generic gl-002-secrets \
  --from-literal=SCADA_USERNAME='new-username' \
  --from-literal=SCADA_PASSWORD='new-password' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Restart to pick up new credentials
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

**Configure Connection Retry:**
```bash
# Edit ConfigMap to increase retry attempts
kubectl edit configmap gl-002-config -n greenlang

# Add/modify:
#   INTEGRATION_RETRY_ATTEMPTS: "5"
#   INTEGRATION_RETRY_DELAY_SECONDS: "10"
#   INTEGRATION_TIMEOUT_SECONDS: "30"

# Restart to apply
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

#### 2. Database Query Timeouts

**Error Message:**
```
psycopg2.errors.QueryCanceled: canceling statement due to statement timeout
SQLSTATE[57014]: Query cancelled: 7 ERROR: canceling statement
```

**Diagnosis:**
```bash
# Check slow queries
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT query, state, wait_event_type, backend_start
                          FROM pg_stat_activity
                          WHERE state != 'idle'
                          ORDER BY backend_start;"

# Check connection pool
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool
```

**Solution:**
```bash
# Increase query timeout
kubectl edit configmap gl-002-config -n greenlang

# Add:
#   DATABASE_QUERY_TIMEOUT: "30"  # seconds
#   DATABASE_POOL_SIZE: "20"
#   DATABASE_MAX_OVERFLOW: "10"

# Analyze and optimize queries
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT schemaname, tablename, last_vacuum, last_autovacuum
                          FROM pg_stat_user_tables
                          WHERE schemaname = 'public';"

# Vacuum and analyze
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Restart deployment
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

#### 3. Rate Limiting

**Error Message:**
```
HTTP 429 Too Many Requests
{"error": "rate_limit_exceeded", "retry_after": 60}
```

**Diagnosis:**
```bash
# Check rate limit metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep rate_limit

# Check Ingress rate limiting
kubectl get ingress gl-002-boiler-efficiency-ingress -n greenlang -o yaml | grep rate
```

**Solution:**
```bash
# Increase rate limits in Ingress
kubectl edit ingress gl-002-boiler-efficiency-ingress -n greenlang

# Modify annotations:
#   nginx.ingress.kubernetes.io/limit-rps: "200"  # from 100
#   nginx.ingress.kubernetes.io/limit-connections: "50"  # from 10

# Or increase application rate limits
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   RATE_LIMIT_REQUESTS_PER_MINUTE: "1000"
#   RATE_LIMIT_BURST: "200"
```

---

## Performance Degradation

### Symptom

Response times increasing, processing slowing down.

### Diagnosis Steps

```bash
# 1. Check response times
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep duration

# 2. Check resource usage
kubectl top pods -n greenlang | grep gl-002

# 3. Check pod count (HPA)
kubectl get hpa -n greenlang

# 4. Check logs for slow operations
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency --tail=1000 | grep -E '(slow|timeout|delay)'
```

### Common Causes and Solutions

#### 1. Cache Miss Issues

**Error Message:**
```
WARNING: Redis cache miss rate: 85% (threshold: 20%)
Cache connection failed, falling back to database
```

**Diagnosis:**
```bash
# Check Redis health
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host ping

# Check cache metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep cache

# Check Redis memory usage
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host INFO memory
```

**Solution:**
```bash
# Increase Redis memory
kubectl edit deployment redis -n greenlang

# Modify resources:
#   resources:
#     limits:
#       memory: "2Gi"  # from 1Gi
#     requests:
#       memory: "1Gi"

# Configure eviction policy
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host CONFIG SET maxmemory-policy allkeys-lru

# Warm up cache
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/cache/warmup

# Increase cache TTL
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   CACHE_TTL_SECONDS: "3600"  # 1 hour
#   CACHE_OPTIMIZATION_RESULTS_TTL: "7200"  # 2 hours
```

#### 2. Resource Constraints (CPU/Memory)

**Error Message:**
```
WARNING: CPU usage at 95%, throttling detected
OOMKilled: Container killed due to memory pressure
```

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n greenlang | grep gl-002

# Check resource limits
kubectl describe pod -n greenlang <pod-name> | grep -A 5 "Limits"

# Check node resources
kubectl describe nodes | grep -A 10 "Allocated resources"

# Check for OOMKilled
kubectl get pods -n greenlang -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}' | grep OOMKilled
```

**Solution:**
```bash
# Option 1: Increase resource limits
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --requests=cpu=1000m,memory=1Gi \
  --limits=cpu=2000m,memory=2Gi

# Option 2: Enable HPA to scale out
kubectl autoscale deployment gl-002-boiler-efficiency -n greenlang \
  --min=3 --max=10 --cpu-percent=70

# Verify HPA is working
kubectl get hpa -n greenlang -w

# Option 3: Optimize code (profile first)
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/debug/profile/start

# Run workload for 5 minutes

kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/debug/profile/stop > profile.json
```

#### 3. Database Connection Pool Exhaustion

**Error Message:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 20 overflow 10 reached
FATAL: remaining connection slots are reserved
```

**Diagnosis:**
```bash
# Check database connections
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'boiler';"

# Check connection pool metrics
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool
```

**Solution:**
```bash
# Increase connection pool size
kubectl edit configmap gl-002-config -n greenlang

# Modify:
#   DATABASE_POOL_SIZE: "30"  # from 20
#   DATABASE_MAX_OVERFLOW: "20"  # from 10
#   DATABASE_POOL_RECYCLE: "3600"  # recycle connections after 1 hour

# Restart deployment
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Monitor pool usage
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  watch -n 5 'curl -s http://localhost:8000/api/v1/metrics | grep db_pool'
```

---

## Determinism Failures

### Symptom

Same inputs producing different outputs, reproducibility issues.

### Diagnosis Steps

```bash
# 1. Check determinism audit logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency | grep -i determinism

# 2. Run determinism test
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  python -m pytest tests/test_determinism.py -v

# 3. Check random seed configuration
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- env | grep SEED

# 4. Check for race conditions
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency | grep -E '(race|concurrent|thread)'
```

### Common Causes and Solutions

#### 1. Random Seed Not Set

**Error Message:**
```
WARNING: Random seed not configured, non-deterministic behavior possible
Optimization results vary between runs with same inputs
```

**Solution:**
```bash
# Set random seed in ConfigMap
kubectl edit configmap gl-002-config -n greenlang

# Add:
#   RANDOM_SEED: "42"
#   NUMPY_SEED: "42"
#   TORCH_SEED: "42"
#   DETERMINISTIC_MODE: "true"

# Restart deployment
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Verify determinism
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/test/determinism \
    -H "Content-Type: application/json" \
    -d '{"iterations": 10}'
```

#### 2. Concurrent Operations (Race Conditions)

**Error Message:**
```
WARNING: Concurrent modification detected in optimization state
Results differ due to parallel execution order
```

**Solution:**
```bash
# Enable single-threaded mode (temporary)
kubectl edit configmap gl-002-config -n greenlang

# Add:
#   WORKER_THREADS: "1"
#   ENABLE_PARALLEL: "false"

# Restart deployment
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# For production, use locking instead:
# Add:
#   ENABLE_DISTRIBUTED_LOCK: "true"
#   LOCK_BACKEND: "redis"
#   LOCK_TIMEOUT_SECONDS: "30"
```

#### 3. Timestamp-Based Calculations

**Error Message:**
```
WARNING: Timestamp-based calculation detected in deterministic mode
Results vary due to current_time() usage
```

**Solution:**
```bash
# Use fixed timestamp in test mode
kubectl edit configmap gl-002-config -n greenlang

# Add:
#   TEST_MODE: "true"
#   FIXED_TIMESTAMP: "2025-01-01T00:00:00Z"

# Or use relative timestamps
# Modify code to use:
#   - relative_time (e.g., seconds since start)
#   - event_sequence_number
#   - deterministic time source
```

---

## Integration Failures

### Symptom

Unable to connect to external systems (SCADA, ERP, Emissions Monitoring).

### Diagnosis Steps

```bash
# 1. Check integration health endpoint
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/integrations/health | jq

# 2. Test connectivity
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -v telnet://scada-server:4840

# 3. Check integration logs
kubectl logs -n greenlang deployment/gl-002-boiler-efficiency | grep -i integration

# 4. Verify credentials
kubectl get secret gl-002-secrets -n greenlang -o jsonpath='{.data}' | jq 'keys'
```

### Common Solutions

#### SCADA (OPC UA) Connection Failures

```bash
# Test OPC UA connectivity
kubectl run -it --rm opcua-client --image=python:3.11 --restart=Never -n greenlang -- \
  bash -c "pip install opcua && python -c 'from opcua import Client; c=Client(\"opc.tcp://scada-server:4840\"); c.connect(); print(\"OK\"); c.disconnect()'"

# Update credentials
kubectl create secret generic gl-002-secrets \
  --from-literal=SCADA_ENDPOINT='opc.tcp://scada-server:4840' \
  --from-literal=SCADA_USERNAME='username' \
  --from-literal=SCADA_PASSWORD='password' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Restart
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
```

#### ERP (SAP/Oracle) Connection Failures

```bash
# Test database connectivity
kubectl run -it --rm dbclient --image=postgres:14 --restart=Never -n greenlang -- \
  psql 'postgresql://erp-host:5432/erp'

# Update ERP credentials
kubectl create secret generic gl-002-secrets \
  --from-literal=ERP_DATABASE_URL='postgresql://user:pass@erp-host:5432/erp' \
  --from-literal=ERP_API_KEY='your-api-key' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -
```

#### Emissions Monitoring API Failures

```bash
# Test API connectivity
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -v https://emissions-api.example.com/health

# Update API credentials
kubectl create secret generic gl-002-secrets \
  --from-literal=EMISSIONS_API_URL='https://emissions-api.example.com' \
  --from-literal=EMISSIONS_API_KEY='your-api-key' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -
```

---

## Database Connection Issues

### Quick Fixes

```bash
# Test connection
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql $DATABASE_URL

# Restart database (if local)
kubectl rollout restart statefulset/postgresql -n greenlang

# Check connection pool
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool

# Increase pool size
kubectl edit configmap gl-002-config -n greenlang
# Set: DATABASE_POOL_SIZE: "30"
```

---

## Redis Cache Issues

### Quick Fixes

```bash
# Test Redis connectivity
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host ping

# Flush cache (careful in production!)
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host FLUSHDB

# Restart Redis
kubectl rollout restart deployment/redis -n greenlang

# Check memory usage
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  redis-cli -h redis-host INFO memory
```

---

## Memory Leaks

### Diagnosis and Fix

```bash
# Check memory usage over time
kubectl top pods -n greenlang | grep gl-002

# Enable memory profiling
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/debug/memory/snapshot

# Force garbage collection
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  curl -X POST http://localhost:8000/api/v1/debug/gc

# Restart pod if memory continues to grow
kubectl delete pod -n greenlang <pod-name>
```

---

## CPU Throttling

### Diagnosis and Fix

```bash
# Check CPU usage
kubectl top pods -n greenlang | grep gl-002

# Check CPU throttling
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled

# Increase CPU limits
kubectl set resources deployment gl-002-boiler-efficiency -n greenlang \
  --limits=cpu=2000m

# Enable HPA for scaling
kubectl autoscale deployment gl-002-boiler-efficiency -n greenlang \
  --min=3 --max=10 --cpu-percent=70
```

---

## Network Connectivity Issues

### Diagnosis and Fix

```bash
# Test DNS resolution
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  nslookup scada-server

# Test connectivity
kubectl exec -n greenlang deployment/gl-002-boiler-efficiency -- \
  telnet scada-server 4840

# Check network policies
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy gl-002-network-policy -n greenlang

# Temporarily disable network policy (testing only!)
kubectl delete networkpolicy gl-002-network-policy -n greenlang

# Re-apply after testing
kubectl apply -f deployment/networkpolicy.yaml -n greenlang
```

---

## Monitoring Dashboards

### Grafana Dashboards

- **Main Dashboard**: https://grafana.greenlang.io/d/gl-002/boiler-efficiency-optimizer
- **Error Rate Dashboard**: https://grafana.greenlang.io/d/gl-002-errors/error-analysis
- **Performance Dashboard**: https://grafana.greenlang.io/d/gl-002-perf/performance-metrics

### Prometheus Queries

```promql
# Error rate (last 5 minutes)
rate(gl_002_http_requests_total{status=~"5.."}[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(gl_002_http_request_duration_seconds_bucket[5m]))

# Memory usage
gl_002_system_memory_usage_bytes{type="rss"}

# CPU usage
gl_002_system_cpu_usage_percent
```

---

## Escalation Contacts

### On-Call Rotation

- **Primary**: DevOps Team (+1-555-0100) - Slack: #gl-002-oncall
- **Secondary**: Platform Engineering (+1-555-0101) - Slack: #platform-oncall
- **Escalation**: Engineering Manager (+1-555-0102)

### Severity Guidelines

- **P0 (Critical)**: Production down, no workaround - Escalate immediately
- **P1 (High)**: Major degradation, limited workaround - Escalate within 1 hour
- **P2 (Medium)**: Minor degradation, workaround available - Escalate within 4 hours
- **P3 (Low)**: Cosmetic issues, no impact - Handle during business hours

### Communication Channels

- **Slack**: #gl-002-alerts (automated), #gl-002-incidents (manual)
- **PagerDuty**: Service ID: GL-002-PROD
- **Email**: gl-002-oncall@greenlang.io
- **Status Page**: https://status.greenlang.io

---

## Additional Resources

- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\deployment\README.md`
- **Architecture**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002\ARCHITECTURE.md`
- **API Documentation**: https://docs.greenlang.io/agents/gl-002
- **Runbooks**:
  - Incident Response: `INCIDENT_RESPONSE.md`
  - Rollback Procedures: `ROLLBACK_PROCEDURE.md`
  - Scaling Guide: `SCALING_GUIDE.md`
