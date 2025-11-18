# GL-003 SteamSystemAnalyzer - Troubleshooting Guide

Production troubleshooting guide for GL-003 SteamSystemAnalyzer. This runbook provides step-by-step solutions for common issues encountered in production environments.

## Table of Contents

1. [Agent Not Starting](#agent-not-starting)
2. [High Error Rates](#high-error-rates)
3. [Performance Degradation](#performance-degradation)
4. [Steam Analysis Issues](#steam-analysis-issues)
5. [Integration Failures](#integration-failures)
6. [Database Connection Issues](#database-connection-issues)
7. [Redis Cache Issues](#redis-cache-issues)
8. [Memory Leaks](#memory-leaks)
9. [CPU Throttling](#cpu-throttling)
10. [Network Connectivity Issues](#network-connectivity-issues)
11. [TimescaleDB Specific Issues](#timescaledb-specific-issues)

---

## Agent Not Starting

### Symptom

Agent pod is in `CrashLoopBackOff` state or fails to start.

### Diagnosis Steps

```bash
# 1. Check pod status
kubectl get pods -n greenlang | grep gl-003

# 2. View recent logs
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=100

# 3. Check previous logs (if pod restarted)
kubectl logs -n greenlang deployment/gl-003-steam-system --previous --tail=100

# 4. Describe pod for events
kubectl describe pod -n greenlang <pod-name>

# 5. Check pod events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-003
```

### Common Causes and Solutions

#### 1. Configuration Errors

**Error Message:**
```
ValueError: Configuration validation failed: steam_pressure_max_bar must be >= steam_pressure_min_bar
ConfigurationError: Invalid leak detection threshold: must be between 0.1 and 10.0 kg/hr
```

**Solution:**
```bash
# Check ConfigMap
kubectl get configmap gl-003-config -n greenlang -o yaml

# Fix configuration
kubectl edit configmap gl-003-config -n greenlang

# Common configuration fixes:
# - STEAM_PRESSURE_MIN_BAR: "1.0"
# - STEAM_PRESSURE_MAX_BAR: "15.0"
# - LEAK_DETECTION_THRESHOLD_KG_HR: "2.5"
# - TRAP_MONITORING_INTERVAL_SECONDS: "300"
# - DISTRIBUTION_EFFICIENCY_TARGET_PERCENT: "90.0"

# Verify fix
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
kubectl rollout status deployment/gl-003-steam-system -n greenlang
```

**Validation:**
```bash
# Watch pod startup
kubectl logs -f -n greenlang deployment/gl-003-steam-system

# Verify health
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -f http://localhost:8000/api/v1/health

# Verify configuration loaded correctly
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/config | jq
```

#### 2. Missing Environment Variables

**Error Message:**
```
KeyError: 'DATABASE_URL'
EnvironmentError: Required environment variable TIMESCALEDB_URL not found
MissingConfigError: SCADA_ENDPOINT not configured
```

**Solution:**
```bash
# Check existing secrets
kubectl get secret gl-003-secrets -n greenlang -o jsonpath='{.data}' | jq 'keys'

# Add missing secrets
kubectl create secret generic gl-003-secrets \
  --from-literal=DATABASE_URL='postgresql://user:pass@db-host:5432/steam' \
  --from-literal=TIMESCALEDB_URL='postgresql://user:pass@timescale-host:5432/steam' \
  --from-literal=REDIS_URL='redis://redis-host:6379/0' \
  --from-literal=SCADA_ENDPOINT='opc.tcp://scada-server:4840' \
  --from-literal=SCADA_USERNAME='scada-user' \
  --from-literal=SCADA_PASSWORD='scada-password' \
  --from-literal=METER_API_KEY='meter-api-key' \
  --from-literal=API_KEY='your-api-key' \
  --from-literal=JWT_SECRET='your-jwt-secret' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Restart deployment to pick up new secrets
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

**Validation:**
```bash
# Verify secrets are mounted
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  env | grep -E '(DATABASE_URL|SCADA_ENDPOINT|METER_API_KEY)'
```

#### 3. Dependency Issues (Database/Redis/SCADA Unreachable)

**Error Message:**
```
psycopg2.OperationalError: could not connect to server: Connection refused
timescaledb.exceptions.ConnectionError: Cannot connect to TimescaleDB
redis.exceptions.ConnectionError: Error connecting to Redis
opcua.exceptions.BadConnectionClosed: OPC UA connection to scada-server failed
```

**Solution:**
```bash
# Test database connectivity from pod
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql 'postgresql://user:pass@db-host:5432/steam'

# Test TimescaleDB extension
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql 'postgresql://user:pass@timescale-host:5432/steam' -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"

# Test Redis connectivity from pod
kubectl run -it --rm debug --image=redis:7 --restart=Never -n greenlang -- \
  redis-cli -h redis-host -p 6379 ping

# Test SCADA connectivity (OPC UA)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# Check network policies
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy gl-003-network-policy -n greenlang

# Verify DNS resolution
kubectl exec -n greenlang deployment/gl-003-steam-system -- nslookup db-host
kubectl exec -n greenlang deployment/gl-003-steam-system -- nslookup redis-host
kubectl exec -n greenlang deployment/gl-003-steam-system -- nslookup scada-server
```

**Fix Network Policy (if blocking):**
```bash
kubectl edit networkpolicy gl-003-network-policy -n greenlang

# Ensure egress rules allow database, Redis, and SCADA
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
#   - to:
#     - namespaceSelector: {}  # Allow SCADA (external)
#     ports:
#     - protocol: TCP
#       port: 4840
```

#### 4. Image Pull Errors

**Error Message:**
```
Failed to pull image "ghcr.io/greenlang/gl-003:latest": rpc error: code = Unknown desc = Error response from daemon: unauthorized
ImagePullBackOff
ErrImagePull
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
kubectl patch deployment gl-003-steam-system -n greenlang \
  -p '{"spec":{"template":{"spec":{"imagePullSecrets":[{"name":"ghcr-secret"}]}}}}'

# Verify image pull
kubectl rollout status deployment/gl-003-steam-system -n greenlang
```

#### 5. Resource Constraints

**Error Message:**
```
0/5 nodes are available: 2 Insufficient cpu, 3 Insufficient memory
FailedScheduling: 0/5 nodes are available: 5 node(s) didn't match Pod's node affinity/selector
```

**Solution:**
```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Option 1: Reduce resource requests
kubectl set resources deployment gl-003-steam-system -n greenlang \
  --requests=cpu=250m,memory=512Mi \
  --limits=cpu=1000m,memory=1Gi

# Option 2: Add more nodes (cloud provider specific)
# AWS EKS:
eksctl scale nodegroup --cluster=greenlang-cluster --nodes=6 --name=standard-workers

# Option 3: Evict low-priority pods
kubectl delete pod <low-priority-pod> -n <namespace>
```

#### 6. TimescaleDB Extension Not Loaded

**Error Message:**
```
psycopg2.errors.UndefinedTable: relation "steam_measurements" does not exist
TimescaleDBError: Hypertable "steam_measurements" not found
ExtensionNotLoaded: TimescaleDB extension not installed
```

**Solution:**
```bash
# Check if TimescaleDB extension is installed
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';"

# Install TimescaleDB extension (if missing)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Create hypertable (if missing)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT create_hypertable('steam_measurements', 'timestamp', if_not_exists => TRUE);"

# Verify hypertable created
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = 'steam_measurements';"

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

---

## High Error Rates

### Symptom

Agent is running but showing high error rates (>5% of requests failing).

### Diagnosis Steps

```bash
# 1. Check error rate in metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(error|fail)'

# 2. Check recent logs for errors
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=500 | grep -i error

# 3. Check Prometheus metrics (if available)
curl -s "http://prometheus:9090/api/v1/query?query=rate(gl_003_http_requests_total{status=~'5..'}[5m])"

# 4. Check Grafana dashboard
# Open: https://grafana.greenlang.io/d/gl-003/steam-system-analyzer

# 5. Check steam system specific errors
kubectl logs -n greenlang deployment/gl-003-steam-system | grep -E '(leak_detection|trap_classification|meter_communication)'
```

### Common Causes and Solutions

#### 1. Integration Failures (SCADA/Steam Meter Connectivity)

**Error Message:**
```
OPC UA Connection Error: Failed to connect to opc.tcp://scada-server:4840
Modbus TCP Error: Connection refused to steam-meter-gateway:502
MQTT Connection Lost: broker.example.com:1883
REST API Error: Steam meter API returned 503 Service Unavailable
```

**Diagnosis:**
```bash
# Check integration health
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/health | jq

# Test SCADA connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# Test steam meter gateway
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  nc -zv steam-meter-gateway 502

# Check MQTT broker
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet broker.example.com 1883

# Test steam meter API
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -v https://meter-api.example.com/health
```

**Solution:**
```bash
# Check firewall rules
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -v telnet://scada-server:4840

# Verify SCADA credentials
kubectl get secret gl-003-secrets -n greenlang -o jsonpath='{.data.SCADA_USERNAME}' | base64 -d
kubectl get secret gl-003-secrets -n greenlang -o jsonpath='{.data.SCADA_PASSWORD}' | base64 -d

# Update credentials if needed
kubectl create secret generic gl-003-secrets \
  --from-literal=SCADA_USERNAME='new-username' \
  --from-literal=SCADA_PASSWORD='new-password' \
  --from-literal=METER_API_KEY='new-api-key' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Restart to pick up new credentials
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

**Configure Connection Retry:**
```bash
# Edit ConfigMap to increase retry attempts
kubectl edit configmap gl-003-config -n greenlang

# Add/modify:
#   INTEGRATION_RETRY_ATTEMPTS: "5"
#   INTEGRATION_RETRY_DELAY_SECONDS: "10"
#   INTEGRATION_TIMEOUT_SECONDS: "30"
#   SCADA_RECONNECT_INTERVAL_SECONDS: "60"
#   METER_POLL_INTERVAL_SECONDS: "5"

# Restart to apply
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

#### 2. Database Query Timeouts (TimescaleDB)

**Error Message:**
```
psycopg2.errors.QueryCanceled: canceling statement due to statement timeout
TimescaleDBError: Query on hypertable 'steam_measurements' exceeded timeout
SQLSTATE[57014]: Query cancelled: 7 ERROR: canceling statement
```

**Diagnosis:**
```bash
# Check slow queries
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT query, state, wait_event_type, query_start, now() - query_start AS duration
                          FROM pg_stat_activity
                          WHERE state != 'idle'
                          AND datname = 'steam'
                          ORDER BY query_start;"

# Check connection pool
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool

# Check TimescaleDB chunk statistics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.chunks ORDER BY range_end DESC LIMIT 10;"

# Check compression status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.compressed_chunk_stats;"
```

**Solution:**
```bash
# Increase query timeout
kubectl edit configmap gl-003-config -n greenlang

# Add:
#   DATABASE_QUERY_TIMEOUT: "60"  # seconds
#   DATABASE_POOL_SIZE: "30"
#   DATABASE_MAX_OVERFLOW: "15"
#   DATABASE_STATEMENT_TIMEOUT: "60000"  # milliseconds

# Optimize TimescaleDB chunks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT set_chunk_time_interval('steam_measurements', INTERVAL '1 day');"

# Enable compression on old chunks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "ALTER TABLE steam_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'meter_id'
  );"

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT add_compression_policy('steam_measurements', INTERVAL '7 days');"

# Vacuum and analyze
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "VACUUM ANALYZE steam_measurements;"

# Create indexes on frequently queried columns
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "CREATE INDEX IF NOT EXISTS idx_steam_meter_id ON steam_measurements (meter_id, timestamp DESC);"

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "CREATE INDEX IF NOT EXISTS idx_steam_location ON steam_measurements (location_id, timestamp DESC);"

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

#### 3. Rate Limiting

**Error Message:**
```
HTTP 429 Too Many Requests
{"error": "rate_limit_exceeded", "retry_after": 60}
RateLimitError: Steam meter API rate limit exceeded: 1000 requests/hour
```

**Diagnosis:**
```bash
# Check rate limit metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep rate_limit

# Check Ingress rate limiting
kubectl get ingress gl-003-steam-system-ingress -n greenlang -o yaml | grep rate

# Check steam meter API usage
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/api-usage | jq
```

**Solution:**
```bash
# Increase rate limits in Ingress
kubectl edit ingress gl-003-steam-system-ingress -n greenlang

# Modify annotations:
#   nginx.ingress.kubernetes.io/limit-rps: "200"  # from 100
#   nginx.ingress.kubernetes.io/limit-connections: "50"  # from 10

# Or increase application rate limits
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   RATE_LIMIT_REQUESTS_PER_MINUTE: "1000"
#   RATE_LIMIT_BURST: "200"
#   METER_API_RATE_LIMIT: "2000"  # requests per hour

# Implement request batching for steam meters
#   METER_BATCH_SIZE: "50"  # Read 50 meters per request
#   METER_BATCH_INTERVAL_SECONDS: "10"
```

---

## Performance Degradation

### Symptom

Response times increasing, processing slowing down, steam analysis taking too long.

### Diagnosis Steps

```bash
# 1. Check response times
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep duration

# 2. Check resource usage
kubectl top pods -n greenlang | grep gl-003

# 3. Check pod count (HPA)
kubectl get hpa -n greenlang

# 4. Check logs for slow operations
kubectl logs -n greenlang deployment/gl-003-steam-system --tail=1000 | grep -E '(slow|timeout|delay|duration)'

# 5. Check steam analysis performance
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(leak_detection_duration|trap_analysis_duration|efficiency_calc_duration)'
```

### Common Causes and Solutions

#### 1. Cache Miss Issues

**Error Message:**
```
WARNING: Redis cache miss rate: 85% (threshold: 15%)
Cache connection failed, falling back to database
CacheError: Redis timeout after 5 seconds
```

**Diagnosis:**
```bash
# Check Redis health
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host ping

# Check cache metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep cache

# Check Redis memory usage
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host INFO memory

# Check cache hit rate by type
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(cache_hit_rate|cache_miss_rate)'
```

**Solution:**
```bash
# Increase Redis memory
kubectl edit deployment redis -n greenlang

# Modify resources:
#   resources:
#     limits:
#       memory: "4Gi"  # from 2Gi
#     requests:
#       memory: "2Gi"

# Configure eviction policy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host CONFIG SET maxmemory-policy allkeys-lru

# Warm up cache for steam system data
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/cache/warmup \
    -H "Content-Type: application/json" \
    -d '{
      "cache_types": [
        "steam_properties",
        "meter_locations",
        "trap_baselines",
        "leak_detection_models"
      ]
    }'

# Increase cache TTL
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   CACHE_TTL_SECONDS: "3600"  # 1 hour
#   CACHE_STEAM_PROPERTIES_TTL: "7200"  # 2 hours (steam properties don't change often)
#   CACHE_METER_DATA_TTL: "300"  # 5 minutes (meter data changes frequently)
#   CACHE_TRAP_BASELINE_TTL: "86400"  # 24 hours (baselines stable)
```

#### 2. Resource Constraints (CPU/Memory)

**Error Message:**
```
WARNING: CPU usage at 95%, throttling detected
OOMKilled: Container killed due to memory pressure
ResourceExhausted: Unable to allocate memory for steam analysis
```

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n greenlang | grep gl-003

# Check resource limits
kubectl describe pod -n greenlang <pod-name> | grep -A 5 "Limits"

# Check node resources
kubectl describe nodes | grep -A 10 "Allocated resources"

# Check for OOMKilled
kubectl get pods -n greenlang -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{end}' | grep OOMKilled

# Check CPU throttling
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled
```

**Solution:**
```bash
# Option 1: Increase resource limits
kubectl set resources deployment gl-003-steam-system -n greenlang \
  --requests=cpu=1000m,memory=2Gi \
  --limits=cpu=2000m,memory=4Gi

# Option 2: Enable HPA to scale out
kubectl autoscale deployment gl-003-steam-system -n greenlang \
  --min=3 --max=10 --cpu-percent=70

# Verify HPA is working
kubectl get hpa -n greenlang -w

# Option 3: Optimize code (profile first)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/debug/profile/start

# Run workload for 5 minutes

kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/debug/profile/stop > profile.json

# Analyze profile.json for hotspots
```

#### 3. Database Connection Pool Exhaustion

**Error Message:**
```
sqlalchemy.exc.TimeoutError: QueuePool limit of size 30 overflow 15 reached
FATAL: remaining connection slots are reserved for non-replication superuser connections
PoolTimeout: Could not acquire database connection after 30 seconds
```

**Diagnosis:**
```bash
# Check database connections
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT count(*) AS active_connections, state
                          FROM pg_stat_activity
                          WHERE datname = 'steam'
                          GROUP BY state;"

# Check max connections
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SHOW max_connections;"

# Check connection pool metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool
```

**Solution:**
```bash
# Increase connection pool size
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   DATABASE_POOL_SIZE: "40"  # from 30
#   DATABASE_MAX_OVERFLOW: "20"  # from 15
#   DATABASE_POOL_RECYCLE: "3600"  # recycle connections after 1 hour
#   DATABASE_POOL_PRE_PING: "true"  # verify connections before use

# Increase database max_connections (if needed)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "ALTER SYSTEM SET max_connections = 200;"

# Restart database to apply (careful!)
# kubectl rollout restart statefulset/postgresql -n greenlang

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang

# Monitor pool usage
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  watch -n 5 'curl -s http://localhost:8000/api/v1/metrics | grep db_pool'
```

#### 4. Slow Steam Analysis Calculations

**Error Message:**
```
WARNING: Leak detection duration: 12.5 seconds (threshold: 5 seconds)
WARNING: Steam trap analysis duration: 8.2 seconds (threshold: 3 seconds)
CalculationTimeout: Distribution efficiency calculation exceeded 10 second limit
```

**Diagnosis:**
```bash
# Check steam analysis performance metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(leak_detection|trap_analysis|efficiency_calculation)_duration'

# Check number of steam meters/traps being analyzed
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/metrics | jq

# Check if calculations are being parallelized
kubectl logs -n greenlang deployment/gl-003-steam-system | grep -E '(parallel|concurrent|thread)'
```

**Solution:**
```bash
# Enable parallel processing
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   ENABLE_PARALLEL_ANALYSIS: "true"
#   ANALYSIS_WORKER_THREADS: "4"  # Number of CPU cores
#   LEAK_DETECTION_BATCH_SIZE: "50"  # Analyze 50 meters at a time
#   TRAP_ANALYSIS_BATCH_SIZE: "100"  # Analyze 100 traps at a time

# Optimize calculation algorithms
#   LEAK_DETECTION_ALGORITHM: "fast"  # Use faster algorithm
#   TRAP_CLASSIFICATION_MODEL: "lightweight"  # Use lighter model
#   ENABLE_CALCULATION_CACHING: "true"  # Cache intermediate results

# Reduce analysis frequency for non-critical items
#   LEAK_DETECTION_INTERVAL_SECONDS: "300"  # Every 5 minutes instead of every minute
#   TRAP_MONITORING_INTERVAL_SECONDS: "600"  # Every 10 minutes
#   EFFICIENCY_CALC_INTERVAL_SECONDS: "60"  # Keep at 1 minute

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

---

## Steam Analysis Issues

### Leak Detection Problems

#### Issue 1: Inaccurate Leak Detection (High False Positives)

**Symptom**: Leak detection reporting many leaks that don't exist upon physical inspection

**Diagnosis:**
```bash
# Check leak detection accuracy metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep leak_detection_accuracy

# Check false positive rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/statistics | jq

# Check leak detection threshold
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/config/leak-detection | jq

# Review recent leak alerts
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/recent | jq
```

**Solution:**
```bash
# Adjust leak detection threshold (increase to reduce false positives)
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   LEAK_DETECTION_THRESHOLD_KG_HR: "5.0"  # from 2.5 (less sensitive)
#   LEAK_DETECTION_CONFIDENCE_THRESHOLD: "0.85"  # from 0.70 (higher confidence required)
#   LEAK_DETECTION_CONSECUTIVE_READINGS: "5"  # Require 5 consecutive readings before alerting

# Recalibrate leak detection algorithm with golden dataset
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/leaks/recalibrate \
    -H "Content-Type: application/json" \
    -d '{
      "use_golden_dataset": true,
      "target_false_positive_rate": 0.02,
      "target_false_negative_rate": 0.01
    }'

# Verify improvement
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep leak_detection_accuracy
```

#### Issue 2: Missed Leaks (High False Negatives)

**Symptom**: Physical leaks discovered that were not detected by the system

**Diagnosis:**
```bash
# Check false negative rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/leaks/statistics | jq '.false_negative_rate'

# Check sensor coverage
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/coverage | jq

# Check if meters near leak location are online
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/status | jq
```

**Solution:**
```bash
# Decrease leak detection threshold (more sensitive)
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   LEAK_DETECTION_THRESHOLD_KG_HR: "1.0"  # from 2.5 (more sensitive)
#   LEAK_DETECTION_CONFIDENCE_THRESHOLD: "0.60"  # from 0.70 (lower confidence acceptable)
#   ENABLE_ACOUSTIC_LEAK_DETECTION: "true"  # Enable acoustic sensors if available

# Add more steam meters in areas with poor coverage
# (Coordinate with facility team to install additional sensors)

# Force immediate leak scan
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/leaks/scan \
    -H "Content-Type: application/json" \
    -d '{"priority": "immediate", "full_network_scan": true}'
```

### Steam Trap Monitoring Issues

#### Issue 1: Steam Trap Misclassification

**Symptom**: Traps classified as "failed" that are actually working correctly

**Diagnosis:**
```bash
# Check trap classification accuracy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep trap_classification_accuracy

# Check trap status distribution
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/traps/statistics | jq

# Review trap baselines
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/traps/baselines | jq

# Check specific trap details
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s "http://localhost:8000/api/v1/analysis/traps/<trap_id>/details" | jq
```

**Solution:**
```bash
# Recalibrate trap classification model
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/traps/recalibrate \
    -H "Content-Type: application/json" \
    -d '{
      "use_verified_data": true,
      "training_duration_days": 30
    }'

# Adjust trap monitoring parameters
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   TRAP_TEMPERATURE_TOLERANCE_C: "5.0"  # from 3.0 (allow more variation)
#   TRAP_CYCLING_FREQUENCY_TOLERANCE: "0.3"  # from 0.2
#   TRAP_FAILED_THRESHOLD_SCORE: "0.75"  # from 0.65 (higher threshold)

# Reset specific trap baseline if needed
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST "http://localhost:8000/api/v1/analysis/traps/<trap_id>/reset-baseline"

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

#### Issue 2: Trap Sensor Communication Failures

**Symptom**: Many traps showing as "offline" or "no data"

**Diagnosis:**
```bash
# Check trap sensor connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/traps/connectivity | jq

# Check network status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/traps/network-status | jq

# Test trap sensor gateway
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  ping -c 5 trap-sensor-gateway.local
```

**Solution:**
```bash
# Check and fix trap sensor gateway configuration
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   TRAP_SENSOR_GATEWAY_URL: "http://trap-sensor-gateway.local:8080"
#   TRAP_SENSOR_POLL_INTERVAL_SECONDS: "30"  # from 10 (reduce polling frequency)
#   TRAP_SENSOR_TIMEOUT_SECONDS: "10"  # from 5 (increase timeout)
#   TRAP_SENSOR_RETRY_ATTEMPTS: "5"  # from 3

# Restart trap sensor gateway (if applicable)
# kubectl rollout restart deployment/trap-sensor-gateway -n greenlang

# Restart GL-003
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

### Distribution Efficiency Issues

#### Issue 1: Efficiency Calculation Errors

**Symptom**: Distribution efficiency showing unrealistic values (>100% or <50%)

**Diagnosis:**
```bash
# Check current efficiency calculation
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/efficiency/current | jq

# Check steam balance
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/balance | jq

# Check meter calibration
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/calibration | jq
```

**Solution:**
```bash
# Recalibrate steam meters (coordinate with facility team)
# Physical calibration required for accurate measurements

# Adjust efficiency calculation parameters
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   EFFICIENCY_CALC_METHOD: "corrected_enthalpy"  # More accurate method
#   EFFICIENCY_STEAM_QUALITY_DEFAULT: "0.98"  # Assume 98% dryness if not measured
#   EFFICIENCY_HEAT_LOSS_FACTOR: "0.05"  # 5% heat loss in distribution

# Force recalculation
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/efficiency/recalculate
```

### Pressure Drop Calculation Issues

#### Issue 1: Inaccurate Pressure Drop Predictions

**Symptom**: Predicted pressure drop doesn't match actual measured values

**Diagnosis:**
```bash
# Check pressure drop calculation
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/pressure-drop | jq

# Compare predicted vs actual
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/pressure-drop/validation | jq
```

**Solution:**
```bash
# Adjust pressure drop calculation parameters
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   PRESSURE_DROP_FRICTION_FACTOR: "0.025"  # Adjust based on pipe roughness
#   PRESSURE_DROP_MINOR_LOSS_COEFFICIENT: "1.5"  # Account for fittings
#   ENABLE_PRESSURE_DROP_LEARNING: "true"  # Use ML to improve accuracy over time

# Recalibrate with actual measurements
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/pressure-drop/recalibrate \
    -H "Content-Type: application/json" \
    -d '{"use_historical_data": true, "calibration_days": 30}'
```

### Condensate Return Optimization Issues

#### Issue 1: Condensate Return Rate Calculation Errors

**Symptom**: Calculated condensate return rate doesn't match actual recovery

**Diagnosis:**
```bash
# Check condensate return metrics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/analysis/condensate/metrics | jq

# Check condensate meter data
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/condensate | jq
```

**Solution:**
```bash
# Verify condensate meters are online and calibrated
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/condensate/status | jq

# Adjust condensate calculation parameters
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   CONDENSATE_EXPECTED_RETURN_RATE: "0.80"  # 80% expected return
#   CONDENSATE_FLASH_STEAM_FACTOR: "0.15"  # 15% flash steam
#   CONDENSATE_HEAT_CONTENT_FACTOR: "0.25"  # 25% of steam enthalpy

# Recalculate condensate optimization
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/analysis/condensate/recalculate
```

---

## Integration Failures

### SCADA Integration Issues

**Symptom**: Unable to connect to SCADA system or data not being received

**Diagnosis:**
```bash
# Check SCADA integration status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/scada/status | jq

# Test OPC UA connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# Check OPC UA logs
kubectl logs -n greenlang deployment/gl-003-steam-system | grep -i "opc ua"
```

**Solution:**
```bash
# Update SCADA credentials
kubectl create secret generic gl-003-secrets \
  --from-literal=SCADA_ENDPOINT='opc.tcp://scada-server:4840' \
  --from-literal=SCADA_USERNAME='updated-username' \
  --from-literal=SCADA_PASSWORD='updated-password' \
  --dry-run=client -o yaml | kubectl apply -n greenlang -f -

# Adjust SCADA connection parameters
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   SCADA_CONNECTION_TIMEOUT_SECONDS: "30"  # from 10
#   SCADA_RECONNECT_INTERVAL_SECONDS: "60"
#   SCADA_SUBSCRIPTION_INTERVAL_MS: "1000"  # 1 second updates
#   SCADA_SECURITY_MODE: "SignAndEncrypt"  # or "None" for testing

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang

# Test connection
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/integrations/scada/test-connection
```

### Steam Meter Integration Issues

**Symptom**: Steam meter data not being received or meters showing offline

**Diagnosis:**
```bash
# Check steam meter connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/integrations/meters/status | jq

# Check meter communication protocol
kubectl logs -n greenlang deployment/gl-003-steam-system | grep -i "modbus\|mqtt\|rest"

# Test meter gateway
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  ping steam-meter-gateway.local
```

**Solution:**
```bash
# Update meter integration configuration
kubectl edit configmap gl-003-config -n greenlang

# Modify:
#   METER_PROTOCOL: "modbus_tcp"  # or "mqtt", "rest_api"
#   METER_GATEWAY_HOST: "steam-meter-gateway.local"
#   METER_GATEWAY_PORT: "502"
#   METER_POLL_INTERVAL_SECONDS: "5"
#   METER_TIMEOUT_SECONDS: "10"
#   METER_BATCH_SIZE: "50"  # Read 50 meters per request

# For Modbus TCP
#   MODBUS_SLAVE_IDS: "1-100"  # Range of slave IDs
#   MODBUS_REGISTER_START: "0"
#   MODBUS_REGISTER_COUNT: "10"

# For MQTT
#   MQTT_BROKER: "broker.example.com:1883"
#   MQTT_TOPIC_PATTERN: "steam/meters/+/data"
#   MQTT_QOS: "1"

# For REST API
#   METER_API_URL: "https://meter-api.example.com"
#   METER_API_KEY_SECRET: "METER_API_KEY"

# Restart deployment
kubectl rollout restart deployment/gl-003-steam-system -n greenlang
```

---

## Database Connection Issues

### Quick Fixes

```bash
# Test connection
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql $DATABASE_URL

# Test TimescaleDB extension
kubectl run -it --rm debug --image=postgres:14 --restart=Never -n greenlang -- \
  psql $DATABASE_URL -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"

# Restart database (if local)
kubectl rollout restart statefulset/postgresql -n greenlang

# Check connection pool
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_pool

# Increase pool size
kubectl edit configmap gl-003-config -n greenlang
# Set: DATABASE_POOL_SIZE: "40"
```

---

## Redis Cache Issues

### Quick Fixes

```bash
# Test Redis connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host ping

# Check Redis memory
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host INFO memory

# Flush cache (careful in production!)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  redis-cli -h redis-host FLUSHDB

# Restart Redis
kubectl rollout restart deployment/redis -n greenlang

# Check cache hit rate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/metrics | grep cache_hit_rate
```

---

## Memory Leaks

### Diagnosis and Fix

```bash
# Check memory usage over time
kubectl top pods -n greenlang | grep gl-003

# Enable memory profiling
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/debug/memory/snapshot

# Check for memory growth patterns
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/debug/memory/growth | jq

# Force garbage collection
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -X POST http://localhost:8000/api/v1/debug/gc

# Restart pod if memory continues to grow
kubectl delete pod -n greenlang <pod-name>

# Check for thread-safe cache leaks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -s http://localhost:8000/api/v1/debug/cache/statistics | jq
```

---

## CPU Throttling

### Diagnosis and Fix

```bash
# Check CPU usage
kubectl top pods -n greenlang | grep gl-003

# Check CPU throttling
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled

# Increase CPU limits
kubectl set resources deployment gl-003-steam-system -n greenlang \
  --limits=cpu=2000m

# Enable HPA for scaling
kubectl autoscale deployment gl-003-steam-system -n greenlang \
  --min=3 --max=10 --cpu-percent=70

# Optimize parallel processing
kubectl edit configmap gl-003-config -n greenlang
# Set: ANALYSIS_WORKER_THREADS: "2"  # Reduce if CPU-bound
```

---

## Network Connectivity Issues

### Diagnosis and Fix

```bash
# Test DNS resolution
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  nslookup scada-server

# Test connectivity
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  telnet scada-server 4840

# Check network policies
kubectl get networkpolicy -n greenlang
kubectl describe networkpolicy gl-003-network-policy -n greenlang

# Temporarily disable network policy (testing only!)
kubectl delete networkpolicy gl-003-network-policy -n greenlang

# Re-apply after testing
kubectl apply -f deployment/networkpolicy.yaml -n greenlang

# Check for firewall blocks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  curl -v telnet://scada-server:4840
```

---

## TimescaleDB Specific Issues

### Issue 1: Hypertable Chunk Bloat

**Symptom**: Queries getting slower over time, disk space increasing rapidly

**Diagnosis:**
```bash
# Check chunk statistics
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.chunks ORDER BY total_bytes DESC LIMIT 20;"

# Check compression status
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.compressed_chunk_stats;"
```

**Solution:**
```bash
# Enable compression on old chunks
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT compress_chunk(i) FROM show_chunks('steam_measurements', older_than => INTERVAL '7 days') i;"

# Set up automatic compression policy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT add_compression_policy('steam_measurements', INTERVAL '7 days');"

# Drop old chunks (if needed)
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT drop_chunks('steam_measurements', older_than => INTERVAL '90 days');"

# Vacuum to reclaim space
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "VACUUM FULL steam_measurements;"
```

### Issue 2: Continuous Aggregate Not Updating

**Symptom**: Real-time aggregation views showing stale data

**Diagnosis:**
```bash
# Check continuous aggregates
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.continuous_aggregates;"

# Check refresh policy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.job_stats WHERE job_id IN (SELECT job_id FROM timescaledb_information.jobs WHERE proc_name = 'policy_refresh_continuous_aggregate');"
```

**Solution:**
```bash
# Manually refresh continuous aggregate
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "CALL refresh_continuous_aggregate('steam_measurements_hourly', NULL, NULL);"

# Update refresh policy
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT alter_job(<job_id>, schedule_interval => INTERVAL '15 minutes');"

# Check if refresh is running
kubectl exec -n greenlang deployment/gl-003-steam-system -- \
  psql $DATABASE_URL -c "SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_refresh_continuous_aggregate';"
```

---

## Monitoring Dashboards

### Grafana Dashboards

- **Main Dashboard**: https://grafana.greenlang.io/d/gl-003/steam-system-analyzer
- **Leak Detection Dashboard**: https://grafana.greenlang.io/d/gl-003-leaks/leak-analysis
- **Steam Trap Dashboard**: https://grafana.greenlang.io/d/gl-003-traps/trap-performance
- **Error Rate Dashboard**: https://grafana.greenlang.io/d/gl-003-errors/error-analysis
- **Performance Dashboard**: https://grafana.greenlang.io/d/gl-003-perf/performance-metrics

### Prometheus Queries

```promql
# Error rate (last 5 minutes)
rate(gl_003_http_requests_total{status=~"5.."}[5m])

# Request latency (p95)
histogram_quantile(0.95, rate(gl_003_http_request_duration_seconds_bucket[5m]))

# Memory usage
gl_003_system_memory_usage_bytes{type="rss"}

# CPU usage
gl_003_system_cpu_usage_percent

# Leak detection accuracy
gl_003_leak_detection_accuracy_ratio

# Steam trap classification accuracy
gl_003_trap_classification_accuracy_ratio

# Steam meter connectivity
gl_003_meter_connectivity_ratio

# Distribution efficiency
gl_003_distribution_efficiency_percent
```

---

## Escalation Contacts

### On-Call Rotation

- **Primary**: DevOps Team (+1-555-0100) - Slack: #gl-003-oncall
- **Secondary**: Platform Engineering (+1-555-0101) - Slack: #platform-oncall
- **Steam System SME**: Steam Engineering Team (+1-555-0105) - Slack: #steam-engineering

### Severity Guidelines

- **P0 (Critical)**: Production down, steam monitoring offline, no workaround - Escalate immediately
- **P1 (High)**: Major degradation, steam analysis accuracy degraded, limited workaround - Escalate within 1 hour
- **P2 (Medium)**: Minor degradation, some steam monitoring issues, workaround available - Escalate within 4 hours
- **P3 (Low)**: Cosmetic issues, no impact - Handle during business hours

### Communication Channels

- **Slack**: #gl-003-alerts (automated), #gl-003-incidents (manual), #steam-engineering (domain questions)
- **PagerDuty**: Service ID: GL-003-PROD
- **Email**: gl-003-oncall@greenlang.io
- **Status Page**: https://status.greenlang.io

---

## Additional Resources

- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\deployment\README.md`
- **Architecture**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\ARCHITECTURE.md`
- **API Documentation**: https://docs.greenlang.io/agents/gl-003
- **Integration Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-003\INTEGRATION_MODULES_DELIVERY.md`
- **Runbooks**:
  - Incident Response: `INCIDENT_RESPONSE.md`
  - Rollback Procedures: `ROLLBACK_PROCEDURE.md`
  - Scaling Guide: `SCALING_GUIDE.md`
- **TimescaleDB Documentation**: https://docs.timescale.com/
- **Steam System Engineering Resources**: https://www.spiraxsarco.com/learn-about-steam
