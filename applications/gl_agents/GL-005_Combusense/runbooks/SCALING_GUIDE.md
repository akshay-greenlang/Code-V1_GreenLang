# GL-005 COMBUSTIONCONTROLAGENT - SCALING GUIDE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang Industrial Control SRE Team

---

## PURPOSE

This guide provides scaling strategies for the GL-005 CombustionControlAgent to handle varying combustion control workloads, from single burner installations (1 control point @ 1 Hz) to large industrial facilities (100+ burners @ 10 Hz), while maintaining <100ms control loop latency and SIL-2 safety integrity.

---

## PERFORMANCE BASELINES

### Current Performance (3 replicas, standard resources)

| Metric | Value | SLA Target |
|--------|-------|------------|
| **Control Loop Latency (P95)** | 85ms | <100ms |
| **Control Frequency** | 10 Hz | >5 Hz |
| **Burner Capacity** | 50 burners/pod | >40 burners/pod |
| **DCS/PLC Read Latency** | 15ms | <50ms |
| **Safety Interlock Response** | <50ms | <100ms |
| **PID Calculation Time** | <5ms | <10ms |
| **Data Acquisition Rate** | 100 Hz | >50 Hz |
| **Memory Usage** | 2.2GB per pod | <4GB |
| **CPU Usage** | 65% average | <80% |

### Agent-Level Performance

| Agent | Latency (P95) | Throughput | SLA |
|-------|---------------|------------|-----|
| **1. Data Intake** | 20ms | 1000 points/sec | <50ms |
| **2. Combustion Analysis** | 15ms | 500 analyses/sec | <30ms |
| **3. Control Optimizer** | 35ms | 200 optimizations/sec | <50ms |
| **4. Command Execution** | 10ms | 300 commands/sec | <20ms |
| **5. Audit & Safety** | 5ms | 1000 events/sec | <10ms |

---

## CAPACITY PLANNING

### Control Point Scaling Guidelines

| Control Points | Frequency | Recommended Replicas | Resources | Expected Latency |
|----------------|-----------|----------------------|-----------|------------------|
| **1-10 burners** | 1-5 Hz | 3 | Standard | <50ms |
| **10-25 burners** | 5-10 Hz | 3 | Standard | <80ms |
| **25-50 burners** | 5-10 Hz | 3-5 | Standard | <100ms |
| **50-75 burners** | 5-10 Hz | 5-8 | Increased | <100ms |
| **75-100 burners** | 5-10 Hz | 8-12 | Increased | <120ms |
| **100+ burners** | 5-10 Hz | 12-20 | High | <150ms |

**Standard Resources:**
- CPU: 2000m (2 cores) request, 4000m limit
- Memory: 2Gi request, 4Gi limit

**Increased Resources:**
- CPU: 4000m (4 cores) request, 8000m limit
- Memory: 4Gi request, 8Gi limit

**High Resources:**
- CPU: 8000m (8 cores) request, 16000m limit
- Memory: 8Gi request, 16Gi limit

### Control Frequency Impact

| Frequency | CPU Impact | Memory Impact | Network Impact |
|-----------|------------|---------------|----------------|
| **1 Hz** | Baseline | Baseline | Baseline |
| **5 Hz** | +150% | +50% | +400% |
| **10 Hz** | +300% | +80% | +900% |

**Recommendation:** Use 5 Hz for most applications. 10 Hz only if required for fast-responding burners or high-precision control.

---

## HORIZONTAL SCALING

### Manual Horizontal Scaling

Scale based on number of control points and required control frequency.

#### Scale Up

```bash
# For moderate burner count increase (50-75 burners)
kubectl scale deployment/gl-005-combustion-control --replicas=5 -n greenlang

# For high burner count (75-100 burners)
kubectl scale deployment/gl-005-combustion-control --replicas=10 -n greenlang

# For very high burner count (100+ burners)
kubectl scale deployment/gl-005-combustion-control --replicas=15 -n greenlang

# Monitor scaling
kubectl get pods -n greenlang -l app=gl-005-combustion-control -w

# Verify control performance after scaling
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_control_loop_duration_seconds{quantile="0.95"}'
```

#### Scale Down

```bash
# After decommissioning burners or reducing control frequency
kubectl scale deployment/gl-005-combustion-control --replicas=3 -n greenlang

# Gradual scale-down (safer for production)
kubectl scale deployment/gl-005-combustion-control --replicas=5 -n greenlang
# Wait 10 minutes, monitor control latency
kubectl scale deployment/gl-005-combustion-control --replicas=3 -n greenlang

# Verify no control degradation
watch -n 5 'curl -s http://gl-005-combustion-control:8001/metrics | grep gl005_control_loop_duration'
```

### Automatic Horizontal Scaling (HPA)

**Setup Horizontal Pod Autoscaler:**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-005-combustion-control-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-005-combustion-control
  minReplicas: 3
  maxReplicas: 20
  metrics:
  # CPU-based scaling
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory-based scaling
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  # Control loop latency-based scaling (custom metric)
  - type: Pods
    pods:
      metric:
        name: gl005_control_loop_duration_seconds_p95
      target:
        type: AverageValue
        averageValue: "0.08"  # 80ms P95 latency
  # Active burner count (custom metric)
  - type: Pods
    pods:
      metric:
        name: gl005_active_burners_per_pod
      target:
        type: AverageValue
        averageValue: "40"  # 40 burners per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 120  # Wait 2 min before scaling up
      policies:
      - type: Percent
        value: 50  # Scale up by 50% at most
        periodSeconds: 60
      - type: Pods
        value: 2  # Or add 2 pods
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 600  # Wait 10 min before scaling down
      policies:
      - type: Percent
        value: 25  # Scale down by 25% at most
        periodSeconds: 120
      - type: Pods
        value: 1  # Or remove 1 pod
        periodSeconds: 120
      selectPolicy: Min
```

**Deploy HPA:**

```bash
kubectl apply -f k8s/hpa.yaml

# Monitor HPA
kubectl get hpa gl-005-combustion-control-hpa -n greenlang -w

# Check current metrics and scaling decisions
kubectl describe hpa gl-005-combustion-control-hpa -n greenlang

# View scaling events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep HorizontalPodAutoscaler
```

**HPA Behavior:**
- **Scale Up Triggers:**
  - CPU >70% for 2 minutes
  - Memory >75% for 2 minutes
  - Control loop latency >80ms P95 for 2 minutes
  - >40 burners per pod average
- **Scale Down Triggers:**
  - CPU <50% for 10 minutes
  - Memory <55% for 10 minutes
  - Control loop latency <60ms P95 for 10 minutes
  - <30 burners per pod average
- **Scaling Rate:**
  - Up: 50% or 2 pods (whichever greater) every minute
  - Down: 25% or 1 pod (whichever smaller) every 2 minutes

---

## VERTICAL SCALING

### When to Use Vertical Scaling

- Individual pods hitting CPU limits (throttling detected)
- Memory saturation causing garbage collection pressure
- High control frequency (10 Hz) on complex burner arrays
- Multi-fuel optimization (simultaneous gas + oil)
- Large historical data buffering requirements

### Increase Pod Resources

```bash
# Method 1: Edit deployment directly
kubectl edit deployment gl-005-combustion-control -n greenlang

# Update resources section:
# resources:
#   requests:
#     cpu: 4000m
#     memory: 4Gi
#   limits:
#     cpu: 8000m
#     memory: 8Gi

# Method 2: Patch deployment
kubectl patch deployment gl-005-combustion-control -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: gl-005-combustion-control
        resources:
          requests:
            cpu: "4000m"
            memory: "4Gi"
          limits:
            cpu: "8000m"
            memory: "8Gi"
'

# Monitor rollout
kubectl rollout status deployment/gl-005-combustion-control -n greenlang

# Verify new resources
kubectl get pods -n greenlang -l app=gl-005-combustion-control \
  -o jsonpath='{.items[0].spec.containers[0].resources}'

# Test control performance with new resources
curl -X POST http://gl-005-combustion-control:8000/test/performance \
  -d '{"duration_sec": 300, "burner_count": 100, "frequency_hz": 10}'
```

### Resource Tiers

**Tier 1: Standard (Default)**
```yaml
resources:
  requests:
    cpu: 2000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 4Gi
```
**Use Case:** <50 burners @ 5 Hz

**Tier 2: Increased**
```yaml
resources:
  requests:
    cpu: 4000m
    memory: 4Gi
  limits:
    cpu: 8000m
    memory: 8Gi
```
**Use Case:** 50-100 burners @ 5-10 Hz

**Tier 3: High**
```yaml
resources:
  requests:
    cpu: 8000m
    memory: 8Gi
  limits:
    cpu: 16000m
    memory: 16Gi
```
**Use Case:** 100+ burners @ 10 Hz or complex multi-fuel optimization

---

## DATABASE SCALING

### PostgreSQL + TimescaleDB Scaling

#### Connection Pool Tuning

```bash
# Increase connection pool for high concurrency
kubectl set env deployment/gl-005-combustion-control \
  DB_POOL_SIZE=30 \
  DB_MAX_OVERFLOW=60 \
  DB_POOL_TIMEOUT=30 \
  -n greenlang

# Monitor connection usage
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "SELECT count(*) as connections, state FROM pg_stat_activity GROUP BY state;"

# Check for connection exhaustion
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "SELECT count(*) as active FROM pg_stat_activity WHERE state = 'active';"
```

**Recommended Settings:**

| Replicas | DB_POOL_SIZE | DB_MAX_OVERFLOW | Total Connections |
|----------|--------------|-----------------|-------------------|
| 3 | 20 | 40 | 180 |
| 5 | 25 | 50 | 375 |
| 10 | 30 | 60 | 900 |
| 15 | 40 | 80 | 1800 |
| 20 | 50 | 100 | 3000 |

**PostgreSQL max_connections** should be set to at least Total Connections + 100 (for admin).

For high replica counts, consider PgBouncer connection pooling:
```bash
kubectl apply -f k8s/pgbouncer.yaml
# Update connection string to use PgBouncer
kubectl set env deployment/gl-005-combustion-control \
  DATABASE_URL="postgresql://user:pass@pgbouncer:5432/greenlang" \
  -n greenlang
```

#### TimescaleDB Optimization

```bash
# Optimize hypertables for time-series data
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
-- Set compression on combustion_data (reduce storage by 90%)
ALTER TABLE combustion_data SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'unit_id',
  timescaledb.compress_orderby = 'timestamp DESC'
);

-- Compress data older than 7 days
SELECT add_compression_policy('combustion_data', INTERVAL '7 days');

-- Retention policy: drop data older than 90 days
SELECT add_retention_policy('combustion_data', INTERVAL '90 days');
"

# Create indexes for common queries
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "
CREATE INDEX IF NOT EXISTS idx_combustion_data_unit_timestamp
  ON combustion_data(unit_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_control_events_unit_timestamp
  ON control_events(unit_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_safety_events_timestamp
  ON safety_events(timestamp DESC);
"

# Vacuum and analyze
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "VACUUM ANALYZE combustion_data;"

# Update statistics for query planner
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql greenlang -c "ANALYZE;"
```

#### Database Vertical Scaling

```bash
# Increase PostgreSQL resources for high-frequency control
kubectl edit statefulset postgres-timescaledb -n greenlang

# Update to:
# resources:
#   requests:
#     cpu: 8000m
#     memory: 16Gi
#   limits:
#     cpu: 16000m
#     memory: 32Gi

# Tune PostgreSQL parameters
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "ALTER SYSTEM SET shared_buffers = '8GB';"
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "ALTER SYSTEM SET effective_cache_size = '24GB';"
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "ALTER SYSTEM SET work_mem = '64MB';"
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "ALTER SYSTEM SET maintenance_work_mem = '2GB';"

# Restart PostgreSQL to apply
kubectl rollout restart statefulset/postgres-timescaledb -n greenlang
```

---

## REDIS SCALING

### Redis for Real-Time State Caching

```bash
# Deploy Redis cluster for high-availability caching
kubectl apply -f k8s/redis-cluster.yaml

# Configure GL-005 to use Redis
kubectl set env deployment/gl-005-combustion-control \
  REDIS_ENABLED=true \
  REDIS_CLUSTER_ENDPOINTS="redis-0:6379,redis-1:6379,redis-2:6379" \
  REDIS_CACHE_TTL_SECONDS=60 \
  -n greenlang

# Monitor Redis performance
kubectl exec -n greenlang deployment/redis-0 -- redis-cli INFO stats

# Check cache hit rate
kubectl exec -n greenlang deployment/redis-0 -- \
  redis-cli INFO stats | grep 'keyspace_hits\|keyspace_misses'

# Target: >95% cache hit rate for control state
```

**What to Cache:**
- PID controller state (Kp, Ki, Kd, integral, derivative)
- Recent sensor readings (last 10 seconds)
- Burner configuration (static data)
- Emission factors (rarely change)
- Safety interlock thresholds (static)

**What NOT to Cache:**
- Raw time-series data (store in TimescaleDB)
- Safety event logs (compliance - must persist)
- Control commands (audit trail required)

---

## APPLICATION-LEVEL OPTIMIZATION

### Control Loop Optimization

```bash
# Reduce control frequency for stable processes
kubectl set env deployment/gl-005-combustion-control \
  CONTROL_FREQUENCY_HZ=5 \
  -n greenlang

# Enable control output rate limiting (smoother control)
kubectl set env deployment/gl-005-combustion-control \
  ENABLE_RATE_LIMITING=true \
  MAX_FUEL_RATE_CHANGE_PERCENT_PER_SEC=5 \
  MAX_AIR_RATE_CHANGE_PERCENT_PER_SEC=8 \
  -n greenlang

# Optimize PID calculation (reduce iterations)
kubectl set env deployment/gl-005-combustion-control \
  PID_OPTIMIZATION_MAX_ITERATIONS=10 \
  PID_CONVERGENCE_TOLERANCE=0.001 \
  -n greenlang
```

### Data Acquisition Optimization

```bash
# Reduce DCS/PLC polling frequency (if acceptable)
kubectl set env deployment/gl-005-combustion-control \
  DCS_POLLING_FREQUENCY_HZ=50 \
  PLC_POLLING_FREQUENCY_HZ=50 \
  -n greenlang

# Enable data compression for network efficiency
kubectl set env deployment/gl-005-combustion-control \
  ENABLE_DATA_COMPRESSION=true \
  COMPRESSION_THRESHOLD_BYTES=1024 \
  -n greenlang

# Batch database inserts
kubectl set env deployment/gl-005-combustion-control \
  DB_BATCH_INSERT_SIZE=100 \
  DB_BATCH_INSERT_INTERVAL_SEC=1 \
  -n greenlang
```

### Performance Tuning Variables

| Variable | Default | High Performance | Description |
|----------|---------|------------------|-------------|
| CONTROL_FREQUENCY_HZ | 10 | 5 | Control loop frequency (Hz) |
| DCS_POLLING_FREQUENCY_HZ | 100 | 50 | DCS data acquisition rate |
| PLC_POLLING_FREQUENCY_HZ | 100 | 50 | PLC data acquisition rate |
| ENABLE_RATE_LIMITING | true | true | Smooth control output changes |
| PID_OPTIMIZATION_MAX_ITERATIONS | 20 | 10 | PID optimization iterations |
| DB_BATCH_INSERT_SIZE | 50 | 100 | Database batch insert size |
| REDIS_ENABLED | false | true | Enable Redis caching |
| ENABLE_DATA_COMPRESSION | false | true | Compress network data |

---

## MONITORING DURING SCALING

### Key Metrics to Watch

```bash
# 1. Pod count and health
kubectl get pods -n greenlang -l app=gl-005-combustion-control

# 2. Resource usage
kubectl top pods -n greenlang -l app=gl-005-combustion-control

# 3. Control loop performance
curl http://gl-005-combustion-control:8001/metrics | \
  grep -E "gl005_control_loop_duration|gl005_control_frequency|gl005_pid_calculation_time"

# 4. Database connections
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 5. Safety system latency
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_safety_interlock_latency_seconds'

# 6. Integration health
curl http://gl-005-combustion-control:8000/integrations/status | \
  jq '.integrations[] | {name: .name, status: .status, latency_ms: .latency_ms}'
```

### Grafana Dashboards

- **GL-005 Agent Performance Dashboard:** http://grafana:3000/d/gl005-perf
  - Control loop latency (P50, P95, P99)
  - Agent execution times
  - Throughput (control points/sec)
  - Pod resource usage

- **GL-005 Combustion Metrics Dashboard:** http://grafana:3000/d/gl005-combustion
  - Heat output tracking
  - Fuel-air ratio optimization
  - Emissions (NOx, CO, CO2)
  - Combustion efficiency
  - Flame stability index

- **GL-005 Safety Monitoring Dashboard:** http://grafana:3000/d/gl005-safety
  - Safety interlock status
  - Emergency shutdown events
  - Temperature/pressure/flow limits
  - SIL-2 compliance tracking

### Alerts

Critical alerts during scaling:

```yaml
# alerts.yml
- alert: GL005ControlLoopLatencyHigh
  expr: gl005_control_loop_duration_seconds{quantile="0.95"} > 0.1
  for: 5m
  annotations:
    summary: "Control loop latency exceeds 100ms"
    description: "P95 latency {{ $value }}s, consider scaling up"

- alert: GL005PodCPUSaturation
  expr: avg(rate(container_cpu_usage_seconds_total{pod=~"gl-005.*"}[5m])) > 0.85
  for: 10m
  annotations:
    summary: "GL-005 pods CPU saturated"
    description: "Average CPU usage {{ $value }}, scale up or optimize"

- alert: GL005DatabaseConnectionExhaustion
  expr: pg_stat_activity_count{database="greenlang"} / pg_settings_max_connections > 0.8
  for: 5m
  annotations:
    summary: "PostgreSQL connection pool near limit"
    description: "{{ $value }}% of max connections used"

- alert: GL005SafetyInterlockLatencyHigh
  expr: gl005_safety_interlock_latency_seconds{quantile="0.95"} > 0.1
  for: 2m
  severity: critical
  annotations:
    summary: "CRITICAL: Safety interlock latency exceeds 100ms"
    description: "Safety system response time {{ $value }}s, immediate investigation required"
```

---

## SCALING PLAYBOOKS

### Playbook 1: New Facility Commissioning (100 burners @ 10 Hz)

**Situation:** Large facility coming online with 100 burners requiring 10 Hz control

**Timeline:** 2 weeks before commissioning

**Actions:**

```bash
# Week -2: Infrastructure preparation
# 1. Scale to 15 replicas
kubectl scale deployment/gl-005-combustion-control --replicas=15 -n greenlang

# 2. Increase resources to High tier
kubectl patch deployment gl-005-combustion-control -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: gl-005-combustion-control
        resources:
          requests: {cpu: "8000m", memory: "8Gi"}
          limits: {cpu: "16000m", memory: "16Gi"}
'

# 3. Scale database
kubectl edit statefulset postgres-timescaledb -n greenlang
# Update to: cpu: 16000m, memory: 32Gi

# 4. Increase database connections
kubectl set env deployment/gl-005-combustion-control \
  DB_POOL_SIZE=50 DB_MAX_OVERFLOW=100 -n greenlang

# 5. Deploy Redis cluster
kubectl apply -f k8s/redis-cluster.yaml

# 6. Enable high-performance mode
kubectl set env deployment/gl-005-combustion-control \
  CONTROL_FREQUENCY_HZ=10 \
  REDIS_ENABLED=true \
  DB_BATCH_INSERT_SIZE=100 \
  -n greenlang

# Week -1: Performance testing
# Load test with simulated 100 burners @ 10 Hz
python tests/load_test.py \
  --burner-count 100 \
  --frequency-hz 10 \
  --duration-sec 3600 \
  --verify-latency

# Week 0: Commissioning
# Monitor closely during ramp-up
# Scale down after stable operation (Week +2)
```

### Playbook 2: Emergency High-Load Response

**Situation:** Additional burners brought online unexpectedly, control latency increasing

**Actions:**

```bash
# Immediate: Max scale
kubectl scale deployment/gl-005-combustion-control --replicas=20 -n greenlang

# Increase resources to High tier
kubectl patch deployment gl-005-combustion-control -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: gl-005-combustion-control
        resources:
          requests: {cpu: "8000m", memory: "8Gi"}
          limits: {cpu: "16000m", memory: "16Gi"}
'

# Optimize for throughput
kubectl set env deployment/gl-005-combustion-control \
  CONTROL_FREQUENCY_HZ=5 \
  DB_BATCH_INSERT_SIZE=200 \
  REDIS_ENABLED=true \
  -n greenlang

# Monitor intensively (every 1 minute)
watch -n 60 'curl -s http://gl-005-combustion-control:8001/metrics | \
  grep gl005_control_loop_duration_seconds'
```

---

## COST OPTIMIZATION

### Right-Sizing After Commissioning

After facility ramp-up complete, reduce resources to match actual load:

```bash
# 1. Analyze actual usage during peak
kubectl top pods -n greenlang -l app=gl-005-combustion-control --containers

# 2. Review control performance metrics
curl http://prometheus:9090/api/v1/query?query=\
  'gl005_control_loop_duration_seconds{quantile="0.95"}'

# 3. Scale down to actual needs
# If P95 latency <60ms with 15 replicas, try 12
kubectl scale deployment/gl-005-combustion-control --replicas=12 -n greenlang

# 4. Monitor for 48 hours
# If latency still <80ms, continue scaling down

# 5. Reduce resources if headroom exists
kubectl patch deployment gl-005-combustion-control -n greenlang --patch '
spec:
  template:
    spec:
      containers:
      - name: gl-005-combustion-control
        resources:
          requests: {cpu: "4000m", memory: "4Gi"}
          limits: {cpu: "8000m", memory: "8Gi"}
'

# 6. Enable HPA for dynamic scaling
kubectl apply -f k8s/hpa.yaml
```

### Resource Requests Optimization

**Requests** = Guaranteed resources (used for scheduling)
**Limits** = Maximum allowed (prevents resource starvation)

**Best Practices:**
- Set requests to 80% of normal usage
- Set limits to 2x requests (allows bursting for transients)
- Monitor actual usage monthly and adjust
- Use HPA to handle variable loads dynamically

---

## TROUBLESHOOTING SCALING ISSUES

### Pods Not Starting After Scale Up

```bash
# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp' | grep gl-005

# Common issue: Insufficient node resources
# Solution: Add more nodes or reduce resource requests

# Add node labels for GL-005 dedicated nodes
kubectl label nodes node-10 workload=gl005
kubectl label nodes node-11 workload=gl005

# Update deployment with node affinity
kubectl patch deployment gl-005-combustion-control -n greenlang --patch '
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: workload
                operator: In
                values:
                - gl005
'
```

### Performance Not Improving After Scaling

```bash
# Check if bottleneck is elsewhere
# 1. Database connections maxed?
kubectl exec -n greenlang deployment/postgres-timescaledb -- \
  psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 2. DCS/PLC network latency?
curl http://gl-005-combustion-control:8000/integrations/status | \
  jq '.integrations[] | select(.latency_ms > 50)'

# 3. Safety interlock processing slow?
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_safety_interlock_latency_seconds{quantile="0.95"}'

# 4. PID calculation slow?
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_pid_calculation_duration_seconds{quantile="0.95"}'

# Profile with py-spy
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  py-spy top --pid 1 --duration 60
```

### Control Loop Latency Increasing

```bash
# Check per-agent latency
curl http://gl-005-combustion-control:8001/metrics | \
  grep 'gl005_agent_duration_seconds' | sort -k3 -n

# Identify slow agent
# If Data Intake slow: Check DCS/PLC connectivity
# If Combustion Analysis slow: Check calculation complexity
# If Control Optimizer slow: Reduce optimization iterations
# If Command Execution slow: Check DCS/PLC write latency
# If Audit & Safety slow: Check database write performance

# Optimize slow component
# Example: Reduce PID optimization iterations
kubectl set env deployment/gl-005-combustion-control \
  PID_OPTIMIZATION_MAX_ITERATIONS=5 \
  -n greenlang
```

---

## RELATED RUNBOOKS

- MAINTENANCE.md - Resource planning and capacity reviews
- INCIDENT_RESPONSE.md - Performance incident handling
- TROUBLESHOOTING.md - Performance degradation diagnosis
- ROLLBACK_PROCEDURE.md - Deployment rollback procedures

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2026-02-18 (Quarterly)
- **Owner:** GL-005 SRE Team

---

*This scaling guide should be reviewed quarterly based on actual facility load patterns and performance data.*
