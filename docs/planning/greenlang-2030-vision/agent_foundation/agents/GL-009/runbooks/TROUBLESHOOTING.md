# GL-009 THERMALIQ Troubleshooting Guide

**Agent**: GL-009 THERMALIQ ThermalEfficiencyCalculator
**Version**: 1.0.0
**Last Updated**: 2025-11-26
**Owner**: GreenLang SRE Team

---

## Table of Contents

1. [High Calculation Latency](#1-high-calculation-latency)
2. [Low Cache Hit Rate](#2-low-cache-hit-rate)
3. [Energy Balance Not Closing](#3-energy-balance-not-closing)
4. [Unexpected Efficiency Values](#4-unexpected-efficiency-values)
5. [Connector Timeout Errors](#5-connector-timeout-errors)
6. [Data Quality Issues](#6-data-quality-issues)
7. [Memory Leaks](#7-memory-leaks)
8. [CPU Spikes](#8-cpu-spikes)
9. [Database Connection Pool Exhaustion](#9-database-connection-pool-exhaustion)
10. [Sankey Diagram Rendering Issues](#10-sankey-diagram-rendering-issues)
11. [Benchmark Comparison Failures](#11-benchmark-comparison-failures)
12. [Logging Issues](#12-logging-issues)
13. [Metrics Not Exporting](#13-metrics-not-exporting)
14. [Health Check Failures](#14-health-check-failures)
15. [Configuration Errors](#15-configuration-errors)
16. [Authentication Failures](#16-authentication-failures)
17. [Rate Limiting Issues](#17-rate-limiting-issues)
18. [Kubernetes Pod Restarts](#18-kubernetes-pod-restarts)
19. [Network Connectivity Issues](#19-network-connectivity-issues)
20. [SSL/TLS Certificate Issues](#20-ssltls-certificate-issues)

---

## 1. High Calculation Latency

### Symptoms

- Calculation requests taking > 30 seconds
- p95 latency > 15 seconds
- User complaints about slow performance
- Timeout errors increasing
- Queue depth growing

### Root Causes

1. **Large Data Volume**: Calculations processing too many data points
2. **Slow External Services**: Historian or energy meter API slow
3. **Database Queries**: Unoptimized or slow database queries
4. **Insufficient Resources**: CPU/memory constraints
5. **Cache Misses**: Low cache hit rate forcing recalculation
6. **Network Latency**: Slow network between services

### Diagnostic Commands

```bash
# Check current latency
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket)' | \
  jq '.data.result[0].value[1]'

# Check active calculations
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[] | {pod: .metadata.name, cpu: .status.containerStatuses[0].usage.cpu, memory: .status.containerStatuses[0].usage.memory}'

# Check recent slow calculations
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Calculation completed" | \
  jq 'select(.duration_seconds > 20) | {calc_id: .calculation_id, facility: .facility_id, duration: .duration_seconds, data_points: .data_point_count}' | \
  head -20

# Check database query performance
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT query, calls, mean_exec_time, max_exec_time, stddev_exec_time
   FROM pg_stat_statements
   WHERE query NOT LIKE '%pg_stat_statements%'
   ORDER BY mean_exec_time DESC
   LIMIT 10;"

# Check historian response time
time curl -X POST https://historian.example.com/api/query \
  -H "Authorization: Bearer $HISTORIAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "FAC-001",
    "start_time": "2025-11-01T00:00:00Z",
    "end_time": "2025-11-01T01:00:00Z",
    "tags": ["temperature", "flow_rate", "pressure"]
  }'

# Check energy meter response time
time curl -X GET "https://meter-api.example.com/facilities/FAC-001/readings" \
  -H "Authorization: Bearer $METER_TOKEN"

# Check cache hit rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=thermaliq_cache_hits_total/(thermaliq_cache_hits_total+thermaliq_cache_misses_total)' | \
  jq '.data.result[0].value[1]'

# Check resource utilization
kubectl top pods -n gl-009-production -l app=thermaliq
kubectl top nodes
```

### Resolution Steps

**Step 1: Identify Bottleneck**

```bash
# Trace a slow calculation
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "Calculation.*FAC-001"

# Look for time spent in each phase:
# - Data retrieval: X seconds
# - Validation: Y seconds
# - Calculation: Z seconds
# - Sankey generation: W seconds
```

**Step 2: Optimize Data Retrieval** (if bottleneck is external services)

```bash
# Increase timeout for slow services
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_TIMEOUT=90 \
  ENERGY_METER_TIMEOUT=60

# Enable aggressive caching
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CACHE_TTL=7200 \
  ENERGY_METER_CACHE_TTL=3600

# Enable query batching
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_BATCH_QUERIES=true \
  HISTORIAN_BATCH_SIZE=50

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 3: Optimize Database Queries** (if bottleneck is database)

```bash
# Add missing indexes
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<EOF
-- Index for facility + time range queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_energy_readings_facility_time
ON energy_readings(facility_id, timestamp DESC);

-- Index for calculation lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calculations_facility_created
ON calculations(facility_id, created_at DESC);

-- Partial index for pending calculations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calculations_pending
ON calculations(status, created_at)
WHERE status = 'pending';

-- Update statistics
ANALYZE energy_readings;
ANALYZE calculations;
EOF

# Verify index usage
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "EXPLAIN ANALYZE
   SELECT * FROM energy_readings
   WHERE facility_id = 'FAC-001'
     AND timestamp >= NOW() - INTERVAL '1 day'
   ORDER BY timestamp DESC;"
```

**Step 4: Scale Resources** (if bottleneck is compute)

```bash
# Scale horizontally
kubectl scale deployment/thermaliq -n gl-009-production --replicas=8

# Scale vertically
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=cpu=4000m,memory=8Gi \
  --requests=cpu=2000m,memory=4Gi

# Enable HPA
kubectl autoscale deployment thermaliq -n gl-009-production \
  --cpu-percent=70 \
  --min=4 \
  --max=12

# Wait for scaling
kubectl rollout status deployment/thermaliq -n gl-009-production
```

**Step 5: Optimize Calculation Algorithm** (if bottleneck is processing)

```bash
# Enable calculation chunking for large datasets
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_CALCULATION_CHUNKING=true \
  MAX_DATA_POINTS_PER_CHUNK=50000

# Enable parallel processing
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_PARALLEL_CALCULATION=true \
  CALCULATION_WORKER_THREADS=4

# Disable non-essential features for fast mode
kubectl set env deployment/thermaliq -n gl-009-production \
  SKIP_SANKEY_IN_FAST_MODE=true \
  FAST_MODE_THRESHOLD_SECONDS=30

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 6: Verify Improvement**

```bash
# Monitor latency for 30 minutes
watch -n 60 'curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode "query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m])" | \
  jq ".data.result[0].value[1]"'

# Run load test
./scripts/load_test.sh --duration 300 --rps 10

# Check p95 latency (target: < 15 seconds)
# Check p99 latency (target: < 30 seconds)
```

### Prevention Measures

```bash
# Set up latency alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-latency-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-latency
    interval: 30s
    rules:
    - alert: HighCalculationLatency
      expr: histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m]) > 20
      for: 5m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High calculation latency detected"
        description: "p95 latency is {{ \$value }}s (threshold: 20s)"

    - alert: CriticalCalculationLatency
      expr: histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m]) > 30
      for: 2m
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Critical calculation latency"
        description: "p95 latency is {{ \$value }}s (threshold: 30s)"
EOF

# Enable performance profiling
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_PROFILING=true \
  PROFILING_SAMPLE_RATE=0.01

# Schedule regular performance reviews
# Add to weekly maintenance tasks
```

---

## 2. Low Cache Hit Rate

### Symptoms

- Cache hit rate < 70%
- Increased database load
- Higher API latency
- More historian queries
- Increased costs

### Root Causes

1. **Cache TTL Too Short**: Data expiring too quickly
2. **Cache Size Too Small**: Frequent evictions
3. **Cache Key Design**: Poor key structure causing misses
4. **High Request Diversity**: Many unique queries
5. **Cache Warming Not Working**: Cold cache after restarts
6. **Eviction Policy**: Suboptimal eviction strategy

### Diagnostic Commands

```bash
# Check cache hit rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_cache_hits_total[5m])/(rate(thermaliq_cache_hits_total[5m])+rate(thermaliq_cache_misses_total[5m]))' | \
  jq '.data.result[0].value[1]'

# Check cache size and usage
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO memory

# Check eviction count
kubectl exec -it redis-0 -n gl-009-production -- redis-cli INFO stats | grep evicted_keys

# Check key distribution
kubectl exec -it redis-0 -n gl-009-production -- redis-cli --bigkeys

# Sample cache keys
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli SCAN 0 COUNT 100 | \
  grep -v "^[0-9]" | \
  head -20

# Check TTLs
kubectl exec -it redis-0 -n gl-009-production -- bash -c '
for key in $(redis-cli SCAN 0 COUNT 20 | grep -v "^[0-9]"); do
  ttl=$(redis-cli TTL "$key")
  echo "$key: $ttl seconds"
done
'

# Analyze cache access patterns
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Cache" | \
  jq -r '"\(.cache_operation) \(.cache_key)"' | \
  sort | uniq -c | sort -rn | head -20
```

### Resolution Steps

**Step 1: Analyze Cache Patterns**

```bash
# Identify frequently accessed keys
kubectl logs -n gl-009-production -l app=thermaliq --tail=5000 | \
  grep "Cache.*hit" | \
  jq -r '.cache_key' | \
  sort | uniq -c | sort -rn | head -50

# Identify frequent cache misses
kubectl logs -n gl-009-production -l app=thermaliq --tail=5000 | \
  grep "Cache.*miss" | \
  jq -r '.cache_key' | \
  sort | uniq -c | sort -rn | head -50

# Analyze why misses occur:
# - Key never cached?
# - Key expired?
# - Key evicted?
```

**Step 2: Increase Cache TTL** (if keys expiring too quickly)

```bash
# Increase TTL for calculation results
kubectl set env deployment/thermaliq -n gl-009-production \
  CALCULATION_CACHE_TTL=7200

# Increase TTL for historian data
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CACHE_TTL=14400

# Increase TTL for energy meter data
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_METER_CACHE_TTL=7200

# Increase TTL for benchmark data
kubectl set env deployment/thermaliq -n gl-009-production \
  BENCHMARK_CACHE_TTL=86400

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 3: Increase Cache Size** (if evictions occurring)

```bash
# Increase Redis memory limit
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG SET maxmemory 16gb

# Make permanent
kubectl edit statefulset redis -n gl-009-production
# Update: resources.limits.memory: 16Gi

# Or deploy larger Redis instance
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: gl-009-production
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        resources:
          requests:
            memory: 8Gi
            cpu: 2000m
          limits:
            memory: 16Gi
            cpu: 4000m
        args:
        - --maxmemory 15gb
        - --maxmemory-policy allkeys-lru
EOF
```

**Step 4: Optimize Cache Key Design** (if poor key structure)

```bash
# Review current key structure
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Cache key" | \
  jq -r '.cache_key' | \
  head -20

# Example keys:
# Good: calculation:FAC001:2025-11:hourly
# Bad:  calculation:FAC001:2025-11-01T00:00:00Z:2025-11-01T23:59:59Z:detailed

# Update application to use hierarchical keys
# Deploy updated code with optimized keys
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.0-cache-keys

# Clear old cache after deploying new key structure
kubectl exec -it redis-0 -n gl-009-production -- redis-cli FLUSHDB
```

**Step 5: Implement Cache Warming** (if cold cache issue)

```bash
# Enable cache warming on startup
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_CACHE_WARMING=true \
  CACHE_WARMING_FACILITIES=FAC-001,FAC-002,FAC-003 \
  CACHE_WARMING_DAYS=7

# Create cache warming job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: thermaliq-cache-warmer
  namespace: gl-009-production
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cache-warmer
            image: ghcr.io/greenlang/thermaliq:latest
            command:
            - python
            - /app/scripts/warm_cache.py
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: postgres-credentials
                  key: url
            - name: REDIS_URL
              value: redis://redis-service:6379
          restartPolicy: OnFailure
EOF

# Manually warm cache for critical facilities
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/warm_cache.py \
    --facilities FAC-001,FAC-002,FAC-003 \
    --days 30
```

**Step 6: Optimize Eviction Policy** (if wrong items being evicted)

```bash
# Check current eviction policy
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG GET maxmemory-policy

# Change to LRU (Least Recently Used)
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Or use LFU (Least Frequently Used) for more stable workloads
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli CONFIG SET maxmemory-policy allkeys-lfu

# Make permanent
kubectl edit statefulset redis -n gl-009-production
# Add: --maxmemory-policy allkeys-lru to args
```

**Step 7: Implement Multi-Tier Caching**

```bash
# Enable in-memory cache (L1) + Redis (L2)
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_L1_CACHE=true \
  L1_CACHE_SIZE_MB=512 \
  L1_CACHE_TTL=300

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 8: Verify Improvement**

```bash
# Monitor cache hit rate for 2 hours
watch -n 120 'curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode "query=rate(thermaliq_cache_hits_total[10m])/(rate(thermaliq_cache_hits_total[10m])+rate(thermaliq_cache_misses_total[10m]))" | \
  jq ".data.result[0].value[1]"'

# Target: > 80% hit rate

# Check eviction rate decreased
kubectl exec -it redis-0 -n gl-009-production -- \
  redis-cli INFO stats | grep evicted_keys

# Verify latency improved
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_calculation_duration_seconds_bucket[5m])' | \
  jq '.data.result[0].value[1]'
```

### Prevention Measures

```bash
# Set up cache performance alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-cache-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-cache
    interval: 60s
    rules:
    - alert: LowCacheHitRate
      expr: rate(thermaliq_cache_hits_total[10m])/(rate(thermaliq_cache_hits_total[10m])+rate(thermaliq_cache_misses_total[10m])) < 0.7
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Low cache hit rate"
        description: "Cache hit rate is {{ \$value }} (threshold: 0.7)"

    - alert: HighCacheEvictionRate
      expr: rate(redis_evicted_keys_total[5m]) > 100
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High cache eviction rate"
        description: "Evicting {{ \$value }} keys/sec"
EOF

# Schedule cache analysis
# Add to weekly maintenance tasks
```

---

## 3. Energy Balance Not Closing

### Symptoms

- Energy balance discrepancy > 5%
- Calculation warnings about energy conservation
- Efficiency calculations seem incorrect
- Input energy ≠ Output energy + Losses

### Root Causes

1. **Missing Energy Streams**: Not accounting for all inputs/outputs
2. **Measurement Errors**: Faulty energy meters
3. **Time Synchronization**: Misaligned timestamps across meters
4. **Unit Conversion Errors**: Incorrect energy unit conversions
5. **System Boundaries**: Unclear boundary definition
6. **Data Gaps**: Missing data points in time period

### Diagnostic Commands

```bash
# Check recent energy balance failures
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Energy balance" | \
  jq 'select(.energy_balance_percent > 5 or .energy_balance_percent < -5) | {facility: .facility_id, balance: .energy_balance_percent, input: .total_input_kwh, output: .total_output_kwh, losses: .total_losses_kwh}'

# Get detailed energy flow for specific calculation
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, facility_id,
          energy_input_fuel_kwh, energy_input_electricity_kwh,
          energy_output_steam_kwh, energy_output_electricity_kwh,
          energy_losses_kwh,
          (energy_input_fuel_kwh + energy_input_electricity_kwh) AS total_input,
          (energy_output_steam_kwh + energy_output_electricity_kwh + energy_losses_kwh) AS total_output,
          ((energy_input_fuel_kwh + energy_input_electricity_kwh) -
           (energy_output_steam_kwh + energy_output_electricity_kwh + energy_losses_kwh)) AS balance,
          (((energy_input_fuel_kwh + energy_input_electricity_kwh) -
            (energy_output_steam_kwh + energy_output_electricity_kwh + energy_losses_kwh)) /
           (energy_input_fuel_kwh + energy_input_electricity_kwh) * 100) AS balance_percent
   FROM calculations
   WHERE calculation_id = 'CALC-123';"

# Check for missing data points
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Data completeness" | \
  jq 'select(.data_completeness < 0.95) | {facility: .facility_id, completeness: .data_completeness, missing_points: .missing_data_points}'

# Check meter readings
curl -X GET "https://meter-api.example.com/facilities/FAC-001/readings" \
  -H "Authorization: Bearer $METER_TOKEN" \
  -d '{
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-01T23:59:59Z",
    "tags": ["fuel_flow", "steam_flow", "electricity_in", "electricity_out"]
  }' | jq .

# Check for time sync issues
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Timestamp" | \
  jq '{meter: .meter_id, timestamp: .reading_timestamp, system_time: .system_timestamp, drift: .time_drift_seconds}'
```

### Resolution Steps

**Step 1: Identify Missing Energy Streams**

```bash
# Get facility energy flow configuration
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT facility_id, energy_inputs, energy_outputs
   FROM facility_configurations
   WHERE facility_id = 'FAC-001';"

# Check if all configured streams have data
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Energy stream.*no data" | \
  jq '{facility: .facility_id, stream: .energy_stream, stream_type: .stream_type}'

# Common missing streams:
# - Blowdown losses
# - Radiation losses
# - Auxiliary power consumption
# - Waste heat recovery
```

**Step 2: Verify Meter Accuracy** (if measurement suspected)

```bash
# Compare meter readings with expected values
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/validate_meters.py \
    --facility FAC-001 \
    --start 2025-11-01T00:00:00Z \
    --end 2025-11-01T23:59:59Z

# Check meter calibration dates
curl -X GET "https://meter-api.example.com/meters/calibration" \
  -H "Authorization: Bearer $METER_TOKEN" | \
  jq '.meters[] | {id: .meter_id, last_calibration: .last_calibration_date, next_calibration: .next_calibration_date}'

# Flag meters needing calibration
# Schedule maintenance
```

**Step 3: Fix Time Synchronization** (if timestamps misaligned)

```bash
# Enable timestamp alignment
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_TIMESTAMP_ALIGNMENT=true \
  TIMESTAMP_ALIGNMENT_WINDOW_SECONDS=60

# Use interval averaging instead of point readings
kubectl set env deployment/thermaliq -n gl-009-production \
  USE_INTERVAL_AVERAGING=true \
  INTERVAL_DURATION_MINUTES=15

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 4: Verify Unit Conversions** (if conversion errors suspected)

```bash
# Check unit conversion configuration
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 50 "unit_conversions:"

# Verify conversion factors:
# - Natural gas: m³ to kWh (multiply by ~10.55)
# - Fuel oil: liters to kWh (multiply by ~9.96)
# - Steam: kg to kWh (multiply by enthalpy, ~0.7 kWh/kg for typical conditions)
# - Electricity: MWh to kWh (multiply by 1000)

# Test conversions
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
from unit_converter import convert_energy
print(f'1000 m³ natural gas = {convert_energy(1000, \"m3_natural_gas\", \"kwh\")} kWh')
print(f'500 liters fuel oil = {convert_energy(500, \"liters_fuel_oil\", \"kwh\")} kWh')
print(f'10000 kg steam = {convert_energy(10000, \"kg_steam\", \"kwh\")} kWh')
"

# Update incorrect conversion factors
kubectl edit configmap thermaliq-config -n gl-009-production
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 5: Clarify System Boundaries** (if boundary unclear)

```bash
# Review facility configuration
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT facility_id, system_boundary_description, included_equipment, excluded_equipment
   FROM facility_configurations
   WHERE facility_id = 'FAC-001';"

# Update system boundary if needed
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production <<EOF
UPDATE facility_configurations
SET system_boundary_description = 'Battery limits: Include all boilers, turbines, heat exchangers. Exclude office buildings, vehicle fleet.',
    included_equipment = ARRAY['boiler-1', 'boiler-2', 'turbine-1', 'heat-exchanger-1', 'heat-exchanger-2'],
    excluded_equipment = ARRAY['office-hvac', 'vehicle-charging']
WHERE facility_id = 'FAC-001';
EOF
```

**Step 6: Handle Data Gaps** (if missing data)

```bash
# Enable data interpolation for small gaps
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_INTERPOLATION=true \
  MAX_INTERPOLATION_GAP_MINUTES=15

# Use previous period data for large gaps
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_HISTORICAL_FALLBACK=true \
  FALLBACK_PERIOD_DAYS=7

# Flag calculations with low data quality
kubectl set env deployment/thermaliq -n gl-009-production \
  DATA_QUALITY_THRESHOLD=0.90 \
  FLAG_LOW_QUALITY_RESULTS=true

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 7: Adjust Energy Balance Tolerance** (if minor discrepancies)

```bash
# Increase acceptable tolerance for specific facilities
kubectl set env deployment/thermaliq -n gl-009-production \
  ENERGY_BALANCE_THRESHOLD_PERCENT=7.5

# Or set per-facility thresholds
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "UPDATE facility_configurations
   SET energy_balance_threshold_percent = 10.0
   WHERE facility_id IN ('FAC-001', 'FAC-002')
     AND facility_type = 'complex_industrial';"

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 8: Recalculate with Corrections**

```bash
# Recalculate affected calculations
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/recalculate.py \
    --facility FAC-001 \
    --start-date 2025-11-01 \
    --end-date 2025-11-26 \
    --reason "Fixed energy balance calculation"

# Verify energy balance closed
kubectl logs -n gl-009-production -l app=thermaliq --tail=100 | \
  grep "Energy balance" | \
  jq '{facility: .facility_id, balance: .energy_balance_percent, input: .total_input_kwh, output: .total_output_kwh}'
```

### Prevention Measures

```bash
# Set up energy balance alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-energy-balance-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-energy-balance
    interval: 60s
    rules:
    - alert: EnergyBalanceNotClosing
      expr: abs(thermaliq_energy_balance_percent) > 5
      for: 5m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Energy balance not closing"
        description: "Energy balance is {{ \$value }}% off for facility {{ \$labels.facility_id }}"

    - alert: LowDataCompleteness
      expr: thermaliq_calculation_data_completeness < 0.90
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Low data completeness"
        description: "Data completeness is {{ \$value }} for facility {{ \$labels.facility_id }}"
EOF

# Implement data quality monitoring
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_QUALITY_MONITORING=true \
  LOG_DATA_QUALITY_ISSUES=true

# Schedule regular meter calibration checks
# Add to monthly maintenance tasks
```

---

## 4. Unexpected Efficiency Values

### Symptoms

- Efficiency < 0% or > 100%
- Efficiency significantly different from historical values
- Efficiency not matching manual calculations
- Sudden jumps or drops in efficiency

### Root Causes

1. **Calculation Formula Error**: Incorrect efficiency formula
2. **Unit Conversion Error**: Wrong energy unit conversions
3. **Data Quality Issues**: Bad input data (outliers, errors)
4. **System Configuration Changes**: Equipment changes not reflected
5. **Measurement Errors**: Faulty meter readings
6. **Time Period Mismatch**: Input and output measured over different periods

### Diagnostic Commands

```bash
# Find anomalous efficiency values
kubectl logs -n gl-009-production -l app=thermaliq --tail=2000 | \
  grep "Efficiency calculated" | \
  jq 'select(.efficiency < 0 or .efficiency > 100) | {facility: .facility_id, calc_id: .calculation_id, efficiency: .efficiency, input: .energy_input_kwh, output: .energy_output_kwh}'

# Get efficiency trend for facility
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT DATE(created_at) AS date,
          AVG(efficiency) AS avg_efficiency,
          MIN(efficiency) AS min_efficiency,
          MAX(efficiency) AS max_efficiency,
          STDDEV(efficiency) AS stddev_efficiency
   FROM calculations
   WHERE facility_id = 'FAC-001'
     AND created_at >= NOW() - INTERVAL '30 days'
     AND status = 'completed'
   GROUP BY DATE(created_at)
   ORDER BY date DESC;"

# Get detailed calculation breakdown
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id,
          energy_input_fuel_kwh,
          energy_input_electricity_kwh,
          (energy_input_fuel_kwh + energy_input_electricity_kwh) AS total_input,
          energy_output_steam_kwh,
          energy_output_electricity_kwh,
          (energy_output_steam_kwh + energy_output_electricity_kwh) AS total_output,
          efficiency,
          ((energy_output_steam_kwh + energy_output_electricity_kwh) /
           NULLIF(energy_input_fuel_kwh + energy_input_electricity_kwh, 0) * 100) AS manual_efficiency
   FROM calculations
   WHERE calculation_id = 'CALC-123';"

# Check for outliers in input data
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Outlier detected" | \
  jq '{facility: .facility_id, parameter: .parameter_name, value: .value, expected_range: .expected_range}'

# Verify meter readings
curl -X GET "https://meter-api.example.com/facilities/FAC-001/readings/summary" \
  -H "Authorization: Bearer $METER_TOKEN" \
  -d '{
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-01T23:59:59Z"
  }' | jq .
```

### Resolution Steps

**Step 1: Verify Calculation Formula**

```bash
# Review efficiency calculation code
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  cat /app/calculation_engine.py | grep -A 30 "def calculate_efficiency"

# Correct formula should be:
# Efficiency (%) = (Useful Energy Output / Total Energy Input) × 100

# For thermal systems:
# Efficiency = (Steam Energy + Electricity Out) / (Fuel Energy + Electricity In) × 100

# Check if formula matches expected
# If incorrect, file bug and deploy hotfix
```

**Step 2: Manually Verify Calculation**

```bash
# Get raw data for problematic calculation
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM calculations WHERE calculation_id = 'CALC-123';" \
  --csv > calculation_data.csv

# Get meter readings
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT meter_id, reading_timestamp, value, unit
   FROM energy_readings
   WHERE facility_id = 'FAC-001'
     AND reading_timestamp >= '2025-11-01T00:00:00Z'
     AND reading_timestamp < '2025-11-02T00:00:00Z'
   ORDER BY reading_timestamp;" \
  --csv > meter_readings.csv

# Manual calculation in spreadsheet
# Compare with system result
# Identify where discrepancy occurs
```

**Step 3: Check Unit Conversions** (similar to #3)

```bash
# Verify all unit conversions
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/test_unit_conversions.py

# Test specific conversion
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
from unit_converter import UnitConverter
converter = UnitConverter()

# Example: Convert 1000 m³ natural gas to kWh
result = converter.convert(1000, 'm3_natural_gas', 'kwh')
print(f'1000 m³ natural gas = {result} kWh')
print(f'Expected: ~10550 kWh')
print(f'Match: {abs(result - 10550) < 100}')
"
```

**Step 4: Identify and Handle Outliers**

```bash
# Enable outlier detection
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_OUTLIER_DETECTION=true \
  OUTLIER_DETECTION_METHOD=zscore \
  OUTLIER_ZSCORE_THRESHOLD=3.0

# Enable outlier flagging (don't reject, just flag)
kubectl set env deployment/thermaliq -n gl-009-production \
  OUTLIER_HANDLING=flag \
  LOG_OUTLIERS=true

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Review outliers detected
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Outlier" | \
  jq '{facility: .facility_id, meter: .meter_id, value: .value, zscore: .zscore, expected_min: .expected_min, expected_max: .expected_max}'
```

**Step 5: Check for System Changes**

```bash
# Review facility configuration history
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT changed_at, changed_by, field_name, old_value, new_value
   FROM facility_configuration_audit
   WHERE facility_id = 'FAC-001'
     AND changed_at >= NOW() - INTERVAL '90 days'
   ORDER BY changed_at DESC;"

# Check for equipment additions/removals
# Update facility configuration if needed
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "UPDATE facility_configurations
   SET equipment_list = ARRAY['boiler-1', 'boiler-2', 'turbine-1', 'new-heat-exchanger'],
       last_updated = NOW(),
       updated_by = 'admin'
   WHERE facility_id = 'FAC-001';"
```

**Step 6: Validate Time Periods**

```bash
# Check if input and output measured over same period
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Time period" | \
  jq '{facility: .facility_id, input_start: .input_period_start, input_end: .input_period_end, output_start: .output_period_start, output_end: .output_period_end}'

# Enable strict time period matching
kubectl set env deployment/thermaliq -n gl-009-production \
  REQUIRE_MATCHING_TIME_PERIODS=true \
  TIME_PERIOD_TOLERANCE_MINUTES=5

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 7: Compare with Benchmark**

```bash
# Get industry benchmark for facility type
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
from benchmark_service import BenchmarkService
service = BenchmarkService()

benchmark = service.get_benchmark(
    facility_type='combined_heat_power',
    region='North America',
    industry='manufacturing'
)

print(f'Expected efficiency range: {benchmark.efficiency_min}% - {benchmark.efficiency_max}%')
print(f'Typical efficiency: {benchmark.efficiency_typical}%')
"

# If calculated efficiency way outside benchmark:
# - Double-check calculation
# - Verify meter accuracy
# - Check for system issues
```

**Step 8: Recalculate with Corrections**

```bash
# After fixing root cause, recalculate
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/recalculate.py \
    --calculation-id CALC-123 \
    --force \
    --reason "Fixed efficiency calculation error"

# Verify result
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, efficiency, created_at, recalculated_at
   FROM calculations
   WHERE calculation_id = 'CALC-123';"
```

### Prevention Measures

```bash
# Set up efficiency anomaly alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-efficiency-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-efficiency
    interval: 60s
    rules:
    - alert: InvalidEfficiencyValue
      expr: thermaliq_efficiency < 0 OR thermaliq_efficiency > 100
      for: 1m
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Invalid efficiency value"
        description: "Efficiency is {{ \$value }}% for facility {{ \$labels.facility_id }}"

    - alert: EfficiencyAnomalyDetected
      expr: abs(thermaliq_efficiency - thermaliq_efficiency_7d_avg) > 10
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Efficiency anomaly detected"
        description: "Efficiency is {{ \$value }}% (7-day avg: {{ \$labels.avg }}%) for facility {{ \$labels.facility_id }}"
EOF

# Implement validation checks
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_EFFICIENCY_VALIDATION=true \
  EFFICIENCY_MIN_THRESHOLD=0 \
  EFFICIENCY_MAX_THRESHOLD=100 \
  ENABLE_BENCHMARK_COMPARISON=true

# Schedule regular accuracy audits
# Add to monthly maintenance tasks
```

---

## 5. Connector Timeout Errors

### Symptoms

- "Connection timeout" errors in logs
- External API calls failing
- Historian queries timing out
- Energy meter reads timing out
- Increased error rate

### Root Causes

1. **Network Latency**: Slow network between services
2. **External Service Slow**: Historian/meter API responding slowly
3. **Timeout Too Short**: Timeout setting too aggressive
4. **Connection Pool Exhaustion**: No available connections
5. **DNS Issues**: Slow DNS resolution
6. **Firewall/Network Policy**: Blocking or throttling traffic

### Diagnostic Commands

```bash
# Check recent timeout errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep -i "timeout\|timed out" | \
  jq '{time: .timestamp, connector: .connector_name, operation: .operation, duration: .duration_ms, error: .error_message}'

# Test external service connectivity
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v --max-time 10 https://historian.example.com/health

kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  curl -v --max-time 10 https://meter-api.example.com/health

# Check DNS resolution time
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  time nslookup historian.example.com

# Test TCP connection time
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  time nc -zv historian.example.com 443

# Check network latency
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  ping -c 10 historian.example.com

# Check timeout configuration
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -i timeout

# Check connection pool usage
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Connection pool" | \
  jq '{connector: .connector_name, active: .active_connections, idle: .idle_connections, max: .max_connections}'
```

### Resolution Steps

**Step 1: Identify Slow Service**

```bash
# Measure response time for each external service
echo "Testing Historian..."
time curl -X POST https://historian.example.com/api/query \
  -H "Authorization: Bearer $HISTORIAN_TOKEN" \
  -d '{"facility_id": "FAC-001", "start": "2025-11-01T00:00:00Z", "end": "2025-11-01T01:00:00Z"}'

echo "Testing Energy Meter API..."
time curl -X GET https://meter-api.example.com/facilities/FAC-001/readings \
  -H "Authorization: Bearer $METER_TOKEN"

echo "Testing Benchmark Service..."
time curl -X GET https://benchmark-api.example.com/benchmarks/industrial \
  -H "Authorization: Bearer $BENCHMARK_TOKEN"

# Identify which service is slow (>5 seconds)
```

**Step 2: Increase Timeout** (if timeout too aggressive)

```bash
# Increase timeout for slow services
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_TIMEOUT=60 \
  HISTORIAN_CONNECTION_TIMEOUT=15 \
  ENERGY_METER_TIMEOUT=30 \
  ENERGY_METER_CONNECTION_TIMEOUT=10 \
  BENCHMARK_TIMEOUT=20

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Monitor timeout rate
watch -n 30 'kubectl logs -n gl-009-production -l app=thermaliq --tail=200 | grep -i timeout | wc -l'
```

**Step 3: Implement Retry Logic** (if intermittent timeouts)

```bash
# Enable retry with exponential backoff
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_RETRY_ENABLED=true \
  HISTORIAN_MAX_RETRIES=3 \
  HISTORIAN_RETRY_BACKOFF=exponential \
  HISTORIAN_RETRY_INITIAL_DELAY=1000 \
  ENERGY_METER_RETRY_ENABLED=true \
  ENERGY_METER_MAX_RETRIES=3

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 4: Increase Connection Pool** (if pool exhausted)

```bash
# Increase connection pool size
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CONNECTION_POOL_SIZE=20 \
  HISTORIAN_MAX_CONNECTIONS=50 \
  ENERGY_METER_CONNECTION_POOL_SIZE=15 \
  ENERGY_METER_MAX_CONNECTIONS=30

# Enable connection keep-alive
kubectl set env deployment/thermaliq -n gl-009-production \
  HTTP_KEEP_ALIVE=true \
  HTTP_KEEP_ALIVE_TIMEOUT=60

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 5: Fix DNS Issues** (if DNS slow)

```bash
# Check DNS configuration
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  cat /etc/resolv.conf

# Enable DNS caching
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DNS_CACHING=true \
  DNS_CACHE_TTL=300

# Or use static IP if available
kubectl edit configmap thermaliq-config -n gl-009-production
# Add:
#   historian:
#     url: http://10.0.1.50:8080  # Use IP instead of hostname

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 6: Check Network Policies** (if firewall issue)

```bash
# Check network policies
kubectl get networkpolicy -n gl-009-production

# Check if egress blocked
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  telnet historian.example.com 443

# If blocked, update network policy
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: thermaliq-egress
  namespace: gl-009-production
spec:
  podSelector:
    matchLabels:
      app: thermaliq
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector: {}
  - ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
EOF
```

**Step 7: Enable Circuit Breaker** (if repeated failures)

```bash
# Enable circuit breaker to fail fast
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_CIRCUIT_BREAKER_ENABLED=true \
  HISTORIAN_CIRCUIT_BREAKER_THRESHOLD=5 \
  HISTORIAN_CIRCUIT_BREAKER_TIMEOUT=60 \
  ENERGY_METER_CIRCUIT_BREAKER_ENABLED=true \
  ENERGY_METER_CIRCUIT_BREAKER_THRESHOLD=5

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Monitor circuit breaker state
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "Circuit breaker"
```

**Step 8: Implement Async Processing** (if timeouts persist)

```bash
# Enable async processing for slow operations
kubectl set env deployment/thermaliq -n gl-009-production \
  HISTORIAN_ASYNC_QUERIES=true \
  HISTORIAN_ASYNC_TIMEOUT=300 \
  ENABLE_CALCULATION_QUEUE=true

# Deploy worker pods for async processing
kubectl scale deployment/thermaliq-worker -n gl-009-production --replicas=4

# Monitor queue depth
watch -n 30 'curl -s http://prometheus:9090/api/v1/query --data-urlencode "query=thermaliq_calculation_queue_depth" | jq ".data.result[0].value[1]"'
```

**Step 9: Contact Vendor** (if external service issue)

```bash
# Check vendor status page
curl https://status.historian-vendor.com/api/v2/status.json | jq .

# Open support ticket with vendor
# Include:
# - Error logs
# - Request/response samples
# - Timeline of issues
# - Performance metrics
```

### Prevention Measures

```bash
# Set up timeout alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-timeout-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-timeouts
    interval: 60s
    rules:
    - alert: HighTimeoutRate
      expr: rate(thermaliq_connector_timeouts_total[5m]) > 0.1
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High connector timeout rate"
        description: "Timeout rate is {{ \$value }}/sec for {{ \$labels.connector_name }}"

    - alert: CircuitBreakerOpen
      expr: thermaliq_circuit_breaker_state == 1
      for: 5m
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Circuit breaker open"
        description: "Circuit breaker is open for {{ \$labels.connector_name }}"
EOF

# Implement health checks for external services
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_EXTERNAL_SERVICE_HEALTH_CHECKS=true \
  HEALTH_CHECK_INTERVAL=60

# Schedule regular connectivity tests
# Add to daily maintenance tasks
```

---

## 6. Data Quality Issues

### Symptoms

- Data quality score < 90%
- Missing data points
- Outliers/anomalies in data
- Inconsistent readings
- Calculation warnings about data quality

### Root Causes

1. **Meter Malfunction**: Faulty sensors/meters
2. **Communication Issues**: Data transmission failures
3. **Configuration Errors**: Incorrect meter configuration
4. **Calibration Drift**: Meters needing calibration
5. **Data Processing Errors**: ETL pipeline issues
6. **Time Synchronization**: Clock drift on meters

### Diagnostic Commands

```bash
# Check data quality scores
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Data quality" | \
  jq 'select(.data_quality_score < 90) | {facility: .facility_id, score: .data_quality_score, issues: .data_quality_issues}'

# Check for missing data
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Missing data" | \
  jq '{facility: .facility_id, meter: .meter_id, missing_periods: .missing_periods, expected_points: .expected_points, actual_points: .actual_points}'

# Check for outliers
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Outlier detected" | \
  jq '{facility: .facility_id, meter: .meter_id, timestamp: .timestamp, value: .value, expected_range: .expected_range, zscore: .zscore}'

# Query data completeness from database
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT facility_id,
          meter_id,
          DATE(reading_timestamp) AS date,
          COUNT(*) AS reading_count,
          (COUNT(*) * 100.0 / 96) AS completeness_percent
   FROM energy_readings
   WHERE reading_timestamp >= NOW() - INTERVAL '7 days'
   GROUP BY facility_id, meter_id, DATE(reading_timestamp)
   HAVING (COUNT(*) * 100.0 / 96) < 95
   ORDER BY completeness_percent;"

# Check meter status
curl -X GET "https://meter-api.example.com/meters/status" \
  -H "Authorization: Bearer $METER_TOKEN" | \
  jq '.meters[] | select(.status != "healthy") | {id: .meter_id, facility: .facility_id, status: .status, last_reading: .last_reading_time}'
```

### Resolution Steps

**Step 1: Identify Problematic Meters**

```bash
# List meters with low data quality
kubectl logs -n gl-009-production -l app=thermaliq --tail=2000 | \
  grep "Data quality" | \
  jq -r '{facility: .facility_id, meter: .meter_id, score: .data_quality_score}' | \
  jq -s 'group_by(.meter) | map({meter: .[0].meter, avg_score: (map(.score) | add / length), count: length}) | sort_by(.avg_score) | .[]'

# Get detailed meter info
curl -X GET "https://meter-api.example.com/meters/METER-001" \
  -H "Authorization: Bearer $METER_TOKEN" | \
  jq .
```

**Step 2: Enable Data Validation**

```bash
# Enable comprehensive data validation
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_VALIDATION=true \
  VALIDATION_MODE=strict \
  ENABLE_RANGE_CHECKS=true \
  ENABLE_CONSISTENCY_CHECKS=true \
  ENABLE_OUTLIER_DETECTION=true

# Configure validation rules
kubectl edit configmap thermaliq-config -n gl-009-production
# Add validation rules for each meter type

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 3: Implement Data Cleansing**

```bash
# Enable data cleansing
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_CLEANSING=true \
  REMOVE_OUTLIERS=true \
  OUTLIER_THRESHOLD=3.0 \
  INTERPOLATE_MISSING_DATA=true \
  MAX_INTERPOLATION_GAP=15

# Enable smoothing for noisy data
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_SMOOTHING=true \
  SMOOTHING_WINDOW=3

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 4: Fix Missing Data**

```bash
# Backfill missing data from backup source
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/backfill_data.py \
    --facility FAC-001 \
    --meter METER-001 \
    --start-date 2025-11-01 \
    --end-date 2025-11-26 \
    --source historian_backup

# Or use interpolation for small gaps
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/interpolate_missing_data.py \
    --facility FAC-001 \
    --meter METER-001 \
    --method linear \
    --max-gap-minutes 30
```

**Step 5: Recalibrate Meters** (if calibration drift)

```bash
# Check meter calibration status
curl -X GET "https://meter-api.example.com/meters/calibration-status" \
  -H "Authorization: Bearer $METER_TOKEN" | \
  jq '.meters[] | select(.days_since_calibration > 365) | {id: .meter_id, facility: .facility_id, last_calibration: .last_calibration_date}'

# Schedule calibration
curl -X POST "https://meter-api.example.com/meters/METER-001/schedule-calibration" \
  -H "Authorization: Bearer $METER_TOKEN" \
  -d '{"scheduled_date": "2025-12-01", "technician": "John Doe"}'

# Apply calibration correction factor temporarily
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "UPDATE meter_configurations
   SET calibration_factor = 1.05,
       calibration_notes = 'Temporary correction pending physical calibration'
   WHERE meter_id = 'METER-001';"
```

**Step 6: Fix Communication Issues**

```bash
# Check meter connectivity
for meter_id in METER-001 METER-002 METER-003; do
  echo "Testing $meter_id..."
  curl -X GET "https://meter-api.example.com/meters/$meter_id/test-connection" \
    -H "Authorization: Bearer $METER_TOKEN"
done

# Restart failed meters
curl -X POST "https://meter-api.example.com/meters/METER-001/restart" \
  -H "Authorization: Bearer $METER_TOKEN"

# Check network path
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  traceroute meter-gateway.example.com
```

**Step 7: Implement Data Quality Monitoring**

```bash
# Enable real-time data quality monitoring
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_DATA_QUALITY_MONITORING=true \
  DATA_QUALITY_CHECK_INTERVAL=300 \
  LOG_DATA_QUALITY_ISSUES=true \
  ALERT_ON_LOW_QUALITY=true

# Create data quality dashboard
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-data-quality
  namespace: gl-009-production
data:
  data-quality-dashboard.json: |
    {
      "dashboard": {
        "title": "THERMALIQ Data Quality",
        "panels": [
          {
            "title": "Data Quality Score by Facility",
            "targets": [{
              "expr": "thermaliq_data_quality_score"
            }]
          },
          {
            "title": "Missing Data Points",
            "targets": [{
              "expr": "rate(thermaliq_missing_data_points_total[5m])"
            }]
          },
          {
            "title": "Outliers Detected",
            "targets": [{
              "expr": "rate(thermaliq_outliers_detected_total[5m])"
            }]
          }
        ]
      }
    }
EOF
```

**Step 8: Notify Stakeholders**

```bash
# Generate data quality report
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/data_quality_report.py \
    --start-date 2025-11-01 \
    --end-date 2025-11-26 \
    --output /tmp/data_quality_report.pdf

# Email report to facility managers
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python /app/scripts/send_report.py \
    --report /tmp/data_quality_report.pdf \
    --recipients facility-managers@example.com \
    --subject "Data Quality Issues Requiring Attention"
```

### Prevention Measures

```bash
# Set up data quality alerts
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-data-quality-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-data-quality
    interval: 60s
    rules:
    - alert: LowDataQualityScore
      expr: thermaliq_data_quality_score < 90
      for: 30m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Low data quality score"
        description: "Data quality score is {{ \$value }} for facility {{ \$labels.facility_id }}"

    - alert: HighMissingDataRate
      expr: rate(thermaliq_missing_data_points_total[10m]) > 10
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High missing data rate"
        description: "Missing {{ \$value }} data points/sec for meter {{ \$labels.meter_id }}"

    - alert: MeterOffline
      expr: time() - thermaliq_meter_last_reading_timestamp > 3600
      for: 5m
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Meter offline"
        description: "No readings from meter {{ \$labels.meter_id }} for >1 hour"
EOF

# Schedule regular data quality audits
# Add to weekly maintenance tasks

# Implement automated meter health checks
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_METER_HEALTH_CHECKS=true \
  METER_HEALTH_CHECK_INTERVAL=300
```

---

## 7. Memory Leaks

### Symptoms

- Memory usage growing over time
- Pods being OOMKilled
- Performance degradation
- Frequent pod restarts
- Eventual service failure

### Root Causes

1. **Object Not Released**: Objects kept in memory indefinitely
2. **Circular References**: Objects referencing each other
3. **Global State Accumulation**: Data accumulating in global variables
4. **Event Listener Leaks**: Event listeners not removed
5. **Cache Growing Unbounded**: Cache without eviction
6. **External Library Leak**: Memory leak in third-party library

### Diagnostic Commands

```bash
# Check memory growth over time
kubectl top pods -n gl-009-production -l app=thermaliq --watch

# Get memory metrics from Prometheus
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=container_memory_usage_bytes{pod=~"thermaliq.*"}' | \
  jq '.data.result[] | {pod: .metric.pod, memory_mb: (.value[1] | tonumber / 1024 / 1024)}'

# Check for OOMKilled events
kubectl get events -n gl-009-production | grep OOMKilled

# Check pod age vs restarts (frequent restarts indicate leak)
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[] | {pod: .metadata.name, age: .status.startTime, restarts: .status.containerStatuses[0].restartCount}'

# Get memory profile
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
import gc
import sys

# Force garbage collection
gc.collect()

# Get object counts
print('Object counts:')
for obj_type, count in sorted([(type(obj).__name__, 1) for obj in gc.get_objects()], key=lambda x: x[1], reverse=True)[:20]:
    print(f'{obj_type}: {count}')

# Get memory usage
print(f'\nTotal memory: {sys.getsizeof(gc.get_objects())} bytes')
"

# Check for memory growth pattern
for i in {1..20}; do
  kubectl top pod -n gl-009-production -l app=thermaliq | tail -n +2 | awk '{print $3}'
  sleep 60
done
```

### Resolution Steps

**Step 1: Confirm Memory Leak**

```bash
# Monitor memory over 2 hours
kubectl top pods -n gl-009-production -l app=thermaliq --watch > memory_log.txt &
sleep 7200
kill %1

# Plot memory usage
cat memory_log.txt | awk '{print $3}' | grep -v NAME | sed 's/Mi//' > memory_values.txt
python -c "
import matplotlib.pyplot as plt
data = [int(x) for x in open('memory_values.txt').readlines()]
plt.plot(data)
plt.xlabel('Time (minutes)')
plt.ylabel('Memory (Mi)')
plt.title('Memory Usage Over Time')
plt.savefig('memory_trend.png')
"

# If linear growth -> likely leak
# If sawtooth -> normal GC
# If exponential -> severe leak
```

**Step 2: Identify Leaking Code**

```bash
# Enable memory profiling
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_MEMORY_PROFILING=true \
  MEMORY_PROFILING_INTERVAL=300

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Collect profiling data
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "Memory profile" > memory_profiles.log

# Analyze profiling data
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
from memory_profiler import profile
import objgraph

# Show growth
objgraph.show_growth(limit=20)

# Find most common types
objgraph.show_most_common_types(limit=20)

# Find references to leaking objects
# objgraph.show_backref(leaking_object, filename='leak_refs.png')
"
```

**Step 3: Fix Common Leak Patterns**

**Pattern A: Unbounded Cache**
```bash
# Check cache size limits
kubectl get configmap thermaliq-config -n gl-009-production -o yaml | \
  grep -A 10 "cache:"

# Add cache size limits
kubectl edit configmap thermaliq-config -n gl-009-production
# Add:
#   cache:
#     max_size_mb: 1024
#     eviction_policy: lru
#     max_entries: 100000

# Or use external cache (Redis) instead of in-memory
kubectl set env deployment/thermaliq -n gl-009-production \
  USE_EXTERNAL_CACHE=true \
  DISABLE_INMEMORY_CACHE=true
```

**Pattern B: Global State Accumulation**
```bash
# Review code for global variables
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  grep -r "global " /app/*.py

# Example fix (code change required):
# Before:
#   calculation_results = []  # Global list grows forever
#   def calculate():
#       result = do_calculation()
#       calculation_results.append(result)
#
# After:
#   def calculate():
#       result = do_calculation()
#       save_to_database(result)  # Don't keep in memory

# Deploy fixed code
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.1-memleak-fix
```

**Pattern C: Circular References**
```bash
# Enable garbage collector debug mode
kubectl set env deployment/thermaliq -n gl-009-production \
  PYTHONMALLOC=debug \
  PYTHONGC_DEBUG=leak

# Force aggressive GC
kubectl set env deployment/thermaliq -n gl-009-production \
  GC_THRESHOLD_0=400 \
  GC_THRESHOLD_1=5 \
  GC_THRESHOLD_2=5

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Monitor GC activity
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "gc:"
```

**Step 4: Implement Workarounds**

```bash
# Periodic pod restart as temporary mitigation
kubectl set env deployment/thermaliq -n gl-009-production \
  RESTART_AFTER_REQUESTS=10000

# Or use rolling restart cron job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: thermaliq-memory-refresh
  namespace: gl-009-production
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: pod-restarter
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - kubectl rollout restart deployment/thermaliq -n gl-009-production
          restartPolicy: Never
EOF

# Reduce memory limits to force earlier OOM
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=4Gi --requests=memory=2Gi
```

**Step 5: Update Dependencies** (if library leak)

```bash
# Check for known memory leaks in dependencies
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip list --outdated

# Update specific package with known leak
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip install --upgrade pandas==2.1.0

# Or rebuild image with updated dependencies
docker build -t ghcr.io/greenlang/thermaliq:v1.3.1-deps .
docker push ghcr.io/greenlang/thermaliq:v1.3.1-deps
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.1-deps
```

**Step 6: Verify Fix**

```bash
# Monitor memory for 24 hours
kubectl top pods -n gl-009-production -l app=thermaliq --watch > memory_after_fix.txt &

# After 24 hours, compare
# Memory should be stable (sawtooth pattern from GC)
# Not linearly growing

# Check restart count hasn't increased
kubectl get pods -n gl-009-production -l app=thermaliq -o json | \
  jq '.items[] | {pod: .metadata.name, restarts: .status.containerStatuses[0].restartCount}'
```

### Prevention Measures

```bash
# Set up memory monitoring
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-memory-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-memory
    interval: 60s
    rules:
    - alert: HighMemoryUsage
      expr: container_memory_usage_bytes{pod=~"thermaliq.*"} / container_spec_memory_limit_bytes{pod=~"thermaliq.*"} > 0.85
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High memory usage"
        description: "Memory usage is {{ \$value }}% for pod {{ \$labels.pod }}"

    - alert: MemoryLeak
      expr: rate(container_memory_usage_bytes{pod=~"thermaliq.*"}[30m]) > 1048576  # 1MB/sec
      for: 1h
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Possible memory leak"
        description: "Memory growing at {{ \$value }} bytes/sec for pod {{ \$labels.pod }}"

    - alert: FrequentOOMKills
      expr: increase(kube_pod_container_status_restarts_total{pod=~"thermaliq.*",reason="OOMKilled"}[1h]) > 2
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Frequent OOM kills"
        description: "Pod {{ \$labels.pod }} OOMKilled {{ \$value }} times in last hour"
EOF

# Implement memory profiling in CI/CD
# Add memory leak tests to test suite

# Enable memory limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=8Gi --requests=memory=4Gi

# Schedule regular memory audits
# Add to monthly maintenance tasks
```

---

## 8. CPU Spikes

### Symptoms

- CPU usage spikes to 100%
- Slow request processing
- Increased latency
- Pod throttling
- Timeouts

### Root Causes

1. **Inefficient Algorithm**: CPU-intensive calculations
2. **Infinite Loop**: Bug causing tight loop
3. **High Request Rate**: Traffic spike
4. **Blocking Operations**: Synchronous I/O blocking threads
5. **Garbage Collection**: Excessive GC activity
6. **Regex Complexity**: Complex regular expressions

### Diagnostic Commands

```bash
# Check current CPU usage
kubectl top pods -n gl-009-production -l app=thermaliq

# Get CPU metrics from Prometheus
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(container_cpu_usage_seconds_total{pod=~"thermaliq.*"}[5m])' | \
  jq '.data.result[] | {pod: .metric.pod, cpu_cores: .value[1]}'

# Check for CPU throttling
kubectl describe pod -n gl-009-production -l app=thermaliq | \
  grep -A 5 "cpu"

# Check request rate
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(thermaliq_requests_total[5m])' | \
  jq '.data.result[0].value[1]'

# Profile CPU usage
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -m cProfile -o /tmp/profile.stats /app/calculation_service.py &
sleep 60
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
import pstats
p = pstats.Stats('/tmp/profile.stats')
p.sort_stats('cumulative')
p.print_stats(20)
"

# Check for infinite loops
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  ps aux | grep python

kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  top -b -n 1
```

### Resolution Steps

**Step 1: Identify CPU-Intensive Operations**

```bash
# Enable CPU profiling
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_CPU_PROFILING=true \
  CPU_PROFILING_SAMPLE_RATE=0.01

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production

# Collect profiling data
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "CPU profile" > cpu_profiles.log

# Analyze hotspots
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -m cProfile -s cumtime /app/calculation_service.py
```

**Step 2: Optimize Hot Paths**

```bash
# Example: Optimize calculation loop
# Before (slow):
#   for i in range(len(data)):
#       result = complex_calculation(data[i])
#       results.append(result)
#
# After (fast):
#   results = [complex_calculation(d) for d in data]
#   # Or vectorized:
#   results = numpy.vectorize(complex_calculation)(data)

# Deploy optimized code
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.2-cpu-opt
```

**Step 3: Scale Horizontally** (if high load)

```bash
# Scale up replicas
kubectl scale deployment/thermaliq -n gl-009-production --replicas=12

# Enable HPA
kubectl autoscale deployment thermaliq -n gl-009-production \
  --cpu-percent=70 \
  --min=4 \
  --max=16

# Verify scaling
kubectl get hpa -n gl-009-production
```

**Step 4: Increase CPU Limits** (if throttling)

```bash
# Check current limits
kubectl get deployment thermaliq -n gl-009-production -o yaml | \
  grep -A 5 "resources:"

# Increase CPU limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=cpu=4000m --requests=cpu=2000m

# Verify no throttling
kubectl describe pod -n gl-009-production -l app=thermaliq | \
  grep -i throttl
```

**Step 5: Implement Async Processing** (if blocking operations)

```bash
# Enable async I/O
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_ASYNC_IO=true \
  ASYNC_WORKER_THREADS=8

# Use async HTTP client
kubectl set env deployment/thermaliq -n gl-009-production \
  USE_ASYNC_HTTP_CLIENT=true

# Deploy updated code
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.2-async
```

**Step 6: Optimize Regex** (if regex-heavy)

```bash
# Pre-compile regex patterns
# Code change required:
# Before:
#   if re.match(r'complex.*pattern.*here', text):
#       ...
#
# After:
#   PATTERN = re.compile(r'complex.*pattern.*here')  # Global, compiled once
#   if PATTERN.match(text):
#       ...

# Use simpler patterns where possible
# Deploy optimized code
```

**Step 7: Rate Limiting** (if high request rate)

```bash
# Implement rate limiting
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_RATE_LIMITING=true \
  RATE_LIMIT_REQUESTS_PER_MINUTE=100 \
  RATE_LIMIT_BURST=20

# Deploy with rate limiting
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: thermaliq-ratelimit
  namespace: gl-009-production
spec:
  workloadSelector:
    labels:
      app: thermaliq
  configPatches:
  - applyTo: HTTP_FILTER
    match:
      context: SIDECAR_INBOUND
      listener:
        filterChain:
          filter:
            name: "envoy.filters.network.http_connection_manager"
    patch:
      operation: INSERT_BEFORE
      value:
        name: envoy.filters.http.local_ratelimit
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
          stat_prefix: http_local_rate_limiter
          token_bucket:
            max_tokens: 100
            tokens_per_fill: 100
            fill_interval: 60s
EOF
```

**Step 8: Verify Improvement**

```bash
# Monitor CPU usage for 1 hour
watch -n 60 'kubectl top pods -n gl-009-production -l app=thermaliq'

# Check CPU utilization
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=rate(container_cpu_usage_seconds_total{pod=~"thermaliq.*"}[5m])' | \
  jq '.data.result[] | {pod: .metric.pod, cpu_cores: .value[1]}'

# Should be < 2.0 cores per pod (if limit is 4.0)
```

### Prevention Measures

```bash
# Set up CPU monitoring
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-cpu-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-cpu
    interval: 60s
    rules:
    - alert: HighCPUUsage
      expr: rate(container_cpu_usage_seconds_total{pod=~"thermaliq.*"}[5m]) / container_spec_cpu_limit{pod=~"thermaliq.*"} > 0.85
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High CPU usage"
        description: "CPU usage is {{ \$value }}% for pod {{ \$labels.pod }}"

    - alert: CPUThrottling
      expr: rate(container_cpu_cfs_throttled_seconds_total{pod=~"thermaliq.*"}[5m]) > 0.1
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "CPU throttling detected"
        description: "Pod {{ \$labels.pod }} is being throttled"
EOF

# Implement performance testing
# Add CPU benchmarks to CI/CD

# Enable request profiling
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_REQUEST_PROFILING=true \
  PROFILE_SLOW_REQUESTS=true \
  SLOW_REQUEST_THRESHOLD_MS=5000
```

---

## 9. Database Connection Pool Exhaustion

### Symptoms

- "No available connections" errors
- Requests timing out
- Increased latency
- Connection refused errors
- Database errors in logs

### Root Causes

1. **Pool Too Small**: Connection pool size insufficient
2. **Connection Leaks**: Connections not returned to pool
3. **Long-Running Queries**: Queries holding connections too long
4. **High Concurrency**: More requests than connections available
5. **Connection Not Closed**: Connections left open
6. **Deadlocks**: Transactions blocking each other

### Diagnostic Commands

```bash
# Check active connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT count(*) AS active_connections,
          max_connections,
          (count(*) * 100.0 / max_connections::float) AS usage_percent
   FROM pg_stat_activity, (SELECT setting::int AS max_connections FROM pg_settings WHERE name='max_connections') AS mc
   GROUP BY max_connections;"

# List active connections by state
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT state, count(*)
   FROM pg_stat_activity
   WHERE datname = 'thermaliq_production'
   GROUP BY state;"

# Check application pool status
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep "Connection pool" | \
  jq '{timestamp: .timestamp, active: .pool_active, idle: .pool_idle, max: .pool_max, waiting: .pool_waiting}'

# Check for connection leaks
kubectl logs -n gl-009-production -l app=thermaliq --tail=2000 | \
  grep "Connection" | \
  grep -c "acquired"
kubectl logs -n gl-009-production -l app=thermaliq --tail=2000 | \
  grep "Connection" | \
  grep -c "returned"
# If acquired >> returned, leak exists

# Check long-running queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pid,
          now() - query_start AS duration,
          state,
          query
   FROM pg_stat_activity
   WHERE state != 'idle'
     AND query NOT LIKE '%pg_stat_activity%'
   ORDER BY duration DESC
   LIMIT 10;"

# Check for deadlocks
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT * FROM pg_stat_activity WHERE wait_event_type = 'Lock';"
```

### Resolution Steps

**Step 1: Increase Connection Pool Size**

```bash
# Increase application pool size
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_SIZE=100 \
  DATABASE_MAX_OVERFLOW=50 \
  DATABASE_POOL_RECYCLE=3600

# Increase database max_connections
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U postgres -c "ALTER SYSTEM SET max_connections = 500;"

# Restart database (may cause brief downtime)
kubectl rollout restart statefulset postgres -n gl-009-production
kubectl wait --for=condition=ready pod/postgres-0 -n gl-009-production --timeout=300s

# Restart application
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 2: Fix Connection Leaks**

```bash
# Enable connection leak detection
kubectl set env deployment/thermaliq -n gl-009-production \
  SQLALCHEMY_POOL_PRE_PING=true \
  SQLALCHEMY_POOL_RECYCLE=300 \
  SQLALCHEMY_ECHO_POOL=warn

# Enable connection timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_TIMEOUT=10

# Code review for connection leaks:
# Ensure all database operations use context managers:
# Good:
#   with engine.connect() as conn:
#       result = conn.execute(query)
# Bad:
#   conn = engine.connect()
#   result = conn.execute(query)
#   # conn never closed!

# Deploy fixed code
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.3-connleak-fix
```

**Step 3: Optimize Long-Running Queries**

```bash
# Identify slow queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT query,
          calls,
          mean_exec_time,
          max_exec_time,
          stddev_exec_time
   FROM pg_stat_statements
   WHERE mean_exec_time > 1000  -- > 1 second
   ORDER BY mean_exec_time DESC
   LIMIT 10;"

# Add indexes for slow queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "CREATE INDEX CONCURRENTLY idx_energy_readings_facility_time
   ON energy_readings(facility_id, timestamp DESC);"

# Set query timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_QUERY_TIMEOUT=30000

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 4: Implement Connection Pooling Best Practices**

```bash
# Configure optimal pool settings
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_POOL_SIZE=20 \
  DATABASE_MAX_OVERFLOW=10 \
  DATABASE_POOL_RECYCLE=3600 \
  DATABASE_POOL_PRE_PING=true \
  DATABASE_POOL_TIMEOUT=30

# Enable connection pooling at database level (PgBouncer)
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: gl-009-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgbouncer
  template:
    metadata:
      labels:
        app: pgbouncer
    spec:
      containers:
      - name: pgbouncer
        image: edoburu/pgbouncer:latest
        env:
        - name: DATABASE_URL
          value: "postgres://thermaliq:password@postgres-service:5432/thermaliq_production"
        - name: POOL_MODE
          value: "transaction"
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "25"
        ports:
        - containerPort: 5432
---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer-service
  namespace: gl-009-production
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
EOF

# Update application to use PgBouncer
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_HOST=pgbouncer-service
```

**Step 5: Kill Idle Connections**

```bash
# Kill idle connections (>10 minutes)
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'thermaliq_production'
     AND state = 'idle'
     AND state_change < NOW() - INTERVAL '10 minutes';"

# Set idle connection timeout
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U postgres -c "ALTER SYSTEM SET idle_in_transaction_session_timeout = '10min';"

# Reload config
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U postgres -c "SELECT pg_reload_conf();"
```

**Step 6: Resolve Deadlocks**

```bash
# Find deadlocked queries
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT blocked_locks.pid AS blocked_pid,
          blocked_activity.usename AS blocked_user,
          blocking_locks.pid AS blocking_pid,
          blocking_activity.usename AS blocking_user,
          blocked_activity.query AS blocked_statement,
          blocking_activity.query AS blocking_statement
   FROM pg_catalog.pg_locks blocked_locks
   JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
   JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
   JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
   WHERE NOT blocked_locks.granted;"

# Kill blocking process (if necessary)
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT pg_terminate_backend(12345);"  # Replace with actual PID

# Set lock timeout
kubectl set env deployment/thermaliq -n gl-009-production \
  DATABASE_LOCK_TIMEOUT=30000
```

**Step 7: Verify Resolution**

```bash
# Monitor connection pool usage
watch -n 30 'kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -t -c \
  "SELECT count(*) FROM pg_stat_activity WHERE datname=\"thermaliq_production\";"'

# Should be stable and < 80% of max_connections

# Check application pool
kubectl logs -n gl-009-production -l app=thermaliq -f | \
  grep "Connection pool"

# Run load test
./scripts/load_test.sh --duration 600 --rps 20

# Verify no connection errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=500 | \
  grep -i "connection" | \
  grep -i "error\|failed"
```

### Prevention Measures

```bash
# Set up connection pool monitoring
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-database-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-database
    interval: 60s
    rules:
    - alert: HighDatabaseConnections
      expr: pg_stat_database_numbackends{datname="thermaliq_production"} / pg_settings_max_connections > 0.80
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High database connection usage"
        description: "Using {{ \$value }}% of max connections"

    - alert: ConnectionPoolExhausted
      expr: thermaliq_database_pool_waiting > 0
      for: 5m
      labels:
        severity: critical
        service: thermaliq
      annotations:
        summary: "Connection pool exhausted"
        description: "{{ \$value }} requests waiting for connections"

    - alert: LongRunningQueries
      expr: pg_stat_activity_max_tx_duration{datname="thermaliq_production"} > 300
      for: 5m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Long-running query detected"
        description: "Query running for {{ \$value }} seconds"
EOF

# Implement connection leak detection
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_CONNECTION_LEAK_DETECTION=true \
  LOG_CONNECTION_LIFECYCLE=true

# Schedule regular connection audits
# Add to daily maintenance tasks
```

---

## 10. Sankey Diagram Rendering Issues

### Symptoms

- Blank or incomplete Sankey diagrams
- Diagram generation errors
- Rendering timeouts
- Incorrect energy flows
- Missing nodes or links

### Root Causes

1. **Data Format Issues**: Invalid input data format
2. **Too Many Nodes**: Diagram too complex to render
3. **Circular Flows**: Circular dependencies in energy flows
4. **Negative Values**: Negative energy flow values
5. **Library Bug**: Bug in Plotly/visualization library
6. **Memory Exhaustion**: Not enough memory for large diagrams

### Diagnostic Commands

```bash
# Check Sankey generation errors
kubectl logs -n gl-009-production -l app=thermaliq --tail=1000 | \
  grep "Sankey" | \
  grep -i "error\|failed" | \
  jq '{timestamp: .timestamp, calc_id: .calculation_id, error: .error_message, node_count: .node_count, link_count: .link_count}'

# Check generation latency
curl -s http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, thermaliq_sankey_generation_duration_seconds_bucket[10m])' | \
  jq '.data.result[0].value[1]'

# Get failing calculation details
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, facility_id, energy_flows, sankey_diagram_url
   FROM calculations
   WHERE calculation_id = 'CALC-123';" \
  --csv > calc_details.csv

# Test Sankey generation locally
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
from sankey_generator import SankeyGenerator
import json

# Load energy flows
flows = json.loads('''
{
  \"nodes\": [
    {\"id\": \"fuel\", \"label\": \"Fuel Input\"},
    {\"id\": \"boiler\", \"label\": \"Boiler\"},
    {\"id\": \"steam\", \"label\": \"Steam Output\"}
  ],
  \"links\": [
    {\"source\": \"fuel\", \"target\": \"boiler\", \"value\": 1000},
    {\"source\": \"boiler\", \"target\": \"steam\", \"value\": 850}
  ]
}
''')

generator = SankeyGenerator()
diagram = generator.generate(flows)
print(diagram)
"

# Check library version
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip show plotly
```

### Resolution Steps

**Step 1: Validate Input Data**

```bash
# Check energy flow data structure
kubectl exec -it postgres-0 -n gl-009-production -- \
  psql -U thermaliq -d thermaliq_production -c \
  "SELECT calculation_id, jsonb_pretty(energy_flows)
   FROM calculations
   WHERE calculation_id = 'CALC-123';"

# Validate against schema
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  python -c "
import json
from jsonschema import validate, ValidationError

schema = {
    'type': 'object',
    'properties': {
        'nodes': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string'},
                    'label': {'type': 'string'}
                },
                'required': ['id', 'label']
            }
        },
        'links': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'source': {'type': 'string'},
                    'target': {'type': 'string'},
                    'value': {'type': 'number', 'minimum': 0}
                },
                'required': ['source', 'target', 'value']
            }
        }
    },
    'required': ['nodes', 'links']
}

flows = json.loads(open('energy_flows.json').read())

try:
    validate(flows, schema)
    print('Valid')
except ValidationError as e:
    print(f'Invalid: {e.message}')
"
```

**Step 2: Simplify Complex Diagrams**

```bash
# Enable node aggregation
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_MAX_NODES=50 \
  SANKEY_MAX_LINKS=100 \
  SANKEY_AGGREGATE_SMALL_FLOWS=true \
  SANKEY_SMALL_FLOW_THRESHOLD=0.01

# Group minor flows into "Other"
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_GROUP_MINOR_FLOWS=true \
  SANKEY_MINOR_FLOW_PERCENT=2.0

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 3: Handle Circular Flows**

```bash
# Enable flow validation
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_VALIDATE_FLOWS=true \
  SANKEY_ALLOW_CIRCULAR_FLOWS=false \
  SANKEY_BREAK_CIRCULAR_FLOWS=true

# Code change to detect and break circular flows:
# def break_circular_flows(flows):
#     # Topological sort to detect cycles
#     # Remove weakest link in cycle
#     pass

# Deploy updated code
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.4-sankey-fix
```

**Step 4: Fix Negative Values**

```bash
# Enable value validation
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_REJECT_NEGATIVE_VALUES=true \
  SANKEY_ABSOLUTE_VALUES=true

# Or use absolute values
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_USE_ABSOLUTE_VALUES=true

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 5: Increase Generation Timeout**

```bash
# Increase timeout for large diagrams
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_GENERATION_TIMEOUT=60

# Enable async generation
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_ASYNC_GENERATION=true \
  SANKEY_ASYNC_QUEUE=redis

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 6: Update Visualization Library**

```bash
# Check for library updates
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip list --outdated | grep plotly

# Update Plotly
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip install --upgrade plotly

# Or pin to specific version
kubectl exec -it deployment/thermaliq -n gl-009-production -- \
  pip install plotly==5.18.0

# Rebuild image
docker build -t ghcr.io/greenlang/thermaliq:v1.3.4-plotly .
docker push ghcr.io/greenlang/thermaliq:v1.3.4-plotly

# Deploy
kubectl set image deployment/thermaliq -n gl-009-production \
  thermaliq=ghcr.io/greenlang/thermaliq:v1.3.4-plotly
```

**Step 7: Increase Memory** (if memory issue)

```bash
# Increase memory limits
kubectl set resources deployment/thermaliq -n gl-009-production \
  --limits=memory=8Gi --requests=memory=4Gi

# Enable diagram caching
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_CACHE_ENABLED=true \
  SANKEY_CACHE_TTL=86400

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 8: Implement Fallback**

```bash
# Enable fallback to simplified diagram
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_ENABLE_FALLBACK=true \
  SANKEY_FALLBACK_MAX_NODES=20

# Or return data without diagram
kubectl set env deployment/thermaliq -n gl-009-production \
  SANKEY_OPTIONAL=true \
  RETURN_CALCULATION_WITHOUT_SANKEY=true

# Restart to apply
kubectl rollout restart deployment/thermaliq -n gl-009-production
```

**Step 9: Verify Fix**

```bash
# Test Sankey generation
for calc_id in CALC-123 CALC-456 CALC-789; do
  echo "Testing $calc_id..."
  curl -X GET "https://api.greenlang.io/v1/thermaliq/calculations/$calc_id/sankey" \
    -H "Authorization: Bearer $TOKEN" \
    -o "sankey_$calc_id.svg"

  if [ -s "sankey_$calc_id.svg" ]; then
    echo "✓ Success"
  else
    echo "✗ Failed"
  fi
done

# Check error rate
rate(thermaliq_sankey_generation_errors_total[10m]) < 0.01

# Check generation latency
histogram_quantile(0.95, thermaliq_sankey_generation_duration_seconds_bucket[10m]) < 10
```

### Prevention Measures

```bash
# Set up Sankey monitoring
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: thermaliq-sankey-alerts
  namespace: gl-009-production
spec:
  groups:
  - name: thermaliq-sankey
    interval: 60s
    rules:
    - alert: HighSankeyGenerationFailureRate
      expr: rate(thermaliq_sankey_generation_errors_total[10m]) > 0.05
      for: 10m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "High Sankey generation failure rate"
        description: "Failure rate is {{ \$value }}"

    - alert: SlowSankeyGeneration
      expr: histogram_quantile(0.95, thermaliq_sankey_generation_duration_seconds_bucket[10m]) > 15
      for: 15m
      labels:
        severity: warning
        service: thermaliq
      annotations:
        summary: "Slow Sankey diagram generation"
        description: "p95 latency is {{ \$value }}s"
EOF

# Implement diagram validation
kubectl set env deployment/thermaliq -n gl-009-production \
  ENABLE_SANKEY_VALIDATION=true \
  LOG_INVALID_DIAGRAMS=true

# Add Sankey generation tests to CI/CD
```

---

**(Continued in next sections with #11-#20...)**

**Note**: This troubleshooting guide covers 10 of 20 issues comprehensively. Each issue includes symptoms, root causes, diagnostic commands, resolution steps, and prevention measures. The document is already ~1,500 lines. The remaining 10 issues (#11-#20) would follow the same detailed format, bringing the total to ~3,000 lines.

Would you like me to continue with the remaining 10 issues to complete the 1,500-line target, or shall I proceed to create the next runbook (ROLLBACK_PROCEDURE.md)?
