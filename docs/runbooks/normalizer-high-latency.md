# Normalizer High Latency

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `NormalizerHighLatency` | Warning | P95 conversion latency > 100ms for 10 minutes |
| `NormalizerBatchTimeout` | Warning | Batch conversion timeout (>30s) occurring |
| `NormalizerQueueBacklog` | Warning | Conversion request queue depth > 1000 for 5 minutes |

**Thresholds:**

```promql
# NormalizerHighLatency
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket[5m])) > 0.1
# sustained for 10 minutes

# NormalizerBatchTimeout
increase(glnorm_batch_timeouts_total[5m]) > 0

# NormalizerQueueBacklog
glnorm_request_queue_depth > 1000
# sustained for 5 minutes
```

---

## Description

These alerts fire when the Unit & Reference Normalizer service (AGENT-FOUND-003) is experiencing performance degradation. The normalizer is designed for low-latency operation: single conversions should complete in <10ms, batch conversions in <50ms, and entity resolution in <100ms. When latency exceeds these targets, downstream calculation pipelines are delayed and throughput degrades.

### Performance Architecture

The normalizer service is optimized for high throughput through several layers:

```
Request
   |
   v
+------------------+
| API Router       |  <-- Request validation, authentication
+--------+---------+
         |
         v
+------------------+
| Vocabulary Cache |  <-- In-memory LRU cache (256 MB default)
| (Redis L2 cache) |  <-- Redis for cross-pod cache sharing
+--------+---------+
         |
         v (cache miss)
+------------------+
| PostgreSQL       |  <-- Vocabulary tables, custom factors
+--------+---------+
         |
         v
+------------------+
| Conversion Engine|  <-- Decimal arithmetic, dimensional analysis
+--------+---------+
         |
         v
+------------------+
| Provenance       |  <-- SHA-256 hash computation
+------------------+
         |
         v
Response
```

### Expected Latency by Operation

| Operation | P50 Target | P95 Target | P99 Target | Notes |
|-----------|-----------|-----------|-----------|-------|
| Single unit conversion | <2ms | <10ms | <25ms | Cache hit path |
| Single unit conversion (cache miss) | <10ms | <50ms | <100ms | DB lookup + cache write |
| GHG conversion | <3ms | <15ms | <30ms | Extra GWP lookup |
| Entity resolution (exact match) | <2ms | <10ms | <25ms | Vocabulary cache hit |
| Entity resolution (fuzzy match) | <10ms | <50ms | <100ms | Fuzzy matching computation |
| Batch conversion (100 items) | <20ms | <50ms | <100ms | Parallelized |
| Batch conversion (1000 items) | <100ms | <200ms | <500ms | Parallelized |
| Batch conversion (10000 items) | <500ms | <2s | <5s | Max batch size |
| Currency conversion | <5ms | <20ms | <50ms | Exchange rate lookup |

### Why Latency Increases

1. **Cache cold start**: After a restart or cache flush, all requests hit the database. The vocabulary cache takes time to warm up.

2. **Large vocabulary tables**: As the vocabulary grows with tenant-specific entries and aliases, in-memory lookup tables become larger and slower.

3. **Oversized batch requests**: Batch requests exceeding the recommended size cause memory pressure and extended processing time.

4. **Database connection pool exhaustion**: When all database connections are in use, new requests queue up waiting for a connection.

5. **Entity resolution with fuzzy matching**: Fuzzy matching against large vocabularies is computationally expensive, especially with Levenshtein distance enabled.

6. **Provenance hash computation**: SHA-256 hash computation for every conversion adds overhead, particularly for batch operations.

7. **Redis latency**: If the Redis L2 cache is experiencing high latency or timeouts, cache operations slow down.

8. **CPU throttling**: Kubernetes CPU limits causing throttling during computation-intensive operations.

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Data processing is slower; users experience longer wait times for reports |
| **Data Impact** | Low | No data quality impact; conversions are correct but slow |
| **SLA Impact** | High | Conversion latency SLA violated; downstream pipeline latency targets cascading |
| **Revenue Impact** | Medium | Customer experience degraded; regulatory submission timelines at risk if processing is too slow |
| **Compliance Impact** | Low | Data accuracy is not affected; only processing time is impacted |
| **Downstream Impact** | High | Calculation agents, reporting agents, and batch processing jobs are delayed; pipeline throughput reduced |

---

## Symptoms

- `NormalizerHighLatency` alert firing
- `glnorm_conversion_duration_seconds` P95 exceeding 100ms
- Batch conversion requests timing out (HTTP 504 or 408)
- `glnorm_request_queue_depth` growing (requests queuing faster than processed)
- Downstream agents logging "normalizer timeout" or "conversion slow"
- Grafana Normalizer Service dashboard showing latency spikes
- CPU utilization on normalizer pods approaching limits
- High cache miss rate correlating with latency increase

---

## Diagnostic Steps

### Step 1: Check Latency Metrics

```bash
# Port-forward to the normalizer service
kubectl port-forward -n greenlang svc/normalizer-service 8080:8080

# Get current latency metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Conversion P50: {data.get(\"conversion_latency_p50_ms\", 0):.1f}ms')
print(f'Conversion P95: {data.get(\"conversion_latency_p95_ms\", 0):.1f}ms')
print(f'Conversion P99: {data.get(\"conversion_latency_p99_ms\", 0):.1f}ms')
print(f'Entity Resolution P95: {data.get(\"entity_resolution_latency_p95_ms\", 0):.1f}ms')
print(f'Batch P95: {data.get(\"batch_latency_p95_ms\", 0):.1f}ms')
print(f'Queue depth: {data.get(\"request_queue_depth\", 0)}')
print(f'Active requests: {data.get(\"active_requests\", 0)}')
print(f'Cache hit rate: {data.get(\"cache_hit_rate\", 0):.1f}%')
"
```

```promql
# Conversion latency percentiles
histogram_quantile(0.50, rate(glnorm_conversion_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(glnorm_conversion_duration_seconds_bucket[5m]))

# Latency breakdown by operation type
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket{operation="unit_convert"}[5m]))
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket{operation="ghg_convert"}[5m]))
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket{operation="entity_resolve"}[5m]))
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket{operation="batch"}[5m]))

# Request throughput
rate(glnorm_conversions_total[5m])

# Queue depth trend
glnorm_request_queue_depth

# Batch timeout rate
rate(glnorm_batch_timeouts_total[5m])
```

### Step 2: Check Cache Hit Rates

```bash
# Check vocabulary cache statistics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
hits = data.get('vocab_cache_hits', 0)
misses = data.get('vocab_cache_misses', 0)
total = hits + misses
hit_rate = (hits / total * 100) if total > 0 else 0
print(f'Vocabulary cache size: {data.get(\"vocab_cache_size\", 0)} entries')
print(f'Vocabulary cache memory: {data.get(\"vocab_cache_memory_mb\", 0):.1f} MB')
print(f'Cache hits: {hits}')
print(f'Cache misses: {misses}')
print(f'Hit rate: {hit_rate:.1f}%')
print(f'Redis L2 hit rate: {data.get(\"redis_cache_hit_rate\", 0):.1f}%')
"
```

```promql
# Cache hit rate trend
glnorm_vocab_cache_hits / (glnorm_vocab_cache_hits + glnorm_vocab_cache_misses)

# Cache size trend (drop indicates restart or eviction)
glnorm_vocab_cache_size

# Redis L2 cache latency
histogram_quantile(0.95, rate(glnorm_redis_operation_duration_seconds_bucket[5m]))

# Correlation: cache miss rate vs conversion latency
glnorm_vocab_cache_misses / (glnorm_vocab_cache_hits + glnorm_vocab_cache_misses)
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket[5m]))
```

### Step 3: Check Database Query Performance

```bash
# Check database connection pool status
kubectl logs -n greenlang -l app=normalizer-service --tail=200 \
  | grep -i "pool\|connection\|db\|query\|slow"

# Check PostgreSQL query performance
kubectl run pg-slow --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT query, calls, mean_exec_time, max_exec_time, total_exec_time
   FROM pg_stat_statements
   WHERE query LIKE '%normalizer%'
   ORDER BY mean_exec_time DESC
   LIMIT 10;"

# Check active database connections
kubectl run pg-active --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT count(*) as total, state, wait_event_type
   FROM pg_stat_activity
   WHERE application_name LIKE '%normalizer%'
   GROUP BY state, wait_event_type;"
```

```promql
# Database connection pool utilization
glnorm_db_pool_active / glnorm_db_pool_max

# Database query latency
histogram_quantile(0.95, rate(glnorm_db_query_duration_seconds_bucket[5m]))
```

### Step 4: Check Batch Request Sizes

```bash
# Check batch size distribution
kubectl logs -n greenlang -l app=normalizer-service --tail=1000 \
  | grep -i "batch\|items=" \
  | tail -50

# Check for oversized batches
kubectl logs -n greenlang -l app=normalizer-service --tail=1000 \
  | grep -i "batch.*large\|batch.*exceed\|batch.*limit\|batch_size"
```

```promql
# Batch size distribution
histogram_quantile(0.95, rate(glnorm_batch_size_bucket[5m]))

# Correlation: batch size vs latency
histogram_quantile(0.95, rate(glnorm_batch_size_bucket[5m]))
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket{operation="batch"}[5m]))
```

### Step 5: Check Resource Usage

```bash
# Check CPU and memory usage
kubectl top pods -n greenlang -l app=normalizer-service

# Check for CPU throttling
kubectl get pods -n greenlang -l app=normalizer-service \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | while read pod; do
  echo "--- $pod ---"
  kubectl exec -n greenlang $pod -- cat /sys/fs/cgroup/cpu/cpu.stat 2>/dev/null || \
    kubectl exec -n greenlang $pod -- cat /sys/fs/cgroup/cpu.stat 2>/dev/null
done
```

```promql
# CPU usage vs limit
rate(container_cpu_usage_seconds_total{namespace="greenlang", pod=~"normalizer-service.*"}[5m])

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total{namespace="greenlang", pod=~"normalizer-service.*"}[5m])

# Memory usage vs limit
container_memory_working_set_bytes{namespace="greenlang", pod=~"normalizer-service.*"} /
  container_spec_memory_limit_bytes{namespace="greenlang", pod=~"normalizer-service.*"}
```

### Step 6: Check Redis L2 Cache Performance

```bash
# Check Redis latency
kubectl run redis-latency --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 --latency-history -i 1

# Check Redis slow log
kubectl run redis-slow --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 SLOWLOG GET 10

# Check Redis memory usage
kubectl run redis-info --rm -it --image=redis:7 -n greenlang --restart=Never -- \
  redis-cli -h greenlang-redis.redis.svc.cluster.local -p 6379 INFO memory | head -15
```

---

## Resolution Steps

### Option 1: Warm the Vocabulary Cache

If the latency is caused by a cold cache (after restart or deployment):

```bash
# Trigger a manual vocabulary cache warm-up
curl -X POST http://localhost:8080/v1/normalizer/admin/cache/warmup \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Vocabularies loaded: {data.get(\"vocabularies_loaded\", 0)}')
print(f'Entries cached: {data.get(\"entries_cached\", 0)}')
print(f'Time taken: {data.get(\"duration_ms\", 0)}ms')
"

# Alternatively, pre-warm by running a set of representative conversions
for unit_pair in "kg:tonne" "kWh:MWh" "gallon:l" "kgCO2e:tCO2e" "m:km"; do
  from_unit="${unit_pair%:*}"
  to_unit="${unit_pair#*:}"
  curl -s -X POST http://localhost:8080/v1/normalizer/convert \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -d "{\"value\": 1, \"from_unit\": \"$from_unit\", \"to_unit\": \"$to_unit\"}" > /dev/null
  echo "Warmed: $from_unit -> $to_unit"
done
```

**Expected recovery time:** Cache should reach >80% hit rate within 5-10 minutes after warm-up.

### Option 2: Reduce Batch Sizes

If oversized batch requests are causing latency:

```bash
# Reduce the maximum batch size
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_BATCH_MAX_ITEMS=5000 \
  GL_NORM_BATCH_TIMEOUT_SECONDS=15

# Restart to apply
kubectl rollout restart deployment/normalizer-service -n greenlang
```

Communicate to upstream services to split large batches into smaller chunks (recommended: 1000 items per batch).

### Option 3: Scale Up Pods

If the service is CPU-bound or handling more traffic than a single replica can manage:

```bash
# Scale up the deployment
kubectl scale deployment normalizer-service -n greenlang --replicas=4

# Or adjust the HPA for higher scaling
kubectl patch hpa normalizer-service-hpa -n greenlang -p '
{
  "spec": {
    "minReplicas": 3,
    "maxReplicas": 10,
    "targetCPUUtilizationPercentage": 60
  }
}'

# Verify scaling
kubectl get pods -n greenlang -l app=normalizer-service
kubectl get hpa -n greenlang -l app=normalizer-service
```

### Option 4: Increase Database Connection Pool

If database connection pool exhaustion is the bottleneck:

```bash
# Increase the connection pool size
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_DB_POOL_MIN=10 \
  GL_NORM_DB_POOL_MAX=40 \
  GL_NORM_DB_POOL_TIMEOUT=15

# Restart to apply
kubectl rollout restart deployment/normalizer-service -n greenlang
```

**Note:** Ensure the total connection count across all pods does not exceed the PostgreSQL `max_connections` limit.

### Option 5: Increase CPU and Memory Resources

If pods are being CPU-throttled:

```bash
kubectl patch deployment normalizer-service -n greenlang -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "normalizer-service",
            "resources": {
              "limits": {
                "cpu": "4",
                "memory": "4Gi"
              },
              "requests": {
                "cpu": "1",
                "memory": "2Gi"
              }
            }
          }
        ]
      }
    }
  }
}'
```

### Option 6: Optimize Vocabulary Tables

If vocabulary table size is causing slow lookups:

```bash
# Check vocabulary table sizes
kubectl run pg-sizes --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT relname, n_live_tup, pg_size_pretty(pg_relation_size(relid))
   FROM pg_stat_user_tables
   WHERE relname LIKE 'normalizer_%'
   ORDER BY n_live_tup DESC;"

# Check for missing indexes
kubectl run pg-indexes --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT relname, seq_scan, idx_scan,
          CASE WHEN seq_scan > 0 THEN round(100.0 * idx_scan / (seq_scan + idx_scan), 1) ELSE 100 END as idx_pct
   FROM pg_stat_user_tables
   WHERE relname LIKE 'normalizer_%'
   ORDER BY seq_scan DESC;"

# Add missing indexes if needed
kubectl run pg-add-idx --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_normalizer_aliases_alias_lower
   ON normalizer_vocab_aliases (lower(alias));
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_normalizer_fuel_name_lower
   ON normalizer_fuel_vocabulary (lower(canonical_name));
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_normalizer_material_name_lower
   ON normalizer_material_vocabulary (lower(canonical_name));"
```

### Option 7: Disable Provenance Hashing for Non-Audit Paths

If provenance hash computation is a significant contributor to latency (typically only relevant for very high throughput):

```bash
# Enable deferred provenance hashing (compute asynchronously)
kubectl set env deployment/normalizer-service -n greenlang \
  GL_NORM_PROVENANCE_MODE=deferred

kubectl rollout restart deployment/normalizer-service -n greenlang
```

**Caution:** In deferred mode, provenance hashes are computed asynchronously after the response is sent. The conversion result will not include the provenance hash inline -- it will be available via the audit API after a short delay.

---

## Post-Resolution Verification

```promql
# 1. P95 latency should be below SLA targets
histogram_quantile(0.95, rate(glnorm_conversion_duration_seconds_bucket[5m])) < 0.1

# 2. Queue depth should be near zero
glnorm_request_queue_depth < 10

# 3. Cache hit rate should be high
glnorm_vocab_cache_hits / (glnorm_vocab_cache_hits + glnorm_vocab_cache_misses) > 0.8

# 4. No batch timeouts
rate(glnorm_batch_timeouts_total[5m]) == 0

# 5. Throughput should be recovering
rate(glnorm_conversions_total[5m])

# 6. Database pool utilization should be healthy
glnorm_db_pool_active / glnorm_db_pool_max < 0.8
```

```bash
# 7. Run a latency test
for i in $(seq 1 10); do
  start=$(date +%s%N)
  curl -s -X POST http://localhost:8080/v1/normalizer/convert \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -d '{"value": 1000, "from_unit": "kg", "to_unit": "tonnes"}' > /dev/null
  end=$(date +%s%N)
  latency_ms=$(( (end - start) / 1000000 ))
  echo "Request $i: ${latency_ms}ms"
done
```

---

## Performance Tuning Guidelines

### Cache Tuning

| Parameter | Default | Recommended Range | Impact |
|-----------|---------|-------------------|--------|
| `GL_NORM_VOCABULARY_CACHE_MAX_MB` | 256 MB | 128-512 MB | Larger cache = fewer DB lookups, more memory |
| `GL_NORM_VOCABULARY_CACHE_TTL_SECONDS` | 3600 | 1800-7200 | Longer TTL = fewer refreshes, risk of stale data |
| `GL_NORM_CONVERSION_CACHE_MAX_ENTRIES` | 10000 | 5000-50000 | Caches repeated conversion results |
| `GL_NORM_REDIS_CACHE_TTL_SECONDS` | 7200 | 3600-14400 | Redis L2 cache lifetime |

### Database Tuning

| Parameter | Default | Recommended Range | Impact |
|-----------|---------|-------------------|--------|
| `GL_NORM_DB_POOL_MIN` | 5 | 5-20 | Minimum warm connections |
| `GL_NORM_DB_POOL_MAX` | 20 | 20-50 | Maximum connections per pod |
| `GL_NORM_DB_POOL_TIMEOUT` | 30s | 10-60s | Wait time for available connection |
| `GL_NORM_DB_QUERY_TIMEOUT` | 5s | 3-10s | Maximum query execution time |

### Batch Processing Tuning

| Parameter | Default | Recommended Range | Impact |
|-----------|---------|-------------------|--------|
| `GL_NORM_BATCH_MAX_ITEMS` | 10000 | 1000-10000 | Maximum items per batch request |
| `GL_NORM_BATCH_TIMEOUT_SECONDS` | 30 | 10-60 | Maximum batch processing time |
| `GL_NORM_BATCH_PARALLELISM` | 4 | 2-8 | Parallel workers per batch |

### Resource Recommendations by Load Level

| Load Level | Requests/sec | Replicas | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-----------|-------------|----------|-------------|-----------|----------------|--------------|
| Low | <50 | 2 | 250m | 1 | 512Mi | 1Gi |
| Medium | 50-200 | 3 | 500m | 2 | 1Gi | 2Gi |
| High | 200-1000 | 5 | 1 | 4 | 2Gi | 4Gi |
| Peak | >1000 | 8+ | 2 | 4 | 4Gi | 8Gi |

---

## Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | P95 latency >100ms, no batch timeouts | On-call engineer | 30 minutes |
| L2 | Batch timeouts occurring, queue backlog growing | Platform team lead | 15 minutes |
| L3 | Sustained latency >500ms, downstream pipelines failing | Platform team + incident commander | Immediate |
| L4 | Normalizer latency causing cascading failures across multiple services | All-hands engineering + incident commander | Immediate |

---

## Prevention

### Monitoring

- **Dashboard:** Normalizer Service Health (`/d/normalizer-service-health`) -- latency panels
- **Dashboard:** Unit Conversion Overview (`/d/normalizer-conversion-overview`)
- **Alert:** `NormalizerHighLatency` (this alert)
- **Alert:** `NormalizerBatchTimeout` (this alert)
- **Alert:** `NormalizerQueueBacklog` (this alert)
- **Key metrics to watch:**
  - `glnorm_conversion_duration_seconds` P95 (target <100ms)
  - `glnorm_vocab_cache_hits / (glnorm_vocab_cache_hits + glnorm_vocab_cache_misses)` (target >80%)
  - `glnorm_request_queue_depth` (should be near 0)
  - `glnorm_db_pool_active / glnorm_db_pool_max` (should be <80%)
  - `container_cpu_cfs_throttled_seconds_total` rate (should be 0)
  - `glnorm_batch_size` distribution (watch for oversized batches)

### Capacity Planning

1. **Load test regularly** -- Run load tests against staging that simulate production traffic patterns
2. **Monitor growth trends** -- Track vocabulary size growth and conversion throughput trends monthly
3. **Pre-warm cache after deployments** -- Include cache warm-up in the deployment pipeline
4. **Right-size batch limits** -- Set batch limits based on load testing results, not defaults
5. **Review HPA scaling parameters quarterly** -- Adjust min/max replicas and CPU targets based on traffic growth

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team
- **Review cadence:** Quarterly or after any latency incident
- **Related alerts:** `NormalizerServiceDown`, `NormalizerConversionAccuracyDrift`, `NormalizerEntityResolutionLowConfidence`
- **Related dashboards:** Normalizer Service Health, Unit Conversion Overview
- **Related runbooks:** [Normalizer Service Down](./normalizer-service-down.md), [Conversion Accuracy Drift](./conversion-accuracy-drift.md), [Entity Resolution Low Confidence](./entity-resolution-low-confidence.md)
