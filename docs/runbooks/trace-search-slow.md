# Trace Search Slow

## Alert

**Alert Name:** `TraceSearchLatencyHigh`

**Severity:** Warning

**Threshold:** `histogram_quantile(0.99, sum(rate(tempo_query_frontend_queries_duration_seconds_bucket[5m])) by (le)) > 10` for 10 minutes

**Duration:** 10 minutes

---

## Description

This alert fires when Grafana Tempo TraceQL search queries are taking longer than 10 seconds at the P99 level. Slow trace searches degrade the developer and SRE experience when investigating incidents and reviewing compliance traces. Common causes include:

1. **Compactor lag**: Blocks are not compacted, causing the querier to scan many small blocks
2. **Large time range queries**: Users searching across many days without filters
3. **Missing bloom filters**: Bloom ShardFactor not configured, forcing full block scans
4. **Resource pressure**: Querier pods CPU/memory constrained
5. **S3 latency**: Object storage backend responding slowly
6. **Cache miss storms**: Query frontend cache cold after restart

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Developers and SREs experience slow trace search in Grafana |
| **Data Impact** | None | Data is intact; only query performance is affected |
| **SLA Impact** | Low | Operational efficiency reduced; no data loss |
| **Compliance Impact** | Low | Audit trace lookups may be slower but still functional |

---

## Symptoms

- TraceQL queries in Grafana Explore take > 10 seconds
- Tempo query frontend P99 latency elevated
- Querier CPU usage near limits
- S3 GET request latency elevated
- Users reporting "timeout" errors on trace search
- Service graph panel loading slowly or showing stale data

---

## Diagnostic Steps

### Step 1: Check Query Performance Metrics

```promql
# Query frontend P99 latency
histogram_quantile(0.99, sum(rate(tempo_query_frontend_queries_duration_seconds_bucket[5m])) by (le))

# Query frontend request rate
sum(rate(tempo_query_frontend_queries_total[5m])) by (status)

# Querier blocks inspected per query
histogram_quantile(0.99, sum(rate(tempo_querier_blocklist_length_bucket[5m])) by (le))

# Querier bytes inspected per query
histogram_quantile(0.99, sum(rate(tempo_querier_bytes_inspected_bucket[5m])) by (le))

# Cache hit rate
sum(rate(tempo_query_frontend_cache_hits_total[5m])) / sum(rate(tempo_query_frontend_cache_requests_total[5m]))
```

### Step 2: Check Compactor Status

```promql
# Compaction rate
sum(rate(tempo_compactor_compactions_total[5m]))

# Outstanding blocks (high = compactor behind)
tempo_compactor_outstanding_blocks

# Compaction duration
histogram_quantile(0.99, sum(rate(tempo_compactor_compaction_duration_seconds_bucket[5m])) by (le))

# Block count in blocklist
tempo_tempodb_blocklist_length
```

```bash
# Check compactor pod health
kubectl get pods -n tracing -l app.kubernetes.io/component=compactor
kubectl top pods -n tracing -l app.kubernetes.io/component=compactor
kubectl logs -n tracing -l app.kubernetes.io/component=compactor --tail=200 | grep -i "error\|halt\|fail"
```

### Step 3: Check Querier Resource Usage

```bash
# Querier CPU and memory
kubectl top pods -n tracing -l app.kubernetes.io/component=querier

# Check resource limits
kubectl get pods -n tracing -l app.kubernetes.io/component=querier \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.memory}{"\t"}{.spec.containers[0].resources.limits.cpu}{"\n"}{end}'

# Check querier logs for slow queries
kubectl logs -n tracing -l app.kubernetes.io/component=querier --tail=500 | grep -i "slow\|timeout\|cancel"
```

### Step 4: Check S3 Backend Latency

```promql
# S3 GET request latency (used by querier)
histogram_quantile(0.99, sum(rate(tempo_tempodb_backend_request_duration_seconds_bucket{operation="GET"}[5m])) by (le))

# S3 request rate by operation
sum(rate(tempo_tempodb_backend_request_duration_seconds_count[5m])) by (operation)

# S3 error rate
sum(rate(tempo_tempodb_backend_request_duration_seconds_count{status_code=~"5.."}[5m])) by (operation)
```

### Step 5: Check Query Frontend Cache

```promql
# Cache hit ratio (should be > 50%)
sum(rate(tempo_query_frontend_cache_hits_total[5m])) /
sum(rate(tempo_query_frontend_cache_requests_total[5m]))

# Cache request rate
sum(rate(tempo_query_frontend_cache_requests_total[5m]))
```

```bash
# Check if Redis cache for Tempo is healthy
kubectl get pods -n cache -l app=redis
kubectl exec -n cache <redis-pod> -- redis-cli info memory | grep used_memory_human
```

---

## Resolution Steps

### Scenario 1: Compactor Lag (Too Many Small Blocks)

**Symptoms:** `tempo_compactor_outstanding_blocks` > 100, `tempo_tempodb_blocklist_length` growing

**Resolution:**

1. **Check if compactor is halted:**

```bash
kubectl logs -n tracing -l app.kubernetes.io/component=compactor --tail=100 | grep -i "halt\|stop"
```

2. **Restart compactor if halted:**

```bash
kubectl rollout restart deployment tempo-compactor -n tracing
```

3. **Increase compactor resources:**

```yaml
# In Helm values
compactor:
  resources:
    requests:
      memory: 4Gi
      cpu: "2"
    limits:
      memory: 8Gi
      cpu: "4"
```

4. **Tune compaction settings:**

```yaml
compactor:
  compaction:
    compaction_window: 1h
    max_block_bytes: 107374182400  # 100GB
    max_compaction_range: 168h     # 7 days
    max_time_per_tenant: 5m
    compacted_block_retention: 1h
```

### Scenario 2: Large Time Range Queries

**Symptoms:** Specific users running queries spanning > 7 days

**Resolution:**

1. **Set query time range limits:**

```yaml
# In Tempo config
query_frontend:
  max_duration: 168h      # 7 days maximum
  max_retries: 2
  search:
    max_duration: 72h     # 3 days for search
    query_timeout: 30s
```

2. **Add query splitting for large ranges:**

```yaml
query_frontend:
  search:
    query_backend_after: 15m
    query_ingesters_until: 30m
```

### Scenario 3: Resource-Constrained Queriers

**Symptoms:** Querier CPU near limits, high query queue depth

**Resolution:**

1. **Scale querier replicas:**

```bash
kubectl scale deployment tempo-querier -n tracing --replicas=4
```

2. **Increase querier resources:**

```yaml
querier:
  resources:
    requests:
      memory: 2Gi
      cpu: "1"
    limits:
      memory: 4Gi
      cpu: "2"
  max_concurrent_queries: 20
```

3. **Enable query frontend for request queuing:**

```yaml
query_frontend:
  max_outstanding_per_tenant: 200
  max_retries: 2
```

### Scenario 4: S3 Backend Latency

**Symptoms:** `tempo_tempodb_backend_request_duration_seconds` P99 > 500ms

**Resolution:**

1. **Verify S3 VPC endpoint is configured:**

```bash
aws ec2 describe-vpc-endpoints --filters "Name=service-name,Values=com.amazonaws.eu-west-1.s3"
```

2. **Enable Tempo block cache (Redis):**

```yaml
storage:
  trace:
    cache: redis
    redis:
      endpoint: gl-redis.cache:6379
      timeout: 500ms
      db: 2
```

3. **Switch to S3 Standard storage class** (not IA):

```bash
# Check current storage classes
aws s3api list-objects-v2 --bucket gl-prod-tempo-traces --max-keys 5 --query "Contents[*].StorageClass"
```

### Scenario 5: Cache Cold Start

**Symptoms:** High latency after collector restart, cache hit ratio < 10%

**Resolution:**

1. **Wait for cache warm-up** (typically 15-30 minutes)

2. **Pre-warm cache by running common queries:**

```bash
# Run common queries via Tempo API
curl -s "http://tempo-query-frontend:3200/api/search?q={service.name=\"api-service\"}&limit=10"
```

3. **Increase cache TTL:**

```yaml
query_frontend:
  search:
    cache_results: true
    cache_min_duration: 5m
    cache_max_duration: 30m
```

---

## Query Optimisation Tips

### For Developers

1. **Always filter by service name first:** `{service.name="api-service"}`
2. **Use time ranges < 24h** when possible
3. **Filter by status** for error investigation: `{status=error}`
4. **Use trace ID search** (fastest): `{traceID="abc123..."}`

### TraceQL Examples (Efficient)

```traceql
# Find errors in a specific service (fast)
{service.name="api-service" && status=error} | duration > 2s

# Find slow requests by route (fast with service filter)
{service.name="api-service" && span.http.route="/api/v1/emissions"} | duration > 5s

# Find compliance agent traces (fast, indexed)
{resource.service.name="eudr-agent" && span.gl.tenant_id="t-corp"}
```

### TraceQL Examples (Slow - Avoid)

```traceql
# BAD: No service filter, scans all blocks
{} | duration > 10s

# BAD: Only attribute filter, no structural query
{span.gl.tenant_id="t-corp"}

# BAD: Very long time range with broad filter
{status=error}  # over 30 days
```

---

## Prevention

### Monitoring

- **Dashboard:** Tempo Operations (`/d/tempo-operations`)
- **Alerts:** `TraceSearchLatencyHigh`, `TempoCompactorHalted`
- **Key metrics:**
  - Query frontend P99 latency (target < 5s)
  - Cache hit ratio (target > 50%)
  - Compactor outstanding blocks (target < 50)
  - S3 GET P99 latency (target < 200ms)

### Capacity Planning

1. **Size querier replicas** at 2x concurrent query load
2. **Size compactor** to handle daily block volume within 24h
3. **Monitor blocklist growth** -- growing blocklist = compactor needs tuning
4. **Plan for 30-day retention** -- ensure S3 lifecycle rules match Tempo config

### Configuration Best Practices

```yaml
# Recommended production query settings
query_frontend:
  max_duration: 168h
  max_retries: 2
  search:
    max_duration: 72h
    query_timeout: 30s
    cache_results: true

querier:
  max_concurrent_queries: 20

compactor:
  compaction:
    compaction_window: 1h
    max_block_bytes: 107374182400
    block_retention: 720h
```

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Observability Team
- **Review cadence:** Quarterly or after any P1 tracing incident
- **Related alerts:** `TempoCompactorHalted`, `TempoStorageErrors`, `TempoDistributorHighLatency`
- **Related dashboards:** Tempo Operations, Trace Analytics
