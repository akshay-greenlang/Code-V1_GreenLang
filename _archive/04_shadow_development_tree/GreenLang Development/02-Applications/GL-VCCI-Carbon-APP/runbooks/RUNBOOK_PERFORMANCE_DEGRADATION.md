# Runbook: Performance Degradation

**Severity**: High
**Owner**: Platform Operations Team
**Last Updated**: 2025-11-09

## Symptoms

### What the User/Operator Sees
- Slow API response times (>3s for calculations)
- UI appears sluggish or unresponsive
- Timeout errors on complex operations
- Alert: `HighLatencyAlert` firing

### Metrics/Alerts That Fire
- **Alert**: `HighLatency` (Severity: High)
- **Metric**: `p95_latency > 3000ms`
- **Metric**: `p99_latency > 5000ms`
- **Metric**: `request_duration_seconds > SLA threshold`
- **Alert**: `SlowDependency` for specific services

## Impact

### User Impact
- **UX**: Frustration with slow responses
- **Productivity**: Delayed calculations
- **Abandonment**: Users may give up and retry

### Business Impact
- **SLA**: Violates p95 < 1s, p99 < 3s SLA
- **Reputation**: Poor user experience
- **Cost**: Wasted compute resources

## Diagnosis

### Step 1: Identify Slow Components
```bash
# Check overall latency
curl http://localhost:8000/metrics | grep -E "latency|duration"

# Sample output:
# api_request_duration_p50 850ms   # OK
# api_request_duration_p95 3200ms  # SLOW!
# api_request_duration_p99 5800ms  # VERY SLOW!

# Check per-endpoint latency
curl http://localhost:8000/metrics | grep "duration.*endpoint"

# Identify slowest endpoints
```

### Step 2: Analyze Request Breakdown
```bash
# Check distributed tracing
curl http://localhost:8000/api/calculate \
  -H "X-Trace: true" \
  -d '{"supplier":"test","spend":1000}' | jq '.trace'

# Sample trace:
{
  "total_duration": 4200,
  "stages": [
    {"name": "validation", "duration": 50},
    {"name": "llm_categorization", "duration": 3800},  # BOTTLENECK!
    {"name": "factor_lookup", "duration": 200},
    {"name": "calculation", "duration": 150}
  ]
}
```

### Step 3: Check Dependency Latency
```bash
# Check external service latency
curl http://localhost:8000/health/dependencies | jq '.[] | {service: .name, latency: .latency_ms}'

# Common slow dependencies:
# - LLM API: 2000-5000ms
# - Factor Broker: 500-1000ms
# - ERP Connector: 1000-3000ms
```

### Step 4: Check Resource Utilization
```bash
# CPU usage
top -bn1 | grep greenlang

# Memory usage
ps aux | grep greenlang | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Database connections
curl http://localhost:8000/metrics | grep db_connections

# Cache hit rate
curl http://localhost:8000/metrics | grep cache_hit_rate
```

### Step 5: Check for Bottlenecks
```bash
# Database slow queries
tail -100 /var/log/postgresql/slow-queries.log

# Redis latency
redis-cli --latency

# Network latency
ping -c 10 api.factor-broker.com

# Check for lock contention
curl http://localhost:8000/admin/locks | jq
```

## Resolution

### Immediate Actions (5 minutes)

#### 1. Enable Caching
```bash
# Enable aggressive caching
curl -X POST http://localhost:8000/admin/cache/config \
  -d '{
    "enabled": true,
    "ttl": 3600,
    "aggressive_mode": true
  }'

# Should improve latency immediately
```

#### 2. Reduce Timeout Values
```yaml
# /etc/greenlang/config.yaml
# Force faster failures to prevent queue buildup
resilience:
  default_timeout: 10  # Reduce from 30
  llm_timeout: 15      # Reduce from 30
```

#### 3. Scale Horizontally
```bash
# Add more workers immediately
kubectl scale deployment greenlang-api --replicas=8

# Or with Docker Compose
docker-compose up -d --scale api=6

# Monitor scaling
kubectl get pods -w
```

### Short-Term Actions (30 minutes)

#### 1. Optimize Slow Dependencies

**LLM API Slow:**
```bash
# Use faster fallback model
curl -X POST http://localhost:8000/admin/llm/config \
  -d '{
    "primary_model": "gpt-3.5-turbo",  # Instead of gpt-4
    "timeout": 10
  }'

# Enable prompt caching
# Edit: /etc/greenlang/config.yaml
llm:
  prompt_cache_enabled: true
  semantic_cache_enabled: true
```

**Database Slow:**
```bash
# Add database indexes
psql -h localhost -U greenlang -d greenlang << EOF
CREATE INDEX CONCURRENTLY idx_calculations_created_at
  ON calculations(created_at DESC);

CREATE INDEX CONCURRENTLY idx_suppliers_category
  ON suppliers(category);
EOF

# Vacuum database
psql -h localhost -U greenlang -d greenlang -c "VACUUM ANALYZE;"
```

**Cache Miss Rate High:**
```bash
# Warm cache with common queries
python scripts/warm_cache.py --preload-common

# Increase cache size
redis-cli CONFIG SET maxmemory 4gb
```

#### 2. Enable Request Batching
```python
# Batch similar requests together
# File: /app/greenlang/api/middleware.py

from greenlang.intelligence.request_batching import AdaptiveBatcher

batcher = AdaptiveBatcher(
    max_batch_size=10,
    max_wait_ms=100
)

# Process requests in batches
results = await batcher.batch_process(requests)
```

#### 3. Implement Rate Limiting
```bash
# Limit concurrent requests to slow services
curl -X POST http://localhost:8000/admin/concurrency \
  -d '{
    "llm_service": {"max_concurrent": 5},
    "factor_broker": {"max_concurrent": 10}
  }'
```

### Long-Term Actions (2-4 hours)

#### 1. Query Optimization
```sql
-- Find slow queries
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Optimize identified queries
-- Add indexes, rewrite queries, etc.
```

#### 2. Code Profiling
```bash
# Profile application
python -m cProfile -o profile.stats \
  -m greenlang.api.main

# Analyze results
python -m pstats profile.stats
>>> sort cumulative
>>> stats 20

# Identify hot spots and optimize
```

#### 3. Async Optimization
```python
# Convert sync calls to async
# Before:
result = slow_sync_function()

# After:
result = await async_slow_function()

# Use async where possible
# Parallelize independent operations
results = await asyncio.gather(
    get_factor(),
    categorize_supplier(),
    fetch_erp_data()
)
```

#### 4. Database Connection Pooling
```yaml
# Optimize pool settings
database:
  pool_size: 20
  max_overflow: 40
  pool_recycle: 3600
  pool_pre_ping: true
```

## Prevention

### 1. Performance Monitoring
```yaml
# Set up comprehensive monitoring
alerts:
  - alert: LatencyP95High
    expr: api_latency_p95 > 1000
    for: 5m
    severity: warning

  - alert: LatencyP99High
    expr: api_latency_p99 > 3000
    for: 2m
    severity: high
```

### 2. Load Testing
```bash
# Weekly load tests
locust -f tests/load/test_api.py \
  --headless -u 100 -r 10 \
  --run-time 10m \
  --html report.html

# Monitor:
# - p95/p99 latency
# - Error rate
# - Throughput (req/s)
```

### 3. Performance Budgets
```yaml
# Define performance budgets
performance_budgets:
  api_endpoints:
    /api/calculate:
      p50: 500ms
      p95: 1000ms
      p99: 2000ms
    /api/categorize:
      p50: 800ms
      p95: 1500ms
      p99: 3000ms

# Alert when budgets exceeded
```

### 4. Caching Strategy
```python
# Multi-layer caching
# L1: In-memory (microseconds)
# L2: Redis (milliseconds)
# L3: Database (hundreds of ms)

@cache_multi_tier(
    l1_ttl=60,      # 1 minute
    l2_ttl=3600,    # 1 hour
    l3_ttl=86400    # 24 hours
)
async def get_emission_factor(category):
    pass
```

### 5. Regular Optimization
- **Weekly**: Review slow query log
- **Monthly**: Performance profiling session
- **Quarterly**: Capacity planning review
- **Annually**: Architecture review

## Runbook Metadata

- **Version**: 1.0
- **Average Resolution Time**: 30-60 minutes
- **Escalation**: >2 hours → Senior Engineer, >4 hours → Architecture Team
- **Related Runbooks**:
  - `RUNBOOK_HIGH_FAILURE_RATE.md`
  - `RUNBOOK_DEPENDENCY_DOWN.md`

## Appendix

### Performance SLAs
| Metric | Target | Maximum |
|--------|--------|---------|
| p50 latency | 200ms | 500ms |
| p95 latency | 800ms | 1500ms |
| p99 latency | 2000ms | 3000ms |
| Throughput | 100 req/s | - |
| Error rate | <0.1% | <1% |

### Useful Commands
```bash
# Real-time latency monitoring
watch -n 1 'curl -s http://localhost:8000/metrics | grep latency'

# Top slow endpoints
curl http://localhost:8000/admin/metrics/slow | jq '.[] | {endpoint, p99}'

# Database query analysis
psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 5;"

# Cache statistics
redis-cli INFO stats | grep hits

# CPU profiling
py-spy top --pid $(pgrep -f greenlang)
```

### Contact Information
- **Performance Team**: perf-team@greenlang.com
- **Database Team**: dba@greenlang.com
- **On-Call**: PagerDuty
