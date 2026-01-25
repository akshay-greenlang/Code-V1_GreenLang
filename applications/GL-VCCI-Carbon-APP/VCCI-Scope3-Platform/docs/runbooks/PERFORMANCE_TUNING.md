# Performance Tuning Runbook

**Scenario**: Optimize system performance through database query tuning, cache configuration, API response time improvements, index optimization, and connection pool management.

**Severity**: P2 (Proactive optimization) / P1 (Performance degradation)

**RTO/RPO**: N/A (Operational procedure)

**Owner**: Platform Team / Database Team

## Prerequisites

- kubectl access to EKS cluster
- Database access with EXPLAIN privileges
- Grafana/Prometheus access
- Jaeger distributed tracing access
- Understanding of application architecture

## Detection

### Performance Degradation Indicators

1. **Application Metrics**:
   - API P95 response time > 2 seconds
   - Database query time > 500ms
   - Cache hit rate < 80%
   - Queue processing lag > 5 minutes

2. **Resource Metrics**:
   - CPU utilization > 70% sustained
   - Memory usage > 85%
   - Disk I/O wait > 20%
   - Network saturation

3. **User Reports**:
   - Slow page loads
   - Timeout errors
   - Calculation delays

### Check Current Performance

```bash
# Check API response times
kubectl logs -n vcci-scope3 deployment/api-gateway --tail=1000 | \
  grep "response_time" | \
  awk '{sum+=$NF; count++} END {print "Average:", sum/count, "ms"}'

# Check database query performance
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  query,
  calls,
  total_exec_time / 1000 AS total_seconds,
  mean_exec_time AS mean_ms,
  max_exec_time AS max_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
EOF

# Check cache hit rates
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli INFO stats | grep hits

# Check resource utilization
kubectl top pods -n vcci-scope3 --sort-by=cpu
kubectl top nodes
```

## Step-by-Step Procedure

### Part 1: Database Query Optimization

#### Step 1: Identify Slow Queries

```bash
# Enable pg_stat_statements if not already enabled
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
EOF

# Find slowest queries
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  substring(query, 1, 100) AS short_query,
  calls,
  ROUND(total_exec_time::numeric, 2) AS total_time_ms,
  ROUND(mean_exec_time::numeric, 2) AS mean_time_ms,
  ROUND(max_exec_time::numeric, 2) AS max_time_ms,
  ROUND((100 * total_exec_time / SUM(total_exec_time) OVER())::numeric, 2) AS percentage
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 20;
EOF

# Find queries with high variation
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  substring(query, 1, 100) AS short_query,
  calls,
  ROUND(mean_exec_time::numeric, 2) AS mean_ms,
  ROUND(stddev_exec_time::numeric, 2) AS stddev_ms,
  ROUND((stddev_exec_time / NULLIF(mean_exec_time, 0))::numeric, 2) AS coefficient_of_variation
FROM pg_stat_statements
WHERE stddev_exec_time > 100
ORDER BY stddev_exec_time DESC
LIMIT 20;
EOF

# Check for full table scans
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  schemaname,
  tablename,
  seq_scan,
  seq_tup_read,
  idx_scan,
  seq_tup_read / NULLIF(seq_scan, 0) AS avg_seq_tup_read
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 20;
EOF
```

#### Step 2: Analyze Query Execution Plans

```bash
# Use EXPLAIN ANALYZE on problematic query
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT
  e.entity_id,
  e.entity_name,
  c.calculation_date,
  c.total_emissions
FROM entity_master e
JOIN calculation_results c ON e.entity_id = c.entity_id
WHERE c.calculation_date >= '2024-01-01'
  AND e.status = 'active'
ORDER BY c.total_emissions DESC
LIMIT 100;
EOF

# Look for:
# - Seq Scan on large tables (bad)
# - Index Scan or Index Only Scan (good)
# - High actual time vs. estimated rows
# - Sort operations (may need index)
# - Hash joins vs. nested loops

# Visualize execution plan
# Copy EXPLAIN output and paste into: https://explain.dalibo.com/
```

**Example Problem Patterns**:
```sql
-- BAD: Sequential scan on large table
Seq Scan on entity_master  (cost=0.00..12567.00 rows=500000 width=120)
  Filter: (status = 'active')

-- GOOD: Index scan
Index Scan using idx_entity_status on entity_master  (cost=0.43..8.45 rows=1 width=120)
  Index Cond: (status = 'active')
```

#### Step 3: Create Missing Indexes

```bash
# Identify missing indexes
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Check for tables without indexes
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  AND tablename NOT IN (
    SELECT DISTINCT tablename
    FROM pg_indexes
    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
  )
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
EOF

# Create indexes for common query patterns
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Index for status filter
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_status
ON entity_master(status) WHERE status = 'active';

-- Index for date range queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calc_date
ON calculation_results(calculation_date DESC);

-- Composite index for common join + filter
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_calc_entity_date
ON calculation_results(entity_id, calculation_date DESC);

-- Index for text search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_name_gin
ON entity_master USING gin(to_tsvector('english', entity_name));

-- Covering index (includes all columns needed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_entity_covering
ON entity_master(entity_id)
INCLUDE (entity_name, status, created_at);
EOF

# Monitor index creation progress
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  phase,
  blocks_done,
  blocks_total,
  ROUND(100.0 * blocks_done / NULLIF(blocks_total, 0), 2) AS percent_complete
FROM pg_stat_progress_create_index;
EOF
```

#### Step 4: Optimize Table Statistics

```bash
# Update statistics for query planner
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Increase statistics target for frequently queried columns
ALTER TABLE entity_master ALTER COLUMN entity_name SET STATISTICS 1000;
ALTER TABLE calculation_results ALTER COLUMN calculation_date SET STATISTICS 1000;

-- Update statistics
ANALYZE entity_master;
ANALYZE calculation_results;
ANALYZE emission_factors;

-- Check statistics freshness
SELECT
  schemaname,
  tablename,
  last_analyze,
  last_autoanalyze,
  n_mod_since_analyze
FROM pg_stat_user_tables
ORDER BY n_mod_since_analyze DESC
LIMIT 20;
EOF

# Configure autovacuum for high-churn tables
aws rds modify-db-parameter-group \
  --db-parameter-group-name vcci-scope3-prod-params \
  --parameters \
    "ParameterName=autovacuum_analyze_scale_factor,ParameterValue=0.05,ApplyMethod=immediate" \
    "ParameterName=autovacuum_vacuum_scale_factor,ParameterValue=0.1,ApplyMethod=immediate"
```

#### Step 5: Query Rewriting

```bash
# Example: Rewrite inefficient query

# BEFORE (slow - using OR)
# SELECT * FROM entity_master WHERE entity_id = '001' OR entity_id = '002' OR entity_id = '003';

# AFTER (fast - using IN)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT * FROM entity_master
WHERE entity_id IN ('001', '002', '003');
EOF

# BEFORE (slow - function on column prevents index use)
# SELECT * FROM calculation_results WHERE YEAR(calculation_date) = 2024;

# AFTER (fast - index can be used)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT * FROM calculation_results
WHERE calculation_date >= '2024-01-01'
  AND calculation_date < '2025-01-01';
EOF

# BEFORE (slow - SELECT *)
# SELECT * FROM entity_master JOIN calculation_results USING (entity_id);

# AFTER (fast - only needed columns)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  e.entity_id,
  e.entity_name,
  c.total_emissions
FROM entity_master e
JOIN calculation_results c USING (entity_id);
EOF
```

### Part 2: Redis Cache Tuning

#### Step 6: Analyze Cache Performance

```bash
# Check cache stats
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli INFO stats

# Key metrics:
# - keyspace_hits: Number of successful key lookups
# - keyspace_misses: Number of failed key lookups
# - Hit rate = hits / (hits + misses)

# Calculate hit rate
STATS=$(kubectl exec -n vcci-scope3 deployment/redis -- redis-cli INFO stats)
HITS=$(echo "$STATS" | grep keyspace_hits | cut -d: -f2 | tr -d '\r')
MISSES=$(echo "$STATS" | grep keyspace_misses | cut -d: -f2 | tr -d '\r')
HIT_RATE=$(echo "scale=2; 100 * $HITS / ($HITS + $MISSES)" | bc)
echo "Cache Hit Rate: $HIT_RATE%"

# Check memory usage
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli INFO memory

# Check slow commands
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli SLOWLOG GET 20
```

#### Step 7: Optimize Cache Configuration

```bash
# Update Redis config
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Set appropriate maxmemory
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli CONFIG SET maxmemory 4gb

# Optimize persistence settings (if using AOF)
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli CONFIG SET appendfsync everysec

# Check current configuration
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli CONFIG GET maxmemory*

# Analyze key patterns
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli --bigkeys

# Find hot keys
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli --hotkeys
```

#### Step 8: Implement Cache Warming

```bash
# Create cache warming script
cat > /tmp/cache_warm.sh << 'EOF'
#!/bin/bash
# Warm cache with frequently accessed data

# Pre-load emission factors
curl -s https://api.vcci-scope3.com/api/v1/emission-factors | jq -r '.items[].id' | while read id; do
  curl -s "https://api.vcci-scope3.com/api/v1/emission-factors/$id" > /dev/null
  echo "Cached emission factor: $id"
done

# Pre-load active entities
curl -s "https://api.vcci-scope3.com/api/v1/entities?status=active&limit=1000" | jq -r '.items[].entity_id' | while read entity_id; do
  curl -s "https://api.vcci-scope3.com/api/v1/entities/$entity_id" > /dev/null
  echo "Cached entity: $entity_id"
done

# Pre-load recent calculations
curl -s "https://api.vcci-scope3.com/api/v1/calculations?limit=500" > /dev/null
echo "Cached recent calculations"
EOF

chmod +x /tmp/cache_warm.sh

# Run after deployment or cache flush
kubectl exec -n vcci-scope3 deployment/api-gateway -- /tmp/cache_warm.sh
```

#### Step 9: Optimize Cache TTL Strategy

```bash
# Review current TTL settings in application code
kubectl exec -n vcci-scope3 deployment/api-gateway -- env | grep CACHE_TTL

# Update cache TTL via ConfigMap
kubectl patch configmap cache-config -n vcci-scope3 --patch '
data:
  CACHE_TTL_ENTITIES: "3600"          # 1 hour for entities
  CACHE_TTL_EMISSION_FACTORS: "86400" # 24 hours for static data
  CACHE_TTL_CALCULATIONS: "300"       # 5 minutes for calculation results
  CACHE_TTL_REPORTS: "1800"           # 30 minutes for reports
'

# Restart pods to pick up changes
kubectl rollout restart deployment/api-gateway -n vcci-scope3
```

### Part 3: API Response Time Optimization

#### Step 10: Analyze API Traces with Jaeger

```bash
# Forward Jaeger UI
kubectl port-forward -n observability svc/jaeger-query 16686:16686 &

# Open in browser: http://localhost:16686

# Query slow traces programmatically
curl -s 'http://localhost:16686/api/traces?service=api-gateway&lookback=1h&minDuration=2s' | \
  jq '.data[].spans[] | select(.duration > 2000000) | {operationName, duration}'

# Find bottlenecks
curl -s 'http://localhost:16686/api/traces?service=api-gateway&lookback=1h' | \
  jq '.data[].spans[] | select(.tags[]? | select(.key == "http.status_code" and .value == 500))'
```

#### Step 11: Implement Response Pagination

```bash
# Review current API pagination
curl -s "https://api.vcci-scope3.com/api/v1/entities" | jq '.pagination'

# Update API to enforce reasonable limits
# Edit application code to add max page size

# Example: Update FastAPI endpoint
cat > /tmp/pagination_optimization.py << 'EOF'
from fastapi import Query

@app.get("/api/v1/entities")
async def get_entities(
    limit: int = Query(default=50, le=1000),  # Max 1000
    offset: int = Query(default=0, ge=0)
):
    # Limit memory usage by capping result size
    return db.query(Entity).offset(offset).limit(limit).all()
EOF
```

#### Step 12: Enable Response Compression

```bash
# Enable gzip compression in NGINX ingress
kubectl patch ingress api-ingress -n vcci-scope3 --patch '
metadata:
  annotations:
    nginx.ingress.kubernetes.io/use-gzip: "true"
    nginx.ingress.kubernetes.io/gzip-types: "application/json,text/plain,text/css,application/javascript"
    nginx.ingress.kubernetes.io/gzip-min-length: "1000"
'

# Verify compression
curl -H "Accept-Encoding: gzip" -I https://api.vcci-scope3.com/api/v1/entities | grep -i content-encoding

# Test compression ratio
UNCOMPRESSED=$(curl -s https://api.vcci-scope3.com/api/v1/entities | wc -c)
COMPRESSED=$(curl -s -H "Accept-Encoding: gzip" https://api.vcci-scope3.com/api/v1/entities | wc -c)
RATIO=$(echo "scale=2; 100 * (1 - $COMPRESSED / $UNCOMPRESSED)" | bc)
echo "Compression ratio: $RATIO%"
```

### Part 4: Connection Pool Tuning

#### Step 13: Optimize Database Connection Pool

```bash
# Check current connection pool settings
kubectl exec -n vcci-scope3 deployment/api-gateway -- env | grep DB_POOL

# Check active connections
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  state,
  COUNT(*) as count,
  application_name
FROM pg_stat_activity
WHERE datname = 'scope3_platform'
GROUP BY state, application_name
ORDER BY count DESC;
EOF

# Update connection pool configuration
kubectl patch configmap database-config -n vcci-scope3 --patch '
data:
  DB_POOL_SIZE: "20"              # Connections per pod
  DB_POOL_MAX_OVERFLOW: "10"      # Extra connections during spikes
  DB_POOL_TIMEOUT: "30"           # Connection timeout (seconds)
  DB_POOL_RECYCLE: "3600"         # Recycle connections after 1 hour
  DB_POOL_PRE_PING: "true"        # Test connections before use
'

# Restart pods
kubectl rollout restart deployment/api-gateway -n vcci-scope3
kubectl rollout restart deployment/data-ingestion -n vcci-scope3
kubectl rollout restart deployment/calculation-engine -n vcci-scope3

# Monitor connection usage
watch -n 5 'psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = '\''scope3_platform'\'';"'
```

#### Step 14: Implement PgBouncer (Connection Pooler)

```bash
# Deploy PgBouncer
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
  namespace: vcci-scope3
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
        image: edoburu/pgbouncer:1.17.0
        ports:
        - containerPort: 5432
        env:
        - name: DB_HOST
          value: "$DB_ENDPOINT"
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: "scope3_platform"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_USER
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DB_PASSWORD
        - name: POOL_MODE
          value: "transaction"
        - name: MAX_CLIENT_CONN
          value: "1000"
        - name: DEFAULT_POOL_SIZE
          value: "25"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: pgbouncer
  namespace: vcci-scope3
spec:
  selector:
    app: pgbouncer
  ports:
  - port: 5432
    targetPort: 5432
EOF

# Update application to use PgBouncer
kubectl patch configmap database-config -n vcci-scope3 --patch '
data:
  DB_HOST: "pgbouncer.vcci-scope3.svc.cluster.local"
'

# Restart applications
kubectl rollout restart deployment -n vcci-scope3
```

### Part 5: Index Maintenance

#### Step 15: Identify and Remove Unused Indexes

```bash
# Find unused indexes
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
  idx_scan AS index_scans
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%_pkey'
ORDER BY pg_relation_size(indexrelid) DESC;
EOF

# Check for duplicate indexes
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  pg_size_pretty(SUM(pg_relation_size(idx))::BIGINT) AS size,
  (array_agg(idx))[1] AS idx1,
  (array_agg(idx))[2] AS idx2,
  (array_agg(idx))[3] AS idx3,
  (array_agg(idx))[4] AS idx4
FROM (
  SELECT
    indexrelid::regclass AS idx,
    (indrelid::text ||E'\n'|| indclass::text ||E'\n'|| indkey::text ||E'\n'||
     COALESCE(indexprs::text,'')||E'\n' || COALESCE(indpred::text,'')) AS key
  FROM pg_index
) sub
GROUP BY key
HAVING COUNT(*) > 1
ORDER BY SUM(pg_relation_size(idx)) DESC;
EOF

# Drop unused index (after verification)
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
DROP INDEX CONCURRENTLY IF EXISTS idx_unused_index_name;
EOF
```

#### Step 16: Rebuild Bloated Indexes

```bash
# Check for index bloat
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
  pg_size_pretty(pg_relation_size(indexrelid) - pg_relation_size(indexrelid, 'main')) AS bloat
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 20;
EOF

# Rebuild bloated indexes
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
-- Rebuild index concurrently (no downtime)
REINDEX INDEX CONCURRENTLY idx_entity_status;
REINDEX INDEX CONCURRENTLY idx_calc_date;

-- Or rebuild all indexes for a table
REINDEX TABLE CONCURRENTLY entity_master;
EOF
```

### Part 6: Application-Level Optimizations

#### Step 17: Implement N+1 Query Detection

```bash
# Check application logs for query patterns
kubectl logs -n vcci-scope3 deployment/api-gateway --tail=1000 | \
  grep "SELECT" | \
  sort | uniq -c | sort -rn | head -20

# If you see many identical queries, likely N+1 problem
# Example fix: Use eager loading instead of lazy loading

# BEFORE (N+1 - one query per entity)
# for entity in entities:
#     calculations = db.query(Calculation).filter_by(entity_id=entity.id).all()

# AFTER (1 query with join)
# entities = db.query(Entity).options(joinedload(Entity.calculations)).all()
```

#### Step 18: Implement Async Processing for Heavy Operations

```bash
# Offload heavy calculations to background workers
# Update API to return job ID immediately

# Example: Move calculation to Celery task
cat > /tmp/async_calculation.py << 'EOF'
@app.post("/api/v1/emissions/calculate")
async def calculate_emissions(request: CalculationRequest):
    # Instead of calculating synchronously:
    # result = perform_calculation(request)

    # Queue async task:
    task = calculate_emissions_task.delay(request.dict())

    return {
        "job_id": task.id,
        "status": "queued",
        "status_url": f"/api/v1/jobs/{task.id}"
    }

@celery_app.task
def calculate_emissions_task(data):
    # Heavy calculation here
    result = perform_calculation(data)
    return result
EOF
```

## Validation

### Performance Improvement Checklist

```bash
# 1. Query performance improved
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT
  ROUND(mean_exec_time::numeric, 2) AS mean_ms
FROM pg_stat_statements
WHERE query LIKE '%entity_master%'
ORDER BY mean_exec_time DESC
LIMIT 10;
EOF

# 2. Cache hit rate acceptable
STATS=$(kubectl exec -n vcci-scope3 deployment/redis -- redis-cli INFO stats)
# Target: > 80% hit rate

# 3. API response times improved
kubectl logs -n vcci-scope3 deployment/api-gateway --tail=1000 | \
  grep "response_time" | \
  awk '{sum+=$NF; count++} END {print "P50:", sum/count, "ms"}'
# Target: < 500ms P50, < 2000ms P95

# 4. Resource utilization optimized
kubectl top pods -n vcci-scope3
# Target: CPU < 70%, Memory < 85%

# 5. Database connections stable
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform -c \
  "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = 'scope3_platform';"
# Target: < 80% of max_connections
```

## Troubleshooting

### Issue 1: Query Still Slow After Index Creation

**Diagnosis**:
```bash
# Check if index is being used
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM entity_master WHERE status = 'active';
EOF

# Check index statistics
psql -h $DB_ENDPOINT -U vcci_admin -d scope3_platform << 'EOF'
SELECT * FROM pg_stat_user_indexes WHERE indexrelname = 'idx_entity_status';
EOF
```

**Resolution**:
- Run ANALYZE to update statistics
- Check if query is using different column
- Consider partial or expression index

### Issue 2: Cache Not Improving Performance

**Diagnosis**:
```bash
# Check if cache is being populated
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli KEYS "*" | wc -l

# Check TTL distribution
kubectl exec -n vcci-scope3 deployment/redis -- redis-cli --scan | \
  head -100 | \
  while read key; do
    kubectl exec -n vcci-scope3 deployment/redis -- redis-cli TTL "$key"
  done | sort -n
```

**Resolution**:
- Verify cache keys are consistent
- Check TTL not too short
- Ensure cache warming after deployment

## Related Documentation

- [Scaling Operations Runbook](./SCALING_OPERATIONS.md)
- [Database Failover Runbook](./DATABASE_FAILOVER.md)
- [Capacity Planning Runbook](./CAPACITY_PLANNING.md)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Best Practices](https://redis.io/docs/management/optimization/)

## Contact Information

- **Platform Team**: platform-team@company.com
- **Database Team**: db-team@company.com
- **Performance Engineering**: perf-eng@company.com
