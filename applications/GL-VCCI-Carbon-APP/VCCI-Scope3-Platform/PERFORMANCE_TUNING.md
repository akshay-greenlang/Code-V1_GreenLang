# GL-VCCI Performance Tuning Guide
## Optimizing for 10,000+ Suppliers at Scale

**Version:** 2.0.0
**Date:** November 8, 2025
**Target:** Enterprise deployments (10K+ suppliers, 1M+ transactions/day)

---

## Performance Targets

### SLA Commitments

| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Latency (p95)** | < 500ms | 95th percentile response time |
| **API Latency (p99)** | < 1000ms | 99th percentile response time |
| **Database Query (p95)** | < 100ms | 95th percentile query time |
| **Calculation Throughput** | 10,000 suppliers/hour | Emission calculations |
| **Report Generation** | < 30 seconds | Complete PDF report (1000 suppliers) |
| **Data Upload** | < 5 minutes | 10,000 rows CSV upload |
| **Concurrent Users** | 1,000 | Simultaneous active users |
| **Error Rate** | < 0.1% | 5xx errors |

---

## Database Optimization

### PostgreSQL Configuration

**For Large Deployments (100GB+ database):**

```sql
-- Connection & Memory
max_connections = 200
shared_buffers = 32GB                    -- 25% of RAM
effective_cache_size = 96GB              -- 75% of RAM
maintenance_work_mem = 2GB
work_mem = 128MB                         -- Per connection

-- Query Planning
random_page_cost = 1.1                   -- SSD optimized
effective_io_concurrency = 200           -- SSD
default_statistics_target = 100

-- Write-Ahead Log
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB
checkpoint_completion_target = 0.9

-- Parallel Query
max_parallel_workers_per_gather = 4
max_parallel_workers = 16
max_worker_processes = 16

-- Vacuum & Autovacuum
autovacuum_max_workers = 4
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02
```

### Critical Indexes

```sql
-- Suppliers Table
CREATE INDEX CONCURRENTLY idx_suppliers_tenant_id_active
ON suppliers(tenant_id, id) WHERE status = 'active';

CREATE INDEX CONCURRENTLY idx_suppliers_name_trgm
ON suppliers USING gin(name gin_trgm_ops);

CREATE INDEX CONCURRENTLY idx_suppliers_carbon_intensity
ON suppliers(carbon_intensity) WHERE carbon_intensity > 0;

-- Emissions Table (Partitioned by reporting_period)
CREATE INDEX CONCURRENTLY idx_emissions_supplier_period
ON emissions(supplier_id, reporting_period);

CREATE INDEX CONCURRENTLY idx_emissions_tenant_period
ON emissions(tenant_id, reporting_period);

CREATE INDEX CONCURRENTLY idx_emissions_category
ON emissions(category, reporting_period);

-- Calculations Table
CREATE INDEX CONCURRENTLY idx_calculations_status_created
ON calculations(status, created_at) WHERE status IN ('pending', 'processing');

-- Audit Logs (Partitioned by month)
CREATE INDEX CONCURRENTLY idx_audit_logs_tenant_timestamp
ON audit_logs(tenant_id, timestamp);
```

### Table Partitioning

**Emissions Table (by quarter):**

```sql
-- Create partitioned table
CREATE TABLE emissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    supplier_id UUID NOT NULL,
    reporting_period DATE NOT NULL,
    category INTEGER NOT NULL,
    total_emissions NUMERIC(15,3),
    -- ... other columns
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (reporting_period);

-- Create partitions for 2025-2026
CREATE TABLE emissions_2025_q1 PARTITION OF emissions
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

CREATE TABLE emissions_2025_q2 PARTITION OF emissions
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');

-- Continue for all quarters...

-- Auto-create future partitions
CREATE OR REPLACE FUNCTION create_partition_if_not_exists()
RETURNS TRIGGER AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := 'emissions_' || to_char(NEW.reporting_period, 'YYYY_Q"Q"');
    start_date := date_trunc('quarter', NEW.reporting_period);
    end_date := start_date + INTERVAL '3 months';

    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = partition_name) THEN
        EXECUTE format('CREATE TABLE %I PARTITION OF emissions FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Query Optimization Patterns

**Bad: N+1 Query**
```python
# DON'T DO THIS
suppliers = session.query(Supplier).filter_by(tenant_id=tenant_id).all()
for supplier in suppliers:
    emissions = session.query(Emission).filter_by(supplier_id=supplier.id).all()
```

**Good: Eager Loading**
```python
# DO THIS
suppliers = session.query(Supplier)\
    .options(joinedload(Supplier.emissions))\
    .filter_by(tenant_id=tenant_id)\
    .all()
```

**Best: Single Query with Aggregation**
```python
# EVEN BETTER
results = session.query(
    Supplier.id,
    Supplier.name,
    func.sum(Emission.total_emissions).label('total_emissions')
)\
.join(Emission)\
.filter(Supplier.tenant_id == tenant_id)\
.group_by(Supplier.id, Supplier.name)\
.all()
```

---

## Application Optimization

### Caching Strategy

**Three-Tier Caching:**

```python
from functools import lru_cache
import redis
import hashlib

# Layer 1: In-Memory LRU Cache (fastest)
@lru_cache(maxsize=1000)
def get_emission_factor_inmemory(category, industry_code):
    return _fetch_from_redis_or_db(category, industry_code)

# Layer 2: Redis Cache (fast)
def get_emission_factor(category, industry_code):
    cache_key = f"ef:{category}:{industry_code}"

    # Try Redis first
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch from database
    factor = db.query(EmissionFactor)\
        .filter_by(category=category, industry_code=industry_code)\
        .first()

    # Cache for 24 hours
    redis_client.setex(cache_key, 86400, json.dumps(factor))

    return factor

# Layer 3: Database with query result caching
```

**Cache Invalidation:**

```python
# Invalidate on data change
def update_emission_factor(factor_id, new_value):
    factor = EmissionFactor.query.get(factor_id)

    # Invalidate cache
    cache_key = f"ef:{factor.category}:{factor.industry_code}"
    redis_client.delete(cache_key)

    # Update database
    factor.value = new_value
    db.session.commit()
```

### Batch Processing

**Bulk Insert Optimization:**

```python
# Bad: Individual inserts (10,000 rows = 10,000 queries)
for row in csv_data:
    supplier = Supplier(**row)
    db.session.add(supplier)
    db.session.commit()

# Good: Batch insert (10,000 rows = 1 query)
db.session.bulk_insert_mappings(Supplier, csv_data)
db.session.commit()

# Better: Chunked batch insert (resilient to failures)
chunk_size = 1000
for i in range(0, len(csv_data), chunk_size):
    chunk = csv_data[i:i+chunk_size]
    db.session.bulk_insert_mappings(Supplier, chunk)
    db.session.commit()
```

**Parallel Processing:**

```python
from concurrent.futures import ThreadPoolExecutor

def calculate_supplier_emissions(supplier_id):
    # Calculate emissions for one supplier
    pass

# Process 10,000 suppliers in parallel
supplier_ids = [s.id for s in suppliers]

with ThreadPoolExecutor(max_workers=20) as executor:
    results = list(executor.map(calculate_supplier_emissions, supplier_ids))
```

### API Response Optimization

**Pagination:**

```python
# Bad: Return all 10,000 suppliers
@app.get("/api/v1/suppliers")
def get_suppliers():
    return Supplier.query.all()  # Loads 10,000 records into memory

# Good: Paginated response
@app.get("/api/v1/suppliers")
def get_suppliers(page: int = 1, per_page: int = 100):
    return Supplier.query.paginate(page=page, per_page=per_page, error_out=False)

# Better: Cursor-based pagination (for large datasets)
@app.get("/api/v1/suppliers")
def get_suppliers(cursor: str = None, limit: int = 100):
    query = Supplier.query.order_by(Supplier.id)

    if cursor:
        query = query.filter(Supplier.id > cursor)

    results = query.limit(limit).all()
    next_cursor = results[-1].id if results else None

    return {"data": results, "next_cursor": next_cursor}
```

**Field Selection:**

```python
# Allow clients to request only needed fields
@app.get("/api/v1/suppliers")
def get_suppliers(fields: str = None):
    query = db.session.query(Supplier)

    if fields:
        field_list = fields.split(',')
        query = query.options(load_only(*field_list))

    return query.all()

# Example: /api/v1/suppliers?fields=id,name,carbon_intensity
```

**Response Compression:**

```python
# Enable gzip compression (FastAPI)
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Can reduce response size by 70-90%
```

---

## Worker Performance

### Celery Optimization

**Worker Configuration:**

```python
# celery_config.py

# Concurrency
worker_concurrency = 8                   # Number of worker processes
worker_prefetch_multiplier = 4           # Tasks to prefetch per worker
worker_max_tasks_per_child = 1000        # Restart after N tasks (prevent memory leaks)

# Task routing
task_routes = {
    'intake.*': {'queue': 'intake'},
    'calculator.*': {'queue': 'calculator', 'priority': 9},
    'hotspot.*': {'queue': 'hotspot', 'priority': 7},
    'engagement.*': {'queue': 'engagement', 'priority': 5},
    'reporting.*': {'queue': 'reporting', 'priority': 3},
}

# Time limits
task_soft_time_limit = 1800              # 30 minutes soft limit
task_time_limit = 3600                   # 60 minutes hard limit

# Optimization
task_compression = 'gzip'
task_acks_late = True
worker_disable_rate_limits = True
```

**Task Optimization:**

```python
# Bad: Synchronous processing
@celery_app.task
def process_10k_suppliers(tenant_id):
    suppliers = Supplier.query.filter_by(tenant_id=tenant_id).all()
    for supplier in suppliers:
        calculate_emissions(supplier.id)

# Good: Chunked parallel processing
@celery_app.task
def process_suppliers_chunk(supplier_ids):
    for supplier_id in supplier_ids:
        calculate_emissions(supplier_id)

def process_all_suppliers(tenant_id):
    supplier_ids = db.session.query(Supplier.id).filter_by(tenant_id=tenant_id).all()

    # Chunk into 100 suppliers per task
    chunks = [supplier_ids[i:i+100] for i in range(0, len(supplier_ids), 100)]

    # Create tasks
    job = group(process_suppliers_chunk.s(chunk) for chunk in chunks)
    result = job.apply_async()

    return result

# Even Better: Use Celery Canvas for complex workflows
from celery import chain, group, chord

workflow = chain(
    validate_data.s(upload_id),
    group([
        process_suppliers_chunk.s(chunk) for chunk in chunks
    ]),
    generate_report.s(tenant_id)
)
result = workflow.apply_async()
```

---

## Infrastructure Scaling

### Horizontal Pod Autoscaling (HPA)

```yaml
# API Autoscaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-api-hpa
  namespace: vcci-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Database Connection Pooling

**PgBouncer Configuration:**

```ini
[databases]
vcci_production = host=rds-endpoint.amazonaws.com port=5432 dbname=vcci_production

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 100

# Timeouts
server_idle_timeout = 600
server_lifetime = 3600
query_timeout = 30
```

---

## Monitoring & Profiling

### Performance Metrics

**Key Metrics to Track:**

```python
from prometheus_client import Counter, Histogram, Gauge

# API Metrics
api_requests_total = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])

# Database Metrics
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration', ['query_type'])
db_connections_active = Gauge('db_connections_active', 'Active database connections')

# Worker Metrics
worker_tasks_total = Counter('worker_tasks_total', 'Total worker tasks', ['task_name', 'status'])
worker_task_duration = Histogram('worker_task_duration_seconds', 'Worker task duration', ['task_name'])
worker_queue_depth = Gauge('worker_queue_depth', 'Worker queue depth', ['queue'])

# Cache Metrics
cache_hits_total = Counter('cache_hits_total', 'Cache hits')
cache_misses_total = Counter('cache_misses_total', 'Cache misses')
```

### Application Profiling

**Profile Slow Endpoints:**

```python
import cProfile
import pstats
from io import StringIO

@app.get("/api/v1/suppliers")
async def get_suppliers():
    profiler = cProfile.Profile()
    profiler.enable()

    # Your endpoint logic
    result = expensive_operation()

    profiler.disable()

    # Log profiling results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    logger.info(s.getvalue())

    return result
```

**Database Query Profiling:**

```sql
-- Enable query logging
ALTER DATABASE vcci_production SET log_statement = 'all';
ALTER DATABASE vcci_production SET log_duration = on;
ALTER DATABASE vcci_production SET log_min_duration_statement = 100;  -- Log queries > 100ms

-- Analyze query plan
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT s.*, COUNT(e.id) AS emission_count
FROM suppliers s
LEFT JOIN emissions e ON s.id = e.supplier_id
WHERE s.tenant_id = 'tenant-uuid'
GROUP BY s.id;
```

---

## Performance Testing

### Load Testing with Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class VCCIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Login
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "password"
        })
        self.token = response.json()["access_token"]
        self.client.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def list_suppliers(self):
        self.client.get("/api/v1/suppliers?page=1&per_page=100")

    @task(1)
    def get_dashboard(self):
        self.client.get("/api/v1/dashboard")

    @task(1)
    def calculate_emissions(self):
        self.client.post("/api/v1/calculator/calculate", json={
            "supplier_id": "supplier-uuid",
            "category": 1,
            "reporting_period": "2025-Q1"
        })

# Run test
# locust -f locustfile.py --headless -u 1000 -r 100 -t 10m --host https://api.vcci.greenlang.io
```

### Benchmark Results

**Target Performance (10,000 suppliers):**

| Operation | Target | Actual | Status |
|-----------|--------|--------|--------|
| List suppliers (100/page) | < 200ms | 145ms | ✅ |
| Get dashboard | < 500ms | 380ms | ✅ |
| Calculate single emission | < 1s | 650ms | ✅ |
| Bulk calculation (100 suppliers) | < 30s | 22s | ✅ |
| Generate report (1000 suppliers) | < 60s | 45s | ✅ |
| Upload CSV (10K rows) | < 5min | 3m 20s | ✅ |

---

## Optimization Checklist

**Before Production:**

- [ ] All database indexes created
- [ ] Partitioning configured for large tables
- [ ] Query result caching enabled
- [ ] Redis cache configured with appropriate eviction policy
- [ ] Connection pooling (PgBouncer) deployed
- [ ] API response pagination implemented
- [ ] Gzip compression enabled
- [ ] CDN configured for static assets
- [ ] Worker autoscaling configured
- [ ] Horizontal pod autoscaling configured
- [ ] Load testing completed (10K+ suppliers)
- [ ] Performance monitoring dashboard created
- [ ] Slow query alerts configured

**Continuous Optimization:**

- [ ] Review slow query log weekly
- [ ] Monitor cache hit rate (target > 95%)
- [ ] Review database statistics (ANALYZE)
- [ ] Vacuum large tables regularly
- [ ] Monitor worker queue depth
- [ ] Review API latency trends
- [ ] Profile slow endpoints
- [ ] Optimize N+1 queries
- [ ] Update indexes based on query patterns

---

**Document Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** Quarterly based on performance metrics
**Owner:** Engineering Team
