# GL-VCCI Scope 3 Platform - Performance Optimization Guide

**Version**: 1.0
**Last Updated**: 2025-11-07
**Platform**: GL-VCCI Carbon Intelligence Platform

---

## Table of Contents

1. [Introduction](#introduction)
2. [Performance Monitoring and Metrics](#performance-monitoring-and-metrics)
3. [Database Optimization](#database-optimization)
4. [API Optimization](#api-optimization)
5. [Caching Strategies](#caching-strategies)
6. [Load Balancer Configuration](#load-balancer-configuration)
7. [Kubernetes Resource Optimization](#kubernetes-resource-optimization)
8. [Application Performance](#application-performance)
9. [Query Optimization](#query-optimization)
10. [Real-World Optimization Examples](#real-world-optimization-examples)
11. [Performance Testing](#performance-testing)
12. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)
13. [Monitoring and Alerting](#monitoring-and-alerting)

---

## Introduction

### Overview

This guide provides comprehensive performance optimization strategies for the GL-VCCI Scope 3 Carbon Intelligence Platform. It covers database, API, caching, load balancing, and Kubernetes resource optimization techniques.

### Performance Goals

**Target Metrics**:
- API response time: < 200ms (p50), < 500ms (p95), < 1s (p99)
- Database query time: < 100ms (average)
- Page load time: < 2s (full page)
- Throughput: 1000+ requests/second
- Uptime: 99.9% availability

### Architecture Overview

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
┌──────▼──────────┐
│  Load Balancer  │
│   (Nginx/HAP)   │
└──────┬──────────┘
       │
┌──────▼──────────┐
│  API Gateway    │
│  (Kong/Envoy)   │
└──────┬──────────┘
       │
   ┌───┴───┐
   │       │
┌──▼───┐ ┌─▼────┐
│ API  │ │Cache │
│Nodes │ │Redis │
└──┬───┘ └──────┘
   │
┌──▼────────┐
│ Database  │
│PostgreSQL │
└───────────┘
```

### Performance Optimization Philosophy

1. **Measure First**: Always profile before optimizing
2. **Optimize Bottlenecks**: Focus on highest-impact areas
3. **Test Changes**: Verify improvements with benchmarks
4. **Monitor Continuously**: Track performance over time
5. **Document Decisions**: Record optimization rationale

---

## Performance Monitoring and Metrics

### Key Performance Indicators (KPIs)

#### Application Metrics

**Response Time Metrics**:
- **p50 (Median)**: 50th percentile response time
- **p95**: 95th percentile (most user requests)
- **p99**: 99th percentile (worst-case scenarios)
- **Max**: Maximum response time observed

**Throughput Metrics**:
- **Requests per second (RPS)**: Total API requests
- **Successful requests**: HTTP 2xx responses
- **Failed requests**: HTTP 4xx/5xx responses
- **Error rate**: Percentage of failed requests

**Resource Utilization**:
- **CPU usage**: Percentage of CPU capacity
- **Memory usage**: RAM consumption
- **Disk I/O**: Read/write operations per second
- **Network I/O**: Bandwidth usage

#### Database Metrics

**Query Performance**:
- **Average query time**: Mean execution time
- **Slow query count**: Queries exceeding threshold
- **Query throughput**: Queries per second
- **Connection pool usage**: Active/idle connections

**Database Health**:
- **Cache hit ratio**: Percentage of queries served from cache
- **Index usage**: Percentage of queries using indexes
- **Lock contention**: Number of lock waits
- **Replication lag**: Time behind primary (if applicable)

#### Infrastructure Metrics

**Container Metrics** (Kubernetes):
- **Pod CPU usage**: Per-pod CPU consumption
- **Pod memory usage**: Per-pod RAM usage
- **Pod restart count**: Container stability
- **Pod readiness**: Healthy pod percentage

**Node Metrics**:
- **Node CPU usage**: Overall node capacity
- **Node memory usage**: Available RAM
- **Node disk usage**: Storage capacity
- **Node network**: Bandwidth utilization

### Monitoring Tools

#### Application Performance Monitoring (APM)

**Recommended Tools**:
1. **New Relic**: Full-stack APM with AI insights
2. **Datadog**: Infrastructure and application monitoring
3. **Dynatrace**: Auto-discovery and root cause analysis
4. **Elastic APM**: Open-source APM with ELK stack

**Implementation Example** (Python with New Relic):
```python
# newrelic.ini configuration
[newrelic]
app_name = GL-VCCI API
license_key = YOUR_LICENSE_KEY
monitor_mode = true
log_level = info

# Enable distributed tracing
distributed_tracing.enabled = true
span_events.enabled = true

# Transaction traces
transaction_tracer.enabled = true
transaction_tracer.transaction_threshold = 0.5
transaction_tracer.record_sql = obfuscated

# Error collection
error_collector.enabled = true
error_collector.ignore_errors = werkzeug.exceptions:NotFound
```

**Application Integration**:
```python
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

from flask import Flask
app = Flask(__name__)

# Wrap Flask app
app = newrelic.agent.WSGIApplicationWrapper(app)

# Custom instrumentation
@newrelic.agent.background_task()
def process_emissions_calculation(transaction_id):
    with newrelic.agent.FunctionTrace('calculate_emissions'):
        result = calculate_emissions(transaction_id)
    return result

# Add custom attributes
@app.route('/api/v1/transactions/<transaction_id>')
def get_transaction(transaction_id):
    newrelic.agent.add_custom_attribute('transaction_id', transaction_id)
    newrelic.agent.add_custom_attribute('user_tier', 'premium')
    # ... endpoint logic
```

#### Database Monitoring

**PostgreSQL Monitoring Tools**:
1. **pgBadger**: Log analyzer for PostgreSQL
2. **pg_stat_statements**: Query statistics extension
3. **pgAdmin**: GUI with performance dashboard
4. **Datadog PostgreSQL Integration**: Full monitoring

**Enable Query Statistics**:
```sql
-- Enable pg_stat_statements
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Configure in postgresql.conf
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.max = 10000
pg_stat_statements.track = all

-- Query slowest queries
SELECT
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    stddev_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Reset statistics
SELECT pg_stat_statements_reset();
```

**Slow Query Logging**:
```sql
-- Enable slow query logging
ALTER SYSTEM SET log_min_duration_statement = 100; -- 100ms
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_lock_waits = on;

-- Reload configuration
SELECT pg_reload_conf();
```

#### Infrastructure Monitoring

**Prometheus + Grafana Setup**:
```yaml
# prometheus-config.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Kubernetes metrics
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token

  # Application metrics
  - job_name: 'gl-vcci-api'
    static_configs:
      - targets: ['api:8080']
    metrics_path: /metrics

  # PostgreSQL metrics
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

**Grafana Dashboards**:
- **API Performance**: Request rates, latencies, error rates
- **Database Health**: Query times, connections, cache hits
- **Infrastructure**: CPU, memory, disk, network
- **Business Metrics**: Transactions processed, emissions calculated

### Custom Metrics

**Application-Level Metrics** (Python with Prometheus):
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

emissions_calculated_total = Counter(
    'emissions_calculated_total',
    'Total emissions calculations',
    ['category', 'method']
)

active_users = Gauge(
    'active_users',
    'Number of active users'
)

# Instrument application
from flask import Flask, request
from time import time

app = Flask(__name__)

@app.before_request
def before_request():
    request.start_time = time()

@app.after_request
def after_request(response):
    request_duration = time() - request.start_time

    api_requests_total.labels(
        method=request.method,
        endpoint=request.endpoint,
        status=response.status_code
    ).inc()

    api_request_duration.labels(
        method=request.method,
        endpoint=request.endpoint
    ).observe(request_duration)

    return response

# Metrics endpoint
@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Setting Performance Baselines

**Baseline Testing Process**:
1. **Load normal test data**: Representative dataset
2. **Run performance tests**: Measure current state
3. **Document baseline metrics**: Record all KPIs
4. **Establish thresholds**: Define acceptable ranges
5. **Set alerts**: Notify on threshold breaches

**Baseline Metrics Template**:
```yaml
baseline_metrics:
  date: 2024-11-07
  environment: production
  dataset_size: 1M_transactions

  api_performance:
    p50_latency_ms: 85
    p95_latency_ms: 245
    p99_latency_ms: 680
    throughput_rps: 450
    error_rate_pct: 0.12

  database_performance:
    avg_query_time_ms: 45
    slow_queries_per_hour: 12
    cache_hit_ratio_pct: 94.5
    connection_pool_usage_pct: 65

  resource_utilization:
    cpu_usage_pct: 45
    memory_usage_pct: 62
    disk_io_ops_per_sec: 1200
    network_mbps: 125
```

---

## Database Optimization

### PostgreSQL Configuration Tuning

#### Memory Settings

**postgresql.conf Configuration**:
```ini
# Memory Configuration
# Base memory settings (for 16GB RAM server)
shared_buffers = 4GB              # 25% of RAM
effective_cache_size = 12GB       # 75% of RAM
work_mem = 32MB                   # Per operation memory
maintenance_work_mem = 1GB        # For VACUUM, CREATE INDEX, etc.

# Query planning
random_page_cost = 1.1            # For SSD storage (lower for SSD)
effective_io_concurrency = 200    # For SSD (higher for SSD)
default_statistics_target = 100   # More detailed statistics

# Write-ahead log (WAL)
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB
checkpoint_completion_target = 0.9

# Connection settings
max_connections = 200
shared_preload_libraries = 'pg_stat_statements'

# Autovacuum tuning
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 30s
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
```

**Memory Calculation Guidelines**:
- **shared_buffers**: 25% of RAM (max 8-16GB)
- **effective_cache_size**: 50-75% of RAM
- **work_mem**: Total RAM / (max_connections * 2-3)
- **maintenance_work_mem**: 5-10% of RAM

#### Connection Pooling

**PgBouncer Configuration**:
```ini
; pgbouncer.ini
[databases]
gl_vcci_db = host=postgres-master port=5432 dbname=gl_vcci_db

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

; Pool configuration
pool_mode = transaction           # transaction, session, or statement
max_client_conn = 1000            # Maximum client connections
default_pool_size = 25            # Connections per database
reserve_pool_size = 5             # Emergency reserve
reserve_pool_timeout = 3          # Seconds

; Timeouts
server_idle_timeout = 600
server_lifetime = 3600
server_connect_timeout = 15
query_timeout = 0
query_wait_timeout = 120

; Performance
max_db_connections = 100
max_user_connections = 100
server_reset_query = DISCARD ALL
```

**Application Connection Pool** (Python SQLAlchemy):
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@pgbouncer:6432/gl_vcci_db',
    poolclass=QueuePool,
    pool_size=20,              # Normal pool size
    max_overflow=10,           # Additional connections if needed
    pool_timeout=30,           # Wait time for connection
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True,        # Verify connection before use
    echo_pool=False,           # Set True for debugging
    connect_args={
        'connect_timeout': 10,
        'application_name': 'gl-vcci-api'
    }
)

# Connection pool monitoring
from sqlalchemy import event

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    connection_record.info['pid'] = dbapi_conn.get_backend_pid()
    logger.info(f"New connection: PID {connection_record.info['pid']}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    logger.debug(f"Connection checkout: {connection_record.info['pid']}")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    logger.debug(f"Connection checkin: {connection_record.info['pid']}")
```

### Index Strategies

#### Index Types and Usage

**B-Tree Indexes** (Default):
```sql
-- Primary key and unique constraints (automatic)
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,  -- Automatic B-tree index
    date DATE NOT NULL,
    supplier_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL
);

-- Single-column indexes
CREATE INDEX idx_transactions_date ON transactions(date);
CREATE INDEX idx_transactions_supplier ON transactions(supplier_id);

-- Multi-column (composite) indexes
-- Column order matters: most selective first
CREATE INDEX idx_transactions_date_supplier
    ON transactions(date, supplier_id);

-- Covering index (includes non-key columns)
CREATE INDEX idx_transactions_covering
    ON transactions(date, supplier_id)
    INCLUDE (amount, ghg_category);
```

**Partial Indexes** (Filtered):
```sql
-- Index only active transactions
CREATE INDEX idx_active_transactions
    ON transactions(date)
    WHERE status = 'active';

-- Index recent transactions only
CREATE INDEX idx_recent_transactions
    ON transactions(date DESC)
    WHERE date >= CURRENT_DATE - INTERVAL '1 year';

-- Index high-value transactions
CREATE INDEX idx_high_value_transactions
    ON transactions(amount DESC)
    WHERE amount > 10000;
```

**Expression Indexes**:
```sql
-- Index on computed values
CREATE INDEX idx_transactions_year_month
    ON transactions(EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date));

-- Case-insensitive search
CREATE INDEX idx_supplier_name_lower
    ON suppliers(LOWER(supplier_name));

-- JSONB field indexing
CREATE INDEX idx_custom_fields_cost_center
    ON transactions((custom_fields->>'cost_center'));
```

**GIN Indexes** (for array and full-text search):
```sql
-- Full-text search
CREATE INDEX idx_products_fts
    ON products USING GIN(to_tsvector('english', product_name || ' ' || description));

-- Array fields
CREATE INDEX idx_transaction_tags
    ON transactions USING GIN(tags);

-- JSONB fields
CREATE INDEX idx_custom_fields_gin
    ON transactions USING GIN(custom_fields);
```

#### Index Maintenance

**Monitor Index Usage**:
```sql
-- Find unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexrelname NOT LIKE 'pg_toast%'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Index size and efficiency
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    CASE WHEN idx_scan > 0
        THEN pg_size_pretty(pg_relation_size(indexrelid)::BIGINT / idx_scan)
        ELSE 'N/A'
    END AS avg_bytes_per_scan
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- Missing indexes (suggested)
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY n_distinct DESC;
```

**Index Bloat Detection and Repair**:
```sql
-- Detect bloated indexes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    CASE WHEN idx_scan > 0 THEN 'used' ELSE 'unused' END AS status
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- Rebuild bloated indexes
REINDEX INDEX CONCURRENTLY idx_transactions_date;

-- Rebuild all indexes on a table
REINDEX TABLE CONCURRENTLY transactions;

-- Rebuild all indexes in database (use carefully)
REINDEX DATABASE gl_vcci_db;
```

**Index Creation Best Practices**:
```sql
-- Create index concurrently (no table lock)
CREATE INDEX CONCURRENTLY idx_transactions_new ON transactions(date);

-- Drop index concurrently
DROP INDEX CONCURRENTLY idx_transactions_old;

-- Conditional index creation (idempotent)
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
```

### Query Optimization

#### EXPLAIN and ANALYZE

**Understanding Query Plans**:
```sql
-- Simple EXPLAIN
EXPLAIN
SELECT * FROM transactions WHERE date >= '2024-01-01';

-- EXPLAIN with execution stats
EXPLAIN ANALYZE
SELECT * FROM transactions WHERE date >= '2024-01-01';

-- Detailed options
EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS)
SELECT
    t.transaction_id,
    t.amount,
    s.supplier_name
FROM transactions t
JOIN suppliers s ON t.supplier_id = s.supplier_id
WHERE t.date >= '2024-01-01'
    AND t.amount > 1000;

-- Format as JSON for parsing
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT * FROM transactions WHERE date >= '2024-01-01';
```

**Reading Query Plans**:
- **Seq Scan**: Full table scan (usually slow)
- **Index Scan**: Uses index (fast)
- **Index Only Scan**: Uses covering index (fastest)
- **Bitmap Heap Scan**: Uses index for large result sets
- **Nested Loop**: Joining tables (fast for small joins)
- **Hash Join**: Fast for large joins
- **Merge Join**: Fast for sorted data

**Performance Indicators**:
- **Execution Time**: Total query duration
- **Planning Time**: Time to create query plan
- **Actual Rows**: Rows returned
- **Estimated Rows**: Planner estimate (should be close to actual)
- **Buffers**: Shared, temp, and local buffer usage

#### Query Optimization Techniques

**Use Proper WHERE Clauses**:
```sql
-- Bad: Non-sargable (can't use index)
SELECT * FROM transactions
WHERE EXTRACT(YEAR FROM date) = 2024;

-- Good: Sargable (can use index)
SELECT * FROM transactions
WHERE date >= '2024-01-01' AND date < '2025-01-01';

-- Bad: Function on indexed column
SELECT * FROM suppliers
WHERE LOWER(supplier_name) = 'acme corp';

-- Good: Use expression index or store lowercase version
CREATE INDEX idx_supplier_name_lower ON suppliers(LOWER(supplier_name));
SELECT * FROM suppliers
WHERE LOWER(supplier_name) = 'acme corp';
```

**Optimize JOINs**:
```sql
-- Bad: JOIN on non-indexed column
SELECT t.*, s.supplier_name
FROM transactions t
JOIN suppliers s ON LOWER(t.supplier_name) = LOWER(s.supplier_name);

-- Good: JOIN on indexed foreign key
SELECT t.*, s.supplier_name
FROM transactions t
JOIN suppliers s ON t.supplier_id = s.supplier_id;

-- Use appropriate JOIN type
-- INNER JOIN for required relationships
-- LEFT JOIN for optional relationships
-- Avoid unnecessary JOINs
```

**Limit Result Sets**:
```sql
-- Always use LIMIT for pagination
SELECT * FROM transactions
ORDER BY date DESC
LIMIT 100 OFFSET 0;

-- Use WHERE to filter before LIMIT
SELECT * FROM transactions
WHERE date >= '2024-01-01'
ORDER BY date DESC
LIMIT 100;

-- Use cursors for large result sets
DECLARE transaction_cursor CURSOR FOR
    SELECT * FROM transactions WHERE date >= '2024-01-01';
```

**Aggregate Optimization**:
```sql
-- Bad: Aggregating large datasets repeatedly
SELECT
    COUNT(*) as total,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions
WHERE date >= '2024-01-01';

-- Good: Use materialized views for frequent aggregations
CREATE MATERIALIZED VIEW transaction_daily_summary AS
SELECT
    DATE(date) as transaction_date,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount,
    MAX(amount) as max_amount,
    MIN(amount) as min_amount
FROM transactions
GROUP BY DATE(date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY transaction_daily_summary;

-- Query the materialized view
SELECT * FROM transaction_daily_summary
WHERE transaction_date >= '2024-01-01';
```

**Subquery Optimization**:
```sql
-- Bad: Correlated subquery (runs for each row)
SELECT
    s.supplier_id,
    s.supplier_name,
    (SELECT COUNT(*) FROM transactions t WHERE t.supplier_id = s.supplier_id) as txn_count
FROM suppliers s;

-- Good: Use JOIN instead
SELECT
    s.supplier_id,
    s.supplier_name,
    COUNT(t.transaction_id) as txn_count
FROM suppliers s
LEFT JOIN transactions t ON s.supplier_id = t.supplier_id
GROUP BY s.supplier_id, s.supplier_name;

-- Use CTE for complex queries
WITH supplier_transactions AS (
    SELECT
        supplier_id,
        COUNT(*) as txn_count,
        SUM(amount) as total_amount
    FROM transactions
    GROUP BY supplier_id
)
SELECT
    s.supplier_name,
    st.txn_count,
    st.total_amount
FROM suppliers s
JOIN supplier_transactions st ON s.supplier_id = st.supplier_id;
```

### Partitioning

#### Table Partitioning Strategies

**Range Partitioning** (by date):
```sql
-- Create partitioned table
CREATE TABLE transactions (
    transaction_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    supplier_id VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    -- other columns
    PRIMARY KEY (transaction_id, date)
) PARTITION BY RANGE (date);

-- Create partitions
CREATE TABLE transactions_2024_q1 PARTITION OF transactions
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE transactions_2024_q2 PARTITION OF transactions
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE transactions_2024_q3 PARTITION OF transactions
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE transactions_2024_q4 PARTITION OF transactions
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

-- Create indexes on partitions
CREATE INDEX idx_transactions_2024_q1_supplier
    ON transactions_2024_q1(supplier_id);
CREATE INDEX idx_transactions_2024_q2_supplier
    ON transactions_2024_q2(supplier_id);
-- repeat for other partitions

-- Default partition for outliers
CREATE TABLE transactions_default PARTITION OF transactions DEFAULT;
```

**List Partitioning** (by category):
```sql
-- Create partitioned table by GHG category
CREATE TABLE emissions (
    emission_id SERIAL,
    transaction_id VARCHAR(50) NOT NULL,
    ghg_category INTEGER NOT NULL,
    co2e_kg DECIMAL(15,6),
    -- other columns
    PRIMARY KEY (emission_id, ghg_category)
) PARTITION BY LIST (ghg_category);

-- Create partitions by category
CREATE TABLE emissions_cat1 PARTITION OF emissions
    FOR VALUES IN (1);

CREATE TABLE emissions_cat4 PARTITION OF emissions
    FOR VALUES IN (4);

CREATE TABLE emissions_cat6 PARTITION OF emissions
    FOR VALUES IN (6);

-- Partition for other categories
CREATE TABLE emissions_other PARTITION OF emissions
    FOR VALUES IN (2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15);
```

**Hash Partitioning** (for even distribution):
```sql
-- Create hash partitioned table
CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- other columns
) PARTITION BY HASH (user_id);

-- Create hash partitions
CREATE TABLE user_sessions_p0 PARTITION OF user_sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE user_sessions_p1 PARTITION OF user_sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE user_sessions_p2 PARTITION OF user_sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE user_sessions_p3 PARTITION OF user_sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

#### Partition Management

**Automated Partition Creation**:
```python
from datetime import datetime, timedelta
import psycopg2

def create_quarterly_partitions(conn, year, quarter):
    """Create quarterly partition for transactions table"""
    quarter_starts = {
        1: (1, 1),
        2: (4, 1),
        3: (7, 1),
        4: (10, 1)
    }

    start_month, start_day = quarter_starts[quarter]
    start_date = f"{year}-{start_month:02d}-{start_day:02d}"

    if quarter == 4:
        end_date = f"{year + 1}-01-01"
    else:
        next_month, next_day = quarter_starts[quarter + 1]
        end_date = f"{year}-{next_month:02d}-{next_day:02d}"

    partition_name = f"transactions_{year}_q{quarter}"

    with conn.cursor() as cur:
        # Check if partition exists
        cur.execute("""
            SELECT COUNT(*) FROM pg_tables
            WHERE tablename = %s
        """, (partition_name,))

        if cur.fetchone()[0] == 0:
            # Create partition
            cur.execute(f"""
                CREATE TABLE {partition_name} PARTITION OF transactions
                FOR VALUES FROM ('{start_date}') TO ('{end_date}')
            """)

            # Create indexes
            cur.execute(f"""
                CREATE INDEX idx_{partition_name}_supplier
                ON {partition_name}(supplier_id)
            """)

            conn.commit()
            print(f"Created partition: {partition_name}")
        else:
            print(f"Partition already exists: {partition_name}")

# Usage
conn = psycopg2.connect("dbname=gl_vcci_db user=postgres")
for year in [2024, 2025]:
    for quarter in [1, 2, 3, 4]:
        create_quarterly_partitions(conn, year, quarter)
conn.close()
```

**Partition Pruning**:
```sql
-- Enable partition pruning
SET enable_partition_pruning = on;

-- Query will only scan relevant partition(s)
SELECT * FROM transactions
WHERE date >= '2024-07-01' AND date < '2024-10-01';

-- EXPLAIN shows partition pruning
EXPLAIN SELECT * FROM transactions
WHERE date >= '2024-07-01' AND date < '2024-10-01';
```

**Detaching and Archiving Old Partitions**:
```sql
-- Detach old partition
ALTER TABLE transactions DETACH PARTITION transactions_2020_q1;

-- Archive to separate tablespace
ALTER TABLE transactions_2020_q1 SET TABLESPACE archive_tablespace;

-- Create compressed table from partition
CREATE TABLE transactions_2020_q1_archived
AS SELECT * FROM transactions_2020_q1;

-- Drop detached partition
DROP TABLE transactions_2020_q1;
```

### Materialized Views

**Creating Materialized Views**:
```sql
-- Daily emissions summary
CREATE MATERIALIZED VIEW daily_emissions_summary AS
SELECT
    DATE(t.date) as emission_date,
    t.ghg_category,
    COUNT(*) as transaction_count,
    SUM(e.co2e_kg) as total_co2e_kg,
    AVG(e.co2e_kg) as avg_co2e_kg,
    SUM(t.amount) as total_spend_usd
FROM transactions t
JOIN emissions e ON t.transaction_id = e.transaction_id
GROUP BY DATE(t.date), t.ghg_category;

-- Create unique index for concurrent refresh
CREATE UNIQUE INDEX idx_daily_emissions_summary_unique
ON daily_emissions_summary(emission_date, ghg_category);

-- Supplier performance summary
CREATE MATERIALIZED VIEW supplier_performance_summary AS
SELECT
    s.supplier_id,
    s.supplier_name,
    COUNT(DISTINCT t.transaction_id) as transaction_count,
    SUM(t.amount) as total_spend_usd,
    SUM(e.co2e_kg) as total_co2e_kg,
    SUM(e.co2e_kg) / NULLIF(SUM(t.amount), 0) as co2e_per_dollar,
    AVG(e.co2e_kg) as avg_co2e_per_transaction,
    MAX(t.date) as last_transaction_date
FROM suppliers s
LEFT JOIN transactions t ON s.supplier_id = t.supplier_id
LEFT JOIN emissions e ON t.transaction_id = e.transaction_id
GROUP BY s.supplier_id, s.supplier_name;

CREATE UNIQUE INDEX idx_supplier_performance_unique
ON supplier_performance_summary(supplier_id);
```

**Refreshing Materialized Views**:
```sql
-- Full refresh (locks the view)
REFRESH MATERIALIZED VIEW daily_emissions_summary;

-- Concurrent refresh (requires unique index)
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_emissions_summary;

-- Scheduled refresh with cron
-- Add to crontab:
-- 0 2 * * * psql -d gl_vcci_db -c "REFRESH MATERIALIZED VIEW CONCURRENTLY daily_emissions_summary;"
```

**Automatic Refresh with Triggers** (Pseudo-realtime):
```python
# Python script for periodic refresh
import schedule
import time
import psycopg2

def refresh_materialized_views():
    conn = psycopg2.connect("dbname=gl_vcci_db user=postgres")
    cur = conn.cursor()

    views = [
        'daily_emissions_summary',
        'supplier_performance_summary',
        'monthly_category_rollup'
    ]

    for view in views:
        try:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
            conn.commit()
            print(f"Refreshed: {view}")
        except Exception as e:
            print(f"Error refreshing {view}: {e}")
            conn.rollback()

    cur.close()
    conn.close()

# Schedule refresh every hour
schedule.every().hour.at(":00").do(refresh_materialized_views)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Vacuum and Analyze

**Autovacuum Configuration**:
```sql
-- Global autovacuum settings (postgresql.conf)
autovacuum = on
autovacuum_max_workers = 4
autovacuum_naptime = 30s
autovacuum_vacuum_threshold = 50
autovacuum_analyze_threshold = 50
autovacuum_vacuum_scale_factor = 0.1
autovacuum_analyze_scale_factor = 0.05
autovacuum_vacuum_cost_delay = 20ms
autovacuum_vacuum_cost_limit = 200

-- Per-table autovacuum tuning
ALTER TABLE transactions SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_vacuum_cost_delay = 10
);
```

**Manual Vacuum**:
```sql
-- Standard vacuum (removes dead tuples)
VACUUM transactions;

-- Vacuum and analyze
VACUUM ANALYZE transactions;

-- Full vacuum (reclaims space, locks table)
VACUUM FULL transactions;

-- Verbose output
VACUUM (VERBOSE, ANALYZE) transactions;

-- Vacuum specific partition
VACUUM ANALYZE transactions_2024_q3;
```

**Monitoring Vacuum**:
```sql
-- Check last vacuum/analyze time
SELECT
    schemaname,
    relname,
    last_vacuum,
    last_autovacuum,
    last_analyze,
    last_autoanalyze,
    n_dead_tup,
    n_live_tup
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY n_dead_tup DESC;

-- Table bloat estimation
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_dead_tup,
    n_live_tup,
    ROUND(n_dead_tup * 100.0 / NULLIF(n_live_tup + n_dead_tup, 0), 2) AS dead_tup_pct
FROM pg_stat_user_tables
WHERE n_live_tup > 0
ORDER BY n_dead_tup DESC;
```

---

## API Optimization

### Response Caching

#### Redis Cache Implementation

**Redis Configuration**:
```python
# redis_config.py
import redis
from redis.sentinel import Sentinel
import json
from datetime import timedelta

class RedisCache:
    def __init__(self, config):
        if config.get('sentinel_enabled'):
            # Sentinel for HA
            sentinel = Sentinel(
                config['sentinel_hosts'],
                socket_timeout=0.5
            )
            self.redis_client = sentinel.master_for(
                config['master_name'],
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                decode_responses=True
            )
        else:
            # Single Redis instance
            self.redis_client = redis.Redis(
                host=config['host'],
                port=config['port'],
                db=config['db'],
                password=config.get('password'),
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                socket_keepalive=True,
                socket_keepalive_options={
                    socket.TCP_KEEPIDLE: 60,
                    socket.TCP_KEEPINTVL: 10,
                    socket.TCP_KEEPCNT: 3
                }
            )

    def get(self, key):
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
        return None

    def set(self, key, value, ttl=3600):
        """Set value in cache with TTL"""
        try:
            self.redis_client.setex(
                key,
                timedelta(seconds=ttl),
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False

    def delete(self, key):
        """Delete key from cache"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    def invalidate_pattern(self, pattern):
        """Invalidate keys matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis INVALIDATE error: {e}")
            return False

# Initialize cache
cache = RedisCache({
    'host': 'redis-master',
    'port': 6379,
    'db': 0,
    'password': os.getenv('REDIS_PASSWORD')
})
```

**Cache Decorator**:
```python
# cache_decorator.py
from functools import wraps
import hashlib
import json

def cached(ttl=3600, key_prefix='', cache_instance=None):
    """
    Decorator for caching function results

    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        cache_instance: Redis cache instance
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]

            # Add args to key
            if args:
                key_parts.extend([str(arg) for arg in args])

            # Add kwargs to key (sorted for consistency)
            if kwargs:
                sorted_kwargs = sorted(kwargs.items())
                key_parts.extend([f"{k}:{v}" for k, v in sorted_kwargs])

            # Create hash of key
            key_string = ":".join(key_parts)
            cache_key = f"cache:{hashlib.md5(key_string.encode()).hexdigest()}"

            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return cached_result

            # Cache miss - execute function
            logger.debug(f"Cache MISS: {cache_key}")
            result = func(*args, **kwargs)

            # Store in cache
            cache_instance.set(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator

# Usage example
@cached(ttl=3600, key_prefix='supplier', cache_instance=cache)
def get_supplier_emissions(supplier_id, date_from, date_to):
    """Get supplier emissions (cached for 1 hour)"""
    query = """
        SELECT
            SUM(e.co2e_kg) as total_co2e,
            COUNT(*) as transaction_count
        FROM transactions t
        JOIN emissions e ON t.transaction_id = e.transaction_id
        WHERE t.supplier_id = %s
            AND t.date >= %s
            AND t.date <= %s
    """
    result = db.execute(query, (supplier_id, date_from, date_to)).fetchone()
    return {
        'supplier_id': supplier_id,
        'total_co2e_kg': float(result[0]) if result[0] else 0,
        'transaction_count': result[1]
    }
```

**Cache Invalidation Strategies**:
```python
# cache_invalidation.py

class CacheInvalidator:
    def __init__(self, cache):
        self.cache = cache

    def invalidate_supplier_cache(self, supplier_id):
        """Invalidate all cache entries for a supplier"""
        pattern = f"cache:*supplier*{supplier_id}*"
        self.cache.invalidate_pattern(pattern)
        logger.info(f"Invalidated cache for supplier: {supplier_id}")

    def invalidate_date_range_cache(self, date_from, date_to):
        """Invalidate cache entries for date range"""
        # This is more complex - may need to store key metadata
        # Or use cache tags/groups
        pass

    def invalidate_on_transaction_create(self, transaction):
        """Invalidate relevant caches when transaction created"""
        # Invalidate supplier caches
        self.invalidate_supplier_cache(transaction['supplier_id'])

        # Invalidate date-based caches
        date_pattern = f"cache:*{transaction['date']}*"
        self.cache.invalidate_pattern(date_pattern)

        # Invalidate category caches
        category_pattern = f"cache:*cat{transaction['ghg_category']}*"
        self.cache.invalidate_pattern(category_pattern)

# Integration with API endpoints
@app.route('/api/v1/transactions', methods=['POST'])
def create_transaction():
    transaction = request.json

    # Save transaction
    db.session.add(Transaction(**transaction))
    db.session.commit()

    # Invalidate relevant caches
    invalidator = CacheInvalidator(cache)
    invalidator.invalidate_on_transaction_create(transaction)

    return jsonify({'status': 'success'}), 201
```

#### HTTP Caching Headers

**Response Caching with Flask**:
```python
from flask import Flask, make_response, request
from datetime import datetime, timedelta
import hashlib

app = Flask(__name__)

def add_cache_headers(response, max_age=3600, public=True):
    """Add HTTP cache headers to response"""
    if public:
        response.headers['Cache-Control'] = f'public, max-age={max_age}'
    else:
        response.headers['Cache-Control'] = f'private, max-age={max_age}'

    # Add ETag
    etag = hashlib.md5(response.get_data()).hexdigest()
    response.headers['ETag'] = f'"{etag}"'

    # Add Last-Modified
    response.headers['Last-Modified'] = datetime.utcnow().strftime(
        '%a, %d %b %Y %H:%M:%S GMT'
    )

    # Add Expires
    expires = datetime.utcnow() + timedelta(seconds=max_age)
    response.headers['Expires'] = expires.strftime(
        '%a, %d %b %Y %H:%M:%S GMT'
    )

    return response

@app.route('/api/v1/suppliers/<supplier_id>')
def get_supplier(supplier_id):
    # Check If-None-Match header (ETag)
    if_none_match = request.headers.get('If-None-Match')

    supplier = get_supplier_data(supplier_id)
    response = make_response(jsonify(supplier))

    # Add cache headers (cache for 1 hour)
    response = add_cache_headers(response, max_age=3600, public=True)

    # Check if ETag matches
    if if_none_match == response.headers.get('ETag'):
        return '', 304  # Not Modified

    return response

@app.route('/api/v1/emissions/summary')
def get_emissions_summary():
    # Public data, cache for 15 minutes
    summary = calculate_emissions_summary()
    response = make_response(jsonify(summary))
    response = add_cache_headers(response, max_age=900, public=True)
    return response

@app.route('/api/v1/user/profile')
def get_user_profile():
    # Private data, cache for 5 minutes
    profile = get_user_profile_data()
    response = make_response(jsonify(profile))
    response = add_cache_headers(response, max_age=300, public=False)
    return response
```

### Query Batching and DataLoader

**Implementing DataLoader Pattern**:
```python
# dataloader.py
from collections import defaultdict
from typing import List, Dict, Any
import asyncio

class DataLoader:
    """
    DataLoader pattern implementation for batch loading
    Reduces N+1 query problems
    """
    def __init__(self, batch_load_fn, cache=True):
        self.batch_load_fn = batch_load_fn
        self.cache_enabled = cache
        self._cache = {}
        self._queue = []
        self._loading = False

    async def load(self, key):
        """Load single item by key"""
        # Check cache
        if self.cache_enabled and key in self._cache:
            return self._cache[key]

        # Add to queue
        future = asyncio.Future()
        self._queue.append((key, future))

        # Schedule batch load if not already scheduled
        if not self._loading:
            asyncio.create_task(self._dispatch_queue())

        return await future

    async def load_many(self, keys):
        """Load multiple items by keys"""
        return await asyncio.gather(*[self.load(key) for key in keys])

    async def _dispatch_queue(self):
        """Process queued requests in batch"""
        self._loading = True

        # Small delay to collect more requests
        await asyncio.sleep(0.001)

        # Get all queued requests
        queue = self._queue
        self._queue = []
        self._loading = False

        if not queue:
            return

        # Extract keys and futures
        keys = [item[0] for item in queue]
        futures = {item[0]: item[1] for item in queue}

        try:
            # Batch load
            results = await self.batch_load_fn(keys)

            # Resolve futures
            for key, value in zip(keys, results):
                if self.cache_enabled:
                    self._cache[key] = value
                futures[key].set_result(value)

        except Exception as e:
            # Reject all futures
            for future in futures.values():
                future.set_exception(e)

    def clear_cache(self, key=None):
        """Clear cache for key or all keys"""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

# Usage example
async def batch_load_suppliers(supplier_ids: List[str]) -> List[Dict]:
    """Batch load suppliers by IDs"""
    placeholders = ','.join(['%s'] * len(supplier_ids))
    query = f"""
        SELECT supplier_id, supplier_name, country, industry
        FROM suppliers
        WHERE supplier_id IN ({placeholders})
    """

    results = await db.fetch(query, supplier_ids)

    # Create lookup dict
    supplier_dict = {row['supplier_id']: row for row in results}

    # Return in same order as input
    return [supplier_dict.get(sid) for sid in supplier_ids]

# Create DataLoader instance
supplier_loader = DataLoader(batch_load_suppliers)

# In API endpoint
@app.route('/api/v1/transactions')
async def get_transactions():
    transactions = await db.fetch("SELECT * FROM transactions LIMIT 100")

    # This will batch load all suppliers in one query
    # instead of 100 separate queries
    for txn in transactions:
        txn['supplier'] = await supplier_loader.load(txn['supplier_id'])

    return jsonify(transactions)
```

### Pagination

**Cursor-Based Pagination** (Efficient for large datasets):
```python
# pagination.py
from flask import request, jsonify, url_for
import base64
import json

class CursorPagination:
    """Cursor-based pagination for efficient large dataset navigation"""

    def __init__(self, query, cursor_column='id', per_page=100):
        self.query = query
        self.cursor_column = cursor_column
        self.per_page = per_page

    def encode_cursor(self, value):
        """Encode cursor value"""
        cursor_data = json.dumps({self.cursor_column: str(value)})
        return base64.b64encode(cursor_data.encode()).decode()

    def decode_cursor(self, cursor):
        """Decode cursor value"""
        try:
            cursor_data = base64.b64decode(cursor.encode()).decode()
            data = json.loads(cursor_data)
            return data[self.cursor_column]
        except:
            return None

    def paginate(self, cursor=None, direction='forward'):
        """
        Paginate results

        Args:
            cursor: Base64 encoded cursor
            direction: 'forward' or 'backward'
        """
        # Decode cursor
        cursor_value = self.decode_cursor(cursor) if cursor else None

        # Build query with cursor
        if cursor_value:
            if direction == 'forward':
                query = self.query.where(
                    getattr(self.query.model, self.cursor_column) > cursor_value
                )
            else:
                query = self.query.where(
                    getattr(self.query.model, self.cursor_column) < cursor_value
                )
        else:
            query = self.query

        # Execute query
        if direction == 'forward':
            items = query.order_by(
                getattr(self.query.model, self.cursor_column).asc()
            ).limit(self.per_page + 1).all()
        else:
            items = query.order_by(
                getattr(self.query.model, self.cursor_column).desc()
            ).limit(self.per_page + 1).all()
            items = list(reversed(items))

        # Check if there are more items
        has_next = len(items) > self.per_page
        items = items[:self.per_page]

        # Generate cursors
        next_cursor = None
        prev_cursor = None

        if items:
            if has_next:
                next_cursor = self.encode_cursor(
                    getattr(items[-1], self.cursor_column)
                )

            if cursor:
                prev_cursor = self.encode_cursor(
                    getattr(items[0], self.cursor_column)
                )

        return {
            'items': items,
            'next_cursor': next_cursor,
            'prev_cursor': prev_cursor,
            'has_next': has_next
        }

# API endpoint with cursor pagination
@app.route('/api/v1/transactions')
def list_transactions():
    cursor = request.args.get('cursor')
    per_page = min(int(request.args.get('per_page', 100)), 1000)

    # Base query
    query = db.session.query(Transaction)

    # Apply filters
    if request.args.get('supplier_id'):
        query = query.filter(
            Transaction.supplier_id == request.args['supplier_id']
        )

    # Paginate
    paginator = CursorPagination(query, cursor_column='transaction_id', per_page=per_page)
    result = paginator.paginate(cursor=cursor)

    # Build response
    response = {
        'data': [item.to_dict() for item in result['items']],
        'pagination': {
            'next_cursor': result['next_cursor'],
            'prev_cursor': result['prev_cursor'],
            'has_next': result['has_next'],
            'per_page': per_page
        }
    }

    # Add navigation links
    if result['next_cursor']:
        response['pagination']['next_url'] = url_for(
            'list_transactions',
            cursor=result['next_cursor'],
            per_page=per_page,
            _external=True
        )

    return jsonify(response)
```

**Offset-Based Pagination** (Simple but less efficient):
```python
# Simple offset pagination
@app.route('/api/v1/suppliers')
def list_suppliers():
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 50)), 100)

    # Calculate offset
    offset = (page - 1) * per_page

    # Query with offset and limit
    query = db.session.query(Supplier)

    # Get total count
    total = query.count()

    # Get page items
    items = query.offset(offset).limit(per_page).all()

    # Build response
    response = {
        'data': [item.to_dict() for item in items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': (total + per_page - 1) // per_page,
            'has_next': offset + per_page < total,
            'has_prev': page > 1
        }
    }

    return jsonify(response)
```

### Rate Limiting

**Redis-Based Rate Limiting**:
```python
# rate_limiter.py
import time
from functools import wraps
from flask import request, jsonify

class RateLimiter:
    """Token bucket rate limiter using Redis"""

    def __init__(self, redis_client):
        self.redis = redis_client

    def check_rate_limit(self, key, max_requests, window_seconds):
        """
        Check if request is within rate limit

        Args:
            key: Unique identifier (user_id, IP, API key)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            tuple: (allowed, remaining, reset_time)
        """
        now = int(time.time())
        window_key = f"ratelimit:{key}:{now // window_seconds}"

        # Get current count
        pipe = self.redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, window_seconds * 2)
        result = pipe.execute()

        current_count = result[0]
        allowed = current_count <= max_requests
        remaining = max(0, max_requests - current_count)
        reset_time = (now // window_seconds + 1) * window_seconds

        return allowed, remaining, reset_time

# Rate limiter instance
rate_limiter = RateLimiter(redis_client)

def rate_limit(max_requests=100, window=3600):
    """
    Rate limiting decorator

    Args:
        max_requests: Maximum requests per window
        window: Time window in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get rate limit key (user, API key, or IP)
            user_id = getattr(request, 'user_id', None)
            api_key = request.headers.get('X-API-Key')

            if user_id:
                key = f"user:{user_id}"
            elif api_key:
                key = f"apikey:{api_key}"
            else:
                key = f"ip:{request.remote_addr}"

            # Check rate limit
            allowed, remaining, reset_time = rate_limiter.check_rate_limit(
                key, max_requests, window
            )

            # Add rate limit headers
            response = None
            if allowed:
                response = func(*args, **kwargs)
                if isinstance(response, tuple):
                    response = make_response(*response)
                else:
                    response = make_response(response)
            else:
                response = make_response(
                    jsonify({
                        'error': 'Rate limit exceeded',
                        'message': f'Maximum {max_requests} requests per {window}s'
                    }),
                    429
                )

            # Add headers
            response.headers['X-RateLimit-Limit'] = str(max_requests)
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            response.headers['X-RateLimit-Reset'] = str(reset_time)
            response.headers['Retry-After'] = str(reset_time - int(time.time()))

            return response

        return wrapper
    return decorator

# Usage
@app.route('/api/v1/emissions/calculate', methods=['POST'])
@rate_limit(max_requests=100, window=3600)  # 100 requests per hour
def calculate_emissions():
    # ... endpoint logic
    pass

@app.route('/api/v1/transactions', methods=['POST'])
@rate_limit(max_requests=1000, window=3600)  # 1000 requests per hour
def create_transaction():
    # ... endpoint logic
    pass
```

### Request/Response Compression

**Enable Gzip Compression**:
```python
# compression.py
from flask import Flask
from flask_compress import Compress

app = Flask(__name__)

# Configure compression
app.config['COMPRESS_MIMETYPES'] = [
    'text/html',
    'text/css',
    'text/xml',
    'application/json',
    'application/javascript',
    'text/plain'
]
app.config['COMPRESS_LEVEL'] = 6  # 1-9, higher = more compression
app.config['COMPRESS_MIN_SIZE'] = 500  # Only compress responses > 500 bytes

# Enable compression
compress = Compress()
compress.init_app(app)

# Compression is now automatic for all responses
@app.route('/api/v1/transactions')
def get_transactions():
    # Large response will be automatically compressed
    transactions = get_large_dataset()
    return jsonify(transactions)
```

**Nginx Compression Configuration**:
```nginx
# nginx.conf
http {
    # Enable gzip
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/rss+xml
        application/atom+xml
        image/svg+xml;

    # Minimum response size to compress
    gzip_min_length 1000;

    # Disable for IE6
    gzip_disable "msie6";

    # Enable brotli (if available)
    brotli on;
    brotli_comp_level 6;
    brotli_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss;
}
```

---

## Caching Strategies

### Multi-Layer Caching Architecture

```
┌─────────────────┐
│  Application    │
│  Memory Cache   │ ← L1: In-process cache (fastest)
└────────┬────────┘
         │
┌────────▼────────┐
│  Redis Cache    │ ← L2: Distributed cache (fast)
└────────┬────────┘
         │
┌────────▼────────┐
│  Database       │ ← L3: Source of truth (slow)
└─────────────────┘
```

### L1: Application Memory Cache

**In-Memory Cache with LRU**:
```python
# memory_cache.py
from functools import lru_cache, wraps
from collections import OrderedDict
import threading
import time

class LRUCache:
    """Thread-safe LRU cache with TTL"""

    def __init__(self, maxsize=1000, ttl=3600):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = threading.RLock()

    def get(self, key):
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None

            value, timestamp = self.cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return value

    def set(self, key, value):
        """Set value in cache"""
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)

            self.cache[key] = (value, time.time())

    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()

    def size(self):
        """Get cache size"""
        with self.lock:
            return len(self.cache)

# Global L1 cache
l1_cache = LRUCache(maxsize=1000, ttl=300)  # 5 minutes

# Decorator for L1 caching
def l1_cached(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Generate cache key
        cache_key = f"{func.__name__}:{args}:{kwargs}"

        # Check L1 cache
        result = l1_cache.get(cache_key)
        if result is not None:
            return result

        # Execute function
        result = func(*args, **kwargs)

        # Store in L1 cache
        l1_cache.set(cache_key, result)

        return result

    return wrapper

# Usage
@l1_cached
def get_emission_factor(product_category, country):
    """Get emission factor (cached in L1)"""
    return db.query(EmissionFactor).filter_by(
        product_category=product_category,
        country=country
    ).first()
```

### L2: Redis Cache

**Multi-Layer Cache Implementation**:
```python
# multi_layer_cache.py

class MultiLayerCache:
    """
    Multi-layer caching with L1 (memory) and L2 (Redis)
    """

    def __init__(self, l1_cache, redis_cache):
        self.l1 = l1_cache
        self.l2 = redis_cache

    def get(self, key):
        """Get value from cache (checks L1, then L2)"""
        # Check L1 cache
        value = self.l1.get(key)
        if value is not None:
            return value

        # Check L2 cache
        value = self.l2.get(key)
        if value is not None:
            # Populate L1 cache
            self.l1.set(key, value)
            return value

        return None

    def set(self, key, value, ttl=3600):
        """Set value in both caches"""
        # Set in L1 (shorter TTL)
        self.l1.set(key, value)

        # Set in L2 (longer TTL)
        self.l2.set(key, value, ttl=ttl)

    def delete(self, key):
        """Delete from both caches"""
        self.l1.cache.pop(key, None)
        self.l2.delete(key)

    def get_or_compute(self, key, compute_fn, ttl=3600):
        """Get value or compute if not cached"""
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = compute_fn()

        # Store in cache
        self.set(key, value, ttl=ttl)

        return value

# Initialize multi-layer cache
ml_cache = MultiLayerCache(l1_cache, redis_cache)

# Usage
def get_supplier_emissions(supplier_id):
    return ml_cache.get_or_compute(
        key=f"supplier_emissions:{supplier_id}",
        compute_fn=lambda: calculate_supplier_emissions(supplier_id),
        ttl=3600
    )
```

### Cache Warming

**Preload Frequently Accessed Data**:
```python
# cache_warming.py
import schedule
import time

class CacheWarmer:
    """Preload cache with frequently accessed data"""

    def __init__(self, cache):
        self.cache = cache

    def warm_emission_factors(self):
        """Preload emission factors"""
        logger.info("Warming emission factors cache...")

        # Get all emission factors
        factors = db.query(EmissionFactor).all()

        for factor in factors:
            key = f"emission_factor:{factor.product_category}:{factor.country}"
            self.cache.set(key, factor.to_dict(), ttl=86400)

        logger.info(f"Warmed {len(factors)} emission factors")

    def warm_supplier_data(self):
        """Preload active supplier data"""
        logger.info("Warming supplier cache...")

        # Get active suppliers
        suppliers = db.query(Supplier).filter_by(status='active').all()

        for supplier in suppliers:
            key = f"supplier:{supplier.supplier_id}"
            self.cache.set(key, supplier.to_dict(), ttl=86400)

        logger.info(f"Warmed {len(suppliers)} suppliers")

    def warm_aggregations(self):
        """Preload common aggregations"""
        logger.info("Warming aggregation cache...")

        # Precompute monthly summaries
        for year in [2023, 2024]:
            for month in range(1, 13):
                key = f"monthly_summary:{year}:{month}"
                value = calculate_monthly_summary(year, month)
                self.cache.set(key, value, ttl=86400)

        logger.info("Warmed aggregation cache")

    def warm_all(self):
        """Warm all caches"""
        self.warm_emission_factors()
        self.warm_supplier_data()
        self.warm_aggregations()

# Schedule cache warming
warmer = CacheWarmer(redis_cache)

# Warm on startup
warmer.warm_all()

# Schedule daily cache warming
schedule.every().day.at("02:00").do(warmer.warm_all)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(60)

# Run in background thread
import threading
threading.Thread(target=run_schedule, daemon=True).start()
```

### Cache-Aside Pattern

```python
# cache_aside.py

def get_transaction_with_cache(transaction_id):
    """
    Cache-aside pattern implementation

    Flow:
    1. Check cache
    2. If miss, query database
    3. Store in cache
    4. Return result
    """
    cache_key = f"transaction:{transaction_id}"

    # 1. Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # 2. Query database
    transaction = db.query(Transaction).filter_by(
        transaction_id=transaction_id
    ).first()

    if transaction is None:
        return None

    # 3. Store in cache
    cache.set(cache_key, transaction.to_dict(), ttl=3600)

    # 4. Return result
    return transaction.to_dict()
```

### Write-Through and Write-Behind

**Write-Through Pattern**:
```python
# write_through.py

def update_transaction_write_through(transaction_id, updates):
    """
    Write-through pattern: Update database and cache simultaneously

    Pros: Cache always consistent with database
    Cons: Higher write latency
    """
    # 1. Update database
    transaction = db.query(Transaction).filter_by(
        transaction_id=transaction_id
    ).first()

    for key, value in updates.items():
        setattr(transaction, key, value)

    db.session.commit()

    # 2. Update cache
    cache_key = f"transaction:{transaction_id}"
    cache.set(cache_key, transaction.to_dict(), ttl=3600)

    return transaction
```

**Write-Behind Pattern**:
```python
# write_behind.py
from queue import Queue
import threading

class WriteBehindCache:
    """
    Write-behind pattern: Update cache first, database asynchronously

    Pros: Lower write latency
    Cons: Risk of data loss if cache fails before DB write
    """

    def __init__(self, cache, db):
        self.cache = cache
        self.db = db
        self.write_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_writes, daemon=True)
        self.worker_thread.start()

    def update(self, key, value, db_update_fn):
        """Update with write-behind"""
        # 1. Update cache immediately
        self.cache.set(key, value, ttl=3600)

        # 2. Queue database update
        self.write_queue.put((key, value, db_update_fn))

        return value

    def _process_writes(self):
        """Background worker to process database writes"""
        while True:
            try:
                key, value, db_update_fn = self.write_queue.get(timeout=1)

                # Update database
                try:
                    db_update_fn(value)
                    logger.debug(f"Wrote to database: {key}")
                except Exception as e:
                    logger.error(f"Database write failed: {e}")
                    # Could implement retry logic here

                self.write_queue.task_done()

            except Empty:
                continue

# Usage
wb_cache = WriteBehindCache(cache, db)

def update_transaction_write_behind(transaction_id, updates):
    """Update transaction with write-behind pattern"""

    def db_update(value):
        transaction = db.query(Transaction).filter_by(
            transaction_id=transaction_id
        ).first()
        for key, val in value.items():
            setattr(transaction, key, val)
        db.session.commit()

    cache_key = f"transaction:{transaction_id}"
    return wb_cache.update(cache_key, updates, db_update)
```

---

## Load Balancer Configuration

### Nginx Load Balancer

**Basic Load Balancing Configuration**:
```nginx
# /etc/nginx/nginx.conf

http {
    # Upstream backend servers
    upstream gl_vcci_api {
        # Load balancing method
        least_conn;  # least_conn, ip_hash, or round_robin (default)

        # Backend servers
        server api-1.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
        server api-2.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
        server api-3.internal:8080 weight=1 max_fails=3 fail_timeout=30s;
        server api-4.internal:8080 weight=1 max_fails=3 fail_timeout=30s backup;

        # Connection keepalive
        keepalive 32;
        keepalive_requests 100;
        keepalive_timeout 60s;
    }

    # Server configuration
    server {
        listen 80;
        listen [::]:80;
        server_name api.gl-vcci.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        listen [::]:443 ssl http2;
        server_name api.gl-vcci.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/gl-vcci.crt;
        ssl_certificate_key /etc/nginx/ssl/gl-vcci.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Logging
        access_log /var/log/nginx/gl-vcci-access.log combined;
        error_log /var/log/nginx/gl-vcci-error.log warn;

        # Client settings
        client_max_body_size 100M;
        client_body_timeout 60s;
        client_header_timeout 60s;

        # Proxy settings
        location / {
            proxy_pass http://gl_vcci_api;
            proxy_http_version 1.1;

            # Headers
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Request-ID $request_id;

            # Connection keepalive
            proxy_set_header Connection "";

            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;

            # Buffering
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            proxy_busy_buffers_size 8k;

            # Retry logic
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
            proxy_next_upstream_tries 2;
            proxy_next_upstream_timeout 60s;
        }

        # Health check endpoint (not proxied)
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # Status endpoint (restricted)
        location /nginx_status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
    }
}
```

### Health Checks

**Active Health Checks** (Nginx Plus or lua):
```nginx
# Active health checks (Nginx Plus)
upstream gl_vcci_api {
    zone backend 64k;

    server api-1.internal:8080;
    server api-2.internal:8080;
    server api-3.internal:8080;

    # Health check configuration
    health_check interval=5s
                 fails=3
                 passes=2
                 uri=/health
                 match=health_ok;
}

# Health check match conditions
match health_ok {
    status 200;
    header Content-Type = "application/json";
    body ~ "\"status\":\"healthy\"";
}
```

**Application Health Endpoint**:
```python
# health_check.py
from flask import Flask, jsonify
import psycopg2
import redis

app = Flask(__name__)

@app.route('/health')
def health_check():
    """
    Health check endpoint for load balancer

    Checks:
    - Application is running
    - Database connection
    - Redis connection
    - Disk space
    """
    health = {
        'status': 'healthy',
        'checks': {}
    }

    # Check database
    try:
        conn = psycopg2.connect(DB_CONNECTION_STRING, connect_timeout=2)
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.close()
        conn.close()
        health['checks']['database'] = 'healthy'
    except Exception as e:
        health['checks']['database'] = f'unhealthy: {str(e)}'
        health['status'] = 'unhealthy'

    # Check Redis
    try:
        redis_client.ping()
        health['checks']['redis'] = 'healthy'
    except Exception as e:
        health['checks']['redis'] = f'unhealthy: {str(e)}'
        health['status'] = 'unhealthy'

    # Check disk space
    import shutil
    stat = shutil.disk_usage('/')
    disk_free_pct = (stat.free / stat.total) * 100
    if disk_free_pct < 10:
        health['checks']['disk'] = f'unhealthy: {disk_free_pct:.1f}% free'
        health['status'] = 'unhealthy'
    else:
        health['checks']['disk'] = f'healthy: {disk_free_pct:.1f}% free'

    # Return appropriate status code
    status_code = 200 if health['status'] == 'healthy' else 503

    return jsonify(health), status_code

@app.route('/readiness')
def readiness_check():
    """
    Readiness check for Kubernetes
    Determines if pod is ready to receive traffic
    """
    ready = {
        'ready': True,
        'checks': {}
    }

    # Check if initialization complete
    if not app_initialized:
        ready['ready'] = False
        ready['checks']['initialization'] = 'not complete'

    # Check database connection pool
    if db_pool.size() == 0:
        ready['ready'] = False
        ready['checks']['database_pool'] = 'no connections'

    status_code = 200 if ready['ready'] else 503
    return jsonify(ready), status_code

@app.route('/liveness')
def liveness_check():
    """
    Liveness check for Kubernetes
    Determines if pod should be restarted
    """
    # Simple check that application is running
    return jsonify({'alive': True}), 200
```

### Session Persistence

**IP Hash for Session Sticky**:
```nginx
upstream gl_vcci_api {
    ip_hash;  # Route requests from same IP to same backend

    server api-1.internal:8080;
    server api-2.internal:8080;
    server api-3.internal:8080;
}
```

**Cookie-Based Session Persistence** (Nginx Plus):
```nginx
upstream gl_vcci_api {
    server api-1.internal:8080 route=a;
    server api-2.internal:8080 route=b;
    server api-3.internal:8080 route=c;

    sticky cookie srv_id expires=1h domain=.gl-vcci.com path=/;
}
```

### Connection Pool Tuning

**Nginx Connection Limits**:
```nginx
http {
    # Worker connections
    events {
        worker_connections 4096;
        use epoll;
        multi_accept on;
    }

    # HTTP settings
    http {
        # Connection timeouts
        keepalive_timeout 65s;
        keepalive_requests 100;

        # Upstream keepalive
        upstream gl_vcci_api {
            server api-1.internal:8080;
            keepalive 32;
            keepalive_requests 1000;
            keepalive_timeout 60s;
        }

        # Limit connections per IP
        limit_conn_zone $binary_remote_addr zone=addr:10m;
        limit_conn addr 10;

        # Limit request rate
        limit_req_zone $binary_remote_addr zone=req_limit:10m rate=10r/s;
        limit_req zone=req_limit burst=20 nodelay;
    }
}
```

---

## Kubernetes Resource Optimization

### Resource Requests and Limits

**Pod Resource Configuration**:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gl-vcci-api
spec:
  replicas: 4
  template:
    metadata:
      labels:
        app: gl-vcci-api
    spec:
      containers:
      - name: api
        image: gl-vcci/api:latest
        ports:
        - containerPort: 8080

        # Resource requests (guaranteed)
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"      # 0.25 CPU cores

          # Resource limits (maximum)
          limits:
            memory: "2Gi"
            cpu: "1000m"     # 1 CPU core

        # Health checks
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

        # Lifecycle hooks
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]

        # Environment variables
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: gl-vcci-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: gl-vcci-secrets
              key: redis-url
        - name: WORKERS
          value: "4"
        - name: THREADS
          value: "2"

      # Pod priority
      priorityClassName: high-priority

      # Affinity rules
      affinity:
        # Pod anti-affinity (spread across nodes)
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - gl-vcci-api
              topologyKey: kubernetes.io/hostname

        # Node affinity (prefer certain nodes)
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 1
            preference:
              matchExpressions:
              - key: node-type
                operator: In
                values:
                - compute-optimized
```

### Horizontal Pod Autoscaler (HPA)

**HPA Configuration**:
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gl-vcci-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-vcci-api

  minReplicas: 4
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
        averageUtilization: 80

  # Custom metric (requests per second)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min

    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

**Vertical Pod Autoscaler (VPA)**:
```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-vcci-api-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-vcci-api

  updatePolicy:
    updateMode: "Auto"  # Or "Recreate", "Initial", "Off"

  resourcePolicy:
    containerPolicies:
    - containerName: api
      minAllowed:
        cpu: 250m
        memory: 512Mi
      maxAllowed:
        cpu: 2000m
        memory: 4Gi
      controlledResources:
      - cpu
      - memory
```

### Pod Disruption Budgets

**PDB Configuration**:
```yaml
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: gl-vcci-api-pdb
spec:
  minAvailable: 2  # Or maxUnavailable: 1
  selector:
    matchLabels:
      app: gl-vcci-api
```

### Resource Quotas

**Namespace Resource Quotas**:
```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gl-vcci-quota
  namespace: gl-vcci
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    pods: "50"
```

### Quality of Service (QoS) Classes

**QoS Classes**:
1. **Guaranteed**: requests == limits for all resources
2. **Burstable**: requests < limits for at least one resource
3. **BestEffort**: No requests or limits set

**Guaranteed QoS Pod**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: api-guaranteed
spec:
  containers:
  - name: api
    image: gl-vcci/api:latest
    resources:
      requests:
        memory: "1Gi"
        cpu: "500m"
      limits:
        memory: "1Gi"
        cpu: "500m"
```

---

## Application Performance

### Python Performance Optimization

**Use Gunicorn with Optimal Settings**:
```python
# gunicorn_config.py
import multiprocessing

# Server socket
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"  # or "sync", "eventlet", "tornado"
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "gl-vcci-api"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Preload app
preload_app = True

# Server hooks
def on_starting(server):
    print("Starting Gunicorn server")

def on_reload(server):
    print("Reloading Gunicorn server")

def when_ready(server):
    print("Gunicorn server is ready")

def worker_int(worker):
    print(f"Worker {worker.pid} received INT signal")

def worker_abort(worker):
    print(f"Worker {worker.pid} aborted")
```

**Run with Gunicorn**:
```bash
gunicorn app:app -c gunicorn_config.py
```

### Profiling

**Profile with py-spy**:
```bash
# Install py-spy
pip install py-spy

# Profile running process
py-spy record -o profile.svg --pid <PID>

# Profile for 60 seconds
py-spy record -o profile.svg --pid <PID> --duration 60

# Top-like view
py-spy top --pid <PID>

# Flame graph
py-spy record -f flamegraph -o flamegraph.svg --pid <PID>
```

**Profile with cProfile**:
```python
# profile_endpoint.py
import cProfile
import pstats
from pstats import SortKey

def profile_function(func):
    """Profile a function"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)

    return result

# Profile specific endpoint
@app.route('/api/v1/emissions/calculate', methods=['POST'])
def calculate_emissions():
    if request.args.get('profile'):
        return profile_function(lambda: _calculate_emissions())
    return _calculate_emissions()

def _calculate_emissions():
    # ... actual implementation
    pass
```

### Async Programming

**Async Endpoints with Flask**:
```python
# async_api.py
from flask import Flask
import asyncio
import asyncpg

app = Flask(__name__)

# Async database connection pool
async def create_db_pool():
    return await asyncpg.create_pool(
        DATABASE_URL,
        min_size=10,
        max_size=20,
        command_timeout=60
    )

db_pool = asyncio.run(create_db_pool())

@app.route('/api/v1/transactions')
async def get_transactions():
    """Async endpoint for better concurrency"""
    async with db_pool.acquire() as conn:
        transactions = await conn.fetch(
            "SELECT * FROM transactions ORDER BY date DESC LIMIT 100"
        )

    return jsonify([dict(t) for t in transactions])

# Run with async-capable server
# gunicorn -k gevent app:app
```

### Background Tasks

**Celery for Async Tasks**:
```python
# celery_config.py
from celery import Celery

app = Celery('gl-vcci',
             broker='redis://redis:6379/0',
             backend='redis://redis:6379/1')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,

    # Concurrency
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,

    # Result backend
    result_expires=3600,
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600
    }
)

# Define tasks
@app.task(bind=True, max_retries=3)
def calculate_emissions_async(self, transaction_ids):
    """Calculate emissions asynchronously"""
    try:
        for txn_id in transaction_ids:
            calculate_emissions_for_transaction(txn_id)
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

@app.task
def generate_report_async(report_id, params):
    """Generate report asynchronously"""
    report = generate_report(params)
    store_report(report_id, report)
    return report_id

# In API endpoint
@app.route('/api/v1/emissions/calculate', methods=['POST'])
def trigger_calculation():
    transaction_ids = request.json['transaction_ids']
    task = calculate_emissions_async.delay(transaction_ids)
    return jsonify({'task_id': task.id}), 202

@app.route('/api/v1/tasks/<task_id>')
def get_task_status(task_id):
    task = calculate_emissions_async.AsyncResult(task_id)
    return jsonify({
        'task_id': task_id,
        'state': task.state,
        'result': task.result if task.ready() else None
    })
```

---

## Query Optimization

### N+1 Query Problem

**Problem**:
```python
# Bad: N+1 queries
transactions = Transaction.query.all()  # 1 query
for txn in transactions:
    supplier = Supplier.query.get(txn.supplier_id)  # N queries
    print(f"{txn.transaction_id}: {supplier.supplier_name}")
```

**Solution 1: Eager Loading**:
```python
# Good: Eager loading with joinedload
from sqlalchemy.orm import joinedload

transactions = Transaction.query.options(
    joinedload(Transaction.supplier)
).all()  # 1 query with JOIN

for txn in transactions:
    print(f"{txn.transaction_id}: {txn.supplier.supplier_name}")
```

**Solution 2: Subquery Load**:
```python
# Good: Subquery loading (for one-to-many)
from sqlalchemy.orm import subqueryload

suppliers = Supplier.query.options(
    subqueryload(Supplier.transactions)
).all()  # 2 queries total

for supplier in suppliers:
    print(f"{supplier.supplier_name}: {len(supplier.transactions)} transactions")
```

**Solution 3: DataLoader (GraphQL-style)**:
```python
# See DataLoader implementation in API Optimization section
```

### Select Only Needed Columns

**Problem**:
```python
# Bad: Select all columns
transactions = db.session.query(Transaction).all()
```

**Solution**:
```python
# Good: Select only needed columns
transactions = db.session.query(
    Transaction.transaction_id,
    Transaction.date,
    Transaction.amount
).all()

# Or with specific columns
from sqlalchemy import select

stmt = select(
    Transaction.transaction_id,
    Transaction.amount,
    Supplier.supplier_name
).join(Supplier).where(
    Transaction.date >= '2024-01-01'
)

results = db.session.execute(stmt).all()
```

### Use EXISTS Instead of COUNT

**Problem**:
```python
# Bad: COUNT to check existence
count = db.session.query(Transaction).filter_by(
    supplier_id=supplier_id
).count()

if count > 0:
    # ...
```

**Solution**:
```python
# Good: EXISTS
from sqlalchemy import exists

has_transactions = db.session.query(
    exists().where(Transaction.supplier_id == supplier_id)
).scalar()

if has_transactions:
    # ...
```

### Batch Operations

**Bulk Insert**:
```python
# Bad: Individual inserts
for data in transaction_data:
    txn = Transaction(**data)
    db.session.add(txn)
    db.session.commit()  # Commit each

# Good: Batch insert
transactions = [Transaction(**data) for data in transaction_data]
db.session.bulk_save_objects(transactions)
db.session.commit()  # Single commit

# Even better: Use COPY for PostgreSQL
from io import StringIO
import csv

# Prepare CSV data
csv_data = StringIO()
writer = csv.writer(csv_data)
for data in transaction_data:
    writer.writerow([data['transaction_id'], data['date'], ...])
csv_data.seek(0)

# COPY to database
with db.engine.connect() as conn:
    cursor = conn.connection.cursor()
    cursor.copy_expert(
        "COPY transactions (transaction_id, date, ...) FROM STDIN WITH CSV",
        csv_data
    )
    conn.connection.commit()
```

### Query Result Streaming

**Stream Large Result Sets**:
```python
# Stream results to avoid loading everything into memory
def stream_transactions(date_from):
    """Stream transactions without loading all into memory"""
    query = db.session.query(Transaction).filter(
        Transaction.date >= date_from
    ).yield_per(1000)

    for txn in query:
        yield txn

# Usage
for txn in stream_transactions('2024-01-01'):
    process_transaction(txn)
```

---

## Real-World Optimization Examples

### Example 1: Slow Transaction List Endpoint

**Problem**:
```python
@app.route('/api/v1/transactions')
def list_transactions():
    # Slow: N+1 queries, no pagination, no caching
    transactions = Transaction.query.all()
    result = []
    for txn in transactions:
        result.append({
            'transaction_id': txn.transaction_id,
            'date': txn.date,
            'supplier_name': txn.supplier.supplier_name,  # N+1 query
            'amount': txn.amount,
            'emissions': txn.emission.co2e_kg if txn.emission else None  # N+1 query
        })
    return jsonify(result)
```

**Optimized Solution**:
```python
@app.route('/api/v1/transactions')
@cached(ttl=300, key_prefix='transactions_list', cache_instance=cache)
def list_transactions():
    # Get pagination params
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 50)), 100)

    # Eager load relationships
    query = db.session.query(Transaction).options(
        joinedload(Transaction.supplier),
        joinedload(Transaction.emission)
    )

    # Apply filters
    if request.args.get('supplier_id'):
        query = query.filter(Transaction.supplier_id == request.args['supplier_id'])

    if request.args.get('date_from'):
        query = query.filter(Transaction.date >= request.args['date_from'])

    # Paginate
    offset = (page - 1) * per_page
    transactions = query.offset(offset).limit(per_page).all()

    # Build response (no additional queries)
    result = []
    for txn in transactions:
        result.append({
            'transaction_id': txn.transaction_id,
            'date': txn.date.isoformat(),
            'supplier_name': txn.supplier.supplier_name,
            'amount': float(txn.amount),
            'emissions': float(txn.emission.co2e_kg) if txn.emission else None
        })

    return jsonify({
        'data': result,
        'pagination': {
            'page': page,
            'per_page': per_page
        }
    })
```

**Performance Improvement**:
- Before: 1000+ queries, 5000ms response time
- After: 1 query, 150ms response time (33x faster)

### Example 2: Emissions Calculation Performance

**Problem**:
```python
def calculate_supplier_emissions(supplier_id):
    # Slow: Multiple database queries
    transactions = Transaction.query.filter_by(supplier_id=supplier_id).all()

    total_emissions = 0
    for txn in transactions:
        # Query emission factor for each transaction
        factor = EmissionFactor.query.filter_by(
            product_category=txn.product_category,
            country=txn.country
        ).first()

        if factor:
            emissions = txn.amount * factor.factor_kg_co2e_per_usd
            total_emissions += emissions

    return total_emissions
```

**Optimized Solution**:
```python
def calculate_supplier_emissions(supplier_id):
    # Use SQL aggregation
    query = """
        SELECT SUM(t.amount * COALESCE(ef.factor_kg_co2e_per_usd, 0)) as total_emissions
        FROM transactions t
        LEFT JOIN emission_factors ef
            ON t.product_category = ef.product_category
            AND t.country = ef.country
        WHERE t.supplier_id = :supplier_id
    """

    result = db.session.execute(
        query,
        {'supplier_id': supplier_id}
    ).fetchone()

    return float(result[0]) if result[0] else 0.0
```

**Performance Improvement**:
- Before: 500 queries, 2500ms calculation time
- After: 1 query, 50ms calculation time (50x faster)

### Example 3: Dashboard Aggregation Performance

**Problem**:
```python
@app.route('/api/v1/dashboard/summary')
def get_dashboard_summary():
    # Slow: Multiple separate queries
    total_transactions = Transaction.query.count()
    total_spend = db.session.query(func.sum(Transaction.amount)).scalar()
    total_emissions = db.session.query(func.sum(Emission.co2e_kg)).scalar()

    # By category (N queries)
    category_breakdown = []
    for category in range(1, 16):
        count = Transaction.query.filter_by(ghg_category=category).count()
        emissions = db.session.query(func.sum(Emission.co2e_kg)).join(
            Transaction
        ).filter(Transaction.ghg_category == category).scalar()

        category_breakdown.append({
            'category': category,
            'count': count,
            'emissions': emissions
        })

    return jsonify({
        'total_transactions': total_transactions,
        'total_spend': total_spend,
        'total_emissions': total_emissions,
        'by_category': category_breakdown
    })
```

**Optimized Solution**:
```python
@app.route('/api/v1/dashboard/summary')
@cached(ttl=300, key_prefix='dashboard_summary', cache_instance=cache)
def get_dashboard_summary():
    # Use materialized view
    query = """
        SELECT
            COUNT(*) as total_transactions,
            SUM(amount) as total_spend,
            SUM(emissions_kg) as total_emissions,
            json_agg(
                json_build_object(
                    'category', ghg_category,
                    'count', category_count,
                    'emissions', category_emissions
                )
            ) as category_breakdown
        FROM (
            SELECT
                ghg_category,
                COUNT(*) as category_count,
                SUM(e.co2e_kg) as category_emissions,
                SUM(t.amount) as amount,
                SUM(e.co2e_kg) as emissions_kg
            FROM transactions t
            LEFT JOIN emissions e ON t.transaction_id = e.transaction_id
            GROUP BY ghg_category
        ) subquery
    """

    result = db.session.execute(query).fetchone()

    return jsonify({
        'total_transactions': result[0],
        'total_spend': float(result[1]) if result[1] else 0,
        'total_emissions': float(result[2]) if result[2] else 0,
        'by_category': result[3]
    })
```

**Even Better: Use Materialized View**:
```sql
-- Create materialized view
CREATE MATERIALIZED VIEW dashboard_summary AS
SELECT
    COUNT(*) as total_transactions,
    SUM(t.amount) as total_spend,
    SUM(e.co2e_kg) as total_emissions,
    t.ghg_category,
    COUNT(*) as category_count,
    SUM(e.co2e_kg) as category_emissions
FROM transactions t
LEFT JOIN emissions e ON t.transaction_id = e.transaction_id
GROUP BY t.ghg_category;

-- Refresh periodically
REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_summary;
```

```python
@app.route('/api/v1/dashboard/summary')
def get_dashboard_summary():
    # Query materialized view (instant)
    summary = db.session.query(DashboardSummary).all()

    return jsonify({
        'total_transactions': summary[0].total_transactions,
        'total_spend': float(summary[0].total_spend),
        'total_emissions': float(summary[0].total_emissions),
        'by_category': [
            {
                'category': row.ghg_category,
                'count': row.category_count,
                'emissions': float(row.category_emissions)
            }
            for row in summary
        ]
    })
```

**Performance Improvement**:
- Before: 20+ queries, 3000ms response time
- After: 1 query on materialized view, 10ms response time (300x faster)

---

## Performance Testing

### Load Testing with Locust

**Locust Test Script**:
```python
# locustfile.py
from locust import HttpUser, task, between
import random

class GLVCCIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Login and get auth token"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test@example.com",
            "password": "testpassword"
        })
        self.token = response.json()['token']
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(10)
    def list_transactions(self):
        """List transactions (most common)"""
        page = random.randint(1, 100)
        self.client.get(
            f"/api/v1/transactions?page={page}&per_page=50",
            headers=self.headers
        )

    @task(5)
    def get_supplier(self):
        """Get supplier details"""
        supplier_id = f"SUP-{random.randint(1000, 2000)}"
        self.client.get(
            f"/api/v1/suppliers/{supplier_id}",
            headers=self.headers
        )

    @task(3)
    def get_emissions_summary(self):
        """Get emissions summary"""
        self.client.get(
            "/api/v1/emissions/summary",
            headers=self.headers
        )

    @task(2)
    def create_transaction(self):
        """Create new transaction"""
        self.client.post(
            "/api/v1/transactions",
            headers=self.headers,
            json={
                "transaction_id": f"TXN-TEST-{random.randint(100000, 999999)}",
                "date": "2024-11-07",
                "supplier_id": f"SUP-{random.randint(1000, 2000)}",
                "amount": random.uniform(100, 10000),
                # ... other fields
            }
        )

    @task(1)
    def generate_report(self):
        """Generate report (heavy operation)"""
        self.client.post(
            "/api/v1/reports/generate",
            headers=self.headers,
            json={
                "report_type": "monthly_summary",
                "date_from": "2024-01-01",
                "date_to": "2024-12-31"
            }
        )
```

**Run Load Test**:
```bash
# Install Locust
pip install locust

# Run web interface
locust -f locustfile.py --host=https://api.gl-vcci.com

# Run headless
locust -f locustfile.py \
    --host=https://api.gl-vcci.com \
    --users=100 \
    --spawn-rate=10 \
    --run-time=10m \
    --headless
```

### Apache Bench (ab)

**Basic Load Test**:
```bash
# 1000 requests, 50 concurrent
ab -n 1000 -c 50 -H "Authorization: Bearer TOKEN" \
    https://api.gl-vcci.com/api/v1/transactions

# POST requests with JSON
ab -n 1000 -c 50 -p transaction.json -T application/json \
    -H "Authorization: Bearer TOKEN" \
    https://api.gl-vcci.com/api/v1/transactions
```

### K6 Load Testing

**K6 Test Script**:
```javascript
// k6_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 200 },  // Ramp up to 200 users
        { duration: '5m', target: 200 },  // Stay at 200 users
        { duration: '2m', target: 0 },    // Ramp down to 0 users
    ],
    thresholds: {
        http_req_duration: ['p(95)<500'],  // 95% of requests must complete below 500ms
        http_req_failed: ['rate<0.01'],    // Error rate must be below 1%
    },
};

const API_BASE = 'https://api.gl-vcci.com';
const TOKEN = 'your_auth_token';

export default function () {
    // List transactions
    let res = http.get(`${API_BASE}/api/v1/transactions?page=1`, {
        headers: { 'Authorization': `Bearer ${TOKEN}` },
    });

    check(res, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });

    sleep(1);
}
```

**Run K6 Test**:
```bash
# Install k6
# https://k6.io/docs/getting-started/installation/

# Run test
k6 run k6_test.js

# Run with specific VUs and duration
k6 run --vus 100 --duration 30s k6_test.js

# Output results to JSON
k6 run --out json=results.json k6_test.js
```

---

## Troubleshooting Performance Issues

### Identifying Slow Queries

**pg_stat_statements Analysis**:
```sql
-- Top 20 slowest queries by average time
SELECT
    query,
    calls,
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    ROUND(mean_exec_time::numeric, 2) as avg_time_ms,
    ROUND(stddev_exec_time::numeric, 2) as stddev_time_ms,
    ROUND((100 * total_exec_time / SUM(total_exec_time) OVER ())::numeric, 2) as pct_total_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Queries with most calls
SELECT
    query,
    calls,
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    ROUND(mean_exec_time::numeric, 2) as avg_time_ms
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 20;

-- Queries consuming most total time
SELECT
    query,
    calls,
    ROUND(total_exec_time::numeric, 2) as total_time_ms,
    ROUND(mean_exec_time::numeric, 2) as avg_time_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
```

### Database Connection Issues

**Check Active Connections**:
```sql
-- Current connections
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    state_change,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY query_start;

-- Connection pool status
SELECT
    datname,
    numbackends,
    xact_commit,
    xact_rollback,
    blks_read,
    blks_hit,
    tup_returned,
    tup_fetched,
    tup_inserted,
    tup_updated,
    tup_deleted
FROM pg_stat_database
WHERE datname = 'gl_vcci_db';

-- Blocking queries
SELECT
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

### Memory Issues

**Monitor Memory Usage**:
```bash
# Overall memory usage
free -h

# Per-process memory
ps aux --sort=-%mem | head -n 20

# Python memory profiling
pip install memory_profiler

# Profile Python function
from memory_profiler import profile

@profile
def calculate_emissions():
    # ... function code
    pass
```

**Kubernetes Memory Monitoring**:
```bash
# Pod memory usage
kubectl top pods -n gl-vcci

# Node memory usage
kubectl top nodes

# Detailed pod metrics
kubectl describe pod <pod-name> -n gl-vcci
```

### CPU Bottlenecks

**Identify CPU-Intensive Processes**:
```bash
# Top CPU consumers
top -o %CPU

# Specific to Python
ps aux | grep python | sort -k 3 -r

# CPU profiling with py-spy
py-spy top --pid <PID>
```

---

## Monitoring and Alerting

### Prometheus Alert Rules

**Alert Rules Configuration**:
```yaml
# alerts.yml
groups:
- name: api_alerts
  interval: 30s
  rules:

  # High error rate
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} (threshold: 0.05)"

  # High response time
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "P95 response time is {{ $value }}s (threshold: 0.5s)"

  # High CPU usage
  - alert: HighCPUUsage
    expr: rate(process_cpu_seconds_total[5m]) > 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }} (threshold: 0.8)"

  # High memory usage
  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }} (threshold: 0.9)"

  # Database connection pool exhaustion
  - alert: DatabaseConnectionPoolExhaustion
    expr: db_connection_pool_used / db_connection_pool_size > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Database connection pool near exhaustion"
      description: "Pool usage is {{ $value }} (threshold: 0.9)"

  # Slow database queries
  - alert: SlowDatabaseQueries
    expr: rate(db_query_duration_seconds_sum[5m]) / rate(db_query_duration_seconds_count[5m]) > 0.1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Slow database queries detected"
      description: "Average query time is {{ $value }}s (threshold: 0.1s)"
```

### Grafana Dashboards

**Dashboard JSON** (abbreviated):
```json
{
  "dashboard": {
    "title": "GL-VCCI API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (P50, P95, P99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## Conclusion

Performance optimization is an ongoing process that requires:

1. **Continuous Monitoring**: Track performance metrics
2. **Regular Profiling**: Identify bottlenecks
3. **Incremental Optimization**: Make targeted improvements
4. **Testing**: Verify improvements with load tests
5. **Documentation**: Record optimization decisions

### Key Takeaways

- **Database**: Proper indexing, connection pooling, query optimization
- **API**: Caching, pagination, rate limiting, compression
- **Infrastructure**: Load balancing, resource limits, autoscaling
- **Application**: Profiling, async programming, background tasks
- **Monitoring**: APM tools, metrics, alerts, dashboards

### Next Steps

1. Implement baseline performance monitoring
2. Identify top 3 performance bottlenecks
3. Apply targeted optimizations
4. Measure improvements with load tests
5. Document changes and lessons learned
6. Repeat the process

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Maintained By**: GL-VCCI Platform Team
