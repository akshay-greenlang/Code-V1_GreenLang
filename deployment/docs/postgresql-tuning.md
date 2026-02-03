# PostgreSQL/TimescaleDB Performance Tuning Guide

## Overview

This guide provides comprehensive documentation for tuning PostgreSQL and TimescaleDB for optimal performance in the GreenLang infrastructure. It covers memory configuration, query optimization, autovacuum tuning, replication setup, and monitoring best practices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Memory Configuration](#memory-configuration)
3. [Connection Management](#connection-management)
4. [WAL Configuration](#wal-configuration)
5. [Query Planner Optimization](#query-planner-optimization)
6. [Parallel Query Execution](#parallel-query-execution)
7. [Autovacuum Tuning](#autovacuum-tuning)
8. [Replication Configuration](#replication-configuration)
9. [TimescaleDB Optimization](#timescaledb-optimization)
10. [Logging and Monitoring](#logging-and-monitoring)
11. [Troubleshooting](#troubleshooting)
12. [Environment-Specific Settings](#environment-specific-settings)

---

## Quick Start

### Using the Settings Calculator

The settings calculator generates optimal PostgreSQL configuration based on your system resources:

```bash
# Basic usage
python calculate-settings.py --ram 32 --cpu 8 --storage ssd

# With specific workload type
python calculate-settings.py --ram 64 --cpu 16 --storage nvme --workload olap

# Using presets
python calculate-settings.py --preset production-large

# Interactive mode
python calculate-settings.py --interactive

# Generate JSON output
python calculate-settings.py --ram 32 --cpu 8 --storage ssd --output json
```

### Available Presets

| Preset | RAM | CPU | Storage | Use Case |
|--------|-----|-----|---------|----------|
| `development` | 4GB | 2 | SSD | Local development |
| `staging` | 16GB | 4 | SSD | QA/Staging environment |
| `production-small` | 32GB | 8 | SSD | Small production |
| `production-medium` | 64GB | 16 | NVMe | Medium production |
| `production-large` | 128GB | 32 | NVMe | Large production |
| `timeseries-small` | 32GB | 8 | NVMe | Time-series workloads |
| `timeseries-large` | 128GB | 32 | NVMe | Large time-series |
| `analytics` | 256GB | 64 | NVMe | Analytics/DW |

---

## Memory Configuration

### Overview

Memory configuration is critical for PostgreSQL performance. The key parameters are:

```
                    ┌─────────────────────────────────────┐
                    │          Total System RAM           │
                    │              (32 GB)                │
                    └─────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
    ┌───────────────┐      ┌─────────────────┐      ┌────────────────┐
    │ shared_buffers│      │   OS/Cache      │      │  Connections   │
    │    (8 GB)     │      │   (20 GB)       │      │   work_mem     │
    │     25%       │      │     ~60%        │      │   (4 GB)       │
    └───────────────┘      └─────────────────┘      └────────────────┘
```

### shared_buffers

The main memory cache for PostgreSQL data pages.

**Formula:** `25% of total RAM` (capped at 8-16GB for most workloads)

```sql
-- Check current shared_buffers
SHOW shared_buffers;

-- Check buffer cache hit ratio (should be > 99%)
SELECT
    sum(heap_blks_read) as heap_read,
    sum(heap_blks_hit) as heap_hit,
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read))::float as ratio
FROM pg_statio_user_tables;
```

| RAM | shared_buffers | Notes |
|-----|----------------|-------|
| 4GB | 1GB | Development |
| 16GB | 4GB | Staging |
| 32GB | 8GB | Production |
| 64GB | 16GB | Large production |
| 128GB+ | 32GB | Diminishing returns beyond 32GB |

### effective_cache_size

Hint to the query planner about available memory for disk caching.

**Formula:** `75% of total RAM`

```sql
-- This affects query planning, not memory allocation
SHOW effective_cache_size;
```

### work_mem

Memory for sort operations and hash tables (per operation, not per connection).

**Formula:** `(Total RAM - shared_buffers) / (max_connections * 2 * workload_factor)`

```sql
-- Check current work_mem
SHOW work_mem;

-- Monitor temporary file usage (indicates work_mem is too low)
SELECT
    datname,
    temp_files,
    pg_size_pretty(temp_bytes) as temp_size
FROM pg_stat_database
WHERE temp_files > 0;
```

**Workload Factors:**
- OLTP: 4 (many concurrent small queries)
- Mixed: 2
- OLAP: 1 (fewer concurrent large queries)

### maintenance_work_mem

Memory for maintenance operations (VACUUM, CREATE INDEX, etc.).

**Formula:** `5% of total RAM`, max 2GB

```sql
-- Temporarily increase for large index builds
SET maintenance_work_mem = '4GB';
CREATE INDEX CONCURRENTLY idx_large ON large_table(column);
RESET maintenance_work_mem;
```

---

## Connection Management

### max_connections

Maximum concurrent connections to the database.

```
                    ┌─────────────────────────────┐
                    │      Connection Pool        │
                    │    (PgBouncer/pgpool)       │
                    │        1000 clients         │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │        PostgreSQL           │
                    │    max_connections = 200    │
                    │   (actual connections)      │
                    └─────────────────────────────┘
```

**Guideline:** Each connection uses ~5-10MB of memory.

```sql
-- Check current connections
SELECT
    count(*) as total,
    state,
    usename
FROM pg_stat_activity
GROUP BY state, usename;

-- Connection utilization
SELECT
    max_conn,
    used,
    res_for_super,
    max_conn - used - res_for_super as available
FROM (
    SELECT
        setting::int as max_conn,
        (SELECT count(*) FROM pg_stat_activity) as used,
        (SELECT setting::int FROM pg_settings WHERE name = 'superuser_reserved_connections') as res_for_super
    FROM pg_settings
    WHERE name = 'max_connections'
) t;
```

### Connection Pooling Recommendation

For production, use a connection pooler:

```yaml
# PgBouncer configuration example
[databases]
greenlang = host=postgresql-prod port=5432 dbname=greenlang

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = scram-sha-256
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 3
```

---

## WAL Configuration

### Understanding WAL

Write-Ahead Logging ensures data durability and enables replication.

```
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Write   │───▶│   WAL    │───▶│ Checkpoint│
   │ Request  │    │ Buffers  │    │  to Disk  │
   └──────────┘    └────┬─────┘    └───────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  WAL Files   │───▶ Replication
                 │ (pg_wal/)    │───▶ PITR
                 └──────────────┘
```

### Key Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `wal_level` | replica | Required for replication |
| `wal_buffers` | 64MB | 3% of shared_buffers |
| `checkpoint_timeout` | 15min | Time between checkpoints |
| `checkpoint_completion_target` | 0.9 | Spread checkpoint I/O |
| `max_wal_size` | 8GB | Maximum WAL before checkpoint |
| `min_wal_size` | 2GB | Minimum WAL to retain |

### Checkpoint Tuning

```sql
-- Monitor checkpoint activity
SELECT * FROM pg_stat_bgwriter;

-- Check if checkpoints are being forced
SELECT
    checkpoints_timed,
    checkpoints_req,
    checkpoints_req / (checkpoints_timed + checkpoints_req)::float as forced_ratio
FROM pg_stat_bgwriter;
-- forced_ratio should be < 0.1 (10%)
```

---

## Query Planner Optimization

### Storage-Based Settings

| Storage Type | random_page_cost | effective_io_concurrency |
|-------------|------------------|--------------------------|
| HDD | 4.0 | 2 |
| SSD | 1.1 | 200 |
| NVMe | 1.0 | 256 |
| Cloud SSD | 1.1 | 200 |

### Statistics Collection

```sql
-- Increase statistics target for important columns
ALTER TABLE sensor_data ALTER COLUMN device_id SET STATISTICS 500;

-- Analyze specific table
ANALYZE sensor_data;

-- Check statistics quality
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE tablename = 'sensor_data';
```

### Query Plan Analysis

```sql
-- Analyze query execution
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM sensor_data
WHERE time > now() - interval '1 day';

-- Check for sequential scans on large tables
SELECT
    schemaname,
    relname,
    seq_scan,
    idx_scan,
    seq_scan / GREATEST(idx_scan, 1)::float as seq_to_idx_ratio
FROM pg_stat_user_tables
WHERE seq_scan > 1000
ORDER BY seq_to_idx_ratio DESC;
```

---

## Parallel Query Execution

### Configuration

```sql
-- Enable parallel query (default settings)
max_parallel_workers_per_gather = 4  -- Workers per query
max_parallel_workers = 8             -- Total parallel workers
max_parallel_maintenance_workers = 4 -- For CREATE INDEX, etc.
```

### When Parallel Query Triggers

Parallel query is used when:
- Table size > `min_parallel_table_scan_size` (8MB)
- Index size > `min_parallel_index_scan_size` (512kB)
- Estimated cost > `parallel_setup_cost` (1000)

```sql
-- Force parallel query for testing
SET force_parallel_mode = on;
EXPLAIN (ANALYZE) SELECT count(*) FROM large_table;
RESET force_parallel_mode;
```

---

## Autovacuum Tuning

### Understanding Autovacuum

Autovacuum maintains table health by:
1. Reclaiming space from dead tuples
2. Updating statistics (ANALYZE)
3. Preventing transaction ID wraparound

### Configuration Matrix

| Setting | OLTP | Mixed | OLAP |
|---------|------|-------|------|
| `autovacuum_vacuum_scale_factor` | 0.01 | 0.02 | 0.1 |
| `autovacuum_analyze_scale_factor` | 0.005 | 0.01 | 0.05 |
| `autovacuum_vacuum_cost_delay` | 2ms | 5ms | 10ms |
| `autovacuum_vacuum_cost_limit` | 1000 | 500 | 200 |
| `autovacuum_max_workers` | 4 | 3 | 2 |

### Per-Table Tuning

```sql
-- High-write table (aggressive vacuum)
ALTER TABLE high_write_table SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_vacuum_threshold = 50
);

-- Large, rarely-updated table (less aggressive)
ALTER TABLE large_archive SET (
    autovacuum_vacuum_scale_factor = 0.2,
    autovacuum_analyze_scale_factor = 0.1
);
```

### Monitoring Autovacuum

```sql
-- Tables needing vacuum
SELECT
    schemaname,
    relname,
    n_live_tup,
    n_dead_tup,
    n_dead_tup / GREATEST(n_live_tup, 1)::float as dead_ratio,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;

-- Currently running autovacuum
SELECT
    pid,
    datname,
    relid::regclass as table_name,
    phase,
    heap_blks_total,
    heap_blks_scanned,
    heap_blks_vacuumed
FROM pg_stat_progress_vacuum;
```

---

## Replication Configuration

### Streaming Replication Setup

```
   ┌────────────────┐         ┌────────────────┐
   │    Primary     │────────▶│   Standby 1    │
   │  (Read/Write)  │   WAL   │  (Read Only)   │
   └────────┬───────┘  Stream └────────────────┘
            │
            │ WAL Stream
            ▼
   ┌────────────────┐
   │   Standby 2    │
   │  (Read Only)   │
   └────────────────┘
```

### Configuration

**Primary:**
```
max_wal_senders = 10
max_replication_slots = 10
wal_keep_size = 4GB
synchronous_commit = on
synchronous_standby_names = 'FIRST 1 (standby1,standby2)'
```

**Standby:**
```
hot_standby = on
hot_standby_feedback = on
primary_conninfo = 'host=primary port=5432 user=replicator'
```

### Monitoring Replication

```sql
-- On Primary: Check replication status
SELECT
    client_addr,
    application_name,
    state,
    sync_state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) as lag_bytes,
    pg_size_pretty(pg_wal_lsn_diff(sent_lsn, replay_lsn)) as lag_size
FROM pg_stat_replication;

-- On Standby: Check receiver status
SELECT
    status,
    received_lsn,
    latest_end_lsn,
    pg_wal_lsn_diff(received_lsn, latest_end_lsn) as lag
FROM pg_stat_wal_receiver;
```

---

## TimescaleDB Optimization

### Hypertable Configuration

```sql
-- Create optimized hypertable
SELECT create_hypertable('sensor_data', 'time',
    chunk_time_interval => INTERVAL '1 day',
    create_default_indexes => TRUE
);

-- Configure compression
ALTER TABLE sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policy
SELECT add_compression_policy('sensor_data', INTERVAL '7 days');

-- Add retention policy
SELECT add_retention_policy('sensor_data', INTERVAL '90 days');
```

### Continuous Aggregates

```sql
-- Create continuous aggregate
CREATE MATERIALIZED VIEW sensor_data_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    device_id,
    AVG(value) AS avg_value,
    MAX(value) AS max_value,
    MIN(value) AS min_value,
    COUNT(*) AS sample_count
FROM sensor_data
GROUP BY bucket, device_id
WITH NO DATA;

-- Add refresh policy
SELECT add_continuous_aggregate_policy('sensor_data_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

### Chunk Management

```sql
-- View chunk information
SELECT
    hypertable_schema,
    hypertable_name,
    chunk_schema,
    chunk_name,
    range_start,
    range_end,
    is_compressed,
    pg_size_pretty(total_bytes) as size
FROM timescaledb_information.chunks
WHERE hypertable_name = 'sensor_data'
ORDER BY range_start DESC
LIMIT 10;

-- Manually compress chunks
SELECT compress_chunk(chunk, if_not_compressed => true)
FROM show_chunks('sensor_data', older_than => INTERVAL '7 days') AS chunk;
```

---

## Logging and Monitoring

### Essential Metrics to Monitor

| Metric | Query | Threshold |
|--------|-------|-----------|
| Connection usage | `SELECT count(*) FROM pg_stat_activity` | < 80% max_connections |
| Cache hit ratio | See query below | > 99% |
| Replication lag | See query below | < 1MB / 1s |
| Dead tuples | `SELECT sum(n_dead_tup) FROM pg_stat_user_tables` | < 10% of live |
| Transaction ID age | See query below | < 500M |

### Cache Hit Ratio

```sql
SELECT
    'index hit ratio' as name,
    sum(idx_blks_hit) / nullif(sum(idx_blks_hit + idx_blks_read), 0) as ratio
FROM pg_statio_user_indexes
UNION ALL
SELECT
    'table hit ratio',
    sum(heap_blks_hit) / nullif(sum(heap_blks_hit + heap_blks_read), 0)
FROM pg_statio_user_tables;
```

### Transaction ID Age Monitoring

```sql
SELECT
    datname,
    age(datfrozenxid) as xid_age,
    CASE
        WHEN age(datfrozenxid) > 1000000000 THEN 'CRITICAL'
        WHEN age(datfrozenxid) > 500000000 THEN 'WARNING'
        ELSE 'OK'
    END as status
FROM pg_database
ORDER BY age(datfrozenxid) DESC;
```

### Slow Query Analysis

```sql
-- Top 10 slowest queries (requires pg_stat_statements)
SELECT
    substr(query, 1, 100) as query_snippet,
    calls,
    round(total_exec_time::numeric, 2) as total_time_ms,
    round(mean_exec_time::numeric, 2) as mean_time_ms,
    round(stddev_exec_time::numeric, 2) as stddev_ms,
    rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;
```

---

## Troubleshooting

### Common Issues and Solutions

#### High CPU Usage

1. Check for inefficient queries:
```sql
SELECT * FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start;
```

2. Look for missing indexes:
```sql
SELECT
    schemaname || '.' || relname as table,
    seq_scan,
    idx_scan,
    seq_tup_read / GREATEST(seq_scan, 1) as avg_seq_read
FROM pg_stat_user_tables
WHERE seq_scan > 100 AND idx_scan < seq_scan
ORDER BY seq_tup_read DESC;
```

#### Memory Pressure

1. Check work_mem usage:
```sql
SELECT * FROM pg_stat_database WHERE temp_files > 0;
```

2. Monitor shared buffer usage:
```sql
SELECT
    c.relname,
    count(*) * 8192 as buffered_bytes,
    pg_size_pretty(count(*) * 8192) as buffered_size
FROM pg_buffercache b
JOIN pg_class c ON b.relfilenode = pg_relation_filenode(c.oid)
GROUP BY c.relname
ORDER BY buffered_bytes DESC
LIMIT 20;
```

#### Lock Contention

```sql
-- Find blocking queries
SELECT
    blocked.pid AS blocked_pid,
    blocked.query AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.query AS blocking_query
FROM pg_stat_activity blocked
JOIN pg_locks blocked_locks ON blocked.pid = blocked_locks.pid
JOIN pg_locks blocking_locks ON blocked_locks.locktype = blocking_locks.locktype
    AND blocked_locks.relation = blocking_locks.relation
    AND blocked_locks.pid != blocking_locks.pid
    AND blocking_locks.granted
JOIN pg_stat_activity blocking ON blocking_locks.pid = blocking.pid
WHERE NOT blocked_locks.granted;
```

#### Replication Lag

```sql
-- Check WAL generation rate vs replay rate
SELECT
    pg_current_wal_lsn() as current_lsn,
    pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes,
    pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)) as lag_size
FROM pg_stat_replication;
```

---

## Environment-Specific Settings

### Development

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_connections` | 50 | Limited local use |
| `shared_buffers` | 1GB | Small dataset |
| `work_mem` | 64MB | Debug queries |
| `log_statement` | 'all' | Full visibility |
| `synchronous_commit` | off | Faster restarts |

### Staging

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_connections` | 100 | Test concurrency |
| `shared_buffers` | 4GB | Match prod ratio |
| `work_mem` | 128MB | Test query plans |
| `log_min_duration_statement` | 500 | Catch slow queries |
| `synchronous_commit` | on | Test prod behavior |

### Production

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_connections` | 200 | Production load |
| `shared_buffers` | 8GB | Optimal caching |
| `work_mem` | 256MB | Complex queries |
| `log_min_duration_statement` | 1000 | Alert on slow |
| `synchronous_commit` | on | Data durability |

---

## References

- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PgTune](https://pgtune.leopard.in.ua/)
- [PostgreSQL Wiki - Tuning](https://wiki.postgresql.org/wiki/Tuning_Your_PostgreSQL_Server)

---

## Configuration Files

The following configuration files are available in the `deployment/database/config/` directory:

| File | Purpose |
|------|---------|
| `postgresql-tuning.conf` | Base PostgreSQL configuration |
| `timescaledb-tuning.conf` | TimescaleDB-specific settings |
| `replication-tuning.conf` | Streaming replication configuration |
| `logging-tuning.conf` | Logging and monitoring settings |

## Kubernetes ConfigMaps

Environment-specific Kubernetes configurations are in `deployment/database/kubernetes/`:

| File | Environment | Resources |
|------|-------------|-----------|
| `configmap-dev.yaml` | Development | 4GB RAM, 2 CPU |
| `configmap-staging.yaml` | Staging | 16GB RAM, 4 CPU |
| `configmap-prod.yaml` | Production | 32GB+ RAM, 8+ CPU |

---

*Last Updated: 2026-02-03*
*GreenLang DevOps Team*
