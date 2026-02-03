# GreenLang Climate OS - Database Schema

Comprehensive TimescaleDB schema for GreenLang Climate OS, optimized for high-volume time-series data from emission measurements, IoT sensors, and audit logs.

## Quick Start

```bash
# Deploy all schema files in order
psql -h localhost -U postgres -d greenlang -f deploy_schema.sql
```

Or deploy individual files:

```bash
for file in 00_extensions.sql 01_schemas.sql 02_core_tables.sql ... 10_roles.sql; do
    psql -h localhost -U postgres -d greenlang -f $file
done
```

## File Overview

| File | Description |
|------|-------------|
| `00_extensions.sql` | PostgreSQL extensions (TimescaleDB, pgcrypto, pg_trgm) |
| `01_schemas.sql` | Database schemas (public, metrics, audit, archive) |
| `02_core_tables.sql` | Core application tables (organizations, users, projects, api_keys) |
| `03_emission_hypertables.sql` | Emission measurement hypertables with 1-day chunks |
| `04_sensor_hypertables.sql` | Sensor reading hypertables with 15-minute chunks |
| `05_audit_hypertables.sql` | Audit log hypertables for compliance |
| `06_continuous_aggregates.sql` | Pre-computed rollups (hourly, daily, monthly) |
| `07_compression_policies.sql` | TimescaleDB compression policies |
| `08_retention_policies.sql` | Data retention and archival policies |
| `09_indexes.sql` | Optimized indexes for common queries |
| `10_roles.sql` | Database roles and permissions |

## Schema Architecture

```
+------------------+     +------------------+     +------------------+
|     public       |     |     metrics      |     |      audit       |
+------------------+     +------------------+     +------------------+
| organizations    |     | emission_sources |     | audit_log        |
| users            |     | emission_meas... |     | api_requests     |
| projects         |     | emission_factors |     | security_events  |
| api_keys         |     | calculation_res..|     |                  |
|                  |     | devices          |     |                  |
|                  |     | sensor_readings  |     |                  |
+------------------+     +------------------+     +------------------+
                                  |
                                  v
                        +------------------+
                        |     archive      |
                        +------------------+
                        | emission_meas... |
                        | audit_log_arch...|
                        | legal_holds      |
                        +------------------+
```

## Hypertables

### Emission Measurements
- **Chunk Interval**: 1 day
- **Compression**: After 7 days
- **Retention**: 7 years
- **Use Case**: Primary emission data storage

### Sensor Readings
- **Chunk Interval**: 15 minutes
- **Compression**: After 1 hour
- **Retention**: 1 year
- **Use Case**: High-frequency IoT sensor data

### Calculation Results
- **Chunk Interval**: 1 hour
- **Compression**: After 1 day
- **Retention**: 2 years
- **Use Case**: Emission calculation outputs

### Audit Log
- **Chunk Interval**: 1 day
- **Compression**: After 7 days
- **Retention**: 10 years
- **Use Case**: Compliance audit trail

### API Requests
- **Chunk Interval**: 1 hour
- **Compression**: After 1 day
- **Retention**: 90 days
- **Use Case**: API monitoring and debugging

## Continuous Aggregates

| Aggregate | Source | Interval | Refresh |
|-----------|--------|----------|---------|
| hourly_emissions | emission_measurements | 1 hour | Every hour |
| daily_emissions | emission_measurements | 1 day | Daily |
| monthly_emissions | emission_measurements | 1 month | Weekly |
| device_statistics_hourly | sensor_readings | 1 hour | Every 15 min |
| device_statistics_daily | sensor_readings | 1 day | Daily |
| api_statistics_hourly | api_requests | 1 hour | Every 15 min |

## Database Roles

| Role | Purpose | Permissions |
|------|---------|-------------|
| `greenlang_app` | Application services | CRUD on all tables |
| `greenlang_readonly` | Analytics/BI | SELECT only |
| `greenlang_admin` | DBA operations | Full access |
| `greenlang_migration` | Schema migrations | DDL + DML |
| `greenlang_backup` | Backup operations | SELECT + replication |

## Row Level Security (RLS)

Multi-tenant tables have RLS enabled. Set organization context before queries:

```sql
-- Set context for session
SELECT set_current_org('org-uuid-here');

-- Or set full user context
SELECT set_current_user_context('user-uuid', 'org-uuid');

-- Clear context before returning connection to pool
SELECT clear_context();
```

## Common Queries

### Recent Emissions by Organization
```sql
SELECT
    time_bucket('1 day', time) AS day,
    scope,
    SUM(emission_value) AS total_emissions
FROM metrics.emission_measurements
WHERE org_id = 'your-org-id'
  AND time > NOW() - INTERVAL '30 days'
GROUP BY day, scope
ORDER BY day DESC;
```

### Use Pre-computed Aggregates for Faster Queries
```sql
-- Use continuous aggregate instead of raw data
SELECT
    bucket AS day,
    scope,
    total_emissions
FROM metrics.daily_emissions
WHERE org_id = 'your-org-id'
  AND bucket > NOW() - INTERVAL '30 days'
ORDER BY bucket DESC;
```

### Device Statistics
```sql
SELECT * FROM metrics.get_device_stats(
    'device-uuid',
    'power_consumption',
    NOW() - INTERVAL '24 hours',
    NOW()
);
```

### Compression Statistics
```sql
SELECT * FROM metrics.get_compression_stats();
```

## Maintenance

### Manual Compression
```sql
SELECT metrics.compress_old_chunks(
    'metrics.sensor_readings',
    INTERVAL '2 hours'
);
```

### Refresh Aggregates
```sql
SELECT metrics.refresh_all_aggregates(
    NOW() - INTERVAL '7 days',
    NOW()
);
```

### Check Retention Deletions
```sql
SELECT * FROM archive.estimate_retention_deletions();
```

### Check Unused Indexes
```sql
SELECT * FROM metrics.get_unused_indexes();
```

## Performance Tuning

### Recommended PostgreSQL Settings

```ini
# Memory
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB

# WAL
wal_level = replica
max_wal_size = 4GB
min_wal_size = 1GB

# Parallelism
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# TimescaleDB
timescaledb.max_background_workers = 8
```

### Index Monitoring

```sql
-- Check index usage
SELECT * FROM metrics.get_index_stats();

-- Find bloated indexes
SELECT
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE schemaname IN ('public', 'metrics', 'audit')
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 20;
```

## Compliance

### Data Retention Summary

| Data Type | Retention | Compliance Requirement |
|-----------|-----------|------------------------|
| Emission Measurements | 7 years | SEC Climate Disclosure, GHG Protocol |
| Audit Logs | 10 years | SOX, SEC |
| Security Events | 7 years | Security Compliance |
| API Requests | 90 days | Operational |
| Sensor Readings | 1 year | Operational |

### Legal Holds

```sql
-- Create a legal hold
INSERT INTO archive.legal_holds (
    hold_name, description, org_id, start_date,
    affected_tables, created_by
) VALUES (
    'Investigation 2024-001',
    'Regulatory investigation',
    'org-uuid',
    '2024-01-01',
    ARRAY['metrics.emission_measurements', 'audit.audit_log'],
    'admin-user-uuid'
);

-- Check if data is under legal hold
SELECT archive.is_under_legal_hold(
    'org-uuid',
    'metrics.emission_measurements',
    '2024-03-15'
);
```

## Backup and Recovery

### Full Backup
```bash
pg_dump -h localhost -U greenlang_backup -Fc greenlang > backup.dump
```

### TimescaleDB-specific Backup
```bash
# Dump with TimescaleDB support
pg_dump -h localhost -U greenlang_backup -Fc \
    --exclude-table-data='_timescaledb_internal.*' \
    greenlang > schema.dump

# Backup chunk data separately
pg_dump -h localhost -U greenlang_backup -Fc \
    --data-only --table='_timescaledb_internal.*' \
    greenlang > chunks.dump
```

## Troubleshooting

### Check Chunk Status
```sql
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    is_compressed
FROM timescaledb_information.chunks
WHERE hypertable_name = 'emission_measurements'
ORDER BY range_start DESC
LIMIT 20;
```

### Check Background Jobs
```sql
SELECT * FROM timescaledb_information.jobs
ORDER BY job_id;
```

### Check Job Errors
```sql
SELECT * FROM timescaledb_information.job_errors
ORDER BY start_time DESC
LIMIT 20;
```
