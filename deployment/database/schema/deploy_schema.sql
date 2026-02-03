-- =============================================================================
-- GreenLang Climate OS - Schema Deployment Script
-- =============================================================================
-- This script deploys all schema files in the correct order.
-- Run with: psql -h localhost -U postgres -d greenlang -f deploy_schema.sql
-- =============================================================================

\echo '============================================================='
\echo 'GreenLang Climate OS - Database Schema Deployment'
\echo '============================================================='
\echo ''

-- Set client encoding
SET client_encoding = 'UTF8';

-- Enable timing for performance tracking
\timing on

-- Transaction wrapper (all or nothing)
BEGIN;

\echo ''
\echo '[1/10] Enabling PostgreSQL extensions...'
\echo '-------------------------------------------------------------'
\i 00_extensions.sql

\echo ''
\echo '[2/10] Creating database schemas...'
\echo '-------------------------------------------------------------'
\i 01_schemas.sql

\echo ''
\echo '[3/10] Creating core application tables...'
\echo '-------------------------------------------------------------'
\i 02_core_tables.sql

\echo ''
\echo '[4/10] Creating emission hypertables...'
\echo '-------------------------------------------------------------'
\i 03_emission_hypertables.sql

\echo ''
\echo '[5/10] Creating sensor hypertables...'
\echo '-------------------------------------------------------------'
\i 04_sensor_hypertables.sql

\echo ''
\echo '[6/10] Creating audit hypertables...'
\echo '-------------------------------------------------------------'
\i 05_audit_hypertables.sql

\echo ''
\echo '[7/10] Creating continuous aggregates...'
\echo '-------------------------------------------------------------'
\i 06_continuous_aggregates.sql

\echo ''
\echo '[8/10] Configuring compression policies...'
\echo '-------------------------------------------------------------'
\i 07_compression_policies.sql

\echo ''
\echo '[9/10] Configuring retention policies...'
\echo '-------------------------------------------------------------'
\i 08_retention_policies.sql

\echo ''
\echo '[10/10] Creating optimized indexes...'
\echo '-------------------------------------------------------------'
\i 09_indexes.sql

\echo ''
\echo '[FINAL] Creating database roles and permissions...'
\echo '-------------------------------------------------------------'
\i 10_roles.sql

-- Commit transaction
COMMIT;

\echo ''
\echo '============================================================='
\echo 'Schema deployment completed successfully!'
\echo '============================================================='
\echo ''

-- Display summary
\echo 'Database Summary:'
\echo '-----------------'

-- Count tables per schema
SELECT
    schemaname AS schema,
    COUNT(*) AS tables
FROM pg_tables
WHERE schemaname IN ('public', 'metrics', 'audit', 'archive')
GROUP BY schemaname
ORDER BY schemaname;

-- List hypertables
\echo ''
\echo 'Hypertables:'
\echo '------------'
SELECT
    hypertable_schema || '.' || hypertable_name AS hypertable,
    num_chunks,
    compression_enabled
FROM timescaledb_information.hypertables
ORDER BY hypertable_schema, hypertable_name;

-- List continuous aggregates
\echo ''
\echo 'Continuous Aggregates:'
\echo '----------------------'
SELECT
    view_schema || '.' || view_name AS aggregate,
    refresh_policy
FROM timescaledb_information.continuous_aggregates
ORDER BY view_schema, view_name;

-- List compression policies
\echo ''
\echo 'Compression Policies:'
\echo '---------------------'
SELECT
    hypertable_schema || '.' || hypertable_name AS hypertable,
    schedule_interval,
    config ->> 'compress_after' AS compress_after
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_compression'
ORDER BY hypertable_schema, hypertable_name;

-- List retention policies
\echo ''
\echo 'Retention Policies:'
\echo '-------------------'
SELECT
    hypertable_schema || '.' || hypertable_name AS hypertable,
    config ->> 'drop_after' AS retention_period
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention'
ORDER BY hypertable_schema, hypertable_name;

-- List roles
\echo ''
\echo 'Database Roles:'
\echo '---------------'
SELECT
    rolname AS role,
    rolconnlimit AS connection_limit,
    rolcanlogin AS can_login,
    rolreplication AS replication
FROM pg_roles
WHERE rolname LIKE 'greenlang_%'
ORDER BY rolname;

\echo ''
\echo '============================================================='
\echo 'IMPORTANT: Set role passwords from secure vault!'
\echo 'Example: ALTER ROLE greenlang_app WITH PASSWORD ''from-vault'';'
\echo '============================================================='
\echo ''

\timing off
