-- =============================================================================
-- V002: TimescaleDB Setup and Configuration
-- =============================================================================
-- Description: Enables TimescaleDB extension, creates hypertables,
--              configures compression and retention policies.
-- Author: GreenLang Data Integration Team
-- Requires: TimescaleDB 2.11+
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Enable TimescaleDB Extension
-- -----------------------------------------------------------------------------

-- Create extension (requires superuser or extension owner privileges)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Verify TimescaleDB version
DO $$
DECLARE
    ts_version TEXT;
BEGIN
    SELECT extversion INTO ts_version
    FROM pg_extension
    WHERE extname = 'timescaledb';

    IF ts_version IS NULL THEN
        RAISE EXCEPTION 'TimescaleDB extension not installed';
    END IF;

    RAISE NOTICE 'TimescaleDB version: %', ts_version;
END $$;

-- -----------------------------------------------------------------------------
-- TimescaleDB License and Telemetry Configuration
-- -----------------------------------------------------------------------------

-- Disable telemetry for data privacy (optional)
SELECT set_config('timescaledb.telemetry_level', 'off', false);

-- -----------------------------------------------------------------------------
-- Helper Function: Safe Hypertable Creation
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.create_hypertable_if_not_exists(
    p_table_name TEXT,
    p_time_column TEXT,
    p_chunk_time_interval INTERVAL DEFAULT INTERVAL '1 day',
    p_partitioning_column TEXT DEFAULT NULL,
    p_number_partitions INTEGER DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    is_hypertable BOOLEAN;
BEGIN
    -- Check if already a hypertable
    SELECT EXISTS (
        SELECT 1
        FROM timescaledb_information.hypertables
        WHERE hypertable_name = p_table_name
    ) INTO is_hypertable;

    IF NOT is_hypertable THEN
        IF p_partitioning_column IS NOT NULL AND p_number_partitions IS NOT NULL THEN
            -- Create with space partitioning
            PERFORM create_hypertable(
                p_table_name,
                p_time_column,
                p_partitioning_column,
                p_number_partitions,
                chunk_time_interval => p_chunk_time_interval,
                if_not_exists => TRUE,
                migrate_data => TRUE
            );
        ELSE
            -- Create with time partitioning only
            PERFORM create_hypertable(
                p_table_name,
                p_time_column,
                chunk_time_interval => p_chunk_time_interval,
                if_not_exists => TRUE,
                migrate_data => TRUE
            );
        END IF;

        RAISE NOTICE 'Created hypertable: %', p_table_name;
    ELSE
        RAISE NOTICE 'Table % is already a hypertable', p_table_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Convert Audit Log to Hypertable
-- -----------------------------------------------------------------------------

-- Convert existing audit_log table to hypertable
SELECT create_hypertable(
    'audit.audit_log',
    'performed_at',
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Set chunk time interval based on data volume expectations
SELECT set_chunk_time_interval('audit.audit_log', INTERVAL '${chunk_interval}');

-- -----------------------------------------------------------------------------
-- Base Metrics Tables (to be converted to hypertables)
-- -----------------------------------------------------------------------------

-- Time-series metrics base table
CREATE TABLE IF NOT EXISTS metrics.time_series_metrics (
    time TIMESTAMPTZ NOT NULL,
    organization_id UUID NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,

    PRIMARY KEY (time, organization_id, metric_name)
);

-- Convert to hypertable with space partitioning
SELECT create_hypertable(
    'metrics.time_series_metrics',
    'time',
    'organization_id',
    4,  -- Number of space partitions
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- System Metrics Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics.system_metrics (
    time TIMESTAMPTZ NOT NULL,
    host VARCHAR(255) NOT NULL,
    service VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,

    PRIMARY KEY (time, host, service, metric_type)
);

SELECT create_hypertable(
    'metrics.system_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Compression Policies
-- -----------------------------------------------------------------------------

-- Enable compression on audit_log
ALTER TABLE audit.audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'table_name',
    timescaledb.compress_orderby = 'performed_at DESC, id'
);

-- Enable compression on time_series_metrics
ALTER TABLE metrics.time_series_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = '${compression_segmentby}',
    timescaledb.compress_orderby = '${compression_orderby}'
);

-- Enable compression on system_metrics
ALTER TABLE metrics.system_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'host, service',
    timescaledb.compress_orderby = 'time DESC'
);

-- Add compression policies
-- Compress audit_log chunks older than 7 days
SELECT add_compression_policy(
    'audit.audit_log',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

-- Compress time_series_metrics chunks older than 7 days
SELECT add_compression_policy(
    'metrics.time_series_metrics',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

-- Compress system_metrics chunks older than 1 day
SELECT add_compression_policy(
    'metrics.system_metrics',
    INTERVAL '1 day',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Retention Policies
-- -----------------------------------------------------------------------------

-- Create retention policy helper function
CREATE OR REPLACE FUNCTION public.add_retention_policy_safe(
    p_table_name REGCLASS,
    p_retention_interval INTERVAL
) RETURNS VOID AS $$
BEGIN
    -- Remove existing policy if any
    BEGIN
        PERFORM remove_retention_policy(p_table_name, if_exists => TRUE);
    EXCEPTION WHEN OTHERS THEN
        NULL; -- Ignore errors
    END;

    -- Add new retention policy
    PERFORM add_retention_policy(
        p_table_name,
        p_retention_interval,
        if_not_exists => TRUE
    );

    RAISE NOTICE 'Added retention policy for % with interval %', p_table_name, p_retention_interval;
END;
$$ LANGUAGE plpgsql;

-- Set retention policies based on environment
-- Note: ${retention_days} is a placeholder from flyway.conf

-- Audit log: Keep for retention period (90 days default)
SELECT add_retention_policy(
    'audit.audit_log',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- Time series metrics: Keep for 1 year
SELECT add_retention_policy(
    'metrics.time_series_metrics',
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- System metrics: Keep for 30 days
SELECT add_retention_policy(
    'metrics.system_metrics',
    INTERVAL '30 days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Chunk Management Functions
-- -----------------------------------------------------------------------------

-- Function to get chunk information
CREATE OR REPLACE FUNCTION public.get_chunk_info(p_hypertable TEXT)
RETURNS TABLE (
    chunk_name TEXT,
    chunk_schema TEXT,
    range_start TIMESTAMPTZ,
    range_end TIMESTAMPTZ,
    is_compressed BOOLEAN,
    total_bytes BIGINT,
    compressed_bytes BIGINT,
    compression_ratio NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.chunk_name::TEXT,
        c.chunk_schema::TEXT,
        c.range_start::TIMESTAMPTZ,
        c.range_end::TIMESTAMPTZ,
        c.is_compressed,
        cs.total_bytes,
        cs.compressed_total_bytes,
        CASE
            WHEN cs.total_bytes > 0 THEN
                ROUND((1 - cs.compressed_total_bytes::NUMERIC / cs.total_bytes) * 100, 2)
            ELSE 0
        END AS compression_ratio
    FROM timescaledb_information.chunks c
    LEFT JOIN timescaledb_information.compressed_chunk_stats cs
        ON c.chunk_name = cs.chunk_name
    WHERE c.hypertable_name = p_hypertable
    ORDER BY c.range_start DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to manually compress old chunks
CREATE OR REPLACE FUNCTION public.compress_chunks_older_than(
    p_hypertable REGCLASS,
    p_older_than INTERVAL
) RETURNS INTEGER AS $$
DECLARE
    chunks_compressed INTEGER := 0;
    chunk_record RECORD;
BEGIN
    FOR chunk_record IN
        SELECT chunk_schema, chunk_name
        FROM timescaledb_information.chunks
        WHERE hypertable_name = p_hypertable::TEXT
          AND range_end < NOW() - p_older_than
          AND NOT is_compressed
    LOOP
        BEGIN
            PERFORM compress_chunk(
                format('%I.%I', chunk_record.chunk_schema, chunk_record.chunk_name)
            );
            chunks_compressed := chunks_compressed + 1;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Failed to compress chunk %: %', chunk_record.chunk_name, SQLERRM;
        END;
    END LOOP;

    RETURN chunks_compressed;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Indexes for Hypertables
-- -----------------------------------------------------------------------------

-- Time series metrics indexes
CREATE INDEX IF NOT EXISTS idx_time_series_metrics_org
    ON metrics.time_series_metrics (organization_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_time_series_metrics_name
    ON metrics.time_series_metrics (metric_name, time DESC);

CREATE INDEX IF NOT EXISTS idx_time_series_metrics_tags
    ON metrics.time_series_metrics USING gin(tags jsonb_path_ops);

-- System metrics indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_service
    ON metrics.system_metrics (service, time DESC);

CREATE INDEX IF NOT EXISTS idx_system_metrics_host
    ON metrics.system_metrics (host, time DESC);

CREATE INDEX IF NOT EXISTS idx_system_metrics_labels
    ON metrics.system_metrics USING gin(labels jsonb_path_ops);

-- -----------------------------------------------------------------------------
-- Background Workers Configuration
-- -----------------------------------------------------------------------------

-- Ensure background workers are enabled for compression and retention
-- These settings should be in postgresql.conf:
-- timescaledb.max_background_workers = 8
-- timescaledb.background_worker_quota = 4

-- Verify job scheduler is running
DO $$
DECLARE
    job_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO job_count
    FROM timescaledb_information.jobs
    WHERE scheduled = true;

    RAISE NOTICE 'Active TimescaleDB jobs: %', job_count;
END $$;

-- -----------------------------------------------------------------------------
-- Monitoring Views
-- -----------------------------------------------------------------------------

-- View for compression statistics
CREATE OR REPLACE VIEW public.compression_stats AS
SELECT
    h.hypertable_schema,
    h.hypertable_name,
    h.compression_enabled,
    COUNT(*) FILTER (WHERE c.is_compressed) AS compressed_chunks,
    COUNT(*) FILTER (WHERE NOT c.is_compressed) AS uncompressed_chunks,
    COALESCE(SUM(cs.total_bytes), 0) AS total_bytes_before,
    COALESCE(SUM(cs.compressed_total_bytes), 0) AS total_bytes_after,
    CASE
        WHEN COALESCE(SUM(cs.total_bytes), 0) > 0 THEN
            ROUND(
                (1 - COALESCE(SUM(cs.compressed_total_bytes), 0)::NUMERIC /
                COALESCE(SUM(cs.total_bytes), 1)) * 100, 2
            )
        ELSE 0
    END AS compression_ratio_pct
FROM timescaledb_information.hypertables h
LEFT JOIN timescaledb_information.chunks c
    ON h.hypertable_name = c.hypertable_name
LEFT JOIN timescaledb_information.compressed_chunk_stats cs
    ON c.chunk_name = cs.chunk_name
GROUP BY h.hypertable_schema, h.hypertable_name, h.compression_enabled;

-- View for retention policy status
CREATE OR REPLACE VIEW public.retention_policy_status AS
SELECT
    j.hypertable_schema,
    j.hypertable_name,
    j.schedule_interval,
    j.config->>'drop_after' AS retention_interval,
    js.last_run_started_at,
    js.last_successful_finish,
    js.next_start,
    js.total_runs,
    js.total_successes,
    js.total_failures
FROM timescaledb_information.jobs j
LEFT JOIN timescaledb_information.job_stats js ON j.job_id = js.job_id
WHERE j.proc_name = 'policy_retention';

-- -----------------------------------------------------------------------------
-- Documentation
-- -----------------------------------------------------------------------------

COMMENT ON TABLE metrics.time_series_metrics IS 'Generic time-series metrics storage with organization-level partitioning';
COMMENT ON TABLE metrics.system_metrics IS 'Infrastructure and application metrics for monitoring';

COMMENT ON FUNCTION public.get_chunk_info(TEXT) IS 'Returns detailed chunk information for a hypertable including compression stats';
COMMENT ON FUNCTION public.compress_chunks_older_than(REGCLASS, INTERVAL) IS 'Manually compress chunks older than specified interval';

COMMENT ON VIEW public.compression_stats IS 'Shows compression statistics for all hypertables';
COMMENT ON VIEW public.retention_policy_status IS 'Shows status of retention policies across hypertables';
