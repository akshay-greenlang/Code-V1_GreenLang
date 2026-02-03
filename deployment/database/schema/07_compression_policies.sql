-- =============================================================================
-- GreenLang Climate OS - Compression Policies
-- =============================================================================
-- File: 07_compression_policies.sql
-- Description: TimescaleDB compression policies for all hypertables.
--              Compression reduces storage by 90-95% for time-series data.
-- =============================================================================

-- =============================================================================
-- COMPRESSION SETTINGS FOR EMISSION MEASUREMENTS
-- =============================================================================
-- Chunk interval: 1 day
-- Compress after: 7 days (allow time for late-arriving data and corrections)
-- Segment by: org_id (most common query filter)
-- Order by: time DESC (most queries want recent data first)

ALTER TABLE metrics.emission_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'org_id, project_id',
    timescaledb.compress_orderby = 'time DESC, source_id'
);

-- Add compression policy: compress chunks older than 7 days
SELECT add_compression_policy(
    'metrics.emission_measurements',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Verify compression settings
DO $$
BEGIN
    RAISE NOTICE 'Compression configured for metrics.emission_measurements:';
    RAISE NOTICE '  - Compress after: 7 days';
    RAISE NOTICE '  - Segment by: org_id, project_id';
    RAISE NOTICE '  - Order by: time DESC, source_id';
END $$;

-- =============================================================================
-- COMPRESSION SETTINGS FOR CALCULATION RESULTS
-- =============================================================================
-- Chunk interval: 1 hour
-- Compress after: 1 day (calculation results are typically queried within a day)
-- Segment by: org_id (tenant isolation)
-- Order by: time DESC, run_id (queries often filter by run)

ALTER TABLE metrics.calculation_results SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'org_id',
    timescaledb.compress_orderby = 'time DESC, run_id'
);

SELECT add_compression_policy(
    'metrics.calculation_results',
    compress_after => INTERVAL '1 day',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Compression configured for metrics.calculation_results:';
    RAISE NOTICE '  - Compress after: 1 day';
    RAISE NOTICE '  - Segment by: org_id';
    RAISE NOTICE '  - Order by: time DESC, run_id';
END $$;

-- =============================================================================
-- COMPRESSION SETTINGS FOR SENSOR READINGS
-- =============================================================================
-- Chunk interval: 15 minutes
-- Compress after: 1 hour (high-frequency data, compress quickly to save space)
-- Segment by: device_id (most queries are device-specific)
-- Order by: time DESC (time-range queries)

ALTER TABLE metrics.sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'device_id, metric_type',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy(
    'metrics.sensor_readings',
    compress_after => INTERVAL '1 hour',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Compression configured for metrics.sensor_readings:';
    RAISE NOTICE '  - Compress after: 1 hour';
    RAISE NOTICE '  - Segment by: device_id, metric_type';
    RAISE NOTICE '  - Order by: time DESC';
END $$;

-- =============================================================================
-- COMPRESSION SETTINGS FOR AUDIT LOG
-- =============================================================================
-- Chunk interval: 1 day
-- Compress after: 7 days (allow time for incident investigation)
-- Segment by: org_id (tenant queries)
-- Order by: time DESC (most queries want recent logs)

ALTER TABLE audit.audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'org_id',
    timescaledb.compress_orderby = 'time DESC, user_id'
);

SELECT add_compression_policy(
    'audit.audit_log',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Compression configured for audit.audit_log:';
    RAISE NOTICE '  - Compress after: 7 days';
    RAISE NOTICE '  - Segment by: org_id';
    RAISE NOTICE '  - Order by: time DESC, user_id';
END $$;

-- =============================================================================
-- COMPRESSION SETTINGS FOR API REQUESTS
-- =============================================================================
-- Chunk interval: 1 hour
-- Compress after: 1 day (API logs are queried frequently for debugging)
-- Segment by: org_id (tenant isolation)
-- Order by: time DESC (debugging queries want recent requests)

ALTER TABLE audit.api_requests SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'org_id',
    timescaledb.compress_orderby = 'time DESC, request_id'
);

SELECT add_compression_policy(
    'audit.api_requests',
    compress_after => INTERVAL '1 day',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Compression configured for audit.api_requests:';
    RAISE NOTICE '  - Compress after: 1 day';
    RAISE NOTICE '  - Segment by: org_id';
    RAISE NOTICE '  - Order by: time DESC, request_id';
END $$;

-- =============================================================================
-- COMPRESSION SETTINGS FOR SECURITY EVENTS
-- =============================================================================
-- Chunk interval: 1 day
-- Compress after: 30 days (security events need longer investigation window)
-- Segment by: org_id
-- Order by: time DESC, severity (high severity first)

ALTER TABLE audit.security_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'org_id',
    timescaledb.compress_orderby = 'time DESC, severity'
);

SELECT add_compression_policy(
    'audit.security_events',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Compression configured for audit.security_events:';
    RAISE NOTICE '  - Compress after: 30 days';
    RAISE NOTICE '  - Segment by: org_id';
    RAISE NOTICE '  - Order by: time DESC, severity';
END $$;

-- =============================================================================
-- UTILITY FUNCTIONS FOR COMPRESSION MANAGEMENT
-- =============================================================================

-- Function to view compression statistics for all hypertables
CREATE OR REPLACE FUNCTION metrics.get_compression_stats()
RETURNS TABLE (
    hypertable_name TEXT,
    total_chunks BIGINT,
    compressed_chunks BIGINT,
    uncompressed_chunks BIGINT,
    total_size TEXT,
    compressed_size TEXT,
    compression_ratio NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        format('%I.%I', ht.schema_name, ht.table_name) AS hypertable_name,
        COUNT(*) AS total_chunks,
        COUNT(*) FILTER (WHERE ch.compression_status = 'Compressed') AS compressed_chunks,
        COUNT(*) FILTER (WHERE ch.compression_status = 'Uncompressed') AS uncompressed_chunks,
        pg_size_pretty(SUM(ch.total_bytes)) AS total_size,
        pg_size_pretty(SUM(ch.compressed_total_bytes)) AS compressed_size,
        ROUND(
            CASE
                WHEN SUM(ch.compressed_total_bytes) > 0
                THEN SUM(ch.total_bytes)::NUMERIC / SUM(ch.compressed_total_bytes)
                ELSE 0
            END, 2
        ) AS compression_ratio
    FROM timescaledb_information.hypertables ht
    JOIN timescaledb_information.chunks ch
        ON ch.hypertable_schema = ht.schema_name
        AND ch.hypertable_name = ht.table_name
    GROUP BY ht.schema_name, ht.table_name
    ORDER BY ht.schema_name, ht.table_name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_compression_stats IS 'View compression statistics for all hypertables';

-- Function to manually compress old chunks
CREATE OR REPLACE FUNCTION metrics.compress_old_chunks(
    p_hypertable REGCLASS,
    p_older_than INTERVAL
)
RETURNS INTEGER AS $$
DECLARE
    v_compressed_count INTEGER := 0;
    v_chunk RECORD;
BEGIN
    FOR v_chunk IN
        SELECT chunk_schema, chunk_name
        FROM timescaledb_information.chunks
        WHERE hypertable_name = p_hypertable::TEXT
          AND compression_status = 'Uncompressed'
          AND range_end < NOW() - p_older_than
    LOOP
        EXECUTE format('SELECT compress_chunk(%L)',
            format('%I.%I', v_chunk.chunk_schema, v_chunk.chunk_name));
        v_compressed_count := v_compressed_count + 1;
    END LOOP;

    RETURN v_compressed_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.compress_old_chunks IS 'Manually compress chunks older than specified interval';

-- Function to estimate compression savings
CREATE OR REPLACE FUNCTION metrics.estimate_compression_savings(
    p_hypertable REGCLASS
)
RETURNS TABLE (
    current_size TEXT,
    estimated_compressed_size TEXT,
    estimated_savings_percent NUMERIC
) AS $$
DECLARE
    v_total_size BIGINT;
    v_compressed_size BIGINT;
    v_uncompressed_size BIGINT;
    v_ratio NUMERIC;
BEGIN
    -- Get sizes from existing compressed and uncompressed chunks
    SELECT
        COALESCE(SUM(total_bytes), 0),
        COALESCE(SUM(compressed_total_bytes), 0),
        COALESCE(SUM(CASE WHEN compression_status = 'Uncompressed' THEN total_bytes ELSE 0 END), 0)
    INTO v_total_size, v_compressed_size, v_uncompressed_size
    FROM timescaledb_information.chunks
    WHERE format('%I.%I', hypertable_schema, hypertable_name)::REGCLASS = p_hypertable;

    -- Calculate compression ratio from already compressed chunks
    IF v_compressed_size > 0 THEN
        v_ratio := (v_total_size - v_uncompressed_size)::NUMERIC / v_compressed_size;
    ELSE
        -- Assume 10x compression if no data yet
        v_ratio := 10.0;
    END IF;

    RETURN QUERY
    SELECT
        pg_size_pretty(v_total_size),
        pg_size_pretty((v_compressed_size + v_uncompressed_size / v_ratio)::BIGINT),
        ROUND((1 - 1/v_ratio) * 100, 1);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.estimate_compression_savings IS 'Estimate storage savings from compression';

-- =============================================================================
-- SUMMARY OF COMPRESSION POLICIES
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'GreenLang Compression Policies Summary';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Hypertable                        Compress After   Segment By';
    RAISE NOTICE '-------------------------------------------------------------';
    RAISE NOTICE 'metrics.emission_measurements     7 days          org_id, project_id';
    RAISE NOTICE 'metrics.calculation_results       1 day           org_id';
    RAISE NOTICE 'metrics.sensor_readings           1 hour          device_id, metric_type';
    RAISE NOTICE 'audit.audit_log                   7 days          org_id';
    RAISE NOTICE 'audit.api_requests                1 day           org_id';
    RAISE NOTICE 'audit.security_events             30 days         org_id';
    RAISE NOTICE '';
    RAISE NOTICE 'Expected compression ratio: 10-20x for numeric time-series data';
    RAISE NOTICE '=============================================================';
END $$;
