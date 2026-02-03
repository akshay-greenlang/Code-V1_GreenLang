-- =============================================================================
-- GreenLang Climate OS - Retention Policies
-- =============================================================================
-- File: 08_retention_policies.sql
-- Description: Data retention policies for all hypertables. Defines how long
--              data is kept before automatic deletion or archival.
-- =============================================================================

-- =============================================================================
-- RETENTION POLICY OVERVIEW
-- =============================================================================
--
-- Table                          Retention    Reason
-- ----------------------------   ---------    --------------------------------
-- emission_measurements          7 years      GHG Protocol verification, SEC
-- calculation_results            2 years      Audit trail for calculations
-- sensor_readings                1 year       Operational data, high volume
-- audit_log                      10 years     Compliance (SOX, SEC, GDPR)
-- api_requests                   90 days      Debugging and monitoring
-- security_events                7 years      Security compliance
--
-- =============================================================================

-- =============================================================================
-- EMISSION MEASUREMENTS - 7 Year Retention
-- =============================================================================
-- Required for:
-- - GHG Protocol verification (5 years recommended)
-- - SEC climate disclosure requirements (7 years)
-- - SOX compliance (7 years)
-- - CDP/SBTi reporting (historical baselines)

SELECT add_retention_policy(
    'metrics.emission_measurements',
    drop_after => INTERVAL '7 years',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for metrics.emission_measurements: 7 years';
    RAISE NOTICE '  Reason: GHG Protocol verification, SEC climate disclosure';
END $$;

-- =============================================================================
-- CALCULATION RESULTS - 2 Year Retention
-- =============================================================================
-- Calculation results can be regenerated if needed.
-- Keep 2 years for audit trail and debugging.

SELECT add_retention_policy(
    'metrics.calculation_results',
    drop_after => INTERVAL '2 years',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for metrics.calculation_results: 2 years';
    RAISE NOTICE '  Reason: Audit trail, calculations can be regenerated';
END $$;

-- =============================================================================
-- SENSOR READINGS - 1 Year Retention
-- =============================================================================
-- High-volume data. Aggregates are kept longer via continuous aggregates.
-- Raw readings needed for:
-- - Anomaly investigation
-- - Recalculation of emissions
-- - Device calibration verification

SELECT add_retention_policy(
    'metrics.sensor_readings',
    drop_after => INTERVAL '1 year',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for metrics.sensor_readings: 1 year';
    RAISE NOTICE '  Reason: High volume, aggregates available for longer periods';
END $$;

-- =============================================================================
-- AUDIT LOG - 10 Year Retention
-- =============================================================================
-- Required for:
-- - SOX compliance (7 years)
-- - SEC requirements (7 years)
-- - GDPR (varies, but audit logs often exempt)
-- - Legal hold requirements
-- Adding 3 years buffer for safety.

SELECT add_retention_policy(
    'audit.audit_log',
    drop_after => INTERVAL '10 years',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for audit.audit_log: 10 years';
    RAISE NOTICE '  Reason: SOX, SEC, legal compliance requirements';
END $$;

-- =============================================================================
-- API REQUESTS - 90 Day Retention
-- =============================================================================
-- Used for:
-- - Debugging
-- - Performance monitoring
-- - Usage analytics (billing)
-- 90 days is sufficient for these purposes.

SELECT add_retention_policy(
    'audit.api_requests',
    drop_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for audit.api_requests: 90 days';
    RAISE NOTICE '  Reason: Debugging, monitoring, billing analytics';
END $$;

-- =============================================================================
-- SECURITY EVENTS - 7 Year Retention
-- =============================================================================
-- Required for:
-- - Security incident investigation
-- - Compliance audits
-- - Forensic analysis

SELECT add_retention_policy(
    'audit.security_events',
    drop_after => INTERVAL '7 years',
    if_not_exists => TRUE
);

DO $$
BEGIN
    RAISE NOTICE 'Retention policy for audit.security_events: 7 years';
    RAISE NOTICE '  Reason: Security compliance, incident investigation';
END $$;

-- =============================================================================
-- ARCHIVE FUNCTIONS FOR COMPLIANCE DATA
-- =============================================================================
-- Some data may need to be archived rather than deleted for compliance.
-- These functions support moving data to the archive schema before deletion.

-- Function to archive emission measurements before deletion
CREATE OR REPLACE FUNCTION archive.archive_emission_measurements(
    p_older_than INTERVAL
)
RETURNS BIGINT AS $$
DECLARE
    v_archived_count BIGINT;
    v_cutoff_time TIMESTAMPTZ;
BEGIN
    v_cutoff_time := NOW() - p_older_than;

    -- Insert into archive table
    INSERT INTO archive.emission_measurements_archive
    SELECT * FROM metrics.emission_measurements
    WHERE time < v_cutoff_time;

    GET DIAGNOSTICS v_archived_count = ROW_COUNT;

    RAISE NOTICE 'Archived % emission measurement records older than %',
        v_archived_count, p_older_than;

    RETURN v_archived_count;
END;
$$ LANGUAGE plpgsql;

-- Create archive table for emission measurements (cold storage)
CREATE TABLE IF NOT EXISTS archive.emission_measurements_archive (
    LIKE metrics.emission_measurements INCLUDING ALL
);

-- Partition archive by year for efficient storage
-- Note: This is a regular partitioned table, not a hypertable
-- because archive data is write-once, read-rarely

COMMENT ON TABLE archive.emission_measurements_archive
    IS 'Archived emission measurements for long-term compliance storage';

-- Function to archive audit logs
CREATE OR REPLACE FUNCTION archive.archive_audit_logs(
    p_older_than INTERVAL
)
RETURNS BIGINT AS $$
DECLARE
    v_archived_count BIGINT;
    v_cutoff_time TIMESTAMPTZ;
BEGIN
    v_cutoff_time := NOW() - p_older_than;

    -- Insert into archive table
    INSERT INTO archive.audit_log_archive
    SELECT * FROM audit.audit_log
    WHERE time < v_cutoff_time
      AND compliance_relevant = true;  -- Only archive compliance-relevant logs

    GET DIAGNOSTICS v_archived_count = ROW_COUNT;

    RAISE NOTICE 'Archived % audit log records older than %',
        v_archived_count, p_older_than;

    RETURN v_archived_count;
END;
$$ LANGUAGE plpgsql;

-- Create archive table for audit logs
CREATE TABLE IF NOT EXISTS archive.audit_log_archive (
    LIKE audit.audit_log INCLUDING ALL
);

COMMENT ON TABLE archive.audit_log_archive
    IS 'Archived audit logs for long-term compliance storage';

-- =============================================================================
-- LEGAL HOLD SUPPORT
-- =============================================================================
-- Function to exclude data from retention policies during legal hold

CREATE TABLE IF NOT EXISTS archive.legal_holds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hold_name VARCHAR(255) NOT NULL,
    description TEXT,
    org_id UUID,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ,
    affected_tables VARCHAR(255)[] NOT NULL,
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ,
    released_by UUID
);

COMMENT ON TABLE archive.legal_holds
    IS 'Legal holds that prevent data deletion during investigations';

-- Function to check if data is under legal hold
CREATE OR REPLACE FUNCTION archive.is_under_legal_hold(
    p_org_id UUID,
    p_table_name VARCHAR,
    p_data_time TIMESTAMPTZ
)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM archive.legal_holds
        WHERE (org_id IS NULL OR org_id = p_org_id)
          AND p_table_name = ANY(affected_tables)
          AND p_data_time >= start_date
          AND (end_date IS NULL OR p_data_time <= end_date)
          AND released_at IS NULL
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive.is_under_legal_hold
    IS 'Check if data is protected by a legal hold';

-- =============================================================================
-- RETENTION POLICY MONITORING
-- =============================================================================

-- View to show current retention policies
CREATE OR REPLACE VIEW archive.retention_policy_status AS
SELECT
    hypertable_schema || '.' || hypertable_name AS hypertable,
    config ->> 'drop_after' AS retention_period,
    schedule_interval,
    job_status,
    last_run_status,
    last_run_started_at,
    next_start
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention'
ORDER BY hypertable_schema, hypertable_name;

COMMENT ON VIEW archive.retention_policy_status
    IS 'Current status of all retention policies';

-- Function to estimate data that will be deleted
CREATE OR REPLACE FUNCTION archive.estimate_retention_deletions()
RETURNS TABLE (
    hypertable TEXT,
    retention_period TEXT,
    chunks_to_delete BIGINT,
    estimated_size_to_delete TEXT,
    oldest_data_time TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    WITH retention_info AS (
        SELECT
            hypertable_schema || '.' || hypertable_name AS ht_name,
            (config ->> 'drop_after')::INTERVAL AS drop_after
        FROM timescaledb_information.jobs
        WHERE proc_name = 'policy_retention'
    )
    SELECT
        ri.ht_name,
        ri.drop_after::TEXT,
        COUNT(*)::BIGINT AS chunks_to_delete,
        pg_size_pretty(SUM(ch.total_bytes)) AS estimated_size,
        MIN(ch.range_start) AS oldest_data
    FROM retention_info ri
    JOIN timescaledb_information.chunks ch
        ON ch.hypertable_schema || '.' || ch.hypertable_name = ri.ht_name
    WHERE ch.range_end < NOW() - ri.drop_after
    GROUP BY ri.ht_name, ri.drop_after
    ORDER BY ri.ht_name;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive.estimate_retention_deletions
    IS 'Estimate how much data will be deleted by retention policies';

-- =============================================================================
-- SUMMARY
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '=============================================================';
    RAISE NOTICE 'GreenLang Retention Policies Summary';
    RAISE NOTICE '=============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Hypertable                        Retention    Archive First';
    RAISE NOTICE '-------------------------------------------------------------';
    RAISE NOTICE 'metrics.emission_measurements     7 years      Yes (compliance)';
    RAISE NOTICE 'metrics.calculation_results       2 years      No';
    RAISE NOTICE 'metrics.sensor_readings           1 year       No';
    RAISE NOTICE 'audit.audit_log                   10 years     Yes (compliance)';
    RAISE NOTICE 'audit.api_requests                90 days      No';
    RAISE NOTICE 'audit.security_events             7 years      No';
    RAISE NOTICE '';
    RAISE NOTICE 'Legal holds can prevent deletion during investigations.';
    RAISE NOTICE 'Use archive.legal_holds table to manage holds.';
    RAISE NOTICE '=============================================================';
END $$;
