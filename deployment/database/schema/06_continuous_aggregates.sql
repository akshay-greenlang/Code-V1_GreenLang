-- =============================================================================
-- GreenLang Climate OS - Continuous Aggregates
-- =============================================================================
-- File: 06_continuous_aggregates.sql
-- Description: TimescaleDB continuous aggregates for pre-computed rollups
--              of emission data and sensor readings for fast dashboards.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Hourly Emissions Aggregate
-- -----------------------------------------------------------------------------
-- Pre-aggregated hourly emission totals by organization, project, and scope.
-- Dramatically speeds up dashboard queries.

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.hourly_emissions
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 hour)
    time_bucket('1 hour', time) AS bucket,

    -- Dimensions
    org_id,
    project_id,
    source_id,
    scope,

    -- Aggregated emission values (kgCO2e)
    SUM(emission_value) AS total_emissions,
    AVG(emission_value) AS avg_emissions,
    MIN(emission_value) AS min_emissions,
    MAX(emission_value) AS max_emissions,

    -- Individual GHG totals
    SUM(co2_value) AS total_co2,
    SUM(ch4_value) AS total_ch4,
    SUM(n2o_value) AS total_n2o,
    SUM(hfc_value) AS total_hfc,
    SUM(pfc_value) AS total_pfc,
    SUM(sf6_value) AS total_sf6,
    SUM(nf3_value) AS total_nf3,

    -- Activity data aggregates
    SUM(activity_value) AS total_activity,

    -- Data quality metrics
    AVG(data_quality_score) AS avg_quality_score,
    COUNT(*) AS record_count,
    COUNT(*) FILTER (WHERE data_quality_score >= 80) AS high_quality_count

FROM metrics.emission_measurements
GROUP BY bucket, org_id, project_id, source_id, scope
WITH NO DATA;

-- Add refresh policy: refresh every hour, covering last 3 hours
SELECT add_continuous_aggregate_policy('metrics.hourly_emissions',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create indexes on the continuous aggregate
CREATE INDEX IF NOT EXISTS idx_hourly_emissions_org
    ON metrics.hourly_emissions(org_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_hourly_emissions_project
    ON metrics.hourly_emissions(project_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_hourly_emissions_scope
    ON metrics.hourly_emissions(scope, bucket DESC);

COMMENT ON MATERIALIZED VIEW metrics.hourly_emissions IS 'Hourly emission rollups for fast dashboard queries';

-- -----------------------------------------------------------------------------
-- Daily Emissions Aggregate
-- -----------------------------------------------------------------------------
-- Pre-aggregated daily emission totals.

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.daily_emissions
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 day)
    time_bucket('1 day', time) AS bucket,

    -- Dimensions
    org_id,
    project_id,
    scope,

    -- Aggregated emission values
    SUM(emission_value) AS total_emissions,
    AVG(emission_value) AS avg_emissions,
    MIN(emission_value) AS min_emissions,
    MAX(emission_value) AS max_emissions,

    -- Individual GHG totals
    SUM(co2_value) AS total_co2,
    SUM(ch4_value) AS total_ch4,
    SUM(n2o_value) AS total_n2o,

    -- Record counts
    COUNT(*) AS record_count,
    COUNT(DISTINCT source_id) AS source_count,

    -- Data quality
    AVG(data_quality_score) AS avg_quality_score

FROM metrics.emission_measurements
GROUP BY bucket, org_id, project_id, scope
WITH NO DATA;

-- Refresh policy: every day, covering last 3 days
SELECT add_continuous_aggregate_policy('metrics.daily_emissions',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_daily_emissions_org
    ON metrics.daily_emissions(org_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_daily_emissions_project
    ON metrics.daily_emissions(project_id, bucket DESC);

COMMENT ON MATERIALIZED VIEW metrics.daily_emissions IS 'Daily emission rollups for reporting';

-- -----------------------------------------------------------------------------
-- Monthly Emissions Aggregate
-- -----------------------------------------------------------------------------
-- Pre-aggregated monthly emission totals for reporting.

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.monthly_emissions
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 month)
    time_bucket('1 month', time) AS bucket,

    -- Dimensions
    org_id,
    project_id,
    scope,

    -- Aggregated emission values
    SUM(emission_value) AS total_emissions,
    AVG(emission_value) AS avg_emissions,

    -- Individual GHG totals
    SUM(co2_value) AS total_co2,
    SUM(ch4_value) AS total_ch4,
    SUM(n2o_value) AS total_n2o,

    -- Record counts
    COUNT(*) AS record_count,
    COUNT(DISTINCT source_id) AS source_count,

    -- Data quality
    AVG(data_quality_score) AS avg_quality_score,

    -- Emissions intensity (if activity data available)
    CASE
        WHEN SUM(activity_value) > 0
        THEN SUM(emission_value) / SUM(activity_value)
        ELSE NULL
    END AS emission_intensity

FROM metrics.emission_measurements
GROUP BY bucket, org_id, project_id, scope
WITH NO DATA;

-- Refresh policy: every week, covering last 2 months
SELECT add_continuous_aggregate_policy('metrics.monthly_emissions',
    start_offset => INTERVAL '2 months',
    end_offset => INTERVAL '1 month',
    schedule_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_monthly_emissions_org
    ON metrics.monthly_emissions(org_id, bucket DESC);

COMMENT ON MATERIALIZED VIEW metrics.monthly_emissions IS 'Monthly emission rollups for annual reporting';

-- -----------------------------------------------------------------------------
-- Device Statistics Aggregate (Hourly)
-- -----------------------------------------------------------------------------
-- Pre-aggregated hourly statistics from sensor readings.

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.device_statistics_hourly
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 hour)
    time_bucket('1 hour', time) AS bucket,

    -- Dimensions
    device_id,
    org_id,
    metric_type,

    -- Statistical aggregates
    COUNT(*) AS reading_count,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    AVG(value) AS avg_value,

    -- Percentiles for distribution analysis
    percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS median_value,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95_value,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY value) AS p99_value,

    -- Standard deviation for anomaly detection
    STDDEV(value) AS stddev_value,

    -- Data quality metrics
    COUNT(*) FILTER (WHERE quality = 'good') AS good_count,
    COUNT(*) FILTER (WHERE quality = 'bad') AS bad_count,
    COUNT(*) FILTER (WHERE quality = 'uncertain') AS uncertain_count,

    -- First and last values for trend analysis
    first(value, time) AS first_value,
    last(value, time) AS last_value,

    -- Time of min/max (useful for peak analysis)
    first(time, value) AS min_value_time,
    last(time, value) AS max_value_time

FROM metrics.sensor_readings
GROUP BY bucket, device_id, org_id, metric_type
WITH NO DATA;

-- Refresh policy: every 15 minutes, covering last 2 hours
SELECT add_continuous_aggregate_policy('metrics.device_statistics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_device_stats_hourly_device
    ON metrics.device_statistics_hourly(device_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_device_stats_hourly_org
    ON metrics.device_statistics_hourly(org_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_device_stats_hourly_metric
    ON metrics.device_statistics_hourly(metric_type, bucket DESC);

COMMENT ON MATERIALIZED VIEW metrics.device_statistics_hourly IS 'Hourly device statistics for monitoring dashboards';

-- -----------------------------------------------------------------------------
-- Device Statistics Aggregate (Daily)
-- -----------------------------------------------------------------------------
-- Daily statistics for longer-term analysis.

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.device_statistics_daily
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 day)
    time_bucket('1 day', time) AS bucket,

    -- Dimensions
    device_id,
    org_id,
    metric_type,

    -- Statistical aggregates
    COUNT(*) AS reading_count,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    AVG(value) AS avg_value,

    -- Percentiles
    percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS median_value,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY value) AS p95_value,

    -- Standard deviation
    STDDEV(value) AS stddev_value,

    -- Data quality (percentage good)
    100.0 * COUNT(*) FILTER (WHERE quality = 'good') / NULLIF(COUNT(*), 0) AS good_quality_percent,

    -- Uptime (percentage of expected readings received)
    -- Assuming 1-minute sampling = 1440 readings/day
    100.0 * COUNT(*) / 1440.0 AS uptime_percent

FROM metrics.sensor_readings
GROUP BY bucket, device_id, org_id, metric_type
WITH NO DATA;

-- Refresh policy: every day, covering last 3 days
SELECT add_continuous_aggregate_policy('metrics.device_statistics_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_device_stats_daily_device
    ON metrics.device_statistics_daily(device_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_device_stats_daily_org
    ON metrics.device_statistics_daily(org_id, bucket DESC);

COMMENT ON MATERIALIZED VIEW metrics.device_statistics_daily IS 'Daily device statistics for trend analysis';

-- -----------------------------------------------------------------------------
-- API Request Statistics (Hourly)
-- -----------------------------------------------------------------------------
-- Hourly API request statistics for monitoring and billing.

CREATE MATERIALIZED VIEW IF NOT EXISTS audit.api_statistics_hourly
WITH (timescaledb.continuous) AS
SELECT
    -- Time bucket (1 hour)
    time_bucket('1 hour', time) AS bucket,

    -- Dimensions
    org_id,
    key_id,
    path,
    method,

    -- Request counts
    COUNT(*) AS request_count,
    COUNT(*) FILTER (WHERE status >= 200 AND status < 300) AS success_count,
    COUNT(*) FILTER (WHERE status >= 400 AND status < 500) AS client_error_count,
    COUNT(*) FILTER (WHERE status >= 500) AS server_error_count,

    -- Performance metrics
    AVG(duration_ms) AS avg_duration_ms,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_duration_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_duration_ms,
    MAX(duration_ms) AS max_duration_ms,

    -- Size metrics
    SUM(request_size) AS total_request_bytes,
    SUM(response_size) AS total_response_bytes,
    AVG(response_size) AS avg_response_bytes

FROM audit.api_requests
GROUP BY bucket, org_id, key_id, path, method
WITH NO DATA;

-- Refresh policy: every 15 minutes, covering last 2 hours
SELECT add_continuous_aggregate_policy('audit.api_statistics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '30 minutes',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_api_stats_hourly_org
    ON audit.api_statistics_hourly(org_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_api_stats_hourly_key
    ON audit.api_statistics_hourly(key_id, bucket DESC)
    WHERE key_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_api_stats_hourly_path
    ON audit.api_statistics_hourly(path, bucket DESC);

COMMENT ON MATERIALIZED VIEW audit.api_statistics_hourly IS 'Hourly API statistics for monitoring and billing';

-- -----------------------------------------------------------------------------
-- Manual Refresh Function
-- -----------------------------------------------------------------------------
-- Function to manually refresh all continuous aggregates (useful for backfilling)

CREATE OR REPLACE FUNCTION metrics.refresh_all_aggregates(
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS VOID AS $$
BEGIN
    -- Refresh emission aggregates
    CALL refresh_continuous_aggregate('metrics.hourly_emissions', p_start_time, p_end_time);
    CALL refresh_continuous_aggregate('metrics.daily_emissions', p_start_time, p_end_time);
    CALL refresh_continuous_aggregate('metrics.monthly_emissions', p_start_time, p_end_time);

    -- Refresh device statistics
    CALL refresh_continuous_aggregate('metrics.device_statistics_hourly', p_start_time, p_end_time);
    CALL refresh_continuous_aggregate('metrics.device_statistics_daily', p_start_time, p_end_time);

    -- Refresh API statistics
    CALL refresh_continuous_aggregate('audit.api_statistics_hourly', p_start_time, p_end_time);

    RAISE NOTICE 'All continuous aggregates refreshed for period % to %', p_start_time, p_end_time;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.refresh_all_aggregates IS 'Manually refresh all continuous aggregates for a time range';
