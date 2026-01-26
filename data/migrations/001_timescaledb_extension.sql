-- =============================================================================
-- GreenLang TimescaleDB Extension Setup
-- =============================================================================
-- INFRA-003: Enable TimescaleDB extension for time-series data management
--
-- This migration:
--   1. Enables TimescaleDB extension
--   2. Creates hypertables for metrics data
--   3. Sets up retention policies
--   4. Configures compression policies
--
-- Prerequisites:
--   - PostgreSQL 14+ with TimescaleDB extension installed
--   - Superuser privileges for extension creation
--
-- Author: GreenLang Team
-- Version: 1.0.0
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable additional useful extensions
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- SCHEMA: metrics
-- =============================================================================
-- Dedicated schema for time-series metrics data
-- =============================================================================
CREATE SCHEMA IF NOT EXISTS metrics;

-- =============================================================================
-- TABLE: metrics.agent_execution_metrics
-- =============================================================================
-- Stores execution metrics for all GreenLang agents
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics.agent_execution_metrics (
    time                    TIMESTAMPTZ NOT NULL,
    tenant_id               UUID NOT NULL,
    agent_id                VARCHAR(64) NOT NULL,
    agent_version           VARCHAR(32) NOT NULL,
    execution_id            UUID NOT NULL,
    pipeline_id             UUID,

    -- Execution metrics
    duration_ms             INTEGER NOT NULL,
    cpu_usage_percent       REAL,
    memory_usage_mb         REAL,
    tokens_processed        INTEGER,

    -- Status
    status                  VARCHAR(32) NOT NULL,
    error_code              VARCHAR(64),

    -- Context
    environment             VARCHAR(32) DEFAULT 'production',
    region                  VARCHAR(32),

    -- Metadata (JSONB for flexibility)
    metadata                JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT agent_execution_metrics_pkey PRIMARY KEY (time, tenant_id, execution_id)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'metrics.agent_execution_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_agent_exec_tenant_time
    ON metrics.agent_execution_metrics (tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_exec_agent_time
    ON metrics.agent_execution_metrics (agent_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_agent_exec_pipeline_time
    ON metrics.agent_execution_metrics (pipeline_id, time DESC)
    WHERE pipeline_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_agent_exec_status
    ON metrics.agent_execution_metrics (status, time DESC);

-- =============================================================================
-- TABLE: metrics.emissions_data
-- =============================================================================
-- Stores calculated emissions data points
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics.emissions_data (
    time                    TIMESTAMPTZ NOT NULL,
    tenant_id               UUID NOT NULL,
    organization_id         UUID NOT NULL,

    -- Emissions data
    scope                   SMALLINT NOT NULL CHECK (scope IN (1, 2, 3)),
    category                VARCHAR(64) NOT NULL,
    source_type             VARCHAR(64) NOT NULL,

    -- Values (in tCO2e)
    emissions_tco2e         DECIMAL(18, 6) NOT NULL,
    uncertainty_percent     REAL,

    -- Data quality
    data_quality_score      REAL CHECK (data_quality_score >= 0 AND data_quality_score <= 1),
    calculation_method      VARCHAR(64) NOT NULL,
    emission_factor_source  VARCHAR(128),

    -- Provenance
    provenance_hash         VARCHAR(64),
    agent_id                VARCHAR(64),

    -- Metadata
    metadata                JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT emissions_data_pkey PRIMARY KEY (time, tenant_id, organization_id, scope, category)
);

-- Convert to hypertable with 7-day chunks (less frequent data)
SELECT create_hypertable(
    'metrics.emissions_data',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_emissions_tenant_org_time
    ON metrics.emissions_data (tenant_id, organization_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_emissions_scope_time
    ON metrics.emissions_data (scope, time DESC);
CREATE INDEX IF NOT EXISTS idx_emissions_category_time
    ON metrics.emissions_data (category, time DESC);

-- =============================================================================
-- TABLE: metrics.carbon_prices
-- =============================================================================
-- Stores carbon market price data
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics.carbon_prices (
    time                    TIMESTAMPTZ NOT NULL,
    market                  VARCHAR(64) NOT NULL,

    -- Price data
    price_usd               DECIMAL(12, 4) NOT NULL,
    price_local             DECIMAL(18, 4),
    local_currency          VARCHAR(3),

    -- Market data
    volume_tonnes           DECIMAL(18, 2),
    open_interest           DECIMAL(18, 2),

    -- Change metrics
    change_1d_percent       REAL,
    change_7d_percent       REAL,
    change_30d_percent      REAL,

    -- Source
    data_source             VARCHAR(64) NOT NULL,

    -- Constraints
    CONSTRAINT carbon_prices_pkey PRIMARY KEY (time, market)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'metrics.carbon_prices',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_carbon_prices_market_time
    ON metrics.carbon_prices (market, time DESC);

-- =============================================================================
-- TABLE: metrics.api_request_metrics
-- =============================================================================
-- Stores API request metrics for monitoring
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics.api_request_metrics (
    time                    TIMESTAMPTZ NOT NULL,
    tenant_id               UUID,

    -- Request info
    endpoint                VARCHAR(256) NOT NULL,
    method                  VARCHAR(10) NOT NULL,
    status_code             SMALLINT NOT NULL,

    -- Performance
    response_time_ms        INTEGER NOT NULL,
    request_size_bytes      INTEGER,
    response_size_bytes     INTEGER,

    -- Client info
    client_id               VARCHAR(64),
    client_ip               INET,
    user_agent              VARCHAR(512),

    -- Tracing
    trace_id                VARCHAR(64),
    span_id                 VARCHAR(64),

    -- Constraints
    CONSTRAINT api_request_metrics_pkey PRIMARY KEY (time, endpoint, method)
);

-- Convert to hypertable with 1-hour chunks (high-volume data)
SELECT create_hypertable(
    'metrics.api_request_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_api_metrics_tenant_time
    ON metrics.api_request_metrics (tenant_id, time DESC)
    WHERE tenant_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_metrics_endpoint_time
    ON metrics.api_request_metrics (endpoint, time DESC);
CREATE INDEX IF NOT EXISTS idx_api_metrics_status_time
    ON metrics.api_request_metrics (status_code, time DESC);

-- =============================================================================
-- TABLE: metrics.system_health_metrics
-- =============================================================================
-- Stores system health and infrastructure metrics
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics.system_health_metrics (
    time                    TIMESTAMPTZ NOT NULL,
    node_name               VARCHAR(128) NOT NULL,
    namespace               VARCHAR(64) NOT NULL,
    pod_name                VARCHAR(256),

    -- Resource usage
    cpu_usage_percent       REAL,
    cpu_limit_percent       REAL,
    memory_usage_mb         REAL,
    memory_limit_mb         REAL,
    disk_usage_percent      REAL,
    network_rx_bytes        BIGINT,
    network_tx_bytes        BIGINT,

    -- Status
    pod_status              VARCHAR(32),
    restart_count           INTEGER,

    -- Metadata
    labels                  JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    CONSTRAINT system_health_metrics_pkey PRIMARY KEY (time, node_name, namespace)
);

-- Convert to hypertable with 1-hour chunks
SELECT create_hypertable(
    'metrics.system_health_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_system_health_namespace_time
    ON metrics.system_health_metrics (namespace, time DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_pod_time
    ON metrics.system_health_metrics (pod_name, time DESC)
    WHERE pod_name IS NOT NULL;

-- =============================================================================
-- COMPRESSION POLICIES
-- =============================================================================
-- Enable compression for older data to save storage
-- =============================================================================

-- Compress agent execution metrics after 7 days
ALTER TABLE metrics.agent_execution_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id, agent_id',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics.agent_execution_metrics', INTERVAL '7 days');

-- Compress emissions data after 30 days
ALTER TABLE metrics.emissions_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id, organization_id, scope',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics.emissions_data', INTERVAL '30 days');

-- Compress carbon prices after 30 days
ALTER TABLE metrics.carbon_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'market',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics.carbon_prices', INTERVAL '30 days');

-- Compress API metrics after 3 days (high volume)
ALTER TABLE metrics.api_request_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'endpoint',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics.api_request_metrics', INTERVAL '3 days');

-- Compress system health after 7 days
ALTER TABLE metrics.system_health_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'namespace, node_name',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('metrics.system_health_metrics', INTERVAL '7 days');

-- =============================================================================
-- RETENTION POLICIES
-- =============================================================================
-- Automatically drop old data to manage storage
-- =============================================================================

-- Keep agent execution metrics for 90 days
SELECT add_retention_policy('metrics.agent_execution_metrics', INTERVAL '90 days');

-- Keep emissions data for 7 years (regulatory requirement)
SELECT add_retention_policy('metrics.emissions_data', INTERVAL '7 years');

-- Keep carbon prices for 5 years
SELECT add_retention_policy('metrics.carbon_prices', INTERVAL '5 years');

-- Keep API metrics for 30 days
SELECT add_retention_policy('metrics.api_request_metrics', INTERVAL '30 days');

-- Keep system health for 30 days
SELECT add_retention_policy('metrics.system_health_metrics', INTERVAL '30 days');

-- =============================================================================
-- CONTINUOUS AGGREGATES
-- =============================================================================
-- Pre-computed aggregates for faster dashboard queries
-- =============================================================================

-- Hourly agent execution summary
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.agent_execution_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    agent_id,
    COUNT(*) AS execution_count,
    COUNT(*) FILTER (WHERE status = 'success') AS success_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failure_count,
    AVG(duration_ms) AS avg_duration_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
    AVG(cpu_usage_percent) AS avg_cpu_percent,
    AVG(memory_usage_mb) AS avg_memory_mb,
    SUM(tokens_processed) AS total_tokens
FROM metrics.agent_execution_metrics
GROUP BY bucket, tenant_id, agent_id
WITH NO DATA;

-- Refresh policy for hourly aggregate
SELECT add_continuous_aggregate_policy('metrics.agent_execution_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- Daily emissions summary
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.emissions_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    tenant_id,
    organization_id,
    scope,
    category,
    SUM(emissions_tco2e) AS total_emissions_tco2e,
    AVG(data_quality_score) AS avg_data_quality,
    COUNT(*) AS data_point_count
FROM metrics.emissions_data
GROUP BY bucket, tenant_id, organization_id, scope, category
WITH NO DATA;

-- Refresh policy for daily emissions
SELECT add_continuous_aggregate_policy('metrics.emissions_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day'
);

-- Hourly API metrics summary
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.api_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    endpoint,
    method,
    COUNT(*) AS request_count,
    COUNT(*) FILTER (WHERE status_code >= 200 AND status_code < 300) AS success_count,
    COUNT(*) FILTER (WHERE status_code >= 500) AS error_count,
    AVG(response_time_ms) AS avg_response_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_response_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99_response_ms,
    SUM(request_size_bytes) AS total_request_bytes,
    SUM(response_size_bytes) AS total_response_bytes
FROM metrics.api_request_metrics
GROUP BY bucket, endpoint, method
WITH NO DATA;

-- Refresh policy for API metrics
SELECT add_continuous_aggregate_policy('metrics.api_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- GRANT PERMISSIONS
-- =============================================================================
-- Grant appropriate permissions to application roles
-- =============================================================================

-- Create application role if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'greenlang_app') THEN
        CREATE ROLE greenlang_app WITH LOGIN PASSWORD 'changeme';
    END IF;
END
$$;

-- Grant schema usage
GRANT USAGE ON SCHEMA metrics TO greenlang_app;

-- Grant table permissions
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA metrics TO greenlang_app;
GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO greenlang_app;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO greenlang_app;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics
    GRANT SELECT, INSERT ON TABLES TO greenlang_app;

-- =============================================================================
-- VERIFICATION
-- =============================================================================
-- Verify TimescaleDB setup
-- =============================================================================

-- Check extension is enabled
SELECT * FROM pg_extension WHERE extname = 'timescaledb';

-- List all hypertables
SELECT * FROM timescaledb_information.hypertables;

-- List all continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;

-- List all policies
SELECT * FROM timescaledb_information.jobs;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB setup completed successfully!';
    RAISE NOTICE 'Created hypertables: agent_execution_metrics, emissions_data, carbon_prices, api_request_metrics, system_health_metrics';
    RAISE NOTICE 'Created continuous aggregates: agent_execution_hourly, emissions_daily, api_metrics_hourly';
    RAISE NOTICE 'Compression and retention policies configured.';
END
$$;
