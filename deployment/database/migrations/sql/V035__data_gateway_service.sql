-- =============================================================================
-- V035: Data Gateway Service Schema
-- =============================================================================
-- Component: AGENT-DATA-004 (API Gateway Agent)
-- Agent ID:  GL-DATA-GW-001
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Data Gateway Agent (GL-DATA-GW-001) with capabilities
-- for unified data source registry, query routing, schema mapping,
-- caching, health monitoring, multi-source aggregation,
-- query metrics tracking, and data catalog management.
-- =============================================================================
-- Tables (10):
--   1. data_sources             - Registry of backend data agents
--   2. source_schemas           - Schema definitions per source
--   3. query_templates          - Reusable query templates
--   4. query_log                - Audit log of all queries (hypertable)
--   5. cache_entries            - Cache metadata and statistics
--   6. source_health_checks     - Health check history (hypertable)
--   7. schema_mappings          - Field mapping between schemas
--   8. aggregation_rules        - Rules for multi-source aggregation
--   9. query_metrics            - Per-query performance metrics (hypertable)
--  10. data_catalog             - Unified data catalog entries
--
-- Continuous Aggregates (2):
--   1. data_gateway_queries_hourly   - Hourly query log aggregates
--   2. data_gateway_health_hourly    - Hourly health check aggregates
--
-- Also includes: 50+ indexes (B-tree, GIN), RLS policies per tenant,
-- retention policies, compression policies, updated_at trigger,
-- security permissions, and seed data registering GL-DATA-GW-001
-- in the agent registry.
-- Previous: V034__eudr_traceability_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS data_gateway_service;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================
-- Reusable trigger function for tables with updated_at columns.

CREATE OR REPLACE FUNCTION data_gateway_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: data_gateway_service.data_sources
-- =============================================================================
-- Registry of backend data agents. Each record captures a data source
-- endpoint, its type (pdf_extractor, excel_normalizer, erp_connector, etc.),
-- health status, capabilities, supported operations, API version, and
-- response time metrics. Core reference table for query routing and
-- health monitoring. Tenant-scoped.

CREATE TABLE data_gateway_service.data_sources (
    source_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    endpoint_url TEXT NOT NULL,
    api_version VARCHAR(16) DEFAULT 'v1',
    status VARCHAR(32) DEFAULT 'unknown',
    capabilities JSONB DEFAULT '[]'::jsonb,
    supported_operations JSONB DEFAULT '[]'::jsonb,
    health_check_url TEXT DEFAULT '',
    last_health_check TIMESTAMPTZ,
    response_time_ms FLOAT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source type constraint
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_source_type
    CHECK (source_type IN (
        'pdf_extractor', 'excel_normalizer', 'erp_connector',
        'eudr_traceability', 'scada_bms', 'fleet_telematics',
        'utility_tariff', 'supplier_portal', 'event_processor',
        'document_classifier', 'ocr_agent', 'email_processor',
        'iot_meter', 'emission_factor', 'custom'
    ));

-- Status constraint
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_status
    CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'unknown', 'maintenance'));

-- Source ID must not be empty
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_source_id_not_empty
    CHECK (LENGTH(TRIM(source_id)) > 0);

-- Name must not be empty
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Endpoint URL must not be empty
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_endpoint_url_not_empty
    CHECK (LENGTH(TRIM(endpoint_url)) > 0);

-- Response time must be non-negative if specified
ALTER TABLE data_gateway_service.data_sources
    ADD CONSTRAINT chk_ds_response_time_non_negative
    CHECK (response_time_ms IS NULL OR response_time_ms >= 0);

-- Updated_at trigger for data_sources
CREATE TRIGGER trg_data_sources_updated_at
    BEFORE UPDATE ON data_gateway_service.data_sources
    FOR EACH ROW
    EXECUTE FUNCTION data_gateway_service.set_updated_at();

-- =============================================================================
-- Table 2: data_gateway_service.source_schemas
-- =============================================================================
-- Schema definitions per source. Each record captures a versioned schema
-- definition with field specifications for a given source type. Used for
-- query validation, field mapping, and data catalog generation.
-- Tenant-scoped.

CREATE TABLE data_gateway_service.source_schemas (
    schema_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    schema_name VARCHAR(255) NOT NULL,
    version VARCHAR(32) NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    fields JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Schema ID must not be empty
ALTER TABLE data_gateway_service.source_schemas
    ADD CONSTRAINT chk_ss_schema_id_not_empty
    CHECK (LENGTH(TRIM(schema_id)) > 0);

-- Schema name must not be empty
ALTER TABLE data_gateway_service.source_schemas
    ADD CONSTRAINT chk_ss_schema_name_not_empty
    CHECK (LENGTH(TRIM(schema_name)) > 0);

-- Version must not be empty
ALTER TABLE data_gateway_service.source_schemas
    ADD CONSTRAINT chk_ss_version_not_empty
    CHECK (LENGTH(TRIM(version)) > 0);

-- Source type must not be empty
ALTER TABLE data_gateway_service.source_schemas
    ADD CONSTRAINT chk_ss_source_type_not_empty
    CHECK (LENGTH(TRIM(source_type)) > 0);

-- =============================================================================
-- Table 3: data_gateway_service.query_templates
-- =============================================================================
-- Reusable query templates. Each template captures a named query plan
-- with parameters, description, and usage tracking. Enables saved queries
-- and parameterized query execution across data sources. Tenant-scoped.

CREATE TABLE data_gateway_service.query_templates (
    template_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    query_plan JSONB NOT NULL,
    parameters JSONB DEFAULT '{}'::jsonb,
    created_by VARCHAR(128) DEFAULT '',
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Template ID must not be empty
ALTER TABLE data_gateway_service.query_templates
    ADD CONSTRAINT chk_qt_template_id_not_empty
    CHECK (LENGTH(TRIM(template_id)) > 0);

-- Name must not be empty
ALTER TABLE data_gateway_service.query_templates
    ADD CONSTRAINT chk_qt_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Usage count must be non-negative
ALTER TABLE data_gateway_service.query_templates
    ADD CONSTRAINT chk_qt_usage_count_non_negative
    CHECK (usage_count >= 0);

-- Updated_at trigger for query_templates
CREATE TRIGGER trg_query_templates_updated_at
    BEFORE UPDATE ON data_gateway_service.query_templates
    FOR EACH ROW
    EXECUTE FUNCTION data_gateway_service.set_updated_at();

-- =============================================================================
-- Table 4: data_gateway_service.query_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording all gateway queries. Each query record
-- captures the sources queried, operations performed, filters applied,
-- execution status, result count, execution time, cache hit flag,
-- complexity score, error messages, and provenance hash. Partitioned by
-- executed_at for time-series queries. Retained for 730 days with
-- compression after 30 days. Tenant-scoped.

CREATE TABLE data_gateway_service.query_log (
    query_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    sources JSONB NOT NULL,
    operations JSONB DEFAULT '[]'::jsonb,
    filters JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(32) DEFAULT 'pending',
    result_count INTEGER DEFAULT 0,
    execution_time_ms FLOAT DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    complexity_score FLOAT DEFAULT 0,
    error_message TEXT DEFAULT '',
    provenance_hash VARCHAR(128) DEFAULT '',
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (query_id, executed_at)
);

-- Create hypertable partitioned by executed_at
SELECT create_hypertable('data_gateway_service.query_log', 'executed_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE data_gateway_service.query_log
    ADD CONSTRAINT chk_ql_status
    CHECK (status IN ('pending', 'executing', 'completed', 'failed', 'timeout', 'cancelled'));

-- Query ID must not be empty
ALTER TABLE data_gateway_service.query_log
    ADD CONSTRAINT chk_ql_query_id_not_empty
    CHECK (LENGTH(TRIM(query_id)) > 0);

-- Result count must be non-negative
ALTER TABLE data_gateway_service.query_log
    ADD CONSTRAINT chk_ql_result_count_non_negative
    CHECK (result_count >= 0);

-- Execution time must be non-negative
ALTER TABLE data_gateway_service.query_log
    ADD CONSTRAINT chk_ql_execution_time_non_negative
    CHECK (execution_time_ms >= 0);

-- Complexity score must be non-negative
ALTER TABLE data_gateway_service.query_log
    ADD CONSTRAINT chk_ql_complexity_score_non_negative
    CHECK (complexity_score >= 0);

-- =============================================================================
-- Table 5: data_gateway_service.cache_entries
-- =============================================================================
-- Cache metadata and statistics. Each entry captures a cache key,
-- query hash, source reference, result count, size, hit count,
-- access timestamps, and expiration time. Used for cache management,
-- eviction decisions, and hit rate monitoring. Tenant-scoped.

CREATE TABLE data_gateway_service.cache_entries (
    cache_key VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    query_hash VARCHAR(128) NOT NULL,
    source_id VARCHAR(64) NOT NULL,
    result_count INTEGER DEFAULT 0,
    size_bytes BIGINT DEFAULT 0,
    hit_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Cache key must not be empty
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_cache_key_not_empty
    CHECK (LENGTH(TRIM(cache_key)) > 0);

-- Query hash must not be empty
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_query_hash_not_empty
    CHECK (LENGTH(TRIM(query_hash)) > 0);

-- Source ID must not be empty
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_source_id_not_empty
    CHECK (LENGTH(TRIM(source_id)) > 0);

-- Result count must be non-negative
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_result_count_non_negative
    CHECK (result_count >= 0);

-- Size bytes must be non-negative
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_size_bytes_non_negative
    CHECK (size_bytes >= 0);

-- Hit count must be non-negative
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_hit_count_non_negative
    CHECK (hit_count >= 0);

-- Expires_at must be after created_at
ALTER TABLE data_gateway_service.cache_entries
    ADD CONSTRAINT chk_ce_expires_after_created
    CHECK (expires_at >= created_at);

-- =============================================================================
-- Table 6: data_gateway_service.source_health_checks (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording health check history for data sources.
-- Each check record captures the source status, response time, error
-- message, and consecutive failure count. Partitioned by checked_at
-- for time-series queries. Retained for 730 days with compression after
-- 30 days. Tenant-scoped.

CREATE TABLE data_gateway_service.source_health_checks (
    source_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    response_time_ms FLOAT DEFAULT 0,
    error_message TEXT DEFAULT '',
    consecutive_failures INTEGER DEFAULT 0,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source_id, checked_at)
);

-- Create hypertable partitioned by checked_at
SELECT create_hypertable('data_gateway_service.source_health_checks', 'checked_at', if_not_exists => TRUE);

-- Status constraint
ALTER TABLE data_gateway_service.source_health_checks
    ADD CONSTRAINT chk_shc_status
    CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'unknown', 'maintenance'));

-- Source ID must not be empty
ALTER TABLE data_gateway_service.source_health_checks
    ADD CONSTRAINT chk_shc_source_id_not_empty
    CHECK (LENGTH(TRIM(source_id)) > 0);

-- Response time must be non-negative
ALTER TABLE data_gateway_service.source_health_checks
    ADD CONSTRAINT chk_shc_response_time_non_negative
    CHECK (response_time_ms >= 0);

-- Consecutive failures must be non-negative
ALTER TABLE data_gateway_service.source_health_checks
    ADD CONSTRAINT chk_shc_consecutive_failures_non_negative
    CHECK (consecutive_failures >= 0);

-- =============================================================================
-- Table 7: data_gateway_service.schema_mappings
-- =============================================================================
-- Field mapping between schemas. Each mapping record defines field-level
-- transformations from a source type to a target type. Used for cross-source
-- query normalization and unified response formatting. Tenant-scoped.

CREATE TABLE data_gateway_service.schema_mappings (
    mapping_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    target_type VARCHAR(64) NOT NULL,
    mappings JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Mapping ID must not be empty
ALTER TABLE data_gateway_service.schema_mappings
    ADD CONSTRAINT chk_sm_mapping_id_not_empty
    CHECK (LENGTH(TRIM(mapping_id)) > 0);

-- Source type must not be empty
ALTER TABLE data_gateway_service.schema_mappings
    ADD CONSTRAINT chk_sm_source_type_not_empty
    CHECK (LENGTH(TRIM(source_type)) > 0);

-- Target type must not be empty
ALTER TABLE data_gateway_service.schema_mappings
    ADD CONSTRAINT chk_sm_target_type_not_empty
    CHECK (LENGTH(TRIM(target_type)) > 0);

-- Updated_at trigger for schema_mappings
CREATE TRIGGER trg_schema_mappings_updated_at
    BEFORE UPDATE ON data_gateway_service.schema_mappings
    FOR EACH ROW
    EXECUTE FUNCTION data_gateway_service.set_updated_at();

-- =============================================================================
-- Table 8: data_gateway_service.aggregation_rules
-- =============================================================================
-- Rules for multi-source aggregation. Each rule defines how to combine
-- results from multiple source types with conflict resolution strategy.
-- Used for cross-source join, merge, and deduplication operations.
-- Tenant-scoped.

CREATE TABLE data_gateway_service.aggregation_rules (
    rule_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    source_types JSONB NOT NULL,
    aggregation_config JSONB NOT NULL,
    conflict_resolution VARCHAR(32) DEFAULT 'latest_wins',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rule ID must not be empty
ALTER TABLE data_gateway_service.aggregation_rules
    ADD CONSTRAINT chk_ar_rule_id_not_empty
    CHECK (LENGTH(TRIM(rule_id)) > 0);

-- Name must not be empty
ALTER TABLE data_gateway_service.aggregation_rules
    ADD CONSTRAINT chk_ar_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Conflict resolution constraint
ALTER TABLE data_gateway_service.aggregation_rules
    ADD CONSTRAINT chk_ar_conflict_resolution
    CHECK (conflict_resolution IN (
        'latest_wins', 'first_wins', 'highest_confidence',
        'manual_review', 'weighted_average', 'union_all'
    ));

-- Updated_at trigger for aggregation_rules
CREATE TRIGGER trg_aggregation_rules_updated_at
    BEFORE UPDATE ON data_gateway_service.aggregation_rules
    FOR EACH ROW
    EXECUTE FUNCTION data_gateway_service.set_updated_at();

-- =============================================================================
-- Table 9: data_gateway_service.query_metrics (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording per-query per-source performance metrics.
-- Each metric captures duration, result count, cache hit, and error flags
-- at the individual source operation level. Partitioned by recorded_at
-- for time-series queries. Retained for 365 days with compression after
-- 14 days. Tenant-scoped.

CREATE TABLE data_gateway_service.query_metrics (
    query_id VARCHAR(64) NOT NULL,
    tenant_id VARCHAR(64) NOT NULL,
    source_id VARCHAR(64) NOT NULL,
    operation VARCHAR(64) NOT NULL,
    duration_ms FLOAT DEFAULT 0,
    result_count INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    error_occurred BOOLEAN DEFAULT FALSE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (query_id, source_id, recorded_at)
);

-- Create hypertable partitioned by recorded_at
SELECT create_hypertable('data_gateway_service.query_metrics', 'recorded_at', if_not_exists => TRUE);

-- Query ID must not be empty
ALTER TABLE data_gateway_service.query_metrics
    ADD CONSTRAINT chk_qm_query_id_not_empty
    CHECK (LENGTH(TRIM(query_id)) > 0);

-- Source ID must not be empty
ALTER TABLE data_gateway_service.query_metrics
    ADD CONSTRAINT chk_qm_source_id_not_empty
    CHECK (LENGTH(TRIM(source_id)) > 0);

-- Operation must not be empty
ALTER TABLE data_gateway_service.query_metrics
    ADD CONSTRAINT chk_qm_operation_not_empty
    CHECK (LENGTH(TRIM(operation)) > 0);

-- Duration must be non-negative
ALTER TABLE data_gateway_service.query_metrics
    ADD CONSTRAINT chk_qm_duration_non_negative
    CHECK (duration_ms >= 0);

-- Result count must be non-negative
ALTER TABLE data_gateway_service.query_metrics
    ADD CONSTRAINT chk_qm_result_count_non_negative
    CHECK (result_count >= 0);

-- =============================================================================
-- Table 10: data_gateway_service.data_catalog
-- =============================================================================
-- Unified data catalog entries. Each entry describes available data from
-- a source, including domain classification, available fields, tags,
-- and sample queries. Used for data discovery, search, and
-- documentation. Tenant-scoped.

CREATE TABLE data_gateway_service.data_catalog (
    catalog_id VARCHAR(64) PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    source_id VARCHAR(64) NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    domain VARCHAR(128) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    tags JSONB DEFAULT '[]'::jsonb,
    fields_available JSONB DEFAULT '[]'::jsonb,
    sample_queries JSONB DEFAULT '[]'::jsonb,
    last_updated TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Catalog ID must not be empty
ALTER TABLE data_gateway_service.data_catalog
    ADD CONSTRAINT chk_dc_catalog_id_not_empty
    CHECK (LENGTH(TRIM(catalog_id)) > 0);

-- Source ID must not be empty
ALTER TABLE data_gateway_service.data_catalog
    ADD CONSTRAINT chk_dc_source_id_not_empty
    CHECK (LENGTH(TRIM(source_id)) > 0);

-- Source type must not be empty
ALTER TABLE data_gateway_service.data_catalog
    ADD CONSTRAINT chk_dc_source_type_not_empty
    CHECK (LENGTH(TRIM(source_type)) > 0);

-- Domain must not be empty
ALTER TABLE data_gateway_service.data_catalog
    ADD CONSTRAINT chk_dc_domain_not_empty
    CHECK (LENGTH(TRIM(domain)) > 0);

-- Name must not be empty
ALTER TABLE data_gateway_service.data_catalog
    ADD CONSTRAINT chk_dc_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- =============================================================================
-- Continuous Aggregate: data_gateway_service.data_gateway_queries_hourly
-- =============================================================================
-- Precomputed hourly query statistics by status and tenant for dashboard
-- queries, trend analysis, cache hit rate monitoring, and SLI tracking.

CREATE MATERIALIZED VIEW data_gateway_service.data_gateway_queries_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', executed_at) AS bucket,
    status,
    tenant_id,
    COUNT(*) AS total_queries,
    AVG(execution_time_ms) AS avg_execution_time_ms,
    COUNT(*) FILTER (WHERE cache_hit = TRUE) AS cache_hit_count,
    COUNT(*) FILTER (WHERE cache_hit = FALSE) AS cache_miss_count,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_count,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_count,
    SUM(result_count) AS total_results
FROM data_gateway_service.query_log
WHERE executed_at IS NOT NULL
GROUP BY bucket, status, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('data_gateway_service.data_gateway_queries_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: data_gateway_service.data_gateway_health_hourly
-- =============================================================================
-- Precomputed hourly health check statistics by source and tenant for
-- monitoring source availability, response time trends, and failure rates.

CREATE MATERIALIZED VIEW data_gateway_service.data_gateway_health_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', checked_at) AS bucket,
    source_id,
    tenant_id,
    AVG(response_time_ms) AS avg_response_time_ms,
    MAX(response_time_ms) AS max_response_time_ms,
    MIN(response_time_ms) AS min_response_time_ms,
    COUNT(*) AS total_checks,
    COUNT(*) FILTER (WHERE status = 'healthy') AS healthy_count,
    COUNT(*) FILTER (WHERE status IN ('unhealthy', 'degraded')) AS failure_count,
    MAX(consecutive_failures) AS max_consecutive_failures
FROM data_gateway_service.source_health_checks
WHERE checked_at IS NOT NULL
GROUP BY bucket, source_id, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('data_gateway_service.data_gateway_health_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- data_sources indexes
CREATE INDEX idx_ds_source_type ON data_gateway_service.data_sources(source_type);
CREATE INDEX idx_ds_status ON data_gateway_service.data_sources(status);
CREATE INDEX idx_ds_api_version ON data_gateway_service.data_sources(api_version);
CREATE INDEX idx_ds_tenant ON data_gateway_service.data_sources(tenant_id);
CREATE INDEX idx_ds_name ON data_gateway_service.data_sources(name);
CREATE INDEX idx_ds_last_health_check ON data_gateway_service.data_sources(last_health_check DESC);
CREATE INDEX idx_ds_response_time ON data_gateway_service.data_sources(response_time_ms);
CREATE INDEX idx_ds_created_at ON data_gateway_service.data_sources(created_at DESC);
CREATE INDEX idx_ds_updated_at ON data_gateway_service.data_sources(updated_at DESC);
CREATE INDEX idx_ds_tenant_type ON data_gateway_service.data_sources(tenant_id, source_type);
CREATE INDEX idx_ds_tenant_status ON data_gateway_service.data_sources(tenant_id, status);
CREATE INDEX idx_ds_capabilities ON data_gateway_service.data_sources USING GIN (capabilities);
CREATE INDEX idx_ds_supported_ops ON data_gateway_service.data_sources USING GIN (supported_operations);
CREATE INDEX idx_ds_metadata ON data_gateway_service.data_sources USING GIN (metadata);

-- source_schemas indexes
CREATE INDEX idx_ss_schema_name ON data_gateway_service.source_schemas(schema_name);
CREATE INDEX idx_ss_version ON data_gateway_service.source_schemas(version);
CREATE INDEX idx_ss_source_type ON data_gateway_service.source_schemas(source_type);
CREATE INDEX idx_ss_tenant ON data_gateway_service.source_schemas(tenant_id);
CREATE INDEX idx_ss_created_at ON data_gateway_service.source_schemas(created_at DESC);
CREATE INDEX idx_ss_tenant_type ON data_gateway_service.source_schemas(tenant_id, source_type);
CREATE INDEX idx_ss_tenant_name ON data_gateway_service.source_schemas(tenant_id, schema_name);
CREATE INDEX idx_ss_fields ON data_gateway_service.source_schemas USING GIN (fields);

-- query_templates indexes
CREATE INDEX idx_qt_name ON data_gateway_service.query_templates(name);
CREATE INDEX idx_qt_created_by ON data_gateway_service.query_templates(created_by);
CREATE INDEX idx_qt_usage_count ON data_gateway_service.query_templates(usage_count DESC);
CREATE INDEX idx_qt_tenant ON data_gateway_service.query_templates(tenant_id);
CREATE INDEX idx_qt_created_at ON data_gateway_service.query_templates(created_at DESC);
CREATE INDEX idx_qt_updated_at ON data_gateway_service.query_templates(updated_at DESC);
CREATE INDEX idx_qt_tenant_name ON data_gateway_service.query_templates(tenant_id, name);
CREATE INDEX idx_qt_query_plan ON data_gateway_service.query_templates USING GIN (query_plan);
CREATE INDEX idx_qt_parameters ON data_gateway_service.query_templates USING GIN (parameters);

-- query_log indexes (hypertable-aware)
CREATE INDEX idx_ql_status ON data_gateway_service.query_log(status, executed_at DESC);
CREATE INDEX idx_ql_tenant ON data_gateway_service.query_log(tenant_id, executed_at DESC);
CREATE INDEX idx_ql_cache_hit ON data_gateway_service.query_log(cache_hit, executed_at DESC);
CREATE INDEX idx_ql_execution_time ON data_gateway_service.query_log(execution_time_ms DESC, executed_at DESC);
CREATE INDEX idx_ql_complexity ON data_gateway_service.query_log(complexity_score DESC, executed_at DESC);
CREATE INDEX idx_ql_tenant_status ON data_gateway_service.query_log(tenant_id, status, executed_at DESC);
CREATE INDEX idx_ql_provenance ON data_gateway_service.query_log(provenance_hash, executed_at DESC);
CREATE INDEX idx_ql_sources ON data_gateway_service.query_log USING GIN (sources);
CREATE INDEX idx_ql_operations ON data_gateway_service.query_log USING GIN (operations);
CREATE INDEX idx_ql_filters ON data_gateway_service.query_log USING GIN (filters);

-- cache_entries indexes
CREATE INDEX idx_ce_query_hash ON data_gateway_service.cache_entries(query_hash);
CREATE INDEX idx_ce_source_id ON data_gateway_service.cache_entries(source_id);
CREATE INDEX idx_ce_tenant ON data_gateway_service.cache_entries(tenant_id);
CREATE INDEX idx_ce_hit_count ON data_gateway_service.cache_entries(hit_count DESC);
CREATE INDEX idx_ce_size_bytes ON data_gateway_service.cache_entries(size_bytes DESC);
CREATE INDEX idx_ce_last_accessed ON data_gateway_service.cache_entries(last_accessed DESC);
CREATE INDEX idx_ce_created_at ON data_gateway_service.cache_entries(created_at DESC);
CREATE INDEX idx_ce_expires_at ON data_gateway_service.cache_entries(expires_at);
CREATE INDEX idx_ce_tenant_source ON data_gateway_service.cache_entries(tenant_id, source_id);
CREATE INDEX idx_ce_tenant_hash ON data_gateway_service.cache_entries(tenant_id, query_hash);

-- source_health_checks indexes (hypertable-aware)
CREATE INDEX idx_shc_source ON data_gateway_service.source_health_checks(source_id, checked_at DESC);
CREATE INDEX idx_shc_status ON data_gateway_service.source_health_checks(status, checked_at DESC);
CREATE INDEX idx_shc_tenant ON data_gateway_service.source_health_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_shc_response_time ON data_gateway_service.source_health_checks(response_time_ms DESC, checked_at DESC);
CREATE INDEX idx_shc_failures ON data_gateway_service.source_health_checks(consecutive_failures DESC, checked_at DESC);
CREATE INDEX idx_shc_tenant_source ON data_gateway_service.source_health_checks(tenant_id, source_id, checked_at DESC);
CREATE INDEX idx_shc_tenant_status ON data_gateway_service.source_health_checks(tenant_id, status, checked_at DESC);

-- schema_mappings indexes
CREATE INDEX idx_sm_source_type ON data_gateway_service.schema_mappings(source_type);
CREATE INDEX idx_sm_target_type ON data_gateway_service.schema_mappings(target_type);
CREATE INDEX idx_sm_tenant ON data_gateway_service.schema_mappings(tenant_id);
CREATE INDEX idx_sm_created_at ON data_gateway_service.schema_mappings(created_at DESC);
CREATE INDEX idx_sm_updated_at ON data_gateway_service.schema_mappings(updated_at DESC);
CREATE INDEX idx_sm_tenant_source ON data_gateway_service.schema_mappings(tenant_id, source_type);
CREATE INDEX idx_sm_tenant_target ON data_gateway_service.schema_mappings(tenant_id, target_type);
CREATE INDEX idx_sm_source_target ON data_gateway_service.schema_mappings(source_type, target_type);
CREATE INDEX idx_sm_mappings ON data_gateway_service.schema_mappings USING GIN (mappings);

-- aggregation_rules indexes
CREATE INDEX idx_ar_name ON data_gateway_service.aggregation_rules(name);
CREATE INDEX idx_ar_conflict_resolution ON data_gateway_service.aggregation_rules(conflict_resolution);
CREATE INDEX idx_ar_tenant ON data_gateway_service.aggregation_rules(tenant_id);
CREATE INDEX idx_ar_created_at ON data_gateway_service.aggregation_rules(created_at DESC);
CREATE INDEX idx_ar_updated_at ON data_gateway_service.aggregation_rules(updated_at DESC);
CREATE INDEX idx_ar_tenant_name ON data_gateway_service.aggregation_rules(tenant_id, name);
CREATE INDEX idx_ar_source_types ON data_gateway_service.aggregation_rules USING GIN (source_types);
CREATE INDEX idx_ar_aggregation_config ON data_gateway_service.aggregation_rules USING GIN (aggregation_config);

-- query_metrics indexes (hypertable-aware)
CREATE INDEX idx_qm_query ON data_gateway_service.query_metrics(query_id, recorded_at DESC);
CREATE INDEX idx_qm_source ON data_gateway_service.query_metrics(source_id, recorded_at DESC);
CREATE INDEX idx_qm_operation ON data_gateway_service.query_metrics(operation, recorded_at DESC);
CREATE INDEX idx_qm_tenant ON data_gateway_service.query_metrics(tenant_id, recorded_at DESC);
CREATE INDEX idx_qm_duration ON data_gateway_service.query_metrics(duration_ms DESC, recorded_at DESC);
CREATE INDEX idx_qm_cache_hit ON data_gateway_service.query_metrics(cache_hit, recorded_at DESC);
CREATE INDEX idx_qm_error ON data_gateway_service.query_metrics(error_occurred, recorded_at DESC);
CREATE INDEX idx_qm_tenant_source ON data_gateway_service.query_metrics(tenant_id, source_id, recorded_at DESC);
CREATE INDEX idx_qm_tenant_operation ON data_gateway_service.query_metrics(tenant_id, operation, recorded_at DESC);

-- data_catalog indexes
CREATE INDEX idx_dc_source_id ON data_gateway_service.data_catalog(source_id);
CREATE INDEX idx_dc_source_type ON data_gateway_service.data_catalog(source_type);
CREATE INDEX idx_dc_domain ON data_gateway_service.data_catalog(domain);
CREATE INDEX idx_dc_name ON data_gateway_service.data_catalog(name);
CREATE INDEX idx_dc_tenant ON data_gateway_service.data_catalog(tenant_id);
CREATE INDEX idx_dc_last_updated ON data_gateway_service.data_catalog(last_updated DESC);
CREATE INDEX idx_dc_created_at ON data_gateway_service.data_catalog(created_at DESC);
CREATE INDEX idx_dc_tenant_source ON data_gateway_service.data_catalog(tenant_id, source_id);
CREATE INDEX idx_dc_tenant_type ON data_gateway_service.data_catalog(tenant_id, source_type);
CREATE INDEX idx_dc_tenant_domain ON data_gateway_service.data_catalog(tenant_id, domain);
CREATE INDEX idx_dc_tags ON data_gateway_service.data_catalog USING GIN (tags);
CREATE INDEX idx_dc_fields_available ON data_gateway_service.data_catalog USING GIN (fields_available);
CREATE INDEX idx_dc_sample_queries ON data_gateway_service.data_catalog USING GIN (sample_queries);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE data_gateway_service.data_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY ds_tenant_read ON data_gateway_service.data_sources
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ds_tenant_write ON data_gateway_service.data_sources
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.source_schemas ENABLE ROW LEVEL SECURITY;
CREATE POLICY ss_tenant_read ON data_gateway_service.source_schemas
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ss_tenant_write ON data_gateway_service.source_schemas
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.query_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY qt_tenant_read ON data_gateway_service.query_templates
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qt_tenant_write ON data_gateway_service.query_templates
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.query_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY ql_tenant_read ON data_gateway_service.query_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ql_tenant_write ON data_gateway_service.query_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.cache_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY ce_tenant_read ON data_gateway_service.cache_entries
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ce_tenant_write ON data_gateway_service.cache_entries
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.source_health_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY shc_tenant_read ON data_gateway_service.source_health_checks
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY shc_tenant_write ON data_gateway_service.source_health_checks
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.schema_mappings ENABLE ROW LEVEL SECURITY;
CREATE POLICY sm_tenant_read ON data_gateway_service.schema_mappings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY sm_tenant_write ON data_gateway_service.schema_mappings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.aggregation_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY ar_tenant_read ON data_gateway_service.aggregation_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ar_tenant_write ON data_gateway_service.aggregation_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.query_metrics ENABLE ROW LEVEL SECURITY;
CREATE POLICY qm_tenant_read ON data_gateway_service.query_metrics
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY qm_tenant_write ON data_gateway_service.query_metrics
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE data_gateway_service.data_catalog ENABLE ROW LEVEL SECURITY;
CREATE POLICY dc_tenant_read ON data_gateway_service.data_catalog
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dc_tenant_write ON data_gateway_service.data_catalog
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA data_gateway_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA data_gateway_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA data_gateway_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON data_gateway_service.data_gateway_queries_hourly TO greenlang_app;
GRANT SELECT ON data_gateway_service.data_gateway_health_hourly TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA data_gateway_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA data_gateway_service TO greenlang_readonly;
GRANT SELECT ON data_gateway_service.data_gateway_queries_hourly TO greenlang_readonly;
GRANT SELECT ON data_gateway_service.data_gateway_health_hourly TO greenlang_readonly;

-- Add data gateway service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'data_gateway:sources:read', 'data_gateway', 'sources_read', 'View data source registry and health status'),
    (gen_random_uuid(), 'data_gateway:sources:write', 'data_gateway', 'sources_write', 'Create and manage data source registrations'),
    (gen_random_uuid(), 'data_gateway:schemas:read', 'data_gateway', 'schemas_read', 'View source schema definitions'),
    (gen_random_uuid(), 'data_gateway:schemas:write', 'data_gateway', 'schemas_write', 'Create and manage source schema definitions'),
    (gen_random_uuid(), 'data_gateway:queries:read', 'data_gateway', 'queries_read', 'View query logs and templates'),
    (gen_random_uuid(), 'data_gateway:queries:write', 'data_gateway', 'queries_write', 'Execute queries and manage query templates'),
    (gen_random_uuid(), 'data_gateway:cache:read', 'data_gateway', 'cache_read', 'View cache entries and statistics'),
    (gen_random_uuid(), 'data_gateway:cache:write', 'data_gateway', 'cache_write', 'Manage cache entries and invalidation'),
    (gen_random_uuid(), 'data_gateway:health:read', 'data_gateway', 'health_read', 'View source health check history'),
    (gen_random_uuid(), 'data_gateway:health:write', 'data_gateway', 'health_write', 'Trigger and manage health checks'),
    (gen_random_uuid(), 'data_gateway:mappings:read', 'data_gateway', 'mappings_read', 'View schema mappings and aggregation rules'),
    (gen_random_uuid(), 'data_gateway:mappings:write', 'data_gateway', 'mappings_write', 'Create and manage schema mappings and aggregation rules'),
    (gen_random_uuid(), 'data_gateway:metrics:read', 'data_gateway', 'metrics_read', 'View query performance metrics'),
    (gen_random_uuid(), 'data_gateway:metrics:write', 'data_gateway', 'metrics_write', 'Record query performance metrics'),
    (gen_random_uuid(), 'data_gateway:catalog:read', 'data_gateway', 'catalog_read', 'View data catalog entries'),
    (gen_random_uuid(), 'data_gateway:catalog:write', 'data_gateway', 'catalog_write', 'Create and manage data catalog entries'),
    (gen_random_uuid(), 'data_gateway:admin', 'data_gateway', 'admin', 'Data gateway service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep query log records for 730 days (2 years)
SELECT add_retention_policy('data_gateway_service.query_log', INTERVAL '730 days');

-- Keep source health check records for 730 days (2 years)
SELECT add_retention_policy('data_gateway_service.source_health_checks', INTERVAL '730 days');

-- Keep query metrics records for 365 days (1 year)
SELECT add_retention_policy('data_gateway_service.query_metrics', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on query_log after 30 days
ALTER TABLE data_gateway_service.query_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'executed_at DESC'
);

SELECT add_compression_policy('data_gateway_service.query_log', INTERVAL '30 days');

-- Enable compression on source_health_checks after 30 days
ALTER TABLE data_gateway_service.source_health_checks SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'checked_at DESC'
);

SELECT add_compression_policy('data_gateway_service.source_health_checks', INTERVAL '30 days');

-- Enable compression on query_metrics after 14 days
ALTER TABLE data_gateway_service.query_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'recorded_at DESC'
);

SELECT add_compression_policy('data_gateway_service.query_metrics', INTERVAL '14 days');

-- =============================================================================
-- Seed: Register the Data Gateway Agent (GL-DATA-GW-001)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-GW-001', 'Data Gateway Agent',
 'Unified API gateway for all GreenLang data agents. Provides a single entry point for querying across PDF extractors, Excel normalizers, ERP connectors, EUDR traceability, SCADA/BMS, fleet telematics, utility tariffs, supplier portals, and other data sources. Features source registry with health monitoring, query routing with caching, schema mapping and cross-source field normalization, multi-source aggregation with configurable conflict resolution, query templates, performance metrics tracking, and a unified data catalog for discovery.',
 2, 'async', true, true, 10, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/data-gateway', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Data Gateway Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-GW-001', '1.0.0',
 '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "1Gi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/data-gateway-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"data", "gateway", "routing", "caching", "aggregation", "catalog", "health-monitoring"}',
 '{"cross-sector", "energy", "manufacturing", "agriculture", "logistics"}',
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Data Gateway Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-DATA-GW-001', '1.0.0', 'source_registry', 'data_management',
 'Register, discover, and manage backend data agent endpoints with health status tracking, capability enumeration, and API version management',
 '{"source_type", "endpoint_url", "name"}', '{"source_id", "status", "capabilities"}',
 '{"supported_source_types": ["pdf_extractor", "excel_normalizer", "erp_connector", "eudr_traceability", "scada_bms", "fleet_telematics", "utility_tariff", "supplier_portal", "event_processor", "document_classifier", "ocr_agent", "email_processor", "iot_meter", "emission_factor", "custom"], "auto_discovery": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'query_routing', 'query',
 'Route queries to appropriate backend data agents based on source type, capabilities, and availability with automatic failover and load balancing',
 '{"sources", "operations", "filters"}', '{"query_id", "results", "execution_time_ms", "cache_hit"}',
 '{"max_sources_per_query": 10, "timeout_seconds": 60, "retry_count": 3, "failover_enabled": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'query_caching', 'performance',
 'Cache query results with configurable TTL, cache invalidation, and hit rate monitoring for improved response times and reduced backend load',
 '{"query_hash", "source_id"}', '{"cache_key", "hit_count", "size_bytes", "expires_at"}',
 '{"default_ttl_seconds": 3600, "max_cache_size_mb": 512, "eviction_policy": "lru", "invalidation_on_source_update": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'health_monitoring', 'monitoring',
 'Continuously monitor data source health with configurable check intervals, consecutive failure tracking, and automatic status updates',
 '{"source_id"}', '{"status", "response_time_ms", "consecutive_failures"}',
 '{"check_interval_seconds": 30, "unhealthy_threshold": 3, "degraded_threshold": 1, "timeout_seconds": 10}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'schema_mapping', 'transformation',
 'Map and normalize fields across different source schemas for cross-source query compatibility and unified response formatting',
 '{"source_type", "target_type", "mappings"}', '{"mapping_id", "normalized_fields"}',
 '{"auto_detect_mappings": true, "fuzzy_matching_threshold": 0.85, "preserve_original_fields": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'multi_source_aggregation', 'aggregation',
 'Aggregate and merge results from multiple data sources with configurable conflict resolution strategies and deduplication',
 '{"source_types", "aggregation_config", "conflict_resolution"}', '{"aggregated_results", "source_count", "conflict_count"}',
 '{"conflict_strategies": ["latest_wins", "first_wins", "highest_confidence", "manual_review", "weighted_average", "union_all"], "dedup_enabled": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'query_templates', 'query',
 'Create, manage, and execute reusable parameterized query templates with usage tracking and parameter validation',
 '{"name", "query_plan", "parameters"}', '{"template_id", "usage_count", "results"}',
 '{"max_templates_per_tenant": 500, "parameter_validation": true, "template_versioning": true}'::jsonb),

('GL-DATA-GW-001', '1.0.0', 'data_catalog', 'discovery',
 'Unified data catalog for discovering available data across all registered sources with domain classification, field inventory, tags, and sample queries',
 '{"domain", "source_type", "search_query"}', '{"catalog_entries", "total_count", "domains"}',
 '{"auto_catalog_on_registration": true, "full_text_search": true, "tag_based_filtering": true}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Data Gateway Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Data Gateway depends on Schema Compiler for query/response validation
('GL-DATA-GW-001', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Query plans, source schemas, and aggregation results are validated against JSON Schema definitions'),

-- Data Gateway depends on Registry for agent discovery
('GL-DATA-GW-001', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent version and capability lookup for source registration and health monitoring'),

-- Data Gateway depends on Access Guard for policy enforcement
('GL-DATA-GW-001', 'GL-FOUND-X-006', '>=1.0.0', false,
 'Data classification and access control enforcement for query routing and cross-tenant isolation'),

-- Data Gateway depends on Observability Agent for metrics
('GL-DATA-GW-001', 'GL-FOUND-X-010', '>=1.0.0', false,
 'Query performance metrics, health check telemetry, and cache statistics are reported to observability'),

-- Data Gateway optionally uses Citations for provenance tracking
('GL-DATA-GW-001', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Query results and aggregated data provenance chains are registered with the citation service for audit trail'),

-- Data Gateway optionally uses Reproducibility for determinism
('GL-DATA-GW-001', 'GL-FOUND-X-008', '>=1.0.0', true,
 'Cached query results and aggregation outputs are verified for reproducibility across re-execution')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Data Gateway Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-GW-001', 'Data Gateway Agent',
 'Unified API gateway for all GreenLang data agents. Single entry point for querying PDF extractors, Excel normalizers, ERP connectors, EUDR traceability, and 10+ other data source types. Features source registry with health monitoring, query routing with caching, schema mapping, multi-source aggregation, query templates, performance metrics, and a unified data catalog.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA data_gateway_service IS 'Data Gateway Agent for GreenLang Climate OS (AGENT-DATA-004) - Unified API gateway for all data agents with source registry, query routing, schema mapping, caching, health monitoring, multi-source aggregation, and data catalog';
COMMENT ON TABLE data_gateway_service.data_sources IS 'Registry of backend data agents with endpoint URLs, source types, health status, capabilities, supported operations, and response time metrics';
COMMENT ON TABLE data_gateway_service.source_schemas IS 'Versioned schema definitions per source type with field specifications for query validation and data catalog generation';
COMMENT ON TABLE data_gateway_service.query_templates IS 'Reusable parameterized query templates with usage tracking for saved query execution across data sources';
COMMENT ON TABLE data_gateway_service.query_log IS 'TimescaleDB hypertable: audit log of all gateway queries with sources, operations, filters, status, execution time, cache hit, and provenance hash';
COMMENT ON TABLE data_gateway_service.cache_entries IS 'Cache metadata and statistics with query hash, result count, size, hit count, and expiration for cache management and eviction';
COMMENT ON TABLE data_gateway_service.source_health_checks IS 'TimescaleDB hypertable: health check history for data sources with status, response time, error messages, and consecutive failure count';
COMMENT ON TABLE data_gateway_service.schema_mappings IS 'Field mapping definitions between source and target schemas for cross-source query normalization and unified response formatting';
COMMENT ON TABLE data_gateway_service.aggregation_rules IS 'Rules for multi-source result aggregation with conflict resolution strategies (latest_wins, first_wins, highest_confidence, etc.)';
COMMENT ON TABLE data_gateway_service.query_metrics IS 'TimescaleDB hypertable: per-query per-source performance metrics with duration, result count, cache hit, and error flags';
COMMENT ON TABLE data_gateway_service.data_catalog IS 'Unified data catalog entries describing available data from each source with domain classification, fields, tags, and sample queries';
COMMENT ON MATERIALIZED VIEW data_gateway_service.data_gateway_queries_hourly IS 'Continuous aggregate: hourly query statistics by status with count, avg execution time, cache hit rate, and result totals for dashboard queries and SLI tracking';
COMMENT ON MATERIALIZED VIEW data_gateway_service.data_gateway_health_hourly IS 'Continuous aggregate: hourly health check statistics by source with avg/max/min response time, failure count, and consecutive failure tracking';

COMMENT ON COLUMN data_gateway_service.data_sources.source_type IS 'Backend data agent type: pdf_extractor, excel_normalizer, erp_connector, eudr_traceability, scada_bms, fleet_telematics, utility_tariff, supplier_portal, event_processor, document_classifier, ocr_agent, email_processor, iot_meter, emission_factor, custom';
COMMENT ON COLUMN data_gateway_service.data_sources.status IS 'Current health status: healthy, degraded, unhealthy, unknown, maintenance';
COMMENT ON COLUMN data_gateway_service.data_sources.capabilities IS 'JSONB array of source capabilities (e.g., search, filter, aggregate, export)';
COMMENT ON COLUMN data_gateway_service.data_sources.supported_operations IS 'JSONB array of supported query operations (e.g., read, list, search, count)';
COMMENT ON COLUMN data_gateway_service.data_sources.response_time_ms IS 'Latest health check response time in milliseconds';
COMMENT ON COLUMN data_gateway_service.query_log.status IS 'Query execution status: pending, executing, completed, failed, timeout, cancelled';
COMMENT ON COLUMN data_gateway_service.query_log.cache_hit IS 'Whether query results were served from cache';
COMMENT ON COLUMN data_gateway_service.query_log.complexity_score IS 'Computed query complexity score based on source count, operations, and filters';
COMMENT ON COLUMN data_gateway_service.query_log.provenance_hash IS 'SHA-256 provenance hash of query inputs and results for reproducibility verification';
COMMENT ON COLUMN data_gateway_service.cache_entries.query_hash IS 'SHA-256 hash of query parameters for cache key matching';
COMMENT ON COLUMN data_gateway_service.cache_entries.size_bytes IS 'Cached result size in bytes for memory management and eviction';
COMMENT ON COLUMN data_gateway_service.cache_entries.expires_at IS 'Cache entry expiration timestamp for TTL-based eviction';
COMMENT ON COLUMN data_gateway_service.source_health_checks.consecutive_failures IS 'Number of consecutive health check failures for circuit breaker logic';
COMMENT ON COLUMN data_gateway_service.schema_mappings.mappings IS 'JSONB field mapping definitions from source to target schema fields';
COMMENT ON COLUMN data_gateway_service.aggregation_rules.conflict_resolution IS 'Conflict resolution strategy: latest_wins, first_wins, highest_confidence, manual_review, weighted_average, union_all';
COMMENT ON COLUMN data_gateway_service.data_catalog.domain IS 'Data domain classification (e.g., emissions, energy, waste, water, supply-chain)';
COMMENT ON COLUMN data_gateway_service.data_catalog.tags IS 'JSONB array of searchable tags for catalog entry discovery';
COMMENT ON COLUMN data_gateway_service.data_catalog.fields_available IS 'JSONB array of available fields from this data source for query planning';
COMMENT ON COLUMN data_gateway_service.data_catalog.sample_queries IS 'JSONB array of example queries demonstrating source usage patterns';
