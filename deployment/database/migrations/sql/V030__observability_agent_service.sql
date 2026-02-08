-- =============================================================================
-- GreenLang Climate OS - Observability Agent Service Schema
-- =============================================================================
-- Migration: V030
-- Component: AGENT-FOUND-010 Observability Agent
-- Description: Creates observability_agent_service schema with metric_definitions,
--              metric_recordings (hypertable), trace_spans (hypertable),
--              log_entries (hypertable), alert_rules, alert_instances,
--              health_check_results, dashboard_configs, slo_definitions,
--              obs_audit_log (hypertable), continuous aggregates for hourly
--              metric stats and hourly audit stats, 50+ indexes, RLS policies,
--              14 security permissions, retention policies, compression, and
--              seed data.
-- Previous: V029__qa_test_harness_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS observability_agent_service;

-- =============================================================================
-- Table: observability_agent_service.metric_definitions
-- =============================================================================
-- Metric definition registry. Each definition describes a named metric with
-- its type (counter, gauge, histogram, summary), label keys, and optional
-- histogram buckets. Metric names are unique within a tenant. Used as the
-- authoritative catalog of all platform and application metrics.

CREATE TABLE observability_agent_service.metric_definitions (
    definition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(20) NOT NULL,
    description TEXT DEFAULT '',
    label_keys TEXT[] DEFAULT '{}',
    buckets FLOAT[] DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Metric type must be one of the supported types
ALTER TABLE observability_agent_service.metric_definitions
    ADD CONSTRAINT chk_metric_def_type
    CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'summary'));

-- Metric name must not be empty
ALTER TABLE observability_agent_service.metric_definitions
    ADD CONSTRAINT chk_metric_def_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Unique metric name per tenant
ALTER TABLE observability_agent_service.metric_definitions
    ADD CONSTRAINT uq_metric_def_name_tenant
    UNIQUE (name, tenant_id);

-- =============================================================================
-- Table: observability_agent_service.metric_recordings (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording individual metric data points. Each
-- recording captures the metric name, numeric value, label set, timestamp,
-- tenant, and provenance hash. Partitioned by recorded_at for time-series
-- queries. Retained for 7 days with compression after 1 day.

CREATE TABLE observability_agent_service.metric_recordings (
    recording_id UUID NOT NULL DEFAULT gen_random_uuid(),
    metric_name VARCHAR(255) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    labels JSONB NOT NULL DEFAULT '{}'::jsonb,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64),
    PRIMARY KEY (recording_id, recorded_at)
);

-- Create hypertable partitioned by recorded_at
SELECT create_hypertable('observability_agent_service.metric_recordings', 'recorded_at', if_not_exists => TRUE);

-- Metric name must not be empty
ALTER TABLE observability_agent_service.metric_recordings
    ADD CONSTRAINT chk_metric_rec_name_not_empty
    CHECK (LENGTH(TRIM(metric_name)) > 0);

-- Provenance hash must be 64-character hex when present
ALTER TABLE observability_agent_service.metric_recordings
    ADD CONSTRAINT chk_metric_rec_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: observability_agent_service.trace_spans (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording distributed trace spans. Each span
-- captures the trace ID, parent span, operation and service names, status,
-- start/end times, duration, attributes, events, tenant, and provenance
-- hash. Partitioned by start_time for time-series queries. Retained for
-- 14 days with compression after 2 days.

CREATE TABLE observability_agent_service.trace_spans (
    span_id UUID NOT NULL DEFAULT gen_random_uuid(),
    trace_id VARCHAR(64) NOT NULL,
    parent_span_id VARCHAR(64) DEFAULT NULL,
    operation_name VARCHAR(255) NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'unset',
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ DEFAULT NULL,
    duration_ms DOUBLE PRECISION DEFAULT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    events JSONB NOT NULL DEFAULT '[]'::jsonb,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    provenance_hash VARCHAR(64),
    PRIMARY KEY (span_id, start_time)
);

-- Create hypertable partitioned by start_time
SELECT create_hypertable('observability_agent_service.trace_spans', 'start_time', if_not_exists => TRUE);

-- Span status constraint
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_status
    CHECK (status IN ('unset', 'ok', 'error'));

-- Operation name must not be empty
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_operation_not_empty
    CHECK (LENGTH(TRIM(operation_name)) > 0);

-- Service name must not be empty
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_service_not_empty
    CHECK (LENGTH(TRIM(service_name)) > 0);

-- Trace ID must not be empty
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_trace_id_not_empty
    CHECK (LENGTH(TRIM(trace_id)) > 0);

-- Duration must be non-negative when present
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- Provenance hash must be 64-character hex when present
ALTER TABLE observability_agent_service.trace_spans
    ADD CONSTRAINT chk_span_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: observability_agent_service.log_entries (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording structured log entries. Each entry
-- captures the log level, message, correlation/trace/span IDs, agent ID,
-- tenant, attributes, and provenance hash. Partitioned by logged_at for
-- time-series queries. Retained for 30 days with compression after 3 days.

CREATE TABLE observability_agent_service.log_entries (
    log_id UUID NOT NULL DEFAULT gen_random_uuid(),
    logged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    correlation_id VARCHAR(64) DEFAULT NULL,
    trace_id VARCHAR(64) DEFAULT NULL,
    span_id VARCHAR(64) DEFAULT NULL,
    agent_id VARCHAR(50) DEFAULT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(64),
    PRIMARY KEY (log_id, logged_at)
);

-- Create hypertable partitioned by logged_at
SELECT create_hypertable('observability_agent_service.log_entries', 'logged_at', if_not_exists => TRUE);

-- Log level constraint
ALTER TABLE observability_agent_service.log_entries
    ADD CONSTRAINT chk_log_level
    CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'));

-- Message must not be empty
ALTER TABLE observability_agent_service.log_entries
    ADD CONSTRAINT chk_log_message_not_empty
    CHECK (LENGTH(TRIM(message)) > 0);

-- Provenance hash must be 64-character hex when present
ALTER TABLE observability_agent_service.log_entries
    ADD CONSTRAINT chk_log_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- =============================================================================
-- Table: observability_agent_service.alert_rules
-- =============================================================================
-- Alert rule definitions. Each rule specifies a metric name, comparison
-- condition, threshold value, evaluation duration, severity, labels,
-- annotations, optional silence window, and enabled flag. Used by the
-- alerting engine to evaluate metric recordings and fire alert instances.

CREATE TABLE observability_agent_service.alert_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    condition VARCHAR(10) NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    duration_seconds INTEGER NOT NULL DEFAULT 60,
    severity VARCHAR(20) NOT NULL DEFAULT 'warning',
    labels JSONB NOT NULL DEFAULT '{}'::jsonb,
    annotations JSONB NOT NULL DEFAULT '{}'::jsonb,
    silenced_until TIMESTAMPTZ DEFAULT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Condition constraint
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT chk_alert_rule_condition
    CHECK (condition IN ('gt', 'lt', 'eq', 'gte', 'lte', 'ne'));

-- Severity constraint
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT chk_alert_rule_severity
    CHECK (severity IN ('info', 'warning', 'error', 'critical'));

-- Duration must be positive
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT chk_alert_rule_duration_positive
    CHECK (duration_seconds > 0);

-- Rule name must not be empty
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT chk_alert_rule_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Metric name must not be empty
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT chk_alert_rule_metric_not_empty
    CHECK (LENGTH(TRIM(metric_name)) > 0);

-- Unique alert rule name per tenant
ALTER TABLE observability_agent_service.alert_rules
    ADD CONSTRAINT uq_alert_rule_name_tenant
    UNIQUE (name, tenant_id);

-- =============================================================================
-- Table: observability_agent_service.alert_instances
-- =============================================================================
-- Alert instance records. Each instance is created when an alert rule fires
-- and tracks the lifecycle of the alert (firing, pending, acknowledged,
-- resolved). References the originating rule, captures severity, metric
-- name, value, threshold, labels, annotations, and timestamps.

CREATE TABLE observability_agent_service.alert_instances (
    instance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id UUID NOT NULL REFERENCES observability_agent_service.alert_rules(rule_id) ON DELETE CASCADE,
    rule_name VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    labels JSONB NOT NULL DEFAULT '{}'::jsonb,
    annotations JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ DEFAULT NULL,
    acknowledged_at TIMESTAMPTZ DEFAULT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Status constraint
ALTER TABLE observability_agent_service.alert_instances
    ADD CONSTRAINT chk_alert_inst_status
    CHECK (status IN ('firing', 'resolved', 'pending', 'acknowledged'));

-- Severity constraint
ALTER TABLE observability_agent_service.alert_instances
    ADD CONSTRAINT chk_alert_inst_severity
    CHECK (severity IN ('info', 'warning', 'error', 'critical'));

-- =============================================================================
-- Table: observability_agent_service.health_check_results
-- =============================================================================
-- Health check probe results. Each result records the probe name, type
-- (liveness, readiness, startup), service name, status (healthy, degraded,
-- unhealthy), message, details, probe duration, and timestamp. Used for
-- service health monitoring and Kubernetes probe status tracking.

CREATE TABLE observability_agent_service.health_check_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    probe_name VARCHAR(255) NOT NULL,
    probe_type VARCHAR(20) NOT NULL,
    service_name VARCHAR(100) DEFAULT NULL,
    status VARCHAR(20) NOT NULL,
    message TEXT DEFAULT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    duration_ms DOUBLE PRECISION DEFAULT NULL,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Probe type constraint
ALTER TABLE observability_agent_service.health_check_results
    ADD CONSTRAINT chk_health_probe_type
    CHECK (probe_type IN ('liveness', 'readiness', 'startup'));

-- Status constraint
ALTER TABLE observability_agent_service.health_check_results
    ADD CONSTRAINT chk_health_status
    CHECK (status IN ('healthy', 'degraded', 'unhealthy'));

-- Probe name must not be empty
ALTER TABLE observability_agent_service.health_check_results
    ADD CONSTRAINT chk_health_probe_name_not_empty
    CHECK (LENGTH(TRIM(probe_name)) > 0);

-- Duration must be non-negative when present
ALTER TABLE observability_agent_service.health_check_results
    ADD CONSTRAINT chk_health_duration_non_negative
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- =============================================================================
-- Table: observability_agent_service.dashboard_configs
-- =============================================================================
-- Dashboard configuration store. Each dashboard defines a name, description,
-- panel layout (JSONB array), default time range, refresh interval,
-- template variables, and tenant scope. Used by the observability UI to
-- render metric visualization dashboards.

CREATE TABLE observability_agent_service.dashboard_configs (
    dashboard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    panels JSONB NOT NULL DEFAULT '[]'::jsonb,
    time_range VARCHAR(20) NOT NULL DEFAULT '1h',
    refresh_interval_seconds INTEGER NOT NULL DEFAULT 30,
    variables JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Dashboard name must not be empty
ALTER TABLE observability_agent_service.dashboard_configs
    ADD CONSTRAINT chk_dashboard_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Refresh interval must be positive
ALTER TABLE observability_agent_service.dashboard_configs
    ADD CONSTRAINT chk_dashboard_refresh_positive
    CHECK (refresh_interval_seconds > 0);

-- =============================================================================
-- Table: observability_agent_service.slo_definitions
-- =============================================================================
-- Service Level Objective definitions. Each SLO specifies a service, SLO
-- type (availability, latency, throughput, error_rate, saturation), target
-- (0-1 range), evaluation window in days, burn rate thresholds, and
-- enabled flag. Used by the SLO engine to calculate error budgets and
-- trigger burn-rate alerts.

CREATE TABLE observability_agent_service.slo_definitions (
    slo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    service_name VARCHAR(100) NOT NULL,
    slo_type VARCHAR(20) NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    window_days INTEGER NOT NULL DEFAULT 30,
    burn_rate_thresholds JSONB NOT NULL DEFAULT '{}'::jsonb,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- SLO type constraint
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT chk_slo_type
    CHECK (slo_type IN ('availability', 'latency', 'throughput', 'error_rate', 'saturation'));

-- Target must be between 0 and 1
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT chk_slo_target_range
    CHECK (target >= 0 AND target <= 1);

-- Window days must be positive
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT chk_slo_window_positive
    CHECK (window_days > 0);

-- SLO name must not be empty
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT chk_slo_name_not_empty
    CHECK (LENGTH(TRIM(name)) > 0);

-- Service name must not be empty
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT chk_slo_service_not_empty
    CHECK (LENGTH(TRIM(service_name)) > 0);

-- Unique SLO name per tenant
ALTER TABLE observability_agent_service.slo_definitions
    ADD CONSTRAINT uq_slo_name_tenant
    UNIQUE (name, tenant_id);

-- =============================================================================
-- Table: observability_agent_service.obs_audit_log (hypertable)
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- observability agent operations. Each event captures the entity being
-- operated on, action, data hashes (current, previous, chain), details
-- (JSONB), user, tenant, and timestamp. Partitioned by created_at for
-- time-series queries. Retained for 365 days with compression after 30
-- days.

CREATE TABLE observability_agent_service.obs_audit_log (
    audit_id UUID NOT NULL DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    action VARCHAR(50) NOT NULL,
    data_hash VARCHAR(64) NOT NULL,
    previous_hash VARCHAR(64) DEFAULT NULL,
    chain_hash VARCHAR(64) NOT NULL,
    user_id VARCHAR(100) NOT NULL DEFAULT 'system',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (audit_id, created_at)
);

-- Create hypertable partitioned by created_at
SELECT create_hypertable('observability_agent_service.obs_audit_log', 'created_at', if_not_exists => TRUE);

-- Entity type constraint
ALTER TABLE observability_agent_service.obs_audit_log
    ADD CONSTRAINT chk_obs_audit_entity_type
    CHECK (entity_type IN (
        'metric_definition', 'metric_recording', 'trace_span',
        'log_entry', 'alert_rule', 'alert_instance',
        'health_check', 'dashboard', 'slo_definition', 'system'
    ));

-- Action constraint
ALTER TABLE observability_agent_service.obs_audit_log
    ADD CONSTRAINT chk_obs_audit_action
    CHECK (action IN (
        'create', 'update', 'delete', 'record', 'query',
        'fire', 'resolve', 'acknowledge', 'silence',
        'evaluate', 'probe', 'export', 'import',
        'activate', 'deactivate', 'system', 'admin'
    ));

-- Data hash must be 64-character hex
ALTER TABLE observability_agent_service.obs_audit_log
    ADD CONSTRAINT chk_obs_audit_data_hash_length
    CHECK (LENGTH(data_hash) = 64);

-- Previous hash must be 64-character hex when present
ALTER TABLE observability_agent_service.obs_audit_log
    ADD CONSTRAINT chk_obs_audit_previous_hash_length
    CHECK (previous_hash IS NULL OR LENGTH(previous_hash) = 64);

-- Chain hash must be 64-character hex
ALTER TABLE observability_agent_service.obs_audit_log
    ADD CONSTRAINT chk_obs_audit_chain_hash_length
    CHECK (LENGTH(chain_hash) = 64);

-- =============================================================================
-- Continuous Aggregate: observability_agent_service.hourly_metric_stats
-- =============================================================================
-- Precomputed hourly metric recording statistics by metric name for
-- dashboard queries, trend analysis, and SLI tracking. Shows the number
-- of recordings, average/min/max values per metric per hour.

CREATE MATERIALIZED VIEW observability_agent_service.hourly_metric_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', recorded_at) AS bucket,
    metric_name,
    tenant_id,
    COUNT(*) AS recording_count,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    SUM(value) AS sum_value
FROM observability_agent_service.metric_recordings
WHERE recorded_at IS NOT NULL
GROUP BY bucket, metric_name, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('observability_agent_service.hourly_metric_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: observability_agent_service.hourly_audit_stats
-- =============================================================================
-- Precomputed hourly counts of audit events by entity type and action
-- for compliance reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW observability_agent_service.hourly_audit_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    entity_type,
    action,
    tenant_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT entity_id) AS unique_entities,
    COUNT(DISTINCT user_id) AS unique_users
FROM observability_agent_service.obs_audit_log
WHERE created_at IS NOT NULL
GROUP BY bucket, entity_type, action, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('observability_agent_service.hourly_audit_stats',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- metric_definitions indexes
CREATE INDEX idx_md_name ON observability_agent_service.metric_definitions(name);
CREATE INDEX idx_md_metric_type ON observability_agent_service.metric_definitions(metric_type);
CREATE INDEX idx_md_created_at ON observability_agent_service.metric_definitions(created_at DESC);
CREATE INDEX idx_md_updated_at ON observability_agent_service.metric_definitions(updated_at DESC);
CREATE INDEX idx_md_tenant ON observability_agent_service.metric_definitions(tenant_id);
CREATE INDEX idx_md_tenant_name ON observability_agent_service.metric_definitions(tenant_id, name);
CREATE INDEX idx_md_tenant_type ON observability_agent_service.metric_definitions(tenant_id, metric_type);
CREATE INDEX idx_md_label_keys ON observability_agent_service.metric_definitions USING GIN (label_keys);

-- metric_recordings indexes (hypertable-aware)
CREATE INDEX idx_mr_metric_name ON observability_agent_service.metric_recordings(metric_name, recorded_at DESC);
CREATE INDEX idx_mr_tenant ON observability_agent_service.metric_recordings(tenant_id, recorded_at DESC);
CREATE INDEX idx_mr_tenant_metric ON observability_agent_service.metric_recordings(tenant_id, metric_name, recorded_at DESC);
CREATE INDEX idx_mr_provenance_hash ON observability_agent_service.metric_recordings(provenance_hash);
CREATE INDEX idx_mr_labels ON observability_agent_service.metric_recordings USING GIN (labels);

-- trace_spans indexes (hypertable-aware)
CREATE INDEX idx_ts_trace_id ON observability_agent_service.trace_spans(trace_id, start_time DESC);
CREATE INDEX idx_ts_parent_span ON observability_agent_service.trace_spans(parent_span_id, start_time DESC);
CREATE INDEX idx_ts_operation ON observability_agent_service.trace_spans(operation_name, start_time DESC);
CREATE INDEX idx_ts_service ON observability_agent_service.trace_spans(service_name, start_time DESC);
CREATE INDEX idx_ts_status ON observability_agent_service.trace_spans(status, start_time DESC);
CREATE INDEX idx_ts_tenant ON observability_agent_service.trace_spans(tenant_id, start_time DESC);
CREATE INDEX idx_ts_tenant_service ON observability_agent_service.trace_spans(tenant_id, service_name, start_time DESC);
CREATE INDEX idx_ts_tenant_status ON observability_agent_service.trace_spans(tenant_id, status, start_time DESC);
CREATE INDEX idx_ts_duration ON observability_agent_service.trace_spans(duration_ms, start_time DESC);
CREATE INDEX idx_ts_provenance_hash ON observability_agent_service.trace_spans(provenance_hash);
CREATE INDEX idx_ts_attributes ON observability_agent_service.trace_spans USING GIN (attributes);
CREATE INDEX idx_ts_events ON observability_agent_service.trace_spans USING GIN (events);

-- log_entries indexes (hypertable-aware)
CREATE INDEX idx_le_level ON observability_agent_service.log_entries(level, logged_at DESC);
CREATE INDEX idx_le_correlation ON observability_agent_service.log_entries(correlation_id, logged_at DESC);
CREATE INDEX idx_le_trace_id ON observability_agent_service.log_entries(trace_id, logged_at DESC);
CREATE INDEX idx_le_span_id ON observability_agent_service.log_entries(span_id, logged_at DESC);
CREATE INDEX idx_le_agent_id ON observability_agent_service.log_entries(agent_id, logged_at DESC);
CREATE INDEX idx_le_tenant ON observability_agent_service.log_entries(tenant_id, logged_at DESC);
CREATE INDEX idx_le_tenant_level ON observability_agent_service.log_entries(tenant_id, level, logged_at DESC);
CREATE INDEX idx_le_tenant_agent ON observability_agent_service.log_entries(tenant_id, agent_id, logged_at DESC);
CREATE INDEX idx_le_provenance_hash ON observability_agent_service.log_entries(provenance_hash);
CREATE INDEX idx_le_attributes ON observability_agent_service.log_entries USING GIN (attributes);

-- alert_rules indexes
CREATE INDEX idx_ar_name ON observability_agent_service.alert_rules(name);
CREATE INDEX idx_ar_metric_name ON observability_agent_service.alert_rules(metric_name);
CREATE INDEX idx_ar_severity ON observability_agent_service.alert_rules(severity);
CREATE INDEX idx_ar_enabled ON observability_agent_service.alert_rules(enabled);
CREATE INDEX idx_ar_created_at ON observability_agent_service.alert_rules(created_at DESC);
CREATE INDEX idx_ar_tenant ON observability_agent_service.alert_rules(tenant_id);
CREATE INDEX idx_ar_tenant_enabled ON observability_agent_service.alert_rules(tenant_id, enabled);
CREATE INDEX idx_ar_tenant_severity ON observability_agent_service.alert_rules(tenant_id, severity);
CREATE INDEX idx_ar_labels ON observability_agent_service.alert_rules USING GIN (labels);
CREATE INDEX idx_ar_annotations ON observability_agent_service.alert_rules USING GIN (annotations);

-- alert_instances indexes
CREATE INDEX idx_ai_rule_id ON observability_agent_service.alert_instances(rule_id);
CREATE INDEX idx_ai_status ON observability_agent_service.alert_instances(status);
CREATE INDEX idx_ai_severity ON observability_agent_service.alert_instances(severity);
CREATE INDEX idx_ai_metric_name ON observability_agent_service.alert_instances(metric_name);
CREATE INDEX idx_ai_started_at ON observability_agent_service.alert_instances(started_at DESC);
CREATE INDEX idx_ai_resolved_at ON observability_agent_service.alert_instances(resolved_at DESC);
CREATE INDEX idx_ai_tenant ON observability_agent_service.alert_instances(tenant_id);
CREATE INDEX idx_ai_tenant_status ON observability_agent_service.alert_instances(tenant_id, status);
CREATE INDEX idx_ai_tenant_severity ON observability_agent_service.alert_instances(tenant_id, severity);
CREATE INDEX idx_ai_labels ON observability_agent_service.alert_instances USING GIN (labels);
CREATE INDEX idx_ai_annotations ON observability_agent_service.alert_instances USING GIN (annotations);

-- health_check_results indexes
CREATE INDEX idx_hcr_probe_name ON observability_agent_service.health_check_results(probe_name);
CREATE INDEX idx_hcr_probe_type ON observability_agent_service.health_check_results(probe_type);
CREATE INDEX idx_hcr_service_name ON observability_agent_service.health_check_results(service_name);
CREATE INDEX idx_hcr_status ON observability_agent_service.health_check_results(status);
CREATE INDEX idx_hcr_checked_at ON observability_agent_service.health_check_results(checked_at DESC);
CREATE INDEX idx_hcr_tenant ON observability_agent_service.health_check_results(tenant_id);
CREATE INDEX idx_hcr_tenant_service ON observability_agent_service.health_check_results(tenant_id, service_name);
CREATE INDEX idx_hcr_tenant_status ON observability_agent_service.health_check_results(tenant_id, status);
CREATE INDEX idx_hcr_details ON observability_agent_service.health_check_results USING GIN (details);

-- dashboard_configs indexes
CREATE INDEX idx_dc_name ON observability_agent_service.dashboard_configs(name);
CREATE INDEX idx_dc_created_at ON observability_agent_service.dashboard_configs(created_at DESC);
CREATE INDEX idx_dc_updated_at ON observability_agent_service.dashboard_configs(updated_at DESC);
CREATE INDEX idx_dc_tenant ON observability_agent_service.dashboard_configs(tenant_id);
CREATE INDEX idx_dc_tenant_name ON observability_agent_service.dashboard_configs(tenant_id, name);
CREATE INDEX idx_dc_panels ON observability_agent_service.dashboard_configs USING GIN (panels);
CREATE INDEX idx_dc_variables ON observability_agent_service.dashboard_configs USING GIN (variables);

-- slo_definitions indexes
CREATE INDEX idx_slo_name ON observability_agent_service.slo_definitions(name);
CREATE INDEX idx_slo_service_name ON observability_agent_service.slo_definitions(service_name);
CREATE INDEX idx_slo_type ON observability_agent_service.slo_definitions(slo_type);
CREATE INDEX idx_slo_enabled ON observability_agent_service.slo_definitions(enabled);
CREATE INDEX idx_slo_created_at ON observability_agent_service.slo_definitions(created_at DESC);
CREATE INDEX idx_slo_tenant ON observability_agent_service.slo_definitions(tenant_id);
CREATE INDEX idx_slo_tenant_service ON observability_agent_service.slo_definitions(tenant_id, service_name);
CREATE INDEX idx_slo_tenant_enabled ON observability_agent_service.slo_definitions(tenant_id, enabled);
CREATE INDEX idx_slo_burn_rate ON observability_agent_service.slo_definitions USING GIN (burn_rate_thresholds);

-- obs_audit_log indexes (hypertable-aware)
CREATE INDEX idx_oal_entity_type ON observability_agent_service.obs_audit_log(entity_type, created_at DESC);
CREATE INDEX idx_oal_entity_id ON observability_agent_service.obs_audit_log(entity_id, created_at DESC);
CREATE INDEX idx_oal_action ON observability_agent_service.obs_audit_log(action, created_at DESC);
CREATE INDEX idx_oal_user ON observability_agent_service.obs_audit_log(user_id, created_at DESC);
CREATE INDEX idx_oal_tenant ON observability_agent_service.obs_audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_oal_data_hash ON observability_agent_service.obs_audit_log(data_hash);
CREATE INDEX idx_oal_chain_hash ON observability_agent_service.obs_audit_log(chain_hash);
CREATE INDEX idx_oal_previous_hash ON observability_agent_service.obs_audit_log(previous_hash);
CREATE INDEX idx_oal_details ON observability_agent_service.obs_audit_log USING GIN (details);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE observability_agent_service.metric_definitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY md_tenant_read ON observability_agent_service.metric_definitions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY md_tenant_write ON observability_agent_service.metric_definitions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.metric_recordings ENABLE ROW LEVEL SECURITY;
CREATE POLICY mr_tenant_read ON observability_agent_service.metric_recordings
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY mr_tenant_write ON observability_agent_service.metric_recordings
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.trace_spans ENABLE ROW LEVEL SECURITY;
CREATE POLICY ts_tenant_read ON observability_agent_service.trace_spans
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ts_tenant_write ON observability_agent_service.trace_spans
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.log_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY le_tenant_read ON observability_agent_service.log_entries
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY le_tenant_write ON observability_agent_service.log_entries
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.alert_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY ar_tenant_read ON observability_agent_service.alert_rules
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ar_tenant_write ON observability_agent_service.alert_rules
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.alert_instances ENABLE ROW LEVEL SECURITY;
CREATE POLICY ai_tenant_read ON observability_agent_service.alert_instances
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY ai_tenant_write ON observability_agent_service.alert_instances
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.health_check_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY hcr_tenant_read ON observability_agent_service.health_check_results
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY hcr_tenant_write ON observability_agent_service.health_check_results
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.dashboard_configs ENABLE ROW LEVEL SECURITY;
CREATE POLICY dc_tenant_read ON observability_agent_service.dashboard_configs
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY dc_tenant_write ON observability_agent_service.dashboard_configs
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.slo_definitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_tenant_read ON observability_agent_service.slo_definitions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY slo_tenant_write ON observability_agent_service.slo_definitions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE observability_agent_service.obs_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY oal_tenant_read ON observability_agent_service.obs_audit_log
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY oal_tenant_write ON observability_agent_service.obs_audit_log
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA observability_agent_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA observability_agent_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA observability_agent_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON observability_agent_service.hourly_metric_stats TO greenlang_app;
GRANT SELECT ON observability_agent_service.hourly_audit_stats TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA observability_agent_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA observability_agent_service TO greenlang_readonly;
GRANT SELECT ON observability_agent_service.hourly_metric_stats TO greenlang_readonly;
GRANT SELECT ON observability_agent_service.hourly_audit_stats TO greenlang_readonly;

-- Add observability agent service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'observability:metrics:read', 'observability', 'metrics_read', 'View metric definitions and recordings'),
    (gen_random_uuid(), 'observability:metrics:write', 'observability', 'metrics_write', 'Create metric definitions and record metric data'),
    (gen_random_uuid(), 'observability:traces:read', 'observability', 'traces_read', 'View trace spans and distributed traces'),
    (gen_random_uuid(), 'observability:traces:write', 'observability', 'traces_write', 'Record trace spans'),
    (gen_random_uuid(), 'observability:logs:read', 'observability', 'logs_read', 'View log entries'),
    (gen_random_uuid(), 'observability:logs:write', 'observability', 'logs_write', 'Record log entries'),
    (gen_random_uuid(), 'observability:alerts:read', 'observability', 'alerts_read', 'View alert rules and instances'),
    (gen_random_uuid(), 'observability:alerts:write', 'observability', 'alerts_write', 'Create and manage alert rules and instances'),
    (gen_random_uuid(), 'observability:health:read', 'observability', 'health_read', 'View health check results'),
    (gen_random_uuid(), 'observability:health:write', 'observability', 'health_write', 'Record health check probe results'),
    (gen_random_uuid(), 'observability:dashboards:read', 'observability', 'dashboards_read', 'View dashboard configurations'),
    (gen_random_uuid(), 'observability:dashboards:write', 'observability', 'dashboards_write', 'Create and update dashboard configurations'),
    (gen_random_uuid(), 'observability:audit:read', 'observability', 'audit_read', 'View observability agent audit event log'),
    (gen_random_uuid(), 'observability:admin', 'observability', 'admin', 'Observability agent service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep metric recordings for 7 days
SELECT add_retention_policy('observability_agent_service.metric_recordings', INTERVAL '7 days');

-- Keep trace spans for 14 days
SELECT add_retention_policy('observability_agent_service.trace_spans', INTERVAL '14 days');

-- Keep log entries for 30 days
SELECT add_retention_policy('observability_agent_service.log_entries', INTERVAL '30 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('observability_agent_service.obs_audit_log', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on metric_recordings after 1 day
ALTER TABLE observability_agent_service.metric_recordings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'recorded_at DESC'
);

SELECT add_compression_policy('observability_agent_service.metric_recordings', INTERVAL '1 day');

-- Enable compression on trace_spans after 2 days
ALTER TABLE observability_agent_service.trace_spans SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'start_time DESC'
);

SELECT add_compression_policy('observability_agent_service.trace_spans', INTERVAL '2 days');

-- Enable compression on log_entries after 3 days
ALTER TABLE observability_agent_service.log_entries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'logged_at DESC'
);

SELECT add_compression_policy('observability_agent_service.log_entries', INTERVAL '3 days');

-- Enable compression on obs_audit_log after 30 days
ALTER TABLE observability_agent_service.obs_audit_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'created_at DESC'
);

SELECT add_compression_policy('observability_agent_service.obs_audit_log', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Observability Agent (GL-FOUND-X-010) in Agent Registry
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-FOUND-X-010', 'Observability & Telemetry Agent',
 'Unified observability agent for GreenLang Climate OS. Provides metric collection and recording, distributed trace span management, structured log aggregation, alert rule evaluation and instance lifecycle, health check probing, dashboard configuration, SLO definition and burn-rate tracking, and comprehensive audit logging with SHA-256 provenance chains.',
 1, 'sync', true, true, 30, '1.0.0', false,
 'GreenLang Platform Team', 'https://docs.greenlang.ai/agents/observability-agent', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Agent Version for Observability Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-FOUND-X-010', '1.0.0',
 '{"cpu_request": "200m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "1Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/observability-agent-service", "tag": "1.0.0", "port": 8080}'::jsonb,
 '{"foundation", "observability", "metrics", "traces", "logs", "alerts", "slo"}',
 '{"cross-sector"}',
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3')
ON CONFLICT (agent_id, version) DO NOTHING;

-- =============================================================================
-- Seed: Agent Capabilities for Observability Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES

('GL-FOUND-X-010', '1.0.0', 'metric_collection', 'data_ingestion',
 'Define, record, and query platform and application metrics with support for counter, gauge, histogram, and summary metric types',
 '{"metric_name", "value", "labels", "metric_type"}', '{"recording_id", "provenance_hash"}',
 '{"supported_types": ["counter", "gauge", "histogram", "summary"], "max_labels": 20, "batch_size": 1000}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'distributed_tracing', 'data_ingestion',
 'Record and query distributed trace spans with parent-child relationships, operation/service classification, attributes, and events',
 '{"trace_id", "operation_name", "service_name", "attributes"}', '{"span_id", "duration_ms", "provenance_hash"}',
 '{"max_span_depth": 50, "auto_duration": true, "status_codes": ["unset", "ok", "error"]}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'log_aggregation', 'data_ingestion',
 'Ingest and query structured log entries with level classification, correlation IDs, trace/span linking, and agent attribution',
 '{"level", "message", "correlation_id", "agent_id"}', '{"log_id", "provenance_hash"}',
 '{"levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "max_message_length": 65536, "auto_correlation": true}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'alert_management', 'analysis',
 'Define alert rules with metric conditions and thresholds, evaluate rules against recordings, manage alert instance lifecycle (firing, acknowledged, resolved)',
 '{"metric_name", "condition", "threshold", "severity"}', '{"alert_instance", "evaluation_result"}',
 '{"conditions": ["gt", "lt", "eq", "gte", "lte", "ne"], "severities": ["info", "warning", "error", "critical"], "default_duration_seconds": 60}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'health_probing', 'validation',
 'Execute liveness, readiness, and startup health check probes against platform services and record results with duration and status',
 '{"probe_name", "probe_type", "service_name"}', '{"health_result", "status", "duration_ms"}',
 '{"probe_types": ["liveness", "readiness", "startup"], "timeout_ms": 5000, "retry_count": 3}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'dashboard_management', 'configuration',
 'Create, update, and retrieve dashboard configurations with panel layouts, time ranges, refresh intervals, and template variables',
 '{"name", "panels", "time_range", "variables"}', '{"dashboard_id", "config"}',
 '{"max_panels": 50, "default_time_range": "1h", "default_refresh_seconds": 30}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'slo_tracking', 'analysis',
 'Define SLOs with targets and burn-rate thresholds, evaluate service performance against objectives, and calculate error budgets',
 '{"service_name", "slo_type", "target", "window_days"}', '{"slo_status", "error_budget_remaining", "burn_rate"}',
 '{"slo_types": ["availability", "latency", "throughput", "error_rate", "saturation"], "default_window_days": 30}'::jsonb),

('GL-FOUND-X-010', '1.0.0', 'audit_provenance', 'validation',
 'Record and verify SHA-256 provenance chains for all observability operations, providing tamper-evident audit trails for compliance',
 '{"entity_type", "entity_id", "action", "data"}', '{"audit_id", "data_hash", "chain_hash"}',
 '{"hash_algorithm": "sha256", "chain_verification": true, "max_chain_depth": 10000}'::jsonb)

ON CONFLICT DO NOTHING;

-- =============================================================================
-- Seed: Agent Dependencies for Observability Agent
-- =============================================================================

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES

-- Observability Agent depends on Registry for agent discovery and health tracking
('GL-FOUND-X-010', 'GL-FOUND-X-007', '>=1.0.0', false,
 'Agent discovery for health probing and service catalog integration'),

-- Observability Agent depends on Schema Compiler for metric/config validation
('GL-FOUND-X-010', 'GL-FOUND-X-002', '>=1.0.0', false,
 'Metric definitions, alert rules, and dashboard configs validated against JSON Schema'),

-- Observability Agent optionally uses Citations for audit provenance verification
('GL-FOUND-X-010', 'GL-FOUND-X-005', '>=1.0.0', true,
 'Audit log provenance chains may be verified via the citation service'),

-- Observability Agent optionally uses Orchestrator for pipeline-level tracing
('GL-FOUND-X-010', 'GL-FOUND-X-001', '>=1.0.0', true,
 'Distributed traces may span orchestrator DAG executions'),

-- Observability Agent optionally uses QA Test Harness for self-testing
('GL-FOUND-X-010', 'GL-FOUND-X-009', '>=1.0.0', true,
 'Self-test capabilities leverage the QA test harness for verification')

ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

-- =============================================================================
-- Seed: Service Catalog Entry for Observability Agent
-- =============================================================================

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-FOUND-X-010', 'Observability & Telemetry Agent',
 'Unified observability agent for GreenLang Climate OS: metric collection, distributed tracing, log aggregation, alert management, health probing, dashboard configuration, SLO tracking, and audit provenance with SHA-256 hash chains.',
 'foundation', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Seed: Pre-defined Platform Metric Definitions
-- =============================================================================

INSERT INTO observability_agent_service.metric_definitions (name, metric_type, description, label_keys, buckets, tenant_id) VALUES

('gl_agent_request_duration_seconds', 'histogram',
 'Duration of agent request processing in seconds. Tracks end-to-end latency for all agent invocations across the platform.',
 '{"agent_id", "operation", "status", "tenant_id"}',
 '{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0}',
 'default'),

('gl_agent_requests_total', 'counter',
 'Total number of agent requests processed. Monotonically increasing counter partitioned by agent, operation, and status code.',
 '{"agent_id", "operation", "status_code", "tenant_id"}',
 NULL,
 'default'),

('gl_pipeline_active_runs', 'gauge',
 'Number of currently active pipeline (DAG) runs across the platform. Used for concurrency monitoring and capacity planning.',
 '{"pipeline_id", "status", "tenant_id"}',
 NULL,
 'default'),

('gl_error_budget_remaining_ratio', 'gauge',
 'Remaining error budget as a ratio (0-1) for each service SLO. Values approaching zero trigger burn-rate alerts.',
 '{"service_name", "slo_name", "slo_type", "tenant_id"}',
 NULL,
 'default'),

('gl_provenance_chain_length', 'summary',
 'Distribution of provenance chain lengths (number of linked hashes) across agent operations. Tracks audit trail depth.',
 '{"agent_id", "entity_type", "tenant_id"}',
 NULL,
 'default')

ON CONFLICT (name, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA observability_agent_service IS 'Observability & Telemetry Agent for GreenLang Climate OS (AGENT-FOUND-010) - metric collection, distributed tracing, log aggregation, alert management, health probing, dashboard configuration, SLO tracking, and audit provenance';
COMMENT ON TABLE observability_agent_service.metric_definitions IS 'Metric definition registry with name, type (counter/gauge/histogram/summary), label keys, and optional histogram buckets';
COMMENT ON TABLE observability_agent_service.metric_recordings IS 'TimescaleDB hypertable: metric data point recordings with name, value, labels, timestamp, and provenance hash';
COMMENT ON TABLE observability_agent_service.trace_spans IS 'TimescaleDB hypertable: distributed trace spans with trace/parent IDs, operation/service names, status, duration, attributes, and events';
COMMENT ON TABLE observability_agent_service.log_entries IS 'TimescaleDB hypertable: structured log entries with level, message, correlation/trace/span IDs, agent ID, and attributes';
COMMENT ON TABLE observability_agent_service.alert_rules IS 'Alert rule definitions with metric name, condition, threshold, duration, severity, labels, and annotations';
COMMENT ON TABLE observability_agent_service.alert_instances IS 'Alert instance lifecycle records with status (firing/resolved/pending/acknowledged), metric value, threshold, and timestamps';
COMMENT ON TABLE observability_agent_service.health_check_results IS 'Health check probe results with probe type (liveness/readiness/startup), status (healthy/degraded/unhealthy), and duration';
COMMENT ON TABLE observability_agent_service.dashboard_configs IS 'Dashboard configuration store with panel layouts, time range, refresh interval, and template variables';
COMMENT ON TABLE observability_agent_service.slo_definitions IS 'SLO definitions with type (availability/latency/throughput/error_rate/saturation), target, window, and burn-rate thresholds';
COMMENT ON TABLE observability_agent_service.obs_audit_log IS 'TimescaleDB hypertable: comprehensive audit events for all observability operations with SHA-256 hash chain integrity';
COMMENT ON MATERIALIZED VIEW observability_agent_service.hourly_metric_stats IS 'Continuous aggregate: hourly metric recording statistics by metric name for dashboard queries and trend analysis';
COMMENT ON MATERIALIZED VIEW observability_agent_service.hourly_audit_stats IS 'Continuous aggregate: hourly audit event counts by entity type and action for compliance reporting';

COMMENT ON COLUMN observability_agent_service.metric_definitions.metric_type IS 'Metric type: counter, gauge, histogram, summary';
COMMENT ON COLUMN observability_agent_service.metric_definitions.buckets IS 'Histogram bucket boundaries (only for histogram type metrics)';
COMMENT ON COLUMN observability_agent_service.metric_recordings.provenance_hash IS 'SHA-256 hash of the recording content for integrity verification';
COMMENT ON COLUMN observability_agent_service.trace_spans.status IS 'Span status: unset, ok, error (follows OpenTelemetry conventions)';
COMMENT ON COLUMN observability_agent_service.trace_spans.duration_ms IS 'Span duration in milliseconds (end_time - start_time)';
COMMENT ON COLUMN observability_agent_service.trace_spans.provenance_hash IS 'SHA-256 hash of the span content for integrity verification';
COMMENT ON COLUMN observability_agent_service.log_entries.level IS 'Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL';
COMMENT ON COLUMN observability_agent_service.log_entries.correlation_id IS 'Correlation ID for grouping related log entries across services';
COMMENT ON COLUMN observability_agent_service.log_entries.provenance_hash IS 'SHA-256 hash of the log entry content for integrity verification';
COMMENT ON COLUMN observability_agent_service.alert_rules.condition IS 'Alert condition operator: gt, lt, eq, gte, lte, ne';
COMMENT ON COLUMN observability_agent_service.alert_rules.severity IS 'Alert severity: info, warning, error, critical';
COMMENT ON COLUMN observability_agent_service.alert_instances.status IS 'Alert instance status: firing, resolved, pending, acknowledged';
COMMENT ON COLUMN observability_agent_service.health_check_results.probe_type IS 'Health probe type: liveness, readiness, startup';
COMMENT ON COLUMN observability_agent_service.health_check_results.status IS 'Health check status: healthy, degraded, unhealthy';
COMMENT ON COLUMN observability_agent_service.slo_definitions.slo_type IS 'SLO type: availability, latency, throughput, error_rate, saturation';
COMMENT ON COLUMN observability_agent_service.slo_definitions.target IS 'SLO target as a ratio (0-1), e.g., 0.999 for 99.9% availability';
COMMENT ON COLUMN observability_agent_service.slo_definitions.burn_rate_thresholds IS 'Burn-rate alert thresholds following Google SRE multi-window approach';
COMMENT ON COLUMN observability_agent_service.obs_audit_log.entity_type IS 'Entity type: metric_definition, metric_recording, trace_span, log_entry, alert_rule, alert_instance, health_check, dashboard, slo_definition, system';
COMMENT ON COLUMN observability_agent_service.obs_audit_log.chain_hash IS 'SHA-256 hash chain linking this event to the previous event for tamper detection';
