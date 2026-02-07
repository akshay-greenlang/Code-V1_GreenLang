-- =============================================================================
-- GreenLang Climate OS - SLO/SLI & Error Budget Management Service Schema
-- =============================================================================
-- Migration: V020
-- Component: OBS-005 SLO/SLI Definitions & Error Budget Management
-- Description: Creates slo schema with definitions, history tracking,
--              error budget snapshots (hypertable), compliance reports,
--              evaluation log (hypertable), and hourly summary continuous
--              aggregate for real-time SLO compliance analytics.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS slo;

-- =============================================================================
-- Table: slo.definitions
-- =============================================================================
-- Core SLO definition table. Each row represents a single Service Level
-- Objective with its SLI query definitions, target, measurement window,
-- owner team, tier classification, and burn rate alert thresholds.
-- Supports versioning to track changes over time.

CREATE TABLE slo.definitions (
    slo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    service VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    sli_type VARCHAR(50) NOT NULL,
    sli_good_events_query TEXT NOT NULL,
    sli_total_events_query TEXT NOT NULL,
    sli_threshold DOUBLE PRECISION,
    target DOUBLE PRECISION NOT NULL,
    window VARCHAR(20) NOT NULL DEFAULT '30d',
    labels JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    owner_team VARCHAR(100) NOT NULL,
    tier VARCHAR(20) NOT NULL DEFAULT 'tier-2',
    burn_rate_alerts JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Category constraint
ALTER TABLE slo.definitions
    ADD CONSTRAINT chk_slo_category
    CHECK (category IN ('availability', 'latency', 'correctness', 'throughput', 'efficiency', 'timeliness'));

-- SLI type constraint
ALTER TABLE slo.definitions
    ADD CONSTRAINT chk_sli_type
    CHECK (sli_type IN ('availability', 'latency', 'accuracy', 'ratio', 'throughput', 'timeliness'));

-- Tier constraint
ALTER TABLE slo.definitions
    ADD CONSTRAINT chk_slo_tier
    CHECK (tier IN ('tier-1', 'tier-2', 'tier-3'));

-- Target must be between 0 and 100
ALTER TABLE slo.definitions
    ADD CONSTRAINT chk_slo_target_range
    CHECK (target >= 0 AND target <= 100);

-- Version must be positive
ALTER TABLE slo.definitions
    ADD CONSTRAINT chk_slo_version_positive
    CHECK (version > 0);

-- =============================================================================
-- Table: slo.definition_history
-- =============================================================================
-- Tracks every change to an SLO definition. Each row captures the state of
-- the SLO at a given version, who changed it, why, and when.

CREATE TABLE slo.definition_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slo_id UUID NOT NULL REFERENCES slo.definitions(slo_id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    window VARCHAR(20) NOT NULL,
    sli_good_events_query TEXT,
    sli_total_events_query TEXT,
    burn_rate_alerts JSONB,
    change_description TEXT NOT NULL,
    changed_by VARCHAR(255) NOT NULL,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Ensure unique version per SLO
ALTER TABLE slo.definition_history
    ADD CONSTRAINT uq_slo_definition_version
    UNIQUE (slo_id, version);

-- =============================================================================
-- Table: slo.error_budget_snapshots
-- =============================================================================
-- TimescaleDB hypertable for time-series error budget tracking. Each row
-- captures a point-in-time snapshot of an SLO's error budget state including
-- total budget, consumed/remaining minutes, burn rates at multiple windows,
-- and the current SLI value.

CREATE TABLE slo.error_budget_snapshots (
    snapshot_time TIMESTAMPTZ NOT NULL,
    slo_id UUID NOT NULL REFERENCES slo.definitions(slo_id) ON DELETE CASCADE,
    total_budget_minutes DOUBLE PRECISION NOT NULL,
    consumed_minutes DOUBLE PRECISION NOT NULL DEFAULT 0,
    remaining_minutes DOUBLE PRECISION NOT NULL,
    consumption_percent DOUBLE PRECISION NOT NULL DEFAULT 0,
    burn_rate_1h DOUBLE PRECISION,
    burn_rate_6h DOUBLE PRECISION,
    burn_rate_3d DOUBLE PRECISION,
    sli_value DOUBLE PRECISION NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'healthy'
);

-- Create hypertable partitioned by snapshot_time
SELECT create_hypertable('slo.error_budget_snapshots', 'snapshot_time');

-- Budget status constraint
ALTER TABLE slo.error_budget_snapshots
    ADD CONSTRAINT chk_budget_status
    CHECK (status IN ('healthy', 'warning', 'critical', 'exhausted'));

-- Consumption percent range
ALTER TABLE slo.error_budget_snapshots
    ADD CONSTRAINT chk_consumption_range
    CHECK (consumption_percent >= 0 AND consumption_percent <= 100);

-- =============================================================================
-- Table: slo.compliance_reports
-- =============================================================================
-- Stores generated compliance reports (weekly, monthly, quarterly). Each
-- report summarizes SLO performance across all definitions for a given
-- time period, including overall compliance, per-SLO breakdowns, and
-- detailed report data in JSONB format.

CREATE TABLE slo.compliance_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(20) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    overall_compliance_percent DOUBLE PRECISION NOT NULL,
    total_slos INTEGER NOT NULL,
    meeting_target INTEGER NOT NULL,
    breached INTEGER NOT NULL,
    report_data JSONB NOT NULL DEFAULT '{}',
    generated_by VARCHAR(255) DEFAULT 'system',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Report type constraint
ALTER TABLE slo.compliance_reports
    ADD CONSTRAINT chk_report_type
    CHECK (report_type IN ('weekly', 'monthly', 'quarterly', 'annual', 'adhoc'));

-- Period end must be after period start
ALTER TABLE slo.compliance_reports
    ADD CONSTRAINT chk_report_period_valid
    CHECK (period_end > period_start);

-- Meeting + breached must equal total
ALTER TABLE slo.compliance_reports
    ADD CONSTRAINT chk_report_slo_counts
    CHECK (meeting_target + breached = total_slos);

-- =============================================================================
-- Table: slo.evaluation_log
-- =============================================================================
-- TimescaleDB hypertable recording every SLO evaluation cycle. Each row
-- captures the SLI value, target comparison, burn rate at three windows
-- (fast/medium/slow), and remaining error budget percentage. Used for
-- detailed SLO trend analysis and debugging.

CREATE TABLE slo.evaluation_log (
    eval_time TIMESTAMPTZ NOT NULL,
    slo_id UUID NOT NULL REFERENCES slo.definitions(slo_id) ON DELETE CASCADE,
    sli_value DOUBLE PRECISION NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    met_target BOOLEAN NOT NULL,
    burn_rate_fast DOUBLE PRECISION,
    burn_rate_medium DOUBLE PRECISION,
    burn_rate_slow DOUBLE PRECISION,
    error_budget_remaining_percent DOUBLE PRECISION
);

-- Create hypertable partitioned by eval_time
SELECT create_hypertable('slo.evaluation_log', 'eval_time');

-- =============================================================================
-- Continuous Aggregate: slo.hourly_summaries
-- =============================================================================
-- Precomputed hourly SLO summaries for efficient dashboard queries.
-- Aggregates evaluation log data into per-SLO hourly statistics including
-- evaluation count, average/min/max SLI values, compliance percentage,
-- average burn rates, and average remaining budget.

CREATE MATERIALIZED VIEW slo.hourly_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', eval_time) AS bucket,
    slo_id,
    COUNT(*) AS evaluation_count,
    AVG(sli_value) AS avg_sli_value,
    MIN(sli_value) AS min_sli_value,
    MAX(sli_value) AS max_sli_value,
    AVG(CASE WHEN met_target THEN 1.0 ELSE 0.0 END) * 100 AS compliance_percent,
    AVG(burn_rate_fast) AS avg_burn_rate_fast,
    AVG(burn_rate_medium) AS avg_burn_rate_medium,
    AVG(burn_rate_slow) AS avg_burn_rate_slow,
    AVG(error_budget_remaining_percent) AS avg_budget_remaining
FROM slo.evaluation_log
WHERE eval_time IS NOT NULL
GROUP BY bucket, slo_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('slo.hourly_summaries',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Definitions indexes
CREATE INDEX idx_slo_definitions_service ON slo.definitions(service);
CREATE INDEX idx_slo_definitions_category ON slo.definitions(category);
CREATE INDEX idx_slo_definitions_owner_team ON slo.definitions(owner_team);
CREATE INDEX idx_slo_definitions_tier ON slo.definitions(tier);
CREATE INDEX idx_slo_definitions_is_active ON slo.definitions(is_active) WHERE is_active = true;
CREATE INDEX idx_slo_definitions_sli_type ON slo.definitions(sli_type);
CREATE INDEX idx_slo_definitions_labels ON slo.definitions USING GIN (labels);
CREATE INDEX idx_slo_definitions_annotations ON slo.definitions USING GIN (annotations);
CREATE INDEX idx_slo_definitions_service_category ON slo.definitions(service, category);

-- Definition history indexes
CREATE INDEX idx_slo_history_slo_id ON slo.definition_history(slo_id);
CREATE INDEX idx_slo_history_changed_at ON slo.definition_history(changed_at DESC);
CREATE INDEX idx_slo_history_changed_by ON slo.definition_history(changed_by);

-- Error budget snapshots indexes (hypertable-aware)
CREATE INDEX idx_budget_snapshots_slo_id ON slo.error_budget_snapshots(slo_id, snapshot_time DESC);
CREATE INDEX idx_budget_snapshots_status ON slo.error_budget_snapshots(status);
CREATE INDEX idx_budget_snapshots_consumption ON slo.error_budget_snapshots(consumption_percent DESC);

-- Compliance reports indexes
CREATE INDEX idx_compliance_reports_type ON slo.compliance_reports(report_type);
CREATE INDEX idx_compliance_reports_period ON slo.compliance_reports(period_start, period_end);
CREATE INDEX idx_compliance_reports_generated_at ON slo.compliance_reports(generated_at DESC);
CREATE INDEX idx_compliance_reports_compliance ON slo.compliance_reports(overall_compliance_percent);

-- Evaluation log indexes (hypertable-aware)
CREATE INDEX idx_eval_log_slo_id ON slo.evaluation_log(slo_id, eval_time DESC);
CREATE INDEX idx_eval_log_met_target ON slo.evaluation_log(met_target);
CREATE INDEX idx_eval_log_budget_remaining ON slo.evaluation_log(error_budget_remaining_percent);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE slo.definitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_definitions_global_read ON slo.definitions
    FOR SELECT USING (true);

CREATE POLICY slo_definitions_team_write ON slo.definitions
    FOR ALL USING (
        owner_team = current_setting('app.current_team', true)
        OR current_setting('app.current_team', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE slo.definition_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_history_global_access ON slo.definition_history
    FOR SELECT USING (true);

ALTER TABLE slo.error_budget_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_budget_global_access ON slo.error_budget_snapshots
    FOR SELECT USING (true);

ALTER TABLE slo.compliance_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_reports_global_access ON slo.compliance_reports
    USING (true);

ALTER TABLE slo.evaluation_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY slo_eval_log_global_access ON slo.evaluation_log
    FOR SELECT USING (true);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA slo TO greenlang_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA slo TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA slo TO greenlang_app;

-- Grant SELECT on the continuous aggregate
GRANT SELECT ON slo.hourly_summaries TO greenlang_app;

-- Add SLO permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'slo:definitions:read', 'slo', 'read', 'View SLO definitions'),
    (gen_random_uuid(), 'slo:definitions:write', 'slo', 'write', 'Create/update SLO definitions'),
    (gen_random_uuid(), 'slo:definitions:delete', 'slo', 'delete', 'Delete SLO definitions'),
    (gen_random_uuid(), 'slo:budgets:read', 'slo', 'budget_read', 'View error budget snapshots'),
    (gen_random_uuid(), 'slo:reports:read', 'slo', 'report_read', 'View compliance reports'),
    (gen_random_uuid(), 'slo:reports:generate', 'slo', 'report_generate', 'Generate compliance reports'),
    (gen_random_uuid(), 'slo:evaluations:read', 'slo', 'eval_read', 'View SLO evaluation logs'),
    (gen_random_uuid(), 'slo:admin', 'slo', 'admin', 'SLO administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Trigger: auto-update updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION slo.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_slo_definitions_updated_at
    BEFORE UPDATE ON slo.definitions
    FOR EACH ROW
    EXECUTE FUNCTION slo.update_updated_at();

-- =============================================================================
-- Trigger: auto-insert history on definition change
-- =============================================================================

CREATE OR REPLACE FUNCTION slo.log_definition_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Only log if target, window, or queries changed
    IF OLD.target IS DISTINCT FROM NEW.target
       OR OLD.window IS DISTINCT FROM NEW.window
       OR OLD.sli_good_events_query IS DISTINCT FROM NEW.sli_good_events_query
       OR OLD.sli_total_events_query IS DISTINCT FROM NEW.sli_total_events_query
       OR OLD.burn_rate_alerts IS DISTINCT FROM NEW.burn_rate_alerts THEN

        INSERT INTO slo.definition_history (
            slo_id, version, target, window,
            sli_good_events_query, sli_total_events_query,
            burn_rate_alerts, change_description, changed_by
        ) VALUES (
            NEW.slo_id, NEW.version, NEW.target, NEW.window,
            NEW.sli_good_events_query, NEW.sli_total_events_query,
            NEW.burn_rate_alerts,
            COALESCE(current_setting('app.change_description', true), 'SLO definition updated'),
            COALESCE(current_setting('app.current_user', true), 'system')
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_slo_definition_history
    AFTER UPDATE ON slo.definitions
    FOR EACH ROW
    EXECUTE FUNCTION slo.log_definition_change();

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep error budget snapshots for 1 year
SELECT add_retention_policy('slo.error_budget_snapshots', INTERVAL '365 days');

-- Keep evaluation log for 90 days
SELECT add_retention_policy('slo.evaluation_log', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on error budget snapshots after 7 days
ALTER TABLE slo.error_budget_snapshots SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'slo_id',
    timescaledb.compress_orderby = 'snapshot_time DESC'
);

SELECT add_compression_policy('slo.error_budget_snapshots', INTERVAL '7 days');

-- Enable compression on evaluation log after 7 days
ALTER TABLE slo.evaluation_log SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'slo_id',
    timescaledb.compress_orderby = 'eval_time DESC'
);

SELECT add_compression_policy('slo.evaluation_log', INTERVAL '7 days');

-- =============================================================================
-- Seed: Default SLO definitions (matching slo_definitions.yaml)
-- =============================================================================

INSERT INTO slo.definitions (name, service, description, category, sli_type, sli_good_events_query, sli_total_events_query, sli_threshold, target, window, owner_team, tier, burn_rate_alerts) VALUES
-- API Gateway SLOs
('api-availability', 'greenlang-api-gateway', 'API Gateway should be available and responding successfully', 'availability', 'availability',
 'sum(rate(http_requests_total{job="greenlang-api-gateway",status!~"5.."}[5m]))',
 'sum(rate(http_requests_total{job="greenlang-api-gateway"}[5m]))',
 NULL, 99.95, '30d', 'platform-infrastructure', 'tier-1',
 '[{"name":"api-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"api-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"api-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('api-latency', 'greenlang-api-gateway', 'API Gateway should respond within 100ms', 'latency', 'latency',
 'sum(rate(http_request_duration_seconds_bucket{job="greenlang-api-gateway",le="0.1"}[5m]))',
 'sum(rate(http_request_duration_seconds_count{job="greenlang-api-gateway"}[5m]))',
 0.1, 99.0, '30d', 'platform-infrastructure', 'tier-1',
 '[{"name":"api-latency-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"api-latency-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"api-latency-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

-- EUDR Agent SLOs
('eudr-availability', 'eudr-agent', 'EUDR Compliance Agent must be highly available for regulatory compliance', 'availability', 'availability',
 'sum(rate(eudr_requests_total{status!~"5.."}[5m]))',
 'sum(rate(eudr_requests_total[5m]))',
 NULL, 99.9, '30d', 'platform-data', 'tier-1',
 '[{"name":"eudr-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"eudr-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"critical"},{"name":"eudr-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"warning"}]'),

('eudr-latency', 'eudr-agent', 'EUDR Agent should respond within 500ms for P95', 'latency', 'latency',
 'sum(rate(eudr_request_duration_seconds_bucket{le="0.5"}[5m]))',
 'sum(rate(eudr_request_duration_seconds_count[5m]))',
 0.5, 99.0, '30d', 'platform-data', 'tier-1',
 '[{"name":"eudr-latency-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"eudr-latency-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"eudr-latency-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('eudr-compliance-accuracy', 'eudr-agent', 'EUDR compliance checks must be 100% accurate', 'correctness', 'accuracy',
 'sum(rate(eudr_compliance_checks_validated_total[5m]))',
 'sum(rate(eudr_compliance_checks_total[5m]))',
 NULL, 100.0, '30d', 'platform-data', 'tier-1',
 '[{"name":"eudr-accuracy-any-error","long_window":"5m","short_window":"1m","burn_rate":0,"severity":"critical"}]'),

-- CBAM Agent SLOs
('cbam-availability', 'cbam-agent', 'CBAM Reporting Agent availability', 'availability', 'availability',
 'sum(rate(cbam_requests_total{status!~"5.."}[5m]))',
 'sum(rate(cbam_requests_total[5m]))',
 NULL, 99.9, '30d', 'platform-data', 'tier-1',
 '[{"name":"cbam-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"cbam-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"cbam-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('cbam-calculation-accuracy', 'cbam-agent', 'CBAM embedded emissions calculations must be accurate', 'correctness', 'accuracy',
 'sum(rate(cbam_calculations_validated_total[5m]))',
 'sum(rate(cbam_calculations_total[5m]))',
 NULL, 99.99, '30d', 'platform-data', 'tier-1',
 '[{"name":"cbam-accuracy-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"cbam-accuracy-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"}]'),

('cbam-report-generation', 'cbam-agent', 'CBAM reports should generate within 30 seconds', 'latency', 'latency',
 'sum(rate(cbam_report_generation_duration_seconds_bucket{le="30"}[5m]))',
 'sum(rate(cbam_report_generation_duration_seconds_count[5m]))',
 30, 95.0, '30d', 'platform-data', 'tier-2',
 '[{"name":"cbam-latency-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"cbam-latency-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"cbam-latency-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

-- SB253 Agent SLOs
('sb253-availability', 'sb253-agent', 'SB253 California Climate Agent availability', 'availability', 'availability',
 'sum(rate(sb253_requests_total{status!~"5.."}[5m]))',
 'sum(rate(sb253_requests_total[5m]))',
 NULL, 99.9, '30d', 'platform-data', 'tier-1',
 '[{"name":"sb253-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"sb253-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"sb253-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('sb253-scope-calculation-accuracy', 'sb253-agent', 'SB253 scope 1/2/3 calculations must be accurate', 'correctness', 'accuracy',
 'sum(rate(sb253_scope_calculations_validated_total[5m]))',
 'sum(rate(sb253_scope_calculations_total[5m]))',
 NULL, 100.0, '30d', 'platform-data', 'tier-1',
 '[{"name":"sb253-accuracy-any-error","long_window":"5m","short_window":"1m","burn_rate":0,"severity":"critical"}]'),

-- Emission Calculator SLOs
('calculator-availability', 'emission-calculator', 'Emission Calculator service availability', 'availability', 'availability',
 'sum(rate(emission_calculation_requests_total{status!~"5.."}[5m]))',
 'sum(rate(emission_calculation_requests_total[5m]))',
 NULL, 99.9, '30d', 'platform-data', 'tier-2',
 '[{"name":"calculator-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"calculator-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"calculator-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('calculator-accuracy', 'emission-calculator', 'Emission calculations must be 100% accurate', 'correctness', 'accuracy',
 'sum(rate(emission_calculations_validated_total[5m]))',
 'sum(rate(emission_calculations_total[5m]))',
 NULL, 100.0, '30d', 'platform-data', 'tier-1',
 '[{"name":"calculator-accuracy-any-error","long_window":"5m","short_window":"1m","burn_rate":0,"severity":"critical"}]'),

('calculator-throughput', 'emission-calculator', 'Calculator should handle at least 1000 calculations per minute', 'throughput', 'throughput',
 'sum(rate(emission_calculations_total[1m])) * 60',
 '1000',
 1000, 99.0, '30d', 'platform-data', 'tier-2',
 '[{"name":"calculator-throughput-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"calculator-throughput-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"}]'),

-- Database SLOs
('database-availability', 'postgresql', 'PostgreSQL database availability', 'availability', 'availability',
 'pg_up', '1',
 NULL, 99.99, '30d', 'platform-infrastructure', 'tier-1',
 '[{"name":"db-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"db-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"critical"},{"name":"db-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"warning"}]'),

('database-query-latency', 'postgresql', 'Database queries should complete within 100ms', 'latency', 'latency',
 'rate(pg_stat_statements_seconds_total[5m]) / rate(pg_stat_statements_calls_total[5m]) * 1000',
 '1',
 100, 95.0, '30d', 'platform-infrastructure', 'tier-1',
 '[{"name":"db-latency-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"db-latency-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"db-latency-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

-- Redis Cache SLOs
('cache-availability', 'redis', 'Redis cache availability', 'availability', 'availability',
 'redis_up', '1',
 NULL, 99.9, '30d', 'platform-infrastructure', 'tier-1',
 '[{"name":"cache-availability-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"critical"},{"name":"cache-availability-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"warning"},{"name":"cache-availability-burn-slow","long_window":"3d","short_window":"6h","burn_rate":1,"severity":"info"}]'),

('cache-hit-ratio', 'redis', 'Cache hit ratio should be above 90%', 'efficiency', 'ratio',
 'rate(redis_keyspace_hits_total[5m])',
 'rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m])',
 NULL, 90.0, '30d', 'platform-infrastructure', 'tier-2',
 '[{"name":"cache-hitratio-burn-fast","long_window":"1h","short_window":"5m","burn_rate":14.4,"severity":"warning"},{"name":"cache-hitratio-burn-medium","long_window":"6h","short_window":"30m","burn_rate":6,"severity":"info"}]'),

-- Business / Compliance SLOs
('compliance-report-delivery', 'compliance-reports', 'Compliance reports must be delivered before regulatory deadlines', 'timeliness', 'timeliness',
 'sum(rate(compliance_reports_delivered_on_time_total[5m]))',
 'sum(rate(compliance_reports_due_total[5m]))',
 NULL, 100.0, '30d', 'platform-data', 'tier-1',
 '[{"name":"compliance-timeliness-any-late","long_window":"5m","short_window":"1m","burn_rate":0,"severity":"critical"}]')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA slo IS 'SLO/SLI definitions and error budget management for GreenLang Climate OS (OBS-005)';
COMMENT ON TABLE slo.definitions IS 'Core SLO definitions with SLI queries, targets, windows, and burn rate alert configurations';
COMMENT ON TABLE slo.definition_history IS 'Audit trail of SLO definition changes, tracked by version number';
COMMENT ON TABLE slo.error_budget_snapshots IS 'TimescaleDB hypertable: point-in-time error budget state per SLO, including multi-window burn rates';
COMMENT ON TABLE slo.compliance_reports IS 'Generated compliance reports summarizing SLO performance over configurable periods';
COMMENT ON TABLE slo.evaluation_log IS 'TimescaleDB hypertable: per-evaluation SLI measurements with target comparisons and burn rate data';
COMMENT ON MATERIALIZED VIEW slo.hourly_summaries IS 'Continuous aggregate: hourly SLO performance summaries for dashboard and trend analysis queries';

COMMENT ON COLUMN slo.definitions.slo_id IS 'Unique identifier for the SLO definition';
COMMENT ON COLUMN slo.definitions.name IS 'Human-readable unique name for the SLO (e.g., api-availability)';
COMMENT ON COLUMN slo.definitions.service IS 'Service that this SLO measures (e.g., greenlang-api-gateway)';
COMMENT ON COLUMN slo.definitions.category IS 'SLO category: availability, latency, correctness, throughput, efficiency, timeliness';
COMMENT ON COLUMN slo.definitions.sli_type IS 'Type of SLI measurement: availability, latency, accuracy, ratio, throughput, timeliness';
COMMENT ON COLUMN slo.definitions.sli_good_events_query IS 'PromQL query for counting good events (numerator)';
COMMENT ON COLUMN slo.definitions.sli_total_events_query IS 'PromQL query for counting total events (denominator)';
COMMENT ON COLUMN slo.definitions.sli_threshold IS 'Threshold value for latency/throughput SLIs (e.g., 0.1 for 100ms)';
COMMENT ON COLUMN slo.definitions.target IS 'SLO target percentage (e.g., 99.9 for three nines)';
COMMENT ON COLUMN slo.definitions.window IS 'Rolling measurement window (e.g., 30d for 30 days)';
COMMENT ON COLUMN slo.definitions.owner_team IS 'Team responsible for maintaining this SLO';
COMMENT ON COLUMN slo.definitions.tier IS 'Service tier classification: tier-1 (critical), tier-2 (important), tier-3 (standard)';
COMMENT ON COLUMN slo.definitions.burn_rate_alerts IS 'JSON array of burn rate alert configurations with window sizes and thresholds';
COMMENT ON COLUMN slo.definitions.version IS 'Monotonically increasing version counter for change tracking';

COMMENT ON COLUMN slo.error_budget_snapshots.burn_rate_1h IS '1-hour burn rate: how fast the error budget is being consumed over the last hour';
COMMENT ON COLUMN slo.error_budget_snapshots.burn_rate_6h IS '6-hour burn rate: medium-term error budget consumption velocity';
COMMENT ON COLUMN slo.error_budget_snapshots.burn_rate_3d IS '3-day burn rate: long-term error budget consumption trend';
COMMENT ON COLUMN slo.error_budget_snapshots.status IS 'Budget health status: healthy (>50%), warning (20-50%), critical (<20%), exhausted (0%)';

COMMENT ON COLUMN slo.evaluation_log.burn_rate_fast IS 'Fast burn rate (1h/5m multi-window)';
COMMENT ON COLUMN slo.evaluation_log.burn_rate_medium IS 'Medium burn rate (6h/30m multi-window)';
COMMENT ON COLUMN slo.evaluation_log.burn_rate_slow IS 'Slow burn rate (3d/6h multi-window)';
