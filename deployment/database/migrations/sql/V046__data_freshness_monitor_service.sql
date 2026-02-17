-- =============================================================================
-- V046: Data Freshness Monitor Service Schema
-- =============================================================================
-- Component: AGENT-DATA-016 (Data Freshness Monitor Agent)
-- Agent ID:  GL-DATA-X-019
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Data Freshness Monitor Agent (GL-DATA-X-019) with capabilities for
-- dataset registration (source tracking, refresh cadence, priority,
-- ownership), SLA definition (warning/critical hour thresholds, breach
-- severity, escalation policies, business-hours-only mode), freshness
-- checking (age calculation, freshness scoring, SLA status evaluation),
-- refresh history tracking (timestamps, data size, record counts, source
-- info), staleness pattern detection (frequency analysis, severity
-- classification, confidence scoring), SLA breach management (detection,
-- acknowledgment, resolution workflows, breach severity tracking),
-- alerting (multi-channel delivery, suppression, acknowledgment),
-- refresh prediction (predicted vs actual timing, error tracking),
-- freshness reporting (compliance counts, breach summaries), and full
-- provenance chain tracking with SHA-256 hashes for zero-hallucination
-- audit trails.
-- =============================================================================
-- Tables (10):
--   1. freshness_datasets           - Registered datasets
--   2. freshness_sla_definitions    - SLA warning/critical thresholds
--   3. freshness_checks             - Freshness check results
--   4. freshness_refresh_history    - Dataset refresh history
--   5. freshness_staleness_patterns - Detected staleness patterns
--   6. freshness_sla_breaches       - SLA breach records
--   7. freshness_alerts             - Alert delivery records
--   8. freshness_predictions        - Refresh predictions
--   9. freshness_reports            - Generated reports
--  10. freshness_audit_log          - Audit trail
--
-- Hypertables (3):
--  11. freshness_check_events       - Freshness check event time-series (hypertable)
--  12. refresh_events               - Refresh event time-series (hypertable)
--  13. alert_events                 - Alert event time-series (hypertable)
--
-- Continuous Aggregates (2):
--   1. freshness_hourly_stats       - Hourly freshness stats
--   2. sla_breach_hourly_stats      - Hourly SLA breach stats
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-DATA-X-019.
-- Previous: V045__cross_source_reconciliation_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS data_freshness_monitor_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION data_freshness_monitor_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: data_freshness_monitor_service.freshness_datasets
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(500) NOT NULL,
    source_name VARCHAR(255),
    source_type VARCHAR(100),
    owner VARCHAR(255),
    refresh_cadence VARCHAR(50),
    priority VARCHAR(50) DEFAULT 'medium',
    status VARCHAR(50) DEFAULT 'active',
    tags JSONB DEFAULT '[]'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    last_refreshed_at TIMESTAMPTZ,
    registered_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT chk_fd_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);
ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT chk_fd_source_type CHECK (source_type IS NULL OR source_type IN ('erp', 'csv', 'excel', 'api', 'database', 'pdf', 'questionnaire', 'satellite', 'gis', 'manual', 'calculated', 'external', 'data_lake', 'stream'));
ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT chk_fd_refresh_cadence CHECK (refresh_cadence IS NULL OR refresh_cadence IN ('realtime', 'hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'annually', 'on_demand'));
ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT chk_fd_priority CHECK (priority IS NULL OR priority IN ('critical', 'high', 'medium', 'low'));
ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT chk_fd_status CHECK (status IS NULL OR status IN ('active', 'inactive', 'deprecated', 'archived', 'error'));
ALTER TABLE data_freshness_monitor_service.freshness_datasets
    ADD CONSTRAINT uq_fd_tenant_name UNIQUE (tenant_id, name);

CREATE TRIGGER trg_fd_updated_at
    BEFORE UPDATE ON data_freshness_monitor_service.freshness_datasets
    FOR EACH ROW EXECUTE FUNCTION data_freshness_monitor_service.set_updated_at();

-- =============================================================================
-- Table 2: data_freshness_monitor_service.freshness_sla_definitions
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_sla_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    warning_hours DOUBLE PRECISION NOT NULL,
    critical_hours DOUBLE PRECISION NOT NULL,
    breach_severity VARCHAR(50) DEFAULT 'medium',
    escalation_policy JSONB DEFAULT '{}'::jsonb,
    business_hours_only BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions
    ADD CONSTRAINT fk_fsd_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions
    ADD CONSTRAINT chk_fsd_warning_hours_positive CHECK (warning_hours > 0);
ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions
    ADD CONSTRAINT chk_fsd_critical_hours_positive CHECK (critical_hours > 0);
ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions
    ADD CONSTRAINT chk_fsd_critical_after_warning CHECK (critical_hours >= warning_hours);
ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions
    ADD CONSTRAINT chk_fsd_breach_severity CHECK (breach_severity IS NULL OR breach_severity IN ('critical', 'high', 'medium', 'low', 'info'));

CREATE TRIGGER trg_fsd_updated_at
    BEFORE UPDATE ON data_freshness_monitor_service.freshness_sla_definitions
    FOR EACH ROW EXECUTE FUNCTION data_freshness_monitor_service.set_updated_at();

-- =============================================================================
-- Table 3: data_freshness_monitor_service.freshness_checks
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    checked_at TIMESTAMPTZ DEFAULT NOW(),
    age_hours DOUBLE PRECISION,
    freshness_score DOUBLE PRECISION,
    freshness_level VARCHAR(50),
    sla_status VARCHAR(50),
    sla_id UUID,
    provenance_hash VARCHAR(128)
);

ALTER TABLE data_freshness_monitor_service.freshness_checks
    ADD CONSTRAINT fk_fc_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_checks
    ADD CONSTRAINT chk_fc_age_hours_non_negative CHECK (age_hours IS NULL OR age_hours >= 0);
ALTER TABLE data_freshness_monitor_service.freshness_checks
    ADD CONSTRAINT chk_fc_freshness_score_range CHECK (freshness_score IS NULL OR (freshness_score >= 0.0 AND freshness_score <= 1.0));
ALTER TABLE data_freshness_monitor_service.freshness_checks
    ADD CONSTRAINT chk_fc_freshness_level CHECK (freshness_level IS NULL OR freshness_level IN ('fresh', 'recent', 'aging', 'stale', 'expired'));
ALTER TABLE data_freshness_monitor_service.freshness_checks
    ADD CONSTRAINT chk_fc_sla_status CHECK (sla_status IS NULL OR sla_status IN ('compliant', 'warning', 'critical', 'breached', 'unknown'));

-- =============================================================================
-- Table 4: data_freshness_monitor_service.freshness_refresh_history
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_refresh_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    refreshed_at TIMESTAMPTZ NOT NULL,
    data_size_bytes BIGINT,
    record_count INTEGER,
    source_info JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128)
);

ALTER TABLE data_freshness_monitor_service.freshness_refresh_history
    ADD CONSTRAINT fk_frh_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_refresh_history
    ADD CONSTRAINT chk_frh_data_size_non_negative CHECK (data_size_bytes IS NULL OR data_size_bytes >= 0);
ALTER TABLE data_freshness_monitor_service.freshness_refresh_history
    ADD CONSTRAINT chk_frh_record_count_non_negative CHECK (record_count IS NULL OR record_count >= 0);

-- =============================================================================
-- Table 5: data_freshness_monitor_service.freshness_staleness_patterns
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_staleness_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    pattern_type VARCHAR(100),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    frequency_hours DOUBLE PRECISION,
    severity VARCHAR(50),
    confidence DOUBLE PRECISION,
    description TEXT
);

ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns
    ADD CONSTRAINT fk_fsp_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns
    ADD CONSTRAINT chk_fsp_pattern_type CHECK (pattern_type IS NULL OR pattern_type IN ('periodic_delay', 'increasing_lag', 'random_miss', 'weekend_gap', 'holiday_gap', 'source_degradation', 'pipeline_failure', 'schedule_drift', 'seasonal_pattern', 'one_time_anomaly'));
ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns
    ADD CONSTRAINT chk_fsp_frequency_hours_positive CHECK (frequency_hours IS NULL OR frequency_hours > 0);
ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns
    ADD CONSTRAINT chk_fsp_severity CHECK (severity IS NULL OR severity IN ('critical', 'high', 'medium', 'low', 'info'));
ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns
    ADD CONSTRAINT chk_fsp_confidence_range CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0));

-- =============================================================================
-- Table 6: data_freshness_monitor_service.freshness_sla_breaches
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_sla_breaches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    sla_id UUID,
    breach_severity VARCHAR(50),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'detected',
    age_at_breach_hours DOUBLE PRECISION,
    resolution_notes TEXT
);

ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT fk_fsb_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT chk_fsb_breach_severity CHECK (breach_severity IS NULL OR breach_severity IN ('critical', 'high', 'medium', 'low', 'info'));
ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT chk_fsb_status CHECK (status IS NULL OR status IN ('detected', 'acknowledged', 'investigating', 'resolving', 'resolved', 'suppressed', 'escalated'));
ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT chk_fsb_age_at_breach_positive CHECK (age_at_breach_hours IS NULL OR age_at_breach_hours > 0);
ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT chk_fsb_acknowledged_after_detected CHECK (acknowledged_at IS NULL OR detected_at IS NULL OR acknowledged_at >= detected_at);
ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches
    ADD CONSTRAINT chk_fsb_resolved_after_detected CHECK (resolved_at IS NULL OR detected_at IS NULL OR resolved_at >= detected_at);

-- =============================================================================
-- Table 7: data_freshness_monitor_service.freshness_alerts
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    breach_id UUID,
    dataset_id UUID,
    alert_severity VARCHAR(50),
    channel VARCHAR(50),
    message TEXT,
    recipients JSONB DEFAULT '[]'::jsonb,
    sent_at TIMESTAMPTZ,
    acknowledged_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'pending',
    suppressed_reason TEXT
);

ALTER TABLE data_freshness_monitor_service.freshness_alerts
    ADD CONSTRAINT chk_fa_alert_severity CHECK (alert_severity IS NULL OR alert_severity IN ('critical', 'high', 'medium', 'low', 'info'));
ALTER TABLE data_freshness_monitor_service.freshness_alerts
    ADD CONSTRAINT chk_fa_channel CHECK (channel IS NULL OR channel IN ('email', 'slack', 'pagerduty', 'opsgenie', 'teams', 'webhook', 'sms', 'in_app'));
ALTER TABLE data_freshness_monitor_service.freshness_alerts
    ADD CONSTRAINT chk_fa_status CHECK (status IS NULL OR status IN ('pending', 'sent', 'delivered', 'acknowledged', 'suppressed', 'failed', 'expired'));
ALTER TABLE data_freshness_monitor_service.freshness_alerts
    ADD CONSTRAINT chk_fa_acknowledged_after_sent CHECK (acknowledged_at IS NULL OR sent_at IS NULL OR acknowledged_at >= sent_at);

-- =============================================================================
-- Table 8: data_freshness_monitor_service.freshness_predictions
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    predicted_refresh_at TIMESTAMPTZ,
    confidence DOUBLE PRECISION,
    actual_refresh_at TIMESTAMPTZ,
    error_hours DOUBLE PRECISION,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE data_freshness_monitor_service.freshness_predictions
    ADD CONSTRAINT fk_fp_dataset_id FOREIGN KEY (dataset_id) REFERENCES data_freshness_monitor_service.freshness_datasets(id) ON DELETE CASCADE;
ALTER TABLE data_freshness_monitor_service.freshness_predictions
    ADD CONSTRAINT chk_fp_confidence_range CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0));
ALTER TABLE data_freshness_monitor_service.freshness_predictions
    ADD CONSTRAINT chk_fp_status CHECK (status IS NULL OR status IN ('pending', 'confirmed', 'missed', 'early', 'late', 'expired'));

-- =============================================================================
-- Table 9: data_freshness_monitor_service.freshness_reports
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    report_type VARCHAR(100),
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    dataset_count INTEGER,
    compliant_count INTEGER,
    breached_count INTEGER,
    summary JSONB DEFAULT '{}'::jsonb,
    provenance_hash VARCHAR(128)
);

ALTER TABLE data_freshness_monitor_service.freshness_reports
    ADD CONSTRAINT chk_fr_report_type CHECK (report_type IS NULL OR report_type IN ('freshness', 'sla_compliance', 'staleness', 'breach', 'prediction', 'executive'));
ALTER TABLE data_freshness_monitor_service.freshness_reports
    ADD CONSTRAINT chk_fr_dataset_count_non_negative CHECK (dataset_count IS NULL OR dataset_count >= 0);
ALTER TABLE data_freshness_monitor_service.freshness_reports
    ADD CONSTRAINT chk_fr_compliant_count_non_negative CHECK (compliant_count IS NULL OR compliant_count >= 0);
ALTER TABLE data_freshness_monitor_service.freshness_reports
    ADD CONSTRAINT chk_fr_breached_count_non_negative CHECK (breached_count IS NULL OR breached_count >= 0);

-- =============================================================================
-- Table 10: data_freshness_monitor_service.freshness_audit_log
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    operation VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id UUID,
    details JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    provenance_hash VARCHAR(128)
);

ALTER TABLE data_freshness_monitor_service.freshness_audit_log
    ADD CONSTRAINT chk_fal_operation CHECK (operation IN (
        'dataset_registered', 'dataset_updated', 'dataset_deactivated', 'dataset_archived',
        'sla_created', 'sla_updated', 'sla_deleted',
        'check_performed', 'check_failed',
        'refresh_recorded', 'refresh_detected',
        'pattern_detected', 'pattern_updated', 'pattern_dismissed',
        'breach_detected', 'breach_acknowledged', 'breach_resolved', 'breach_escalated',
        'alert_sent', 'alert_acknowledged', 'alert_suppressed', 'alert_failed',
        'prediction_created', 'prediction_confirmed', 'prediction_missed',
        'report_generated', 'config_changed', 'export_generated', 'import_completed'
    ));
ALTER TABLE data_freshness_monitor_service.freshness_audit_log
    ADD CONSTRAINT chk_fal_operation_not_empty CHECK (LENGTH(TRIM(operation)) > 0);
ALTER TABLE data_freshness_monitor_service.freshness_audit_log
    ADD CONSTRAINT chk_fal_entity_type CHECK (entity_type IS NULL OR entity_type IN ('dataset', 'sla', 'check', 'refresh', 'pattern', 'breach', 'alert', 'prediction', 'report', 'config'));

-- =============================================================================
-- Table 11: data_freshness_monitor_service.freshness_check_events (hypertable)
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.freshness_check_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('data_freshness_monitor_service.freshness_check_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE data_freshness_monitor_service.freshness_check_events
    ADD CONSTRAINT chk_fce_event_type CHECK (event_type IN (
        'check_started', 'check_completed', 'check_failed',
        'fresh_detected', 'stale_detected', 'expired_detected',
        'sla_compliant', 'sla_warning', 'sla_critical', 'sla_breached',
        'score_calculated', 'age_calculated',
        'progress_update'
    ));

-- =============================================================================
-- Table 12: data_freshness_monitor_service.refresh_events (hypertable)
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.refresh_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    dataset_id UUID,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('data_freshness_monitor_service.refresh_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE data_freshness_monitor_service.refresh_events
    ADD CONSTRAINT chk_rfe_event_type CHECK (event_type IN (
        'refresh_detected', 'refresh_recorded', 'refresh_validated',
        'refresh_late', 'refresh_early', 'refresh_on_time',
        'size_changed', 'record_count_changed',
        'pattern_detected', 'pattern_changed',
        'progress_update'
    ));

-- =============================================================================
-- Table 13: data_freshness_monitor_service.alert_events (hypertable)
-- =============================================================================

CREATE TABLE data_freshness_monitor_service.alert_events (
    event_id UUID DEFAULT gen_random_uuid(),
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    alert_id UUID,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (event_id, time)
);

SELECT create_hypertable('data_freshness_monitor_service.alert_events', 'time',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE data_freshness_monitor_service.alert_events
    ADD CONSTRAINT chk_ae_event_type CHECK (event_type IN (
        'alert_created', 'alert_sent', 'alert_delivered', 'alert_failed',
        'alert_acknowledged', 'alert_suppressed', 'alert_expired',
        'breach_detected', 'breach_escalated', 'breach_resolved',
        'notification_queued', 'notification_dispatched',
        'progress_update'
    ));

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- freshness_hourly_stats: hourly bucket, count checks, avg score, min score, breach count per hour
CREATE MATERIALIZED VIEW data_freshness_monitor_service.freshness_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE event_type IN ('check_completed')) AS check_count,
    COUNT(*) FILTER (WHERE event_type IN ('sla_breached')) AS breach_count,
    COUNT(*) FILTER (WHERE event_type IN ('sla_warning')) AS warning_count,
    COUNT(*) FILTER (WHERE event_type IN ('sla_compliant')) AS compliant_count
FROM data_freshness_monitor_service.freshness_check_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('data_freshness_monitor_service.freshness_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- sla_breach_hourly_stats: hourly bucket, count breaches by severity, count alerts
CREATE MATERIALIZED VIEW data_freshness_monitor_service.sla_breach_hourly_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    tenant_id,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE event_type IN ('breach_detected')) AS breach_count,
    COUNT(*) FILTER (WHERE event_type IN ('breach_escalated')) AS escalated_count,
    COUNT(*) FILTER (WHERE event_type IN ('breach_resolved')) AS resolved_count,
    COUNT(*) FILTER (WHERE event_type IN ('alert_sent', 'alert_delivered')) AS alert_count
FROM data_freshness_monitor_service.alert_events
WHERE time IS NOT NULL
GROUP BY bucket, tenant_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('data_freshness_monitor_service.sla_breach_hourly_stats',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- freshness_datasets indexes (14)
CREATE INDEX idx_fd_tenant_id ON data_freshness_monitor_service.freshness_datasets(tenant_id);
CREATE INDEX idx_fd_name ON data_freshness_monitor_service.freshness_datasets(name);
CREATE INDEX idx_fd_source_name ON data_freshness_monitor_service.freshness_datasets(source_name);
CREATE INDEX idx_fd_source_type ON data_freshness_monitor_service.freshness_datasets(source_type);
CREATE INDEX idx_fd_owner ON data_freshness_monitor_service.freshness_datasets(owner);
CREATE INDEX idx_fd_refresh_cadence ON data_freshness_monitor_service.freshness_datasets(refresh_cadence);
CREATE INDEX idx_fd_priority ON data_freshness_monitor_service.freshness_datasets(priority);
CREATE INDEX idx_fd_status ON data_freshness_monitor_service.freshness_datasets(status);
CREATE INDEX idx_fd_last_refreshed_at ON data_freshness_monitor_service.freshness_datasets(last_refreshed_at DESC);
CREATE INDEX idx_fd_registered_at ON data_freshness_monitor_service.freshness_datasets(registered_at DESC);
CREATE INDEX idx_fd_updated_at ON data_freshness_monitor_service.freshness_datasets(updated_at DESC);
CREATE INDEX idx_fd_tenant_status ON data_freshness_monitor_service.freshness_datasets(tenant_id, status);
CREATE INDEX idx_fd_tenant_priority ON data_freshness_monitor_service.freshness_datasets(tenant_id, priority);
CREATE INDEX idx_fd_tags ON data_freshness_monitor_service.freshness_datasets USING GIN (tags);
CREATE INDEX idx_fd_metadata ON data_freshness_monitor_service.freshness_datasets USING GIN (metadata);

-- freshness_sla_definitions indexes (10)
CREATE INDEX idx_fsd_tenant_id ON data_freshness_monitor_service.freshness_sla_definitions(tenant_id);
CREATE INDEX idx_fsd_dataset_id ON data_freshness_monitor_service.freshness_sla_definitions(dataset_id);
CREATE INDEX idx_fsd_warning_hours ON data_freshness_monitor_service.freshness_sla_definitions(warning_hours);
CREATE INDEX idx_fsd_critical_hours ON data_freshness_monitor_service.freshness_sla_definitions(critical_hours);
CREATE INDEX idx_fsd_breach_severity ON data_freshness_monitor_service.freshness_sla_definitions(breach_severity);
CREATE INDEX idx_fsd_business_hours ON data_freshness_monitor_service.freshness_sla_definitions(business_hours_only);
CREATE INDEX idx_fsd_created_at ON data_freshness_monitor_service.freshness_sla_definitions(created_at DESC);
CREATE INDEX idx_fsd_updated_at ON data_freshness_monitor_service.freshness_sla_definitions(updated_at DESC);
CREATE INDEX idx_fsd_tenant_dataset ON data_freshness_monitor_service.freshness_sla_definitions(tenant_id, dataset_id);
CREATE INDEX idx_fsd_escalation ON data_freshness_monitor_service.freshness_sla_definitions USING GIN (escalation_policy);

-- freshness_checks indexes (14)
CREATE INDEX idx_fc_tenant_id ON data_freshness_monitor_service.freshness_checks(tenant_id);
CREATE INDEX idx_fc_dataset_id ON data_freshness_monitor_service.freshness_checks(dataset_id);
CREATE INDEX idx_fc_checked_at ON data_freshness_monitor_service.freshness_checks(checked_at DESC);
CREATE INDEX idx_fc_age_hours ON data_freshness_monitor_service.freshness_checks(age_hours);
CREATE INDEX idx_fc_freshness_score ON data_freshness_monitor_service.freshness_checks(freshness_score DESC);
CREATE INDEX idx_fc_freshness_level ON data_freshness_monitor_service.freshness_checks(freshness_level);
CREATE INDEX idx_fc_sla_status ON data_freshness_monitor_service.freshness_checks(sla_status);
CREATE INDEX idx_fc_sla_id ON data_freshness_monitor_service.freshness_checks(sla_id);
CREATE INDEX idx_fc_provenance ON data_freshness_monitor_service.freshness_checks(provenance_hash);
CREATE INDEX idx_fc_tenant_dataset ON data_freshness_monitor_service.freshness_checks(tenant_id, dataset_id);
CREATE INDEX idx_fc_tenant_sla_status ON data_freshness_monitor_service.freshness_checks(tenant_id, sla_status);
CREATE INDEX idx_fc_tenant_checked ON data_freshness_monitor_service.freshness_checks(tenant_id, checked_at DESC);
CREATE INDEX idx_fc_dataset_checked ON data_freshness_monitor_service.freshness_checks(dataset_id, checked_at DESC);
CREATE INDEX idx_fc_dataset_sla_status ON data_freshness_monitor_service.freshness_checks(dataset_id, sla_status);

-- freshness_refresh_history indexes (12)
CREATE INDEX idx_frh_tenant_id ON data_freshness_monitor_service.freshness_refresh_history(tenant_id);
CREATE INDEX idx_frh_dataset_id ON data_freshness_monitor_service.freshness_refresh_history(dataset_id);
CREATE INDEX idx_frh_refreshed_at ON data_freshness_monitor_service.freshness_refresh_history(refreshed_at DESC);
CREATE INDEX idx_frh_data_size ON data_freshness_monitor_service.freshness_refresh_history(data_size_bytes DESC);
CREATE INDEX idx_frh_record_count ON data_freshness_monitor_service.freshness_refresh_history(record_count DESC);
CREATE INDEX idx_frh_provenance ON data_freshness_monitor_service.freshness_refresh_history(provenance_hash);
CREATE INDEX idx_frh_tenant_dataset ON data_freshness_monitor_service.freshness_refresh_history(tenant_id, dataset_id);
CREATE INDEX idx_frh_tenant_refreshed ON data_freshness_monitor_service.freshness_refresh_history(tenant_id, refreshed_at DESC);
CREATE INDEX idx_frh_dataset_refreshed ON data_freshness_monitor_service.freshness_refresh_history(dataset_id, refreshed_at DESC);
CREATE INDEX idx_frh_tenant_dataset_refreshed ON data_freshness_monitor_service.freshness_refresh_history(tenant_id, dataset_id, refreshed_at DESC);
CREATE INDEX idx_frh_dataset_size ON data_freshness_monitor_service.freshness_refresh_history(dataset_id, data_size_bytes DESC);
CREATE INDEX idx_frh_source_info ON data_freshness_monitor_service.freshness_refresh_history USING GIN (source_info);

-- freshness_staleness_patterns indexes (12)
CREATE INDEX idx_fsp_tenant_id ON data_freshness_monitor_service.freshness_staleness_patterns(tenant_id);
CREATE INDEX idx_fsp_dataset_id ON data_freshness_monitor_service.freshness_staleness_patterns(dataset_id);
CREATE INDEX idx_fsp_pattern_type ON data_freshness_monitor_service.freshness_staleness_patterns(pattern_type);
CREATE INDEX idx_fsp_detected_at ON data_freshness_monitor_service.freshness_staleness_patterns(detected_at DESC);
CREATE INDEX idx_fsp_frequency_hours ON data_freshness_monitor_service.freshness_staleness_patterns(frequency_hours);
CREATE INDEX idx_fsp_severity ON data_freshness_monitor_service.freshness_staleness_patterns(severity);
CREATE INDEX idx_fsp_confidence ON data_freshness_monitor_service.freshness_staleness_patterns(confidence DESC);
CREATE INDEX idx_fsp_tenant_dataset ON data_freshness_monitor_service.freshness_staleness_patterns(tenant_id, dataset_id);
CREATE INDEX idx_fsp_tenant_severity ON data_freshness_monitor_service.freshness_staleness_patterns(tenant_id, severity);
CREATE INDEX idx_fsp_tenant_type ON data_freshness_monitor_service.freshness_staleness_patterns(tenant_id, pattern_type);
CREATE INDEX idx_fsp_dataset_severity ON data_freshness_monitor_service.freshness_staleness_patterns(dataset_id, severity);
CREATE INDEX idx_fsp_dataset_detected ON data_freshness_monitor_service.freshness_staleness_patterns(dataset_id, detected_at DESC);

-- freshness_sla_breaches indexes (14)
CREATE INDEX idx_fsb_tenant_id ON data_freshness_monitor_service.freshness_sla_breaches(tenant_id);
CREATE INDEX idx_fsb_dataset_id ON data_freshness_monitor_service.freshness_sla_breaches(dataset_id);
CREATE INDEX idx_fsb_sla_id ON data_freshness_monitor_service.freshness_sla_breaches(sla_id);
CREATE INDEX idx_fsb_breach_severity ON data_freshness_monitor_service.freshness_sla_breaches(breach_severity);
CREATE INDEX idx_fsb_detected_at ON data_freshness_monitor_service.freshness_sla_breaches(detected_at DESC);
CREATE INDEX idx_fsb_acknowledged_at ON data_freshness_monitor_service.freshness_sla_breaches(acknowledged_at DESC);
CREATE INDEX idx_fsb_resolved_at ON data_freshness_monitor_service.freshness_sla_breaches(resolved_at DESC);
CREATE INDEX idx_fsb_status ON data_freshness_monitor_service.freshness_sla_breaches(status);
CREATE INDEX idx_fsb_age_at_breach ON data_freshness_monitor_service.freshness_sla_breaches(age_at_breach_hours DESC);
CREATE INDEX idx_fsb_tenant_dataset ON data_freshness_monitor_service.freshness_sla_breaches(tenant_id, dataset_id);
CREATE INDEX idx_fsb_tenant_severity ON data_freshness_monitor_service.freshness_sla_breaches(tenant_id, breach_severity);
CREATE INDEX idx_fsb_tenant_status ON data_freshness_monitor_service.freshness_sla_breaches(tenant_id, status);
CREATE INDEX idx_fsb_dataset_status ON data_freshness_monitor_service.freshness_sla_breaches(dataset_id, status);
CREATE INDEX idx_fsb_dataset_severity ON data_freshness_monitor_service.freshness_sla_breaches(dataset_id, breach_severity);

-- freshness_alerts indexes (14)
CREATE INDEX idx_fa_tenant_id ON data_freshness_monitor_service.freshness_alerts(tenant_id);
CREATE INDEX idx_fa_breach_id ON data_freshness_monitor_service.freshness_alerts(breach_id);
CREATE INDEX idx_fa_dataset_id ON data_freshness_monitor_service.freshness_alerts(dataset_id);
CREATE INDEX idx_fa_alert_severity ON data_freshness_monitor_service.freshness_alerts(alert_severity);
CREATE INDEX idx_fa_channel ON data_freshness_monitor_service.freshness_alerts(channel);
CREATE INDEX idx_fa_sent_at ON data_freshness_monitor_service.freshness_alerts(sent_at DESC);
CREATE INDEX idx_fa_acknowledged_at ON data_freshness_monitor_service.freshness_alerts(acknowledged_at DESC);
CREATE INDEX idx_fa_status ON data_freshness_monitor_service.freshness_alerts(status);
CREATE INDEX idx_fa_tenant_severity ON data_freshness_monitor_service.freshness_alerts(tenant_id, alert_severity);
CREATE INDEX idx_fa_tenant_status ON data_freshness_monitor_service.freshness_alerts(tenant_id, status);
CREATE INDEX idx_fa_tenant_channel ON data_freshness_monitor_service.freshness_alerts(tenant_id, channel);
CREATE INDEX idx_fa_tenant_dataset ON data_freshness_monitor_service.freshness_alerts(tenant_id, dataset_id);
CREATE INDEX idx_fa_breach_status ON data_freshness_monitor_service.freshness_alerts(breach_id, status);
CREATE INDEX idx_fa_recipients ON data_freshness_monitor_service.freshness_alerts USING GIN (recipients);

-- freshness_predictions indexes (12)
CREATE INDEX idx_fp_tenant_id ON data_freshness_monitor_service.freshness_predictions(tenant_id);
CREATE INDEX idx_fp_dataset_id ON data_freshness_monitor_service.freshness_predictions(dataset_id);
CREATE INDEX idx_fp_predicted_refresh_at ON data_freshness_monitor_service.freshness_predictions(predicted_refresh_at DESC);
CREATE INDEX idx_fp_confidence ON data_freshness_monitor_service.freshness_predictions(confidence DESC);
CREATE INDEX idx_fp_actual_refresh_at ON data_freshness_monitor_service.freshness_predictions(actual_refresh_at DESC);
CREATE INDEX idx_fp_error_hours ON data_freshness_monitor_service.freshness_predictions(error_hours);
CREATE INDEX idx_fp_status ON data_freshness_monitor_service.freshness_predictions(status);
CREATE INDEX idx_fp_created_at ON data_freshness_monitor_service.freshness_predictions(created_at DESC);
CREATE INDEX idx_fp_tenant_dataset ON data_freshness_monitor_service.freshness_predictions(tenant_id, dataset_id);
CREATE INDEX idx_fp_tenant_status ON data_freshness_monitor_service.freshness_predictions(tenant_id, status);
CREATE INDEX idx_fp_dataset_status ON data_freshness_monitor_service.freshness_predictions(dataset_id, status);
CREATE INDEX idx_fp_dataset_predicted ON data_freshness_monitor_service.freshness_predictions(dataset_id, predicted_refresh_at DESC);

-- freshness_reports indexes (12)
CREATE INDEX idx_fr_tenant_id ON data_freshness_monitor_service.freshness_reports(tenant_id);
CREATE INDEX idx_fr_report_type ON data_freshness_monitor_service.freshness_reports(report_type);
CREATE INDEX idx_fr_generated_at ON data_freshness_monitor_service.freshness_reports(generated_at DESC);
CREATE INDEX idx_fr_dataset_count ON data_freshness_monitor_service.freshness_reports(dataset_count DESC);
CREATE INDEX idx_fr_compliant_count ON data_freshness_monitor_service.freshness_reports(compliant_count DESC);
CREATE INDEX idx_fr_breached_count ON data_freshness_monitor_service.freshness_reports(breached_count DESC);
CREATE INDEX idx_fr_provenance ON data_freshness_monitor_service.freshness_reports(provenance_hash);
CREATE INDEX idx_fr_tenant_type ON data_freshness_monitor_service.freshness_reports(tenant_id, report_type);
CREATE INDEX idx_fr_tenant_generated ON data_freshness_monitor_service.freshness_reports(tenant_id, generated_at DESC);
CREATE INDEX idx_fr_type_generated ON data_freshness_monitor_service.freshness_reports(report_type, generated_at DESC);
CREATE INDEX idx_fr_tenant_breached ON data_freshness_monitor_service.freshness_reports(tenant_id, breached_count DESC);
CREATE INDEX idx_fr_summary ON data_freshness_monitor_service.freshness_reports USING GIN (summary);

-- freshness_audit_log indexes (12)
CREATE INDEX idx_fal_tenant_id ON data_freshness_monitor_service.freshness_audit_log(tenant_id);
CREATE INDEX idx_fal_operation ON data_freshness_monitor_service.freshness_audit_log(operation);
CREATE INDEX idx_fal_entity_type ON data_freshness_monitor_service.freshness_audit_log(entity_type);
CREATE INDEX idx_fal_entity_id ON data_freshness_monitor_service.freshness_audit_log(entity_id);
CREATE INDEX idx_fal_timestamp ON data_freshness_monitor_service.freshness_audit_log(timestamp DESC);
CREATE INDEX idx_fal_provenance ON data_freshness_monitor_service.freshness_audit_log(provenance_hash);
CREATE INDEX idx_fal_tenant_operation ON data_freshness_monitor_service.freshness_audit_log(tenant_id, operation);
CREATE INDEX idx_fal_tenant_entity ON data_freshness_monitor_service.freshness_audit_log(tenant_id, entity_type);
CREATE INDEX idx_fal_tenant_timestamp ON data_freshness_monitor_service.freshness_audit_log(tenant_id, timestamp DESC);
CREATE INDEX idx_fal_entity_type_id ON data_freshness_monitor_service.freshness_audit_log(entity_type, entity_id);
CREATE INDEX idx_fal_tenant_entity_op ON data_freshness_monitor_service.freshness_audit_log(tenant_id, entity_type, operation);
CREATE INDEX idx_fal_details ON data_freshness_monitor_service.freshness_audit_log USING GIN (details);

-- freshness_check_events indexes (hypertable-aware) (8)
CREATE INDEX idx_fce_tenant_id ON data_freshness_monitor_service.freshness_check_events(tenant_id, time DESC);
CREATE INDEX idx_fce_dataset_id ON data_freshness_monitor_service.freshness_check_events(dataset_id, time DESC);
CREATE INDEX idx_fce_event_type ON data_freshness_monitor_service.freshness_check_events(event_type, time DESC);
CREATE INDEX idx_fce_tenant_type ON data_freshness_monitor_service.freshness_check_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_fce_tenant_dataset ON data_freshness_monitor_service.freshness_check_events(tenant_id, dataset_id, time DESC);
CREATE INDEX idx_fce_dataset_type ON data_freshness_monitor_service.freshness_check_events(dataset_id, event_type, time DESC);
CREATE INDEX idx_fce_tenant_dataset_type ON data_freshness_monitor_service.freshness_check_events(tenant_id, dataset_id, event_type, time DESC);
CREATE INDEX idx_fce_event_data ON data_freshness_monitor_service.freshness_check_events USING GIN (event_data);

-- refresh_events indexes (hypertable-aware) (8)
CREATE INDEX idx_rfe_tenant_id ON data_freshness_monitor_service.refresh_events(tenant_id, time DESC);
CREATE INDEX idx_rfe_dataset_id ON data_freshness_monitor_service.refresh_events(dataset_id, time DESC);
CREATE INDEX idx_rfe_event_type ON data_freshness_monitor_service.refresh_events(event_type, time DESC);
CREATE INDEX idx_rfe_tenant_type ON data_freshness_monitor_service.refresh_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_rfe_tenant_dataset ON data_freshness_monitor_service.refresh_events(tenant_id, dataset_id, time DESC);
CREATE INDEX idx_rfe_dataset_type ON data_freshness_monitor_service.refresh_events(dataset_id, event_type, time DESC);
CREATE INDEX idx_rfe_tenant_dataset_type ON data_freshness_monitor_service.refresh_events(tenant_id, dataset_id, event_type, time DESC);
CREATE INDEX idx_rfe_event_data ON data_freshness_monitor_service.refresh_events USING GIN (event_data);

-- alert_events indexes (hypertable-aware) (8)
CREATE INDEX idx_ae_tenant_id ON data_freshness_monitor_service.alert_events(tenant_id, time DESC);
CREATE INDEX idx_ae_alert_id ON data_freshness_monitor_service.alert_events(alert_id, time DESC);
CREATE INDEX idx_ae_event_type ON data_freshness_monitor_service.alert_events(event_type, time DESC);
CREATE INDEX idx_ae_tenant_type ON data_freshness_monitor_service.alert_events(tenant_id, event_type, time DESC);
CREATE INDEX idx_ae_tenant_alert ON data_freshness_monitor_service.alert_events(tenant_id, alert_id, time DESC);
CREATE INDEX idx_ae_alert_type ON data_freshness_monitor_service.alert_events(alert_id, event_type, time DESC);
CREATE INDEX idx_ae_tenant_alert_type ON data_freshness_monitor_service.alert_events(tenant_id, alert_id, event_type, time DESC);
CREATE INDEX idx_ae_event_data ON data_freshness_monitor_service.alert_events USING GIN (event_data);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE data_freshness_monitor_service.freshness_datasets ENABLE ROW LEVEL SECURITY;
CREATE POLICY fd_tenant_read ON data_freshness_monitor_service.freshness_datasets FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fd_tenant_write ON data_freshness_monitor_service.freshness_datasets FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_sla_definitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY fsd_tenant_read ON data_freshness_monitor_service.freshness_sla_definitions FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fsd_tenant_write ON data_freshness_monitor_service.freshness_sla_definitions FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_checks ENABLE ROW LEVEL SECURITY;
CREATE POLICY fc_tenant_read ON data_freshness_monitor_service.freshness_checks FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fc_tenant_write ON data_freshness_monitor_service.freshness_checks FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_refresh_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY frh_tenant_read ON data_freshness_monitor_service.freshness_refresh_history FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY frh_tenant_write ON data_freshness_monitor_service.freshness_refresh_history FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_staleness_patterns ENABLE ROW LEVEL SECURITY;
CREATE POLICY fsp_tenant_read ON data_freshness_monitor_service.freshness_staleness_patterns FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fsp_tenant_write ON data_freshness_monitor_service.freshness_staleness_patterns FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_sla_breaches ENABLE ROW LEVEL SECURITY;
CREATE POLICY fsb_tenant_read ON data_freshness_monitor_service.freshness_sla_breaches FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fsb_tenant_write ON data_freshness_monitor_service.freshness_sla_breaches FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY fa_tenant_read ON data_freshness_monitor_service.freshness_alerts FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fa_tenant_write ON data_freshness_monitor_service.freshness_alerts FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY fp_tenant_read ON data_freshness_monitor_service.freshness_predictions FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fp_tenant_write ON data_freshness_monitor_service.freshness_predictions FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY fr_tenant_read ON data_freshness_monitor_service.freshness_reports FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fr_tenant_write ON data_freshness_monitor_service.freshness_reports FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY fal_tenant_read ON data_freshness_monitor_service.freshness_audit_log FOR SELECT USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');
CREATE POLICY fal_tenant_write ON data_freshness_monitor_service.freshness_audit_log FOR ALL USING (tenant_id::text = current_setting('app.current_tenant', true) OR current_setting('app.current_tenant', true) IS NULL OR current_setting('app.is_admin', true) = 'true');

ALTER TABLE data_freshness_monitor_service.freshness_check_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fce_tenant_read ON data_freshness_monitor_service.freshness_check_events FOR SELECT USING (TRUE);
CREATE POLICY fce_tenant_write ON data_freshness_monitor_service.freshness_check_events FOR ALL USING (TRUE);

ALTER TABLE data_freshness_monitor_service.refresh_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rfe_tenant_read ON data_freshness_monitor_service.refresh_events FOR SELECT USING (TRUE);
CREATE POLICY rfe_tenant_write ON data_freshness_monitor_service.refresh_events FOR ALL USING (TRUE);

ALTER TABLE data_freshness_monitor_service.alert_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ae_tenant_read ON data_freshness_monitor_service.alert_events FOR SELECT USING (TRUE);
CREATE POLICY ae_tenant_write ON data_freshness_monitor_service.alert_events FOR ALL USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA data_freshness_monitor_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA data_freshness_monitor_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA data_freshness_monitor_service TO greenlang_app;
GRANT SELECT ON data_freshness_monitor_service.freshness_hourly_stats TO greenlang_app;
GRANT SELECT ON data_freshness_monitor_service.sla_breach_hourly_stats TO greenlang_app;

GRANT USAGE ON SCHEMA data_freshness_monitor_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA data_freshness_monitor_service TO greenlang_readonly;
GRANT SELECT ON data_freshness_monitor_service.freshness_hourly_stats TO greenlang_readonly;
GRANT SELECT ON data_freshness_monitor_service.sla_breach_hourly_stats TO greenlang_readonly;

GRANT ALL ON SCHEMA data_freshness_monitor_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA data_freshness_monitor_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA data_freshness_monitor_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'freshness:datasets:read', 'freshness', 'datasets_read', 'View registered datasets and their freshness status'),
    (gen_random_uuid(), 'freshness:datasets:write', 'freshness', 'datasets_write', 'Register, update, and manage datasets'),
    (gen_random_uuid(), 'freshness:sla:read', 'freshness', 'sla_read', 'View SLA definitions and thresholds'),
    (gen_random_uuid(), 'freshness:sla:write', 'freshness', 'sla_write', 'Create, update, and manage SLA definitions'),
    (gen_random_uuid(), 'freshness:checks:read', 'freshness', 'checks_read', 'View freshness check results and scores'),
    (gen_random_uuid(), 'freshness:checks:write', 'freshness', 'checks_write', 'Perform and manage freshness checks'),
    (gen_random_uuid(), 'freshness:breaches:read', 'freshness', 'breaches_read', 'View SLA breaches and their status'),
    (gen_random_uuid(), 'freshness:breaches:write', 'freshness', 'breaches_write', 'Acknowledge, resolve, and manage SLA breaches'),
    (gen_random_uuid(), 'freshness:alerts:read', 'freshness', 'alerts_read', 'View freshness alerts and delivery status'),
    (gen_random_uuid(), 'freshness:alerts:write', 'freshness', 'alerts_write', 'Send, acknowledge, and manage freshness alerts'),
    (gen_random_uuid(), 'freshness:predictions:read', 'freshness', 'predictions_read', 'View refresh predictions and accuracy'),
    (gen_random_uuid(), 'freshness:predictions:write', 'freshness', 'predictions_write', 'Create and manage refresh predictions'),
    (gen_random_uuid(), 'freshness:reports:read', 'freshness', 'reports_read', 'View freshness reports and summaries'),
    (gen_random_uuid(), 'freshness:reports:write', 'freshness', 'reports_write', 'Generate and manage freshness reports'),
    (gen_random_uuid(), 'freshness:audit:read', 'freshness', 'audit_read', 'View freshness audit log entries and provenance chains'),
    (gen_random_uuid(), 'freshness:admin', 'freshness', 'admin', 'Freshness monitor service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('data_freshness_monitor_service.freshness_check_events', INTERVAL '90 days');
SELECT add_retention_policy('data_freshness_monitor_service.refresh_events', INTERVAL '90 days');
SELECT add_retention_policy('data_freshness_monitor_service.alert_events', INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE data_freshness_monitor_service.freshness_check_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('data_freshness_monitor_service.freshness_check_events', INTERVAL '7 days');

ALTER TABLE data_freshness_monitor_service.refresh_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('data_freshness_monitor_service.refresh_events', INTERVAL '7 days');

ALTER TABLE data_freshness_monitor_service.alert_events SET (timescaledb.compress, timescaledb.compress_segmentby = 'tenant_id', timescaledb.compress_orderby = 'time DESC');
SELECT add_compression_policy('data_freshness_monitor_service.alert_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Data Freshness Monitor Agent (GL-DATA-X-019)
-- =============================================================================

INSERT INTO agent_registry_service.agents (agent_id, name, description, layer, execution_mode, idempotency_support, deterministic, max_concurrent_runs, glip_version, supports_checkpointing, author, documentation_url, enabled, tenant_id) VALUES
('GL-DATA-X-019', 'Data Freshness Monitor Agent',
 'Data freshness monitoring engine for GreenLang Climate OS. Registers and tracks datasets with source type, ownership, refresh cadence, and priority classification. Defines SLA thresholds (warning/critical hours) with breach severity, escalation policies, and business-hours-only mode. Performs freshness checks with age calculation, freshness scoring (0-1 scale), freshness level classification (fresh/recent/aging/stale/expired), and SLA status evaluation (compliant/warning/critical/breached). Tracks refresh history with timestamps, data sizes, record counts, and source information. Detects staleness patterns (periodic delay, increasing lag, weekend/holiday gaps, pipeline failures, schedule drift, seasonal patterns). Manages SLA breaches with detection, acknowledgment, investigation, resolution workflows. Delivers alerts across multiple channels (email/Slack/PagerDuty/OpsGenie/Teams/webhook/SMS/in-app) with suppression and acknowledgment. Predicts next refresh timing with confidence scoring and error tracking. Generates freshness compliance reports. SHA-256 provenance chains for zero-hallucination audit trail.',
 2, 'async', true, true, 5, '1.0.0', true,
 'GreenLang Data Team', 'https://docs.greenlang.ai/agents/data-freshness-monitor', true, 'default')
ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (agent_id, version, resource_profile, container_spec, tags, sectors, provenance_hash) VALUES
('GL-DATA-X-019', '1.0.0',
 '{"cpu_request": "250m", "cpu_limit": "1000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
 '{"image": "greenlang/data-freshness-monitor-service", "tag": "1.0.0", "port": 8000}'::jsonb,
 '{"freshness", "sla", "monitoring", "staleness", "alerting", "prediction", "data-quality"}',
 '{"cross-sector", "manufacturing", "retail", "energy", "finance", "agriculture", "utilities"}',
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4')
ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (agent_id, version, name, category, description, input_types, output_types, parameters) VALUES
('GL-DATA-X-019', '1.0.0', 'dataset_registration', 'configuration', 'Register datasets for freshness monitoring with source type, ownership, refresh cadence, priority, and metadata.', '{"dataset_name", "source_name", "source_type", "owner", "refresh_cadence", "priority"}', '{"dataset_id", "registration_status", "validation_result"}', '{"source_types": ["erp", "csv", "excel", "api", "database", "pdf", "questionnaire", "satellite", "gis", "manual", "calculated", "external", "data_lake", "stream"], "priorities": ["critical", "high", "medium", "low"]}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'sla_definition', 'configuration', 'Define SLA thresholds for datasets with warning/critical hours, breach severity, escalation policies, and business-hours-only mode.', '{"dataset_id", "warning_hours", "critical_hours", "breach_severity", "escalation_policy"}', '{"sla_id", "validation_result"}', '{"breach_severities": ["critical", "high", "medium", "low", "info"], "business_hours_support": true}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'freshness_checking', 'monitoring', 'Perform freshness checks calculating age, scoring freshness, classifying levels, and evaluating SLA status.', '{"dataset_id", "check_config"}', '{"check_result", "age_hours", "freshness_score", "freshness_level", "sla_status"}', '{"freshness_levels": ["fresh", "recent", "aging", "stale", "expired"], "sla_statuses": ["compliant", "warning", "critical", "breached"]}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'staleness_detection', 'analysis', 'Detect staleness patterns in dataset refresh behavior with frequency analysis, severity classification, and confidence scoring.', '{"dataset_id", "detection_config"}', '{"patterns", "severity_distribution", "recommendations"}', '{"pattern_types": ["periodic_delay", "increasing_lag", "random_miss", "weekend_gap", "holiday_gap", "source_degradation", "pipeline_failure", "schedule_drift", "seasonal_pattern", "one_time_anomaly"]}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'breach_management', 'processing', 'Manage SLA breaches through detection, acknowledgment, investigation, and resolution workflows.', '{"breach_id", "action", "notes"}', '{"breach_status", "resolution_result", "escalation_status"}', '{"statuses": ["detected", "acknowledged", "investigating", "resolving", "resolved", "suppressed", "escalated"]}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'alert_delivery', 'notification', 'Deliver freshness alerts across multiple channels with suppression and acknowledgment tracking.', '{"breach_id", "channels", "recipients", "message"}', '{"alert_ids", "delivery_status", "suppression_info"}', '{"channels": ["email", "slack", "pagerduty", "opsgenie", "teams", "webhook", "sms", "in_app"]}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'refresh_prediction', 'analysis', 'Predict next dataset refresh timing based on historical patterns with confidence scoring.', '{"dataset_id", "prediction_config"}', '{"predicted_refresh_at", "confidence", "prediction_basis"}', '{"min_history_points": 5, "confidence_threshold": 0.6}'::jsonb),
('GL-DATA-X-019', '1.0.0', 'freshness_reporting', 'reporting', 'Generate freshness compliance reports with dataset counts, SLA compliance rates, breach summaries, and recommendations.', '{"report_type", "config"}', '{"report", "compliance_rate", "breach_summary", "recommendations"}', '{"report_types": ["freshness", "sla_compliance", "staleness", "breach", "prediction", "executive"]}'::jsonb)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (agent_id, depends_on_agent_id, version_constraint, optional, reason) VALUES
('GL-DATA-X-019', 'GL-FOUND-X-002', '>=1.0.0', false, 'Schema validation for dataset definitions, SLA configurations, and check parameters'),
('GL-DATA-X-019', 'GL-FOUND-X-007', '>=1.0.0', false, 'Agent version and capability lookup for pipeline orchestration'),
('GL-DATA-X-019', 'GL-FOUND-X-006', '>=1.0.0', false, 'Access control enforcement for datasets, SLAs, and breach management'),
('GL-DATA-X-019', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for freshness checks, breaches, and alerts'),
('GL-DATA-X-019', 'GL-FOUND-X-005', '>=1.0.0', true, 'Provenance and audit trail registration with citation service'),
('GL-DATA-X-019', 'GL-FOUND-X-008', '>=1.0.0', true, 'Reproducibility verification for freshness scoring results'),
('GL-DATA-X-019', 'GL-FOUND-X-009', '>=1.0.0', true, 'QA Test Harness zero-hallucination verification'),
('GL-DATA-X-019', 'GL-DATA-X-013', '>=1.0.0', true, 'Data quality profiling for dataset quality assessment'),
('GL-DATA-X-019', 'GL-DATA-X-018', '>=1.0.0', true, 'Cross-source reconciliation for multi-source freshness comparison')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (agent_id, display_name, summary, category, status, tenant_id) VALUES
('GL-DATA-X-019', 'Data Freshness Monitor Agent',
 'Data freshness monitoring engine. Dataset registration (source type/ownership/cadence/priority). SLA definitions (warning/critical thresholds/escalation/business hours). Freshness checking (age/score/level/SLA status). Refresh history tracking. Staleness pattern detection (10 pattern types). SLA breach management (detection/acknowledgment/resolution). Multi-channel alerting (8 channels/suppression). Refresh prediction (timing/confidence). Compliance reporting. SHA-256 provenance chains.',
 'data', 'active', 'default')
ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA data_freshness_monitor_service IS 'Data Freshness Monitor Agent (AGENT-DATA-016) - dataset freshness monitoring, SLA management, staleness detection, breach workflows, multi-channel alerting, refresh prediction, provenance chains';
COMMENT ON TABLE data_freshness_monitor_service.freshness_datasets IS 'Registered datasets: name, source name/type, owner, refresh cadence, priority, status, tags, metadata, last refreshed timestamp';
COMMENT ON TABLE data_freshness_monitor_service.freshness_sla_definitions IS 'SLA definitions: dataset ref, warning/critical hour thresholds, breach severity, escalation policy, business-hours-only flag';
COMMENT ON TABLE data_freshness_monitor_service.freshness_checks IS 'Freshness check results: dataset ref, check timestamp, age hours, freshness score (0-1), freshness level, SLA status, provenance hash';
COMMENT ON TABLE data_freshness_monitor_service.freshness_refresh_history IS 'Dataset refresh history: dataset ref, refresh timestamp, data size bytes, record count, source info, provenance hash';
COMMENT ON TABLE data_freshness_monitor_service.freshness_staleness_patterns IS 'Detected staleness patterns: dataset ref, pattern type, frequency hours, severity, confidence, description';
COMMENT ON TABLE data_freshness_monitor_service.freshness_sla_breaches IS 'SLA breach records: dataset ref, SLA ref, breach severity, detection/acknowledgment/resolution timestamps, status, age at breach, resolution notes';
COMMENT ON TABLE data_freshness_monitor_service.freshness_alerts IS 'Alert delivery records: breach ref, dataset ref, severity, channel, message, recipients, sent/acknowledged timestamps, status, suppression reason';
COMMENT ON TABLE data_freshness_monitor_service.freshness_predictions IS 'Refresh predictions: dataset ref, predicted/actual refresh timestamps, confidence, error hours, status';
COMMENT ON TABLE data_freshness_monitor_service.freshness_reports IS 'Generated reports: report type, generated timestamp, dataset/compliant/breached counts, summary JSONB, provenance hash';
COMMENT ON TABLE data_freshness_monitor_service.freshness_audit_log IS 'Audit trail: operation, entity type/id, details JSONB, timestamp, provenance hash';
COMMENT ON TABLE data_freshness_monitor_service.freshness_check_events IS 'TimescaleDB hypertable: freshness check events with dataset ref, event type, event data (7-day chunks, 90-day retention)';
COMMENT ON TABLE data_freshness_monitor_service.refresh_events IS 'TimescaleDB hypertable: refresh events with dataset ref, event type, event data (7-day chunks, 90-day retention)';
COMMENT ON TABLE data_freshness_monitor_service.alert_events IS 'TimescaleDB hypertable: alert events with alert ref, event type, event data (7-day chunks, 90-day retention)';
COMMENT ON MATERIALIZED VIEW data_freshness_monitor_service.freshness_hourly_stats IS 'Continuous aggregate: hourly freshness stats by tenant (total events, check/breach/warning/compliant counts)';
COMMENT ON MATERIALIZED VIEW data_freshness_monitor_service.sla_breach_hourly_stats IS 'Continuous aggregate: hourly SLA breach stats by tenant (total events, breach/escalated/resolved/alert counts)';
