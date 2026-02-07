-- =============================================================================
-- GreenLang Climate OS - Alerting Service Schema
-- =============================================================================
-- Migration: V019
-- Component: OBS-004 Unified Alerting & Notification Platform
-- Description: Creates alerting schema with alert tracking, notification log,
--              escalation history, on-call cache, and MTTA/MTTR analytics.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS alerting;

-- =============================================================================
-- Table: alerting.alerts
-- =============================================================================
-- Core alert tracking table. Each row represents a unique alert instance
-- identified by its fingerprint. Tracks full lifecycle from firing through
-- acknowledgment to resolution.

CREATE TABLE alerting.alerts (
    alert_id UUID PRIMARY KEY,
    fingerprint VARCHAR(64) NOT NULL,
    source VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'firing',
    title TEXT NOT NULL,
    description TEXT,
    labels JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    tenant_id VARCHAR(50),
    team VARCHAR(100),
    service VARCHAR(100),
    environment VARCHAR(20),
    fired_at TIMESTAMPTZ NOT NULL,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    escalation_level INTEGER DEFAULT 0,
    notification_count INTEGER DEFAULT 0,
    runbook_url TEXT,
    dashboard_url TEXT,
    related_trace_id VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Status constraint
ALTER TABLE alerting.alerts
    ADD CONSTRAINT chk_alert_status
    CHECK (status IN ('firing', 'acknowledged', 'resolved', 'silenced', 'suppressed'));

-- Severity constraint
ALTER TABLE alerting.alerts
    ADD CONSTRAINT chk_alert_severity
    CHECK (severity IN ('critical', 'warning', 'info', 'page'));

-- =============================================================================
-- Table: alerting.notification_log
-- =============================================================================
-- TimescaleDB hypertable for time-series notification delivery analytics.
-- Tracks every notification attempt with channel, status, latency, and errors.

CREATE TABLE alerting.notification_log (
    id BIGSERIAL,
    alert_id UUID REFERENCES alerting.alerts(alert_id),
    channel VARCHAR(30) NOT NULL,
    status VARCHAR(20) NOT NULL,
    recipient VARCHAR(255),
    duration_ms INTEGER,
    response_code INTEGER,
    error_message TEXT,
    sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('alerting.notification_log', 'sent_at');

-- Channel constraint
ALTER TABLE alerting.notification_log
    ADD CONSTRAINT chk_notification_channel
    CHECK (channel IN ('pagerduty', 'opsgenie', 'slack', 'email', 'webhook', 'teams', 'sns'));

-- Status constraint
ALTER TABLE alerting.notification_log
    ADD CONSTRAINT chk_notification_status
    CHECK (status IN ('sent', 'delivered', 'failed', 'rate_limited', 'suppressed'));

-- =============================================================================
-- Table: alerting.escalation_log
-- =============================================================================
-- Tracks escalation events when alerts are escalated from one level to another.

CREATE TABLE alerting.escalation_log (
    id BIGSERIAL,
    alert_id UUID REFERENCES alerting.alerts(alert_id),
    from_level INTEGER NOT NULL,
    to_level INTEGER NOT NULL,
    reason VARCHAR(100),
    escalated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Table: alerting.oncall_cache
-- =============================================================================
-- Caches on-call schedule lookups from PagerDuty/Opsgenie to reduce API calls.
-- Entries are refreshed on a schedule; valid_until tracks cache freshness.

CREATE TABLE alerting.oncall_cache (
    schedule_id VARCHAR(100) PRIMARY KEY,
    provider VARCHAR(20) NOT NULL,
    oncall_user_id VARCHAR(100),
    oncall_user_name VARCHAR(255),
    oncall_user_email VARCHAR(255),
    valid_until TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Provider constraint
ALTER TABLE alerting.oncall_cache
    ADD CONSTRAINT chk_oncall_provider
    CHECK (provider IN ('pagerduty', 'opsgenie', 'manual'));

-- =============================================================================
-- Table: alerting.routing_rules
-- =============================================================================
-- Configurable routing rules that map alert labels to notification channels.
-- Rules are evaluated in priority order; first match wins.

CREATE TABLE alerting.routing_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    match_labels JSONB NOT NULL DEFAULT '{}',
    match_severity VARCHAR(20),
    match_team VARCHAR(100),
    channels TEXT[] NOT NULL,
    escalation_policy_id UUID,
    silence_period_minutes INTEGER DEFAULT 0,
    group_by TEXT[] DEFAULT ARRAY['alertname', 'team'],
    group_wait_seconds INTEGER DEFAULT 30,
    group_interval_seconds INTEGER DEFAULT 300,
    repeat_interval_seconds INTEGER DEFAULT 3600,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Table: alerting.escalation_policies
-- =============================================================================
-- Defines escalation policies with multiple levels. Each level specifies the
-- channel, wait time, and optional on-call schedule.

CREATE TABLE alerting.escalation_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    levels JSONB NOT NULL DEFAULT '[]',
    repeat_count INTEGER DEFAULT 0,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Continuous Aggregate: alerting.alert_response_metrics_hourly
-- =============================================================================
-- Precomputed MTTA/MTTR metrics per hour, team, and severity.
-- Used by the alerting dashboard for response time trend analysis.

CREATE MATERIALIZED VIEW alerting.alert_response_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', fired_at) AS bucket,
    team,
    severity,
    COUNT(*) AS alert_count,
    AVG(EXTRACT(EPOCH FROM (acknowledged_at - fired_at))) AS avg_mtta_seconds,
    AVG(EXTRACT(EPOCH FROM (resolved_at - fired_at))) AS avg_mttr_seconds,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (acknowledged_at - fired_at))) AS p95_mtta,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (resolved_at - fired_at))) AS p95_mttr
FROM alerting.alerts
WHERE fired_at IS NOT NULL
GROUP BY bucket, team, severity
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('alerting.alert_response_metrics_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- Alerts indexes
CREATE INDEX idx_alerts_status ON alerting.alerts(status);
CREATE INDEX idx_alerts_severity ON alerting.alerts(severity);
CREATE INDEX idx_alerts_team ON alerting.alerts(team);
CREATE INDEX idx_alerts_fingerprint ON alerting.alerts(fingerprint);
CREATE INDEX idx_alerts_fired_at ON alerting.alerts(fired_at DESC);
CREATE INDEX idx_alerts_tenant ON alerting.alerts(tenant_id);
CREATE INDEX idx_alerts_service ON alerting.alerts(service);
CREATE INDEX idx_alerts_environment ON alerting.alerts(environment);
CREATE INDEX idx_alerts_status_severity ON alerting.alerts(status, severity);
CREATE INDEX idx_alerts_labels ON alerting.alerts USING GIN (labels);
CREATE INDEX idx_alerts_annotations ON alerting.alerts USING GIN (annotations);

-- Notification log indexes (hypertable-aware)
CREATE INDEX idx_notification_log_alert ON alerting.notification_log(alert_id);
CREATE INDEX idx_notification_log_channel ON alerting.notification_log(channel);
CREATE INDEX idx_notification_log_status ON alerting.notification_log(status);

-- Escalation log indexes
CREATE INDEX idx_escalation_log_alert ON alerting.escalation_log(alert_id);

-- Routing rules indexes
CREATE INDEX idx_routing_rules_priority ON alerting.routing_rules(priority);
CREATE INDEX idx_routing_rules_enabled ON alerting.routing_rules(enabled) WHERE enabled = true;

-- On-call cache indexes
CREATE INDEX idx_oncall_cache_valid ON alerting.oncall_cache(valid_until);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE alerting.alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY alerting_tenant_isolation ON alerting.alerts
    USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
    );

ALTER TABLE alerting.notification_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY notification_log_tenant_isolation ON alerting.notification_log
    USING (
        alert_id IN (
            SELECT alert_id FROM alerting.alerts
            WHERE tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
        )
    );

ALTER TABLE alerting.escalation_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_log_tenant_isolation ON alerting.escalation_log
    USING (
        alert_id IN (
            SELECT alert_id FROM alerting.alerts
            WHERE tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
        )
    );

ALTER TABLE alerting.routing_rules ENABLE ROW LEVEL SECURITY;
CREATE POLICY routing_rules_global_access ON alerting.routing_rules
    USING (true);

ALTER TABLE alerting.escalation_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY escalation_policies_global_access ON alerting.escalation_policies
    USING (true);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA alerting TO greenlang_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA alerting TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA alerting TO greenlang_app;

-- Add alerting permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'alerting:alerts:read', 'alerting', 'read', 'View alerts'),
    (gen_random_uuid(), 'alerting:alerts:write', 'alerting', 'write', 'Create/update alerts'),
    (gen_random_uuid(), 'alerting:alerts:acknowledge', 'alerting', 'acknowledge', 'Acknowledge alerts'),
    (gen_random_uuid(), 'alerting:alerts:resolve', 'alerting', 'resolve', 'Resolve alerts'),
    (gen_random_uuid(), 'alerting:alerts:escalate', 'alerting', 'escalate', 'Escalate alerts'),
    (gen_random_uuid(), 'alerting:analytics:read', 'alerting', 'analytics', 'View alert analytics'),
    (gen_random_uuid(), 'alerting:oncall:read', 'alerting', 'oncall_read', 'View on-call schedules'),
    (gen_random_uuid(), 'alerting:admin', 'alerting', 'admin', 'Alert administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Trigger: auto-update updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION alerting.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_alerts_updated_at
    BEFORE UPDATE ON alerting.alerts
    FOR EACH ROW
    EXECUTE FUNCTION alerting.update_updated_at();

CREATE TRIGGER trg_routing_rules_updated_at
    BEFORE UPDATE ON alerting.routing_rules
    FOR EACH ROW
    EXECUTE FUNCTION alerting.update_updated_at();

CREATE TRIGGER trg_escalation_policies_updated_at
    BEFORE UPDATE ON alerting.escalation_policies
    FOR EACH ROW
    EXECUTE FUNCTION alerting.update_updated_at();

-- =============================================================================
-- Retention policy for notification_log (90 days)
-- =============================================================================

SELECT add_retention_policy('alerting.notification_log', INTERVAL '90 days');

-- =============================================================================
-- Seed: Default escalation policy
-- =============================================================================

INSERT INTO alerting.escalation_policies (name, description, levels) VALUES
    ('default-critical', 'Default critical alert escalation', '[
        {"level": 0, "channels": ["slack"], "wait_minutes": 0},
        {"level": 1, "channels": ["pagerduty"], "wait_minutes": 5},
        {"level": 2, "channels": ["pagerduty", "opsgenie"], "wait_minutes": 15},
        {"level": 3, "channels": ["pagerduty", "opsgenie", "email"], "wait_minutes": 30}
    ]'),
    ('default-warning', 'Default warning alert escalation', '[
        {"level": 0, "channels": ["slack"], "wait_minutes": 0},
        {"level": 1, "channels": ["slack", "email"], "wait_minutes": 30},
        {"level": 2, "channels": ["pagerduty"], "wait_minutes": 60}
    ]'),
    ('default-info', 'Default info alert routing (no escalation)', '[
        {"level": 0, "channels": ["slack"], "wait_minutes": 0}
    ]');
