-- =============================================================================
-- GreenLang Climate OS - Access & Policy Guard Agent Service Schema
-- =============================================================================
-- Migration: V026
-- Component: AGENT-FOUND-006 Access & Policy Guard Agent
-- Description: Creates access_guard_service schema with policy definitions,
--              policy rules, access decisions (hypertable), audit events
--              (hypertable), rate limit state, data classifications, OPA Rego
--              policy storage, compliance reports, continuous aggregates for
--              hourly decision counts and daily audit counts, 30+ indexes,
--              RLS policies per tenant, 14 security permissions, retention
--              policies (90-day decisions, 365-day audit), compression,
--              and seed data for default policies, rate limits, and
--              classification patterns.
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS access_guard_service;

-- =============================================================================
-- Table: access_guard_service.policies
-- =============================================================================
-- Top-level policy definitions that group and organize access control rules.
-- Each policy has a unique ID, name, version, parent for hierarchical
-- inheritance, tenant scoping, provenance hash, and soft-delete support.
-- The applies_to array specifies which resource types the policy covers.

CREATE TABLE access_guard_service.policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    parent_policy_id UUID,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    applies_to TEXT[] DEFAULT '{}',
    allow_override BOOLEAN NOT NULL DEFAULT FALSE,
    provenance_hash CHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

-- Self-referencing FK for policy hierarchy
ALTER TABLE access_guard_service.policies
    ADD CONSTRAINT fk_policy_parent
    FOREIGN KEY (parent_policy_id)
    REFERENCES access_guard_service.policies(policy_id)
    ON DELETE SET NULL;

-- Provenance hash must be 64-character hex when present
ALTER TABLE access_guard_service.policies
    ADD CONSTRAINT chk_policy_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- Version must be positive
ALTER TABLE access_guard_service.policies
    ADD CONSTRAINT chk_policy_version_positive
    CHECK (version > 0);

-- =============================================================================
-- Table: access_guard_service.policy_rules
-- =============================================================================
-- Individual rules within a policy. Each rule specifies an effect (allow/deny),
-- the actions it covers, the resources it applies to, the principals it
-- targets, optional conditions (JSONB), time constraints, geographic
-- constraints, and classification level limits. Priority determines
-- evaluation order (lower = higher priority). Tags for organization.

CREATE TABLE access_guard_service.policy_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID NOT NULL REFERENCES access_guard_service.policies(policy_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT DEFAULT '',
    policy_type VARCHAR(50) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    effect VARCHAR(10) NOT NULL,
    actions TEXT[] DEFAULT '{}',
    resources TEXT[] DEFAULT '{}',
    principals TEXT[] DEFAULT '{}',
    conditions JSONB DEFAULT '{}'::jsonb,
    time_constraints JSONB DEFAULT '{}'::jsonb,
    geographic_constraints TEXT[] DEFAULT '{}',
    classification_max VARCHAR(30),
    version INTEGER NOT NULL DEFAULT 1,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Policy type constraint
ALTER TABLE access_guard_service.policy_rules
    ADD CONSTRAINT chk_rule_policy_type
    CHECK (policy_type IN (
        'identity', 'resource', 'organization', 'network',
        'time_based', 'classification', 'custom', 'composite'
    ));

-- Effect constraint (deny-wins model)
ALTER TABLE access_guard_service.policy_rules
    ADD CONSTRAINT chk_rule_effect
    CHECK (effect IN ('allow', 'deny'));

-- Classification max constraint
ALTER TABLE access_guard_service.policy_rules
    ADD CONSTRAINT chk_rule_classification_max
    CHECK (classification_max IS NULL OR classification_max IN (
        'public', 'internal', 'confidential', 'restricted', 'top_secret'
    ));

-- Priority must be positive
ALTER TABLE access_guard_service.policy_rules
    ADD CONSTRAINT chk_rule_priority_positive
    CHECK (priority > 0);

-- Version must be positive
ALTER TABLE access_guard_service.policy_rules
    ADD CONSTRAINT chk_rule_version_positive
    CHECK (version > 0);

-- =============================================================================
-- Table: access_guard_service.access_decisions
-- =============================================================================
-- TimescaleDB hypertable recording every access decision made by the guard
-- agent. Each row captures who (principal) tried to do what (action) on
-- which resource, the decision (allow/deny), matching rules, deny reasons,
-- evaluation time in milliseconds, and a SHA-256 decision hash for
-- tamper-evident audit. Partitioned by timestamp for time-series queries.
-- Retained for 90 days with compression after 7 days.

CREATE TABLE access_guard_service.access_decisions (
    decision_id UUID DEFAULT gen_random_uuid(),
    request_id VARCHAR(255) NOT NULL,
    principal_id VARCHAR(255) NOT NULL,
    principal_type VARCHAR(50) NOT NULL,
    principal_tenant VARCHAR(100) NOT NULL DEFAULT 'default',
    resource_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_tenant VARCHAR(100) NOT NULL DEFAULT 'default',
    action VARCHAR(100) NOT NULL,
    decision VARCHAR(20) NOT NULL,
    allowed BOOLEAN NOT NULL,
    matching_rules TEXT[] DEFAULT '{}',
    deny_reasons TEXT[] DEFAULT '{}',
    evaluation_time_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    decision_hash CHAR(64),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    PRIMARY KEY (decision_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('access_guard_service.access_decisions', 'timestamp', if_not_exists => TRUE);

-- Principal type constraint
ALTER TABLE access_guard_service.access_decisions
    ADD CONSTRAINT chk_decision_principal_type
    CHECK (principal_type IN (
        'user', 'service', 'agent', 'api_key', 'system', 'anonymous'
    ));

-- Decision constraint
ALTER TABLE access_guard_service.access_decisions
    ADD CONSTRAINT chk_decision_result
    CHECK (decision IN (
        'allow', 'deny', 'rate_limited', 'challenge', 'defer'
    ));

-- Decision hash must be 64-character hex when present
ALTER TABLE access_guard_service.access_decisions
    ADD CONSTRAINT chk_decision_hash_length
    CHECK (decision_hash IS NULL OR LENGTH(decision_hash) = 64);

-- Evaluation time must be non-negative
ALTER TABLE access_guard_service.access_decisions
    ADD CONSTRAINT chk_decision_eval_time_positive
    CHECK (evaluation_time_ms >= 0);

-- =============================================================================
-- Table: access_guard_service.audit_events
-- =============================================================================
-- TimescaleDB hypertable recording comprehensive audit events for all
-- access control operations. Each event captures the type, principal,
-- resource, action, decision, details (JSONB), source IP, user agent,
-- and configurable retention. Partitioned by timestamp for time-series
-- queries. Retained for 365 days with compression after 30 days.

CREATE TABLE access_guard_service.audit_events (
    event_id UUID DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    principal_id VARCHAR(255),
    resource_id VARCHAR(255),
    action VARCHAR(100),
    decision VARCHAR(20),
    decision_hash CHAR(64),
    details JSONB DEFAULT '{}'::jsonb,
    source_ip VARCHAR(45),
    user_agent TEXT,
    retention_days INTEGER NOT NULL DEFAULT 365,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_id, timestamp)
);

-- Create hypertable partitioned by timestamp
SELECT create_hypertable('access_guard_service.audit_events', 'timestamp', if_not_exists => TRUE);

-- Event type constraint
ALTER TABLE access_guard_service.audit_events
    ADD CONSTRAINT chk_audit_event_type
    CHECK (event_type IN (
        'access_request', 'access_granted', 'access_denied',
        'policy_created', 'policy_updated', 'policy_deleted',
        'rule_created', 'rule_updated', 'rule_deleted',
        'rate_limit_hit', 'rate_limit_reset',
        'classification_set', 'classification_changed',
        'rego_compiled', 'rego_evaluation_failed',
        'tenant_violation', 'cross_tenant_attempt',
        'compliance_report_generated', 'admin_action',
        'simulation_run', 'cache_invalidated'
    ));

-- Decision hash must be 64-character hex when present
ALTER TABLE access_guard_service.audit_events
    ADD CONSTRAINT chk_audit_decision_hash_length
    CHECK (decision_hash IS NULL OR LENGTH(decision_hash) = 64);

-- Retention days must be positive
ALTER TABLE access_guard_service.audit_events
    ADD CONSTRAINT chk_audit_retention_positive
    CHECK (retention_days > 0);

-- =============================================================================
-- Table: access_guard_service.rate_limit_state
-- =============================================================================
-- Rate limit counters per principal per tenant. Tracks requests per minute,
-- per hour, and per day with window start timestamps. Used by the rate
-- limiting middleware to enforce per-role and per-tenant request quotas.

CREATE TABLE access_guard_service.rate_limit_state (
    bucket_key VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    principal_id VARCHAR(255) NOT NULL,
    requests_minute INTEGER NOT NULL DEFAULT 0,
    requests_hour INTEGER NOT NULL DEFAULT 0,
    requests_day INTEGER NOT NULL DEFAULT 0,
    minute_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hour_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    day_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Request counts must be non-negative
ALTER TABLE access_guard_service.rate_limit_state
    ADD CONSTRAINT chk_rate_requests_minute_positive
    CHECK (requests_minute >= 0);

ALTER TABLE access_guard_service.rate_limit_state
    ADD CONSTRAINT chk_rate_requests_hour_positive
    CHECK (requests_hour >= 0);

ALTER TABLE access_guard_service.rate_limit_state
    ADD CONSTRAINT chk_rate_requests_day_positive
    CHECK (requests_day >= 0);

-- =============================================================================
-- Table: access_guard_service.data_classifications
-- =============================================================================
-- Resource classification registry tracking the data sensitivity level of
-- each resource. Supports classification levels from public to top_secret.
-- Used by classification-based policy rules to restrict access based on
-- data sensitivity.

CREATE TABLE access_guard_service.data_classifications (
    resource_id VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    classification VARCHAR(30) NOT NULL,
    classified_by VARCHAR(255),
    classified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    reason TEXT DEFAULT '',
    PRIMARY KEY (resource_id, resource_type, tenant_id)
);

-- Classification level constraint
ALTER TABLE access_guard_service.data_classifications
    ADD CONSTRAINT chk_classification_level
    CHECK (classification IN (
        'public', 'internal', 'confidential', 'restricted', 'top_secret'
    ));

-- =============================================================================
-- Table: access_guard_service.rego_policies
-- =============================================================================
-- OPA Rego policy storage. Each row stores the Rego source code for a
-- policy module, its SHA-256 hash for integrity verification, version
-- number, and tenant scoping. Used by the OPA evaluation engine for
-- complex policy evaluation beyond simple rule matching.

CREATE TABLE access_guard_service.rego_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    rego_source TEXT NOT NULL,
    source_hash VARCHAR(64) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default'
);

-- Source hash must be 64-character hex
ALTER TABLE access_guard_service.rego_policies
    ADD CONSTRAINT chk_rego_source_hash_length
    CHECK (LENGTH(source_hash) = 64);

-- Version must be positive
ALTER TABLE access_guard_service.rego_policies
    ADD CONSTRAINT chk_rego_version_positive
    CHECK (version > 0);

-- =============================================================================
-- Table: access_guard_service.compliance_reports
-- =============================================================================
-- Generated compliance reports summarizing access control activity over a
-- specified period. Each report includes total requests, allow/deny/rate
-- limited counts, decisions by type, top denial reasons, evaluated policies,
-- access by classification level, and a provenance hash.

CREATE TABLE access_guard_service.compliance_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    total_requests BIGINT NOT NULL DEFAULT 0,
    allowed BIGINT NOT NULL DEFAULT 0,
    denied BIGINT NOT NULL DEFAULT 0,
    rate_limited BIGINT NOT NULL DEFAULT 0,
    decisions_by_type JSONB DEFAULT '{}'::jsonb,
    top_denial_reasons JSONB DEFAULT '[]'::jsonb,
    policies_evaluated TEXT[] DEFAULT '{}',
    access_by_classification JSONB DEFAULT '{}'::jsonb,
    provenance_hash CHAR(64),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    generated_by VARCHAR(255)
);

-- Period end must be after period start
ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_period_range
    CHECK (period_end > period_start);

-- Provenance hash must be 64-character hex when present
ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_provenance_hash_length
    CHECK (provenance_hash IS NULL OR LENGTH(provenance_hash) = 64);

-- Request counts must be non-negative
ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_total_positive
    CHECK (total_requests >= 0);

ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_allowed_positive
    CHECK (allowed >= 0);

ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_denied_positive
    CHECK (denied >= 0);

ALTER TABLE access_guard_service.compliance_reports
    ADD CONSTRAINT chk_report_rate_limited_positive
    CHECK (rate_limited >= 0);

-- =============================================================================
-- Continuous Aggregate: access_guard_service.hourly_decision_counts
-- =============================================================================
-- Precomputed hourly counts of access decisions by decision type for
-- dashboard queries, trend analysis, and real-time monitoring of
-- allow/deny/rate_limited ratios.

CREATE MATERIALIZED VIEW access_guard_service.hourly_decision_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    decision,
    tenant_id,
    COUNT(*) AS total_decisions,
    COUNT(DISTINCT principal_id) AS unique_principals,
    COUNT(DISTINCT resource_id) AS unique_resources,
    AVG(evaluation_time_ms) AS avg_evaluation_ms,
    MAX(evaluation_time_ms) AS max_evaluation_ms
FROM access_guard_service.access_decisions
WHERE timestamp IS NOT NULL
GROUP BY bucket, decision, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 15 minutes, covering the last 2 hours
SELECT add_continuous_aggregate_policy('access_guard_service.hourly_decision_counts',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- =============================================================================
-- Continuous Aggregate: access_guard_service.daily_audit_counts
-- =============================================================================
-- Precomputed daily counts of audit events by event type for compliance
-- reporting, dashboard queries, and long-term trend analysis.

CREATE MATERIALIZED VIEW access_guard_service.daily_audit_counts
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    event_type,
    tenant_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT principal_id) AS unique_principals,
    COUNT(DISTINCT resource_id) AS unique_resources
FROM access_guard_service.audit_events
WHERE timestamp IS NOT NULL
GROUP BY bucket, event_type, tenant_id
WITH NO DATA;

-- Refresh policy: refresh every 30 minutes, covering the last 2 days
SELECT add_continuous_aggregate_policy('access_guard_service.daily_audit_counts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- policies indexes
CREATE INDEX idx_policies_name ON access_guard_service.policies(name);
CREATE INDEX idx_policies_tenant ON access_guard_service.policies(tenant_id);
CREATE INDEX idx_policies_enabled ON access_guard_service.policies(enabled, tenant_id);
CREATE INDEX idx_policies_parent ON access_guard_service.policies(parent_policy_id);
CREATE INDEX idx_policies_version ON access_guard_service.policies(version);
CREATE INDEX idx_policies_created_at ON access_guard_service.policies(created_at DESC);
CREATE INDEX idx_policies_updated_at ON access_guard_service.policies(updated_at DESC);
CREATE INDEX idx_policies_created_by ON access_guard_service.policies(created_by);
CREATE INDEX idx_policies_is_deleted ON access_guard_service.policies(is_deleted);
CREATE INDEX idx_policies_provenance ON access_guard_service.policies(provenance_hash);
CREATE INDEX idx_policies_applies_to ON access_guard_service.policies USING GIN (applies_to);

-- policy_rules indexes
CREATE INDEX idx_rules_policy ON access_guard_service.policy_rules(policy_id);
CREATE INDEX idx_rules_name ON access_guard_service.policy_rules(name);
CREATE INDEX idx_rules_type ON access_guard_service.policy_rules(policy_type);
CREATE INDEX idx_rules_priority ON access_guard_service.policy_rules(priority);
CREATE INDEX idx_rules_enabled ON access_guard_service.policy_rules(enabled, policy_id);
CREATE INDEX idx_rules_effect ON access_guard_service.policy_rules(effect);
CREATE INDEX idx_rules_classification ON access_guard_service.policy_rules(classification_max);
CREATE INDEX idx_rules_created_at ON access_guard_service.policy_rules(created_at DESC);
CREATE INDEX idx_rules_created_by ON access_guard_service.policy_rules(created_by);
CREATE INDEX idx_rules_actions ON access_guard_service.policy_rules USING GIN (actions);
CREATE INDEX idx_rules_resources ON access_guard_service.policy_rules USING GIN (resources);
CREATE INDEX idx_rules_principals ON access_guard_service.policy_rules USING GIN (principals);
CREATE INDEX idx_rules_conditions ON access_guard_service.policy_rules USING GIN (conditions);
CREATE INDEX idx_rules_tags ON access_guard_service.policy_rules USING GIN (tags);

-- access_decisions indexes (hypertable-aware)
CREATE INDEX idx_decisions_request ON access_guard_service.access_decisions(request_id, timestamp DESC);
CREATE INDEX idx_decisions_principal ON access_guard_service.access_decisions(principal_id, timestamp DESC);
CREATE INDEX idx_decisions_principal_type ON access_guard_service.access_decisions(principal_type, timestamp DESC);
CREATE INDEX idx_decisions_resource ON access_guard_service.access_decisions(resource_id, timestamp DESC);
CREATE INDEX idx_decisions_resource_type ON access_guard_service.access_decisions(resource_type, timestamp DESC);
CREATE INDEX idx_decisions_action ON access_guard_service.access_decisions(action, timestamp DESC);
CREATE INDEX idx_decisions_decision ON access_guard_service.access_decisions(decision, timestamp DESC);
CREATE INDEX idx_decisions_allowed ON access_guard_service.access_decisions(allowed, timestamp DESC);
CREATE INDEX idx_decisions_tenant ON access_guard_service.access_decisions(tenant_id, timestamp DESC);
CREATE INDEX idx_decisions_principal_tenant ON access_guard_service.access_decisions(principal_tenant, timestamp DESC);
CREATE INDEX idx_decisions_resource_tenant ON access_guard_service.access_decisions(resource_tenant, timestamp DESC);
CREATE INDEX idx_decisions_hash ON access_guard_service.access_decisions(decision_hash);
CREATE INDEX idx_decisions_eval_time ON access_guard_service.access_decisions(evaluation_time_ms, timestamp DESC);
CREATE INDEX idx_decisions_matching_rules ON access_guard_service.access_decisions USING GIN (matching_rules);
CREATE INDEX idx_decisions_deny_reasons ON access_guard_service.access_decisions USING GIN (deny_reasons);

-- audit_events indexes (hypertable-aware)
CREATE INDEX idx_audit_event_type ON access_guard_service.audit_events(event_type, timestamp DESC);
CREATE INDEX idx_audit_tenant ON access_guard_service.audit_events(tenant_id, timestamp DESC);
CREATE INDEX idx_audit_principal ON access_guard_service.audit_events(principal_id, timestamp DESC);
CREATE INDEX idx_audit_resource ON access_guard_service.audit_events(resource_id, timestamp DESC);
CREATE INDEX idx_audit_action ON access_guard_service.audit_events(action, timestamp DESC);
CREATE INDEX idx_audit_decision ON access_guard_service.audit_events(decision, timestamp DESC);
CREATE INDEX idx_audit_decision_hash ON access_guard_service.audit_events(decision_hash);
CREATE INDEX idx_audit_source_ip ON access_guard_service.audit_events(source_ip, timestamp DESC);
CREATE INDEX idx_audit_retention ON access_guard_service.audit_events(retention_days);
CREATE INDEX idx_audit_details ON access_guard_service.audit_events USING GIN (details);

-- rate_limit_state indexes
CREATE INDEX idx_rate_tenant ON access_guard_service.rate_limit_state(tenant_id);
CREATE INDEX idx_rate_principal ON access_guard_service.rate_limit_state(principal_id);
CREATE INDEX idx_rate_updated ON access_guard_service.rate_limit_state(updated_at DESC);

-- data_classifications indexes
CREATE INDEX idx_class_resource ON access_guard_service.data_classifications(resource_id);
CREATE INDEX idx_class_resource_type ON access_guard_service.data_classifications(resource_type);
CREATE INDEX idx_class_classification ON access_guard_service.data_classifications(classification);
CREATE INDEX idx_class_tenant ON access_guard_service.data_classifications(tenant_id);
CREATE INDEX idx_class_classified_by ON access_guard_service.data_classifications(classified_by);
CREATE INDEX idx_class_classified_at ON access_guard_service.data_classifications(classified_at DESC);

-- rego_policies indexes
CREATE INDEX idx_rego_enabled ON access_guard_service.rego_policies(enabled);
CREATE INDEX idx_rego_tenant ON access_guard_service.rego_policies(tenant_id);
CREATE INDEX idx_rego_source_hash ON access_guard_service.rego_policies(source_hash);
CREATE INDEX idx_rego_created_at ON access_guard_service.rego_policies(created_at DESC);
CREATE INDEX idx_rego_created_by ON access_guard_service.rego_policies(created_by);

-- compliance_reports indexes
CREATE INDEX idx_reports_tenant ON access_guard_service.compliance_reports(tenant_id);
CREATE INDEX idx_reports_period ON access_guard_service.compliance_reports(period_start, period_end);
CREATE INDEX idx_reports_generated_at ON access_guard_service.compliance_reports(generated_at DESC);
CREATE INDEX idx_reports_generated_by ON access_guard_service.compliance_reports(generated_by);
CREATE INDEX idx_reports_provenance ON access_guard_service.compliance_reports(provenance_hash);
CREATE INDEX idx_reports_policies ON access_guard_service.compliance_reports USING GIN (policies_evaluated);
CREATE INDEX idx_reports_decisions ON access_guard_service.compliance_reports USING GIN (decisions_by_type);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

ALTER TABLE access_guard_service.policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY policies_tenant_read ON access_guard_service.policies
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY policies_tenant_write ON access_guard_service.policies
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.policy_rules ENABLE ROW LEVEL SECURITY;
-- Rules inherit tenant context via their parent policy
CREATE POLICY rules_read ON access_guard_service.policy_rules
    FOR SELECT USING (true);
CREATE POLICY rules_write ON access_guard_service.policy_rules
    FOR ALL USING (
        current_setting('app.is_admin', true) = 'true'
        OR current_setting('app.current_tenant', true) IS NULL
        OR EXISTS (
            SELECT 1 FROM access_guard_service.policies p
            WHERE p.policy_id = policy_rules.policy_id
            AND (
                p.tenant_id = current_setting('app.current_tenant', true)
                OR current_setting('app.current_tenant', true) IS NULL
            )
        )
    );

ALTER TABLE access_guard_service.access_decisions ENABLE ROW LEVEL SECURITY;
CREATE POLICY decisions_tenant_read ON access_guard_service.access_decisions
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY decisions_tenant_write ON access_guard_service.access_decisions
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.audit_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_tenant_read ON access_guard_service.audit_events
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY audit_tenant_write ON access_guard_service.audit_events
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.rate_limit_state ENABLE ROW LEVEL SECURITY;
CREATE POLICY rate_limit_tenant_read ON access_guard_service.rate_limit_state
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY rate_limit_tenant_write ON access_guard_service.rate_limit_state
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.data_classifications ENABLE ROW LEVEL SECURITY;
CREATE POLICY classifications_tenant_read ON access_guard_service.data_classifications
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY classifications_tenant_write ON access_guard_service.data_classifications
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.rego_policies ENABLE ROW LEVEL SECURITY;
CREATE POLICY rego_tenant_read ON access_guard_service.rego_policies
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY rego_tenant_write ON access_guard_service.rego_policies
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

ALTER TABLE access_guard_service.compliance_reports ENABLE ROW LEVEL SECURITY;
CREATE POLICY reports_tenant_read ON access_guard_service.compliance_reports
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );
CREATE POLICY reports_tenant_write ON access_guard_service.compliance_reports
    FOR ALL USING (
        tenant_id = current_setting('app.current_tenant', true)
        OR current_setting('app.current_tenant', true) IS NULL
        OR current_setting('app.is_admin', true) = 'true'
    );

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA access_guard_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA access_guard_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA access_guard_service TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON access_guard_service.hourly_decision_counts TO greenlang_app;
GRANT SELECT ON access_guard_service.daily_audit_counts TO greenlang_app;

-- Read-only role
GRANT USAGE ON SCHEMA access_guard_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA access_guard_service TO greenlang_readonly;
GRANT SELECT ON access_guard_service.hourly_decision_counts TO greenlang_readonly;
GRANT SELECT ON access_guard_service.daily_audit_counts TO greenlang_readonly;

-- Add access guard service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'access_guard:policies:read', 'access_guard', 'policies_read', 'View access policies and their rules'),
    (gen_random_uuid(), 'access_guard:policies:write', 'access_guard', 'policies_write', 'Create and update access policies'),
    (gen_random_uuid(), 'access_guard:policies:delete', 'access_guard', 'policies_delete', 'Delete access policies'),
    (gen_random_uuid(), 'access_guard:rules:read', 'access_guard', 'rules_read', 'View policy rules'),
    (gen_random_uuid(), 'access_guard:rules:write', 'access_guard', 'rules_write', 'Create and update policy rules'),
    (gen_random_uuid(), 'access_guard:decisions:read', 'access_guard', 'decisions_read', 'View access decision history'),
    (gen_random_uuid(), 'access_guard:audit:read', 'access_guard', 'audit_read', 'View access audit event log'),
    (gen_random_uuid(), 'access_guard:rate_limit:read', 'access_guard', 'rate_limit_read', 'View rate limit state and configuration'),
    (gen_random_uuid(), 'access_guard:rate_limit:write', 'access_guard', 'rate_limit_write', 'Manage rate limit configuration'),
    (gen_random_uuid(), 'access_guard:classifications:read', 'access_guard', 'classifications_read', 'View data classification registry'),
    (gen_random_uuid(), 'access_guard:classifications:write', 'access_guard', 'classifications_write', 'Manage data classifications'),
    (gen_random_uuid(), 'access_guard:rego:write', 'access_guard', 'rego_write', 'Manage OPA Rego policies'),
    (gen_random_uuid(), 'access_guard:reports:read', 'access_guard', 'reports_read', 'View compliance reports'),
    (gen_random_uuid(), 'access_guard:admin', 'access_guard', 'admin', 'Access Guard service administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep access decisions for 90 days
SELECT add_retention_policy('access_guard_service.access_decisions', INTERVAL '90 days');

-- Keep audit events for 365 days
SELECT add_retention_policy('access_guard_service.audit_events', INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on access_decisions after 7 days
ALTER TABLE access_guard_service.access_decisions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('access_guard_service.access_decisions', INTERVAL '7 days');

-- Enable compression on audit_events after 30 days
ALTER TABLE access_guard_service.audit_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tenant_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('access_guard_service.audit_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Default Policies
-- =============================================================================

INSERT INTO access_guard_service.policies (policy_id, name, description, version, enabled, tenant_id, applies_to, allow_override, provenance_hash, created_by) VALUES

-- Tenant Isolation Policy
('b0000001-0001-4000-8000-000000000001', 'Tenant Isolation Policy',
 'Enforces strict tenant isolation ensuring principals can only access resources within their own tenant. Cross-tenant access is denied by default unless explicitly granted by an admin override rule.',
 1, true, 'default', '{"*"}', false,
 'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2',
 'system'),

-- Classification Access Policy
('b0000001-0001-4000-8000-000000000002', 'Classification-Based Access Policy',
 'Restricts access to resources based on their data classification level. Ensures that principals can only access resources up to their authorized classification clearance (public < internal < confidential < restricted < top_secret).',
 1, true, 'default', '{"emissions", "calculations", "reports", "assumptions", "citations"}', false,
 'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3',
 'system'),

-- Default Viewer Policy
('b0000001-0001-4000-8000-000000000003', 'Default Viewer Policy',
 'Grants read-only access to public and internal resources for principals with the viewer role. Viewers can view emissions data, calculation results, and published reports but cannot modify or delete any data.',
 1, true, 'default', '{"emissions", "calculations", "reports"}', true,
 'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4',
 'system'),

-- Default Analyst Policy
('b0000001-0001-4000-8000-000000000004', 'Default Analyst Policy',
 'Grants read/write access to emissions data, calculations, assumptions, and citations for principals with the analyst role. Analysts can create and modify calculations but cannot delete data or manage policies.',
 1, true, 'default', '{"emissions", "calculations", "assumptions", "citations", "reports"}', true,
 'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5',
 'system'),

-- Default Admin Policy
('b0000001-0001-4000-8000-000000000005', 'Default Admin Policy',
 'Grants full administrative access to all resources for principals with the admin role. Admins can manage policies, rules, classifications, rate limits, Rego policies, and generate compliance reports.',
 1, true, 'default', '{"*"}', false,
 'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6',
 'system')

ON CONFLICT (policy_id) DO NOTHING;

-- =============================================================================
-- Seed: Default Policy Rules
-- =============================================================================

INSERT INTO access_guard_service.policy_rules (rule_id, policy_id, name, description, policy_type, priority, enabled, effect, actions, resources, principals, conditions, time_constraints, geographic_constraints, classification_max, version, tags, created_by) VALUES

-- Tenant Isolation: Deny cross-tenant access
('c0000001-0001-4000-8000-000000000001', 'b0000001-0001-4000-8000-000000000001',
 'deny-cross-tenant', 'Deny all cross-tenant resource access. Principal tenant must match resource tenant.',
 'organization', 1, true, 'deny', '{"*"}', '{"*"}', '{"*"}',
 '{"condition": "principal_tenant != resource_tenant"}'::jsonb,
 '{}'::jsonb, '{}', NULL, 1,
 '{"tenant-isolation", "security", "default"}', 'system'),

-- Tenant Isolation: Allow same-tenant access
('c0000001-0001-4000-8000-000000000002', 'b0000001-0001-4000-8000-000000000001',
 'allow-same-tenant', 'Allow access when principal and resource are in the same tenant.',
 'organization', 10, true, 'allow', '{"*"}', '{"*"}', '{"*"}',
 '{"condition": "principal_tenant == resource_tenant"}'::jsonb,
 '{}'::jsonb, '{}', NULL, 1,
 '{"tenant-isolation", "default"}', 'system'),

-- Classification: Deny access above clearance
('c0000001-0001-4000-8000-000000000003', 'b0000001-0001-4000-8000-000000000002',
 'deny-above-clearance', 'Deny access to resources classified above the principal clearance level.',
 'classification', 1, true, 'deny', '{"read", "write", "delete", "export"}', '{"*"}', '{"*"}',
 '{"condition": "resource_classification > principal_clearance"}'::jsonb,
 '{}'::jsonb, '{}', NULL, 1,
 '{"classification", "security", "default"}', 'system'),

-- Classification: Deny restricted/top_secret to non-admin
('c0000001-0001-4000-8000-000000000004', 'b0000001-0001-4000-8000-000000000002',
 'deny-restricted-non-admin', 'Deny access to restricted and top_secret resources for non-admin principals.',
 'classification', 2, true, 'deny', '{"*"}', '{"*"}', '{"role:viewer", "role:analyst"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', 'confidential', 1,
 '{"classification", "security", "restricted"}', 'system'),

-- Viewer: Allow read on public/internal
('c0000001-0001-4000-8000-000000000005', 'b0000001-0001-4000-8000-000000000003',
 'viewer-read-public', 'Allow viewers to read public and internal resources.',
 'identity', 50, true, 'allow', '{"read", "list", "search"}', '{"emissions:*", "calculations:*", "reports:*"}', '{"role:viewer"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', 'internal', 1,
 '{"viewer", "read-only", "default"}', 'system'),

-- Viewer: Deny write operations
('c0000001-0001-4000-8000-000000000006', 'b0000001-0001-4000-8000-000000000003',
 'viewer-deny-write', 'Deny all write, update, and delete operations for viewers.',
 'identity', 5, true, 'deny', '{"write", "create", "update", "delete", "admin"}', '{"*"}', '{"role:viewer"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', NULL, 1,
 '{"viewer", "deny-write", "default"}', 'system'),

-- Analyst: Allow read/write on data resources
('c0000001-0001-4000-8000-000000000007', 'b0000001-0001-4000-8000-000000000004',
 'analyst-read-write', 'Allow analysts to read, create, and update emissions data, calculations, assumptions, and citations.',
 'identity', 50, true, 'allow',
 '{"read", "list", "search", "create", "update", "export"}',
 '{"emissions:*", "calculations:*", "assumptions:*", "citations:*", "reports:*"}',
 '{"role:analyst"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', 'confidential', 1,
 '{"analyst", "read-write", "default"}', 'system'),

-- Analyst: Deny delete and admin
('c0000001-0001-4000-8000-000000000008', 'b0000001-0001-4000-8000-000000000004',
 'analyst-deny-admin', 'Deny delete and admin operations for analysts.',
 'identity', 5, true, 'deny', '{"delete", "admin", "manage_policies", "manage_rules"}', '{"*"}', '{"role:analyst"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', NULL, 1,
 '{"analyst", "deny-admin", "default"}', 'system'),

-- Admin: Allow all operations
('c0000001-0001-4000-8000-000000000009', 'b0000001-0001-4000-8000-000000000005',
 'admin-full-access', 'Allow admin principals full access to all resources and actions.',
 'identity', 100, true, 'allow', '{"*"}', '{"*"}', '{"role:admin", "role:super_admin"}',
 '{}'::jsonb,
 '{}'::jsonb, '{}', 'top_secret', 1,
 '{"admin", "full-access", "default"}', 'system'),

-- Time-based: Deny outside business hours for non-admin
('c0000001-0001-4000-8000-000000000010', 'b0000001-0001-4000-8000-000000000005',
 'time-restrict-writes', 'Restrict write operations to business hours (06:00-22:00 UTC) for non-admin principals.',
 'time_based', 200, false, 'deny', '{"create", "update", "delete"}', '{"*"}', '{"role:viewer", "role:analyst"}',
 '{}'::jsonb,
 '{"start_hour": 22, "end_hour": 6, "timezone": "UTC", "days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}'::jsonb,
 '{}', NULL, 1,
 '{"time-based", "business-hours", "optional"}', 'system')

ON CONFLICT (rule_id) DO NOTHING;

-- =============================================================================
-- Seed: Default Rate Limit Configurations
-- =============================================================================

INSERT INTO access_guard_service.rate_limit_state (bucket_key, tenant_id, principal_id, requests_minute, requests_hour, requests_day) VALUES
('default:role:viewer:rate_config', 'default', 'role:viewer', 0, 0, 0),
('default:role:analyst:rate_config', 'default', 'role:analyst', 0, 0, 0),
('default:role:admin:rate_config', 'default', 'role:admin', 0, 0, 0),
('default:role:service:rate_config', 'default', 'role:service', 0, 0, 0),
('default:role:agent:rate_config', 'default', 'role:agent', 0, 0, 0)
ON CONFLICT (bucket_key) DO NOTHING;

-- =============================================================================
-- Seed: Default Data Classification Patterns
-- =============================================================================

INSERT INTO access_guard_service.data_classifications (resource_id, resource_type, classification, classified_by, tenant_id, reason) VALUES

-- PII data is restricted
('pattern:pii:*', 'pii_data', 'restricted', 'system', 'default',
 'All personally identifiable information (PII) is classified as restricted. Access requires explicit authorization and is subject to GDPR, CCPA, and other privacy regulations.'),

-- Financial data is confidential
('pattern:financial:*', 'financial_data', 'confidential', 'system', 'default',
 'Financial data including revenue, costs, carbon credit pricing, and economic parameters is classified as confidential. Access limited to analysts and admins.'),

-- Emission data is internal
('pattern:emission:*', 'emission_data', 'internal', 'system', 'default',
 'Emission factor data, calculation results, and scope 1/2/3 emissions are classified as internal. Available to authenticated users within the tenant.'),

-- Compliance reports are confidential
('pattern:compliance:*', 'compliance_report', 'confidential', 'system', 'default',
 'Compliance reports including CSRD, CBAM, SEC Climate, and SOC 2 reports are classified as confidential. Access limited to compliance team and admins.'),

-- Personal employee data is restricted
('pattern:personal:*', 'personal_data', 'restricted', 'system', 'default',
 'Personal employee data used in commuting and business travel calculations is classified as restricted. Subject to HR data protection policies.')

ON CONFLICT (resource_id, resource_type, tenant_id) DO NOTHING;

-- =============================================================================
-- Seed: Default OPA Rego Policies
-- =============================================================================

INSERT INTO access_guard_service.rego_policies (policy_id, rego_source, source_hash, version, enabled, created_by, tenant_id) VALUES

('rego-tenant-isolation',
'package greenlang.access.tenant_isolation

import future.keywords.if
import future.keywords.in

default allow := false

allow if {
    input.principal.tenant_id == input.resource.tenant_id
}

deny if {
    input.principal.tenant_id != input.resource.tenant_id
    not input.principal.is_admin
}

violation["cross_tenant_access_attempt"] if {
    input.principal.tenant_id != input.resource.tenant_id
}',
'f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2', 1, true, 'system', 'default'),

('rego-classification-access',
'package greenlang.access.classification

import future.keywords.if
import future.keywords.in

classification_levels := {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
    "top_secret": 4
}

default allow := false

allow if {
    clearance := classification_levels[input.principal.clearance]
    resource_level := classification_levels[input.resource.classification]
    clearance >= resource_level
}

deny if {
    clearance := classification_levels[input.principal.clearance]
    resource_level := classification_levels[input.resource.classification]
    clearance < resource_level
}

violation["classification_breach"] if {
    clearance := classification_levels[input.principal.clearance]
    resource_level := classification_levels[input.resource.classification]
    clearance < resource_level
    resource_level >= 3
}',
'a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3', 1, true, 'system', 'default'),

('rego-rate-limiting',
'package greenlang.access.rate_limit

import future.keywords.if
import future.keywords.in

rate_limits := {
    "viewer":  {"minute": 60,  "hour": 500,  "day": 5000},
    "analyst": {"minute": 120, "hour": 2000, "day": 20000},
    "admin":   {"minute": 300, "hour": 5000, "day": 50000},
    "service": {"minute": 500, "hour": 10000, "day": 100000},
    "agent":   {"minute": 200, "hour": 3000, "day": 30000}
}

default allow := true

deny if {
    limits := rate_limits[input.principal.role]
    input.rate_state.requests_minute > limits.minute
}

deny if {
    limits := rate_limits[input.principal.role]
    input.rate_state.requests_hour > limits.hour
}

deny if {
    limits := rate_limits[input.principal.role]
    input.rate_state.requests_day > limits.day
}

violation["rate_limit_exceeded"] if {
    limits := rate_limits[input.principal.role]
    input.rate_state.requests_minute > limits.minute
}',
'b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4', 1, true, 'system', 'default')

ON CONFLICT (policy_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA access_guard_service IS 'Access & Policy Guard Agent service for GreenLang Climate OS (AGENT-FOUND-006) - policy-based access control, tenant isolation, data classification, rate limiting, OPA Rego evaluation, compliance reporting, and audit trail';
COMMENT ON TABLE access_guard_service.policies IS 'Top-level policy definitions grouping access control rules with versioning, tenant scoping, hierarchical inheritance, and provenance hashes';
COMMENT ON TABLE access_guard_service.policy_rules IS 'Individual access control rules within policies specifying effect (allow/deny), actions, resources, principals, conditions, time/geo constraints, and classification limits';
COMMENT ON TABLE access_guard_service.access_decisions IS 'TimescaleDB hypertable: every access decision with principal, resource, action, result, matching rules, deny reasons, evaluation time, and decision hash';
COMMENT ON TABLE access_guard_service.audit_events IS 'TimescaleDB hypertable: comprehensive audit log of all access control events with details, source IP, user agent, and configurable retention';
COMMENT ON TABLE access_guard_service.rate_limit_state IS 'Rate limit counters per principal per tenant tracking requests per minute/hour/day with sliding window timestamps';
COMMENT ON TABLE access_guard_service.data_classifications IS 'Resource classification registry mapping resources to sensitivity levels (public through top_secret) for classification-based access control';
COMMENT ON TABLE access_guard_service.rego_policies IS 'OPA Rego policy storage for complex policy evaluation beyond simple rule matching with source hash integrity verification';
COMMENT ON TABLE access_guard_service.compliance_reports IS 'Generated compliance reports summarizing access control activity with decision counts, denial reasons, policy coverage, and provenance hashes';
COMMENT ON MATERIALIZED VIEW access_guard_service.hourly_decision_counts IS 'Continuous aggregate: hourly access decision counts by decision type and tenant for real-time monitoring and trend analysis';
COMMENT ON MATERIALIZED VIEW access_guard_service.daily_audit_counts IS 'Continuous aggregate: daily audit event counts by event type and tenant for compliance reporting and long-term trend analysis';

COMMENT ON COLUMN access_guard_service.policies.policy_id IS 'Unique identifier for the policy (UUID)';
COMMENT ON COLUMN access_guard_service.policies.applies_to IS 'Array of resource types this policy covers (use * for all)';
COMMENT ON COLUMN access_guard_service.policies.allow_override IS 'Whether tenant-specific rules can override this policy';
COMMENT ON COLUMN access_guard_service.policies.provenance_hash IS 'SHA-256 hash of the policy content for integrity verification';

COMMENT ON COLUMN access_guard_service.policy_rules.effect IS 'Rule effect: allow or deny (deny-wins evaluation model)';
COMMENT ON COLUMN access_guard_service.policy_rules.policy_type IS 'Rule type: identity, resource, organization, network, time_based, classification, custom, composite';
COMMENT ON COLUMN access_guard_service.policy_rules.priority IS 'Evaluation priority (lower = higher priority, deny rules typically priority 1-10)';
COMMENT ON COLUMN access_guard_service.policy_rules.conditions IS 'JSONB conditions for dynamic rule evaluation (e.g., tenant matching, IP range, custom expressions)';
COMMENT ON COLUMN access_guard_service.policy_rules.time_constraints IS 'JSONB time window constraints (start_hour, end_hour, timezone, days of week)';
COMMENT ON COLUMN access_guard_service.policy_rules.classification_max IS 'Maximum resource classification level this rule allows access to';

COMMENT ON COLUMN access_guard_service.access_decisions.decision IS 'Decision result: allow, deny, rate_limited, challenge, defer';
COMMENT ON COLUMN access_guard_service.access_decisions.matching_rules IS 'Array of rule IDs that matched this request during evaluation';
COMMENT ON COLUMN access_guard_service.access_decisions.deny_reasons IS 'Array of human-readable reasons for denial (empty for allow decisions)';
COMMENT ON COLUMN access_guard_service.access_decisions.evaluation_time_ms IS 'Time taken to evaluate this access decision in milliseconds';
COMMENT ON COLUMN access_guard_service.access_decisions.decision_hash IS 'SHA-256 hash of the decision for tamper-evident audit trail';

COMMENT ON COLUMN access_guard_service.audit_events.event_type IS 'Audit event type: access_request, access_granted, access_denied, policy_created, rate_limit_hit, tenant_violation, etc.';
COMMENT ON COLUMN access_guard_service.audit_events.retention_days IS 'Number of days to retain this audit event (default 365)';

COMMENT ON COLUMN access_guard_service.data_classifications.classification IS 'Data classification level: public, internal, confidential, restricted, top_secret';

COMMENT ON COLUMN access_guard_service.rego_policies.source_hash IS 'SHA-256 hash of the Rego source code for integrity verification';

COMMENT ON COLUMN access_guard_service.compliance_reports.provenance_hash IS 'SHA-256 hash of the report content for integrity verification and audit';
