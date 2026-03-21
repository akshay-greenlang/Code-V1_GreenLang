-- =============================================================================
-- V202: PACK-029 Interim Targets Pack - Progress Alerts
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    007 of 015
-- Date:         March 2026
--
-- Progress alert system with RAG (Red/Amber/Green) status, alert reasons,
-- severity levels, escalation workflow tracking, and resolution management
-- for proactive interim target monitoring.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_progress_alerts
--
-- Previous: V201__PACK029_corrective_actions.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_progress_alerts
-- =============================================================================
-- Progress alert records with RAG classification, alert type categorization,
-- severity levels, escalation tracking, and resolution workflow for
-- proactive identification of off-track interim targets.

CREATE TABLE pack029_interim_targets.gl_progress_alerts (
    alert_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    variance_id                 UUID            REFERENCES pack029_interim_targets.gl_variance_analysis(variance_id) ON DELETE SET NULL,
    -- Time context
    year                        INTEGER         NOT NULL,
    quarter                     VARCHAR(2),
    -- Alert classification
    alert_type                  VARCHAR(10)     NOT NULL,
    alert_reason                VARCHAR(40)     NOT NULL,
    alert_code                  VARCHAR(20),
    alert_title                 VARCHAR(200)    NOT NULL,
    alert_description           TEXT,
    -- Severity
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    impact_score                DECIMAL(5,2),
    -- Scope context
    scope                       VARCHAR(20),
    category                    VARCHAR(60),
    -- Threshold breach details
    threshold_value             DECIMAL(18,4),
    actual_value                DECIMAL(18,4),
    breach_pct                  DECIMAL(8,4),
    -- Timeline
    triggered_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    acknowledged_at             TIMESTAMPTZ,
    acknowledged_by             VARCHAR(255),
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    resolution_notes            TEXT,
    resolution_action_id        UUID,
    -- Escalation
    escalated                   BOOLEAN         DEFAULT FALSE,
    escalation_level            INTEGER         DEFAULT 0,
    escalated_at                TIMESTAMPTZ,
    escalated_to                VARCHAR(255),
    escalation_reason           TEXT,
    auto_escalation_date        DATE,
    -- Notification
    notification_sent           BOOLEAN         DEFAULT FALSE,
    notification_channels       TEXT[]          DEFAULT '{}',
    notification_recipients     TEXT[]          DEFAULT '{}',
    last_notification_at        TIMESTAMPTZ,
    -- Recurrence
    is_recurring                BOOLEAN         DEFAULT FALSE,
    recurrence_count            INTEGER         DEFAULT 1,
    first_occurrence_at         TIMESTAMPTZ,
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_suppressed               BOOLEAN         DEFAULT FALSE,
    suppressed_until            TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_pa_alert_type CHECK (
        alert_type IN ('RED', 'AMBER', 'GREEN')
    ),
    CONSTRAINT chk_p029_pa_alert_reason CHECK (
        alert_reason IN (
            'VARIANCE_EXCEEDED', 'MILESTONE_MISSED', 'TREND_OFF_TRACK',
            'BUDGET_OVERSHOOT', 'DATA_QUALITY_LOW', 'INITIATIVE_DELAYED',
            'FORECAST_OFF_TARGET', 'CONSECUTIVE_MISS', 'CRITICAL_PATH_RISK',
            'VALIDATION_EXPIRED', 'SBTI_NON_COMPLIANT', 'RESTATEMENT_NEEDED',
            'BACK_ON_TRACK', 'MILESTONE_ACHIEVED', 'TARGET_MET', 'IMPROVEMENT_TREND'
        )
    ),
    CONSTRAINT chk_p029_pa_severity CHECK (
        severity IN ('CRITICAL', 'WARNING', 'INFO', 'SUCCESS')
    ),
    CONSTRAINT chk_p029_pa_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_pa_quarter CHECK (
        quarter IS NULL OR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_pa_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_pa_escalation_level CHECK (
        escalation_level >= 0 AND escalation_level <= 5
    ),
    CONSTRAINT chk_p029_pa_recurrence_count CHECK (
        recurrence_count >= 1
    ),
    CONSTRAINT chk_p029_pa_impact_score CHECK (
        impact_score IS NULL OR (impact_score >= 0 AND impact_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_pa_tenant             ON pack029_interim_targets.gl_progress_alerts(tenant_id);
CREATE INDEX idx_p029_pa_org                ON pack029_interim_targets.gl_progress_alerts(organization_id);
CREATE INDEX idx_p029_pa_target             ON pack029_interim_targets.gl_progress_alerts(target_id);
CREATE INDEX idx_p029_pa_variance           ON pack029_interim_targets.gl_progress_alerts(variance_id);
CREATE INDEX idx_p029_pa_org_triggered      ON pack029_interim_targets.gl_progress_alerts(organization_id, triggered_at DESC);
CREATE INDEX idx_p029_pa_org_year_qtr       ON pack029_interim_targets.gl_progress_alerts(organization_id, year, quarter);
CREATE INDEX idx_p029_pa_alert_type         ON pack029_interim_targets.gl_progress_alerts(alert_type);
CREATE INDEX idx_p029_pa_alert_reason       ON pack029_interim_targets.gl_progress_alerts(alert_reason);
CREATE INDEX idx_p029_pa_severity           ON pack029_interim_targets.gl_progress_alerts(severity);
CREATE INDEX idx_p029_pa_active_unresolved  ON pack029_interim_targets.gl_progress_alerts(organization_id, alert_type) WHERE resolved_at IS NULL AND is_active = TRUE;
CREATE INDEX idx_p029_pa_red_active         ON pack029_interim_targets.gl_progress_alerts(organization_id, triggered_at DESC) WHERE alert_type = 'RED' AND resolved_at IS NULL;
CREATE INDEX idx_p029_pa_critical_active    ON pack029_interim_targets.gl_progress_alerts(organization_id) WHERE severity = 'CRITICAL' AND resolved_at IS NULL;
CREATE INDEX idx_p029_pa_escalated          ON pack029_interim_targets.gl_progress_alerts(organization_id, escalation_level) WHERE escalated = TRUE;
CREATE INDEX idx_p029_pa_auto_escalation    ON pack029_interim_targets.gl_progress_alerts(auto_escalation_date) WHERE auto_escalation_date IS NOT NULL AND resolved_at IS NULL;
CREATE INDEX idx_p029_pa_unacknowledged     ON pack029_interim_targets.gl_progress_alerts(organization_id) WHERE acknowledged_at IS NULL AND resolved_at IS NULL;
CREATE INDEX idx_p029_pa_recurring          ON pack029_interim_targets.gl_progress_alerts(organization_id, recurrence_count DESC) WHERE is_recurring = TRUE;
CREATE INDEX idx_p029_pa_resolved_at        ON pack029_interim_targets.gl_progress_alerts(resolved_at DESC) WHERE resolved_at IS NOT NULL;
CREATE INDEX idx_p029_pa_created            ON pack029_interim_targets.gl_progress_alerts(created_at DESC);
CREATE INDEX idx_p029_pa_channels           ON pack029_interim_targets.gl_progress_alerts USING GIN(notification_channels);
CREATE INDEX idx_p029_pa_recipients         ON pack029_interim_targets.gl_progress_alerts USING GIN(notification_recipients);
CREATE INDEX idx_p029_pa_metadata           ON pack029_interim_targets.gl_progress_alerts USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_progress_alerts_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_progress_alerts
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_progress_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_pa_tenant_isolation
    ON pack029_interim_targets.gl_progress_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_pa_service_bypass
    ON pack029_interim_targets.gl_progress_alerts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_progress_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_progress_alerts IS
    'Progress alert records with RAG (Red/Amber/Green) classification, alert reasons, severity levels, escalation workflow tracking, and resolution management for proactive interim target monitoring.';

COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.alert_id IS 'Unique progress alert identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.organization_id IS 'Reference to the organization this alert pertains to.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.alert_type IS 'RAG classification: RED (off-track), AMBER (at-risk), GREEN (on-track).';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.alert_reason IS 'Specific reason for alert trigger (e.g., VARIANCE_EXCEEDED, MILESTONE_MISSED).';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.severity IS 'Alert severity: CRITICAL, WARNING, INFO, SUCCESS.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.triggered_at IS 'Timestamp when the alert was triggered.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.resolved_at IS 'Timestamp when the alert was resolved (NULL for active alerts).';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.escalated IS 'Whether this alert has been escalated to higher management.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.escalation_level IS 'Current escalation level (0=none, 1-5=escalation tiers).';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.is_recurring IS 'Whether this alert type has occurred multiple times.';
COMMENT ON COLUMN pack029_interim_targets.gl_progress_alerts.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
