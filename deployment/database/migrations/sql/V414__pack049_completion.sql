-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V414 - Completion Tracking
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates completion tracking tables for monitoring site-level data
-- submission progress, detecting gaps, managing deadlines, and sending
-- reminders. The completion engine drives the "traffic light" dashboard
-- showing which sites are on track, at risk, or overdue.
--
-- Tables (5):
--   1. ghg_multisite.gl_ms_completion_status
--   2. ghg_multisite.gl_ms_submission_tracker
--   3. ghg_multisite.gl_ms_gap_detections
--   4. ghg_multisite.gl_ms_reminders
--   5. ghg_multisite.gl_ms_deadlines
--
-- Also includes: indexes, RLS, comments.
-- Previous: V413__pack049_comparison_quality.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_completion_status
-- =============================================================================
-- Overall completion status per site per reporting period. Provides the
-- "traffic light" view (GREEN/AMBER/RED) and completion percentage.

CREATE TABLE ghg_multisite.gl_ms_completion_status (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    traffic_light               VARCHAR(10)     NOT NULL DEFAULT 'RED',
    completion_pct              NUMERIC(10,4)   NOT NULL DEFAULT 0.0000,
    scope1_complete             BOOLEAN         NOT NULL DEFAULT false,
    scope2_complete             BOOLEAN         NOT NULL DEFAULT false,
    scope3_complete             BOOLEAN         NOT NULL DEFAULT false,
    data_sources_expected       INTEGER         NOT NULL DEFAULT 0,
    data_sources_received       INTEGER         NOT NULL DEFAULT 0,
    months_expected             INTEGER         NOT NULL DEFAULT 0,
    months_received             INTEGER         NOT NULL DEFAULT 0,
    categories_expected         INTEGER         NOT NULL DEFAULT 0,
    categories_received         INTEGER         NOT NULL DEFAULT 0,
    validation_passed           BOOLEAN         NOT NULL DEFAULT false,
    review_status               VARCHAR(20)     NOT NULL DEFAULT 'NOT_STARTED',
    days_until_deadline         INTEGER,
    is_overdue                  BOOLEAN         NOT NULL DEFAULT false,
    last_updated_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    last_submission_at          TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_cs_traffic CHECK (
        traffic_light IN ('GREEN', 'AMBER', 'RED', 'GREY')
    ),
    CONSTRAINT chk_p049_cs_completion CHECK (
        completion_pct >= 0 AND completion_pct <= 100
    ),
    CONSTRAINT chk_p049_cs_review CHECK (
        review_status IN (
            'NOT_STARTED', 'IN_PROGRESS', 'SUBMITTED',
            'UNDER_REVIEW', 'APPROVED', 'REJECTED'
        )
    ),
    CONSTRAINT chk_p049_cs_ds_expected CHECK (data_sources_expected >= 0),
    CONSTRAINT chk_p049_cs_ds_received CHECK (data_sources_received >= 0),
    CONSTRAINT chk_p049_cs_months_exp CHECK (months_expected >= 0),
    CONSTRAINT chk_p049_cs_months_rec CHECK (months_received >= 0),
    CONSTRAINT chk_p049_cs_cats_exp CHECK (categories_expected >= 0),
    CONSTRAINT chk_p049_cs_cats_rec CHECK (categories_received >= 0),
    CONSTRAINT uq_p049_cs_site_period UNIQUE (site_id, period_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_cs_tenant          ON ghg_multisite.gl_ms_completion_status(tenant_id);
CREATE INDEX idx_p049_cs_config          ON ghg_multisite.gl_ms_completion_status(config_id);
CREATE INDEX idx_p049_cs_period          ON ghg_multisite.gl_ms_completion_status(period_id);
CREATE INDEX idx_p049_cs_site            ON ghg_multisite.gl_ms_completion_status(site_id);
CREATE INDEX idx_p049_cs_traffic         ON ghg_multisite.gl_ms_completion_status(traffic_light);
CREATE INDEX idx_p049_cs_red             ON ghg_multisite.gl_ms_completion_status(period_id, traffic_light)
    WHERE traffic_light = 'RED';
CREATE INDEX idx_p049_cs_amber           ON ghg_multisite.gl_ms_completion_status(period_id, traffic_light)
    WHERE traffic_light = 'AMBER';
CREATE INDEX idx_p049_cs_overdue         ON ghg_multisite.gl_ms_completion_status(period_id)
    WHERE is_overdue = true;
CREATE INDEX idx_p049_cs_review          ON ghg_multisite.gl_ms_completion_status(review_status);
CREATE INDEX idx_p049_cs_incomplete      ON ghg_multisite.gl_ms_completion_status(period_id, completion_pct)
    WHERE completion_pct < 100;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_completion_status ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_cs_tenant_isolation ON ghg_multisite.gl_ms_completion_status
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_submission_tracker
-- =============================================================================
-- Detailed submission activity log per site. Tracks each action taken
-- (upload, edit, submit, approve, reject) with timestamps and actor.

CREATE TABLE ghg_multisite.gl_ms_submission_tracker (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    completion_status_id        UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_completion_status(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    action                      VARCHAR(30)     NOT NULL,
    action_description          TEXT,
    actor_id                    UUID,
    actor_name                  VARCHAR(255),
    actor_role                  VARCHAR(50),
    records_affected            INTEGER         NOT NULL DEFAULT 0,
    previous_status             VARCHAR(20),
    new_status                  VARCHAR(20),
    completion_before_pct       NUMERIC(10,4),
    completion_after_pct        NUMERIC(10,4),
    ip_address                  VARCHAR(45),
    user_agent                  VARCHAR(500),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_tracker_action CHECK (
        action IN (
            'UPLOAD', 'MANUAL_ENTRY', 'EDIT', 'DELETE',
            'SUBMIT', 'RECALL', 'APPROVE', 'REJECT',
            'REQUEST_REVISION', 'COMMENT', 'ASSIGN_REVIEWER',
            'LOCK', 'UNLOCK', 'EXPORT'
        )
    ),
    CONSTRAINT chk_p049_tracker_records CHECK (records_affected >= 0),
    CONSTRAINT chk_p049_tracker_before CHECK (
        completion_before_pct IS NULL OR (completion_before_pct >= 0 AND completion_before_pct <= 100)
    ),
    CONSTRAINT chk_p049_tracker_after CHECK (
        completion_after_pct IS NULL OR (completion_after_pct >= 0 AND completion_after_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_tracker_tenant     ON ghg_multisite.gl_ms_submission_tracker(tenant_id);
CREATE INDEX idx_p049_tracker_cs         ON ghg_multisite.gl_ms_submission_tracker(completion_status_id);
CREATE INDEX idx_p049_tracker_site       ON ghg_multisite.gl_ms_submission_tracker(site_id);
CREATE INDEX idx_p049_tracker_action     ON ghg_multisite.gl_ms_submission_tracker(action);
CREATE INDEX idx_p049_tracker_actor      ON ghg_multisite.gl_ms_submission_tracker(actor_id)
    WHERE actor_id IS NOT NULL;
CREATE INDEX idx_p049_tracker_created    ON ghg_multisite.gl_ms_submission_tracker(created_at);
CREATE INDEX idx_p049_tracker_submits    ON ghg_multisite.gl_ms_submission_tracker(site_id, action)
    WHERE action IN ('SUBMIT', 'APPROVE', 'REJECT');

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_submission_tracker ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_tracker_tenant_isolation ON ghg_multisite.gl_ms_submission_tracker
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_gap_detections
-- =============================================================================
-- Detected data gaps for a site within a reporting period. Gaps represent
-- missing data points (missing months, missing scopes, missing categories)
-- that prevent the site from reaching 100% completion.

CREATE TABLE ghg_multisite.gl_ms_gap_detections (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    completion_status_id        UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_completion_status(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    gap_type                    VARCHAR(30)     NOT NULL,
    gap_description             TEXT            NOT NULL,
    scope                       VARCHAR(10),
    category                    VARCHAR(100),
    missing_months              JSONB           DEFAULT '[]',
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    impact_on_total_pct         NUMERIC(10,4),
    suggested_action            TEXT,
    estimation_available        BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(100),
    estimated_value_tco2e       NUMERIC(20,6),
    is_resolved                 BOOLEAN         NOT NULL DEFAULT false,
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 UUID,
    resolution_method           VARCHAR(50),
    detected_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_gap_type CHECK (
        gap_type IN (
            'MISSING_MONTH', 'MISSING_SCOPE', 'MISSING_CATEGORY',
            'MISSING_SOURCE', 'INCOMPLETE_DATA', 'STALE_DATA',
            'BELOW_THRESHOLD', 'ANOMALY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_gap_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p049_gap_severity CHECK (
        severity IN ('CRITICAL', 'ERROR', 'WARNING', 'INFO')
    ),
    CONSTRAINT chk_p049_gap_impact CHECK (
        impact_on_total_pct IS NULL OR (impact_on_total_pct >= 0 AND impact_on_total_pct <= 100)
    ),
    CONSTRAINT chk_p049_gap_estimated CHECK (
        estimated_value_tco2e IS NULL OR estimated_value_tco2e >= 0
    ),
    CONSTRAINT chk_p049_gap_resolution CHECK (
        resolution_method IS NULL OR resolution_method IN (
            'DATA_RECEIVED', 'ESTIMATED', 'EXCLUDED', 'NOT_APPLICABLE', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_gap_tenant         ON ghg_multisite.gl_ms_gap_detections(tenant_id);
CREATE INDEX idx_p049_gap_cs             ON ghg_multisite.gl_ms_gap_detections(completion_status_id);
CREATE INDEX idx_p049_gap_site           ON ghg_multisite.gl_ms_gap_detections(site_id);
CREATE INDEX idx_p049_gap_type           ON ghg_multisite.gl_ms_gap_detections(gap_type);
CREATE INDEX idx_p049_gap_severity       ON ghg_multisite.gl_ms_gap_detections(severity);
CREATE INDEX idx_p049_gap_scope          ON ghg_multisite.gl_ms_gap_detections(scope)
    WHERE scope IS NOT NULL;
CREATE INDEX idx_p049_gap_unresolved     ON ghg_multisite.gl_ms_gap_detections(completion_status_id)
    WHERE is_resolved = false;
CREATE INDEX idx_p049_gap_critical       ON ghg_multisite.gl_ms_gap_detections(completion_status_id, severity)
    WHERE severity = 'CRITICAL' AND is_resolved = false;
CREATE INDEX idx_p049_gap_estimable      ON ghg_multisite.gl_ms_gap_detections(completion_status_id)
    WHERE estimation_available = true AND is_resolved = false;
CREATE INDEX idx_p049_gap_months         ON ghg_multisite.gl_ms_gap_detections USING gin(missing_months);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_gap_detections ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_gap_tenant_isolation ON ghg_multisite.gl_ms_gap_detections
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_reminders
-- =============================================================================
-- Reminder records sent to site contacts regarding data submission.
-- Tracks reminder type, delivery channel, and outcome.

CREATE TABLE ghg_multisite.gl_ms_reminders (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    reminder_type               VARCHAR(30)     NOT NULL,
    delivery_channel            VARCHAR(20)     NOT NULL DEFAULT 'EMAIL',
    recipient_email             VARCHAR(255),
    recipient_name              VARCHAR(255),
    subject                     VARCHAR(500),
    message_body                TEXT,
    template_id                 VARCHAR(100),
    scheduled_at                TIMESTAMPTZ     NOT NULL,
    sent_at                     TIMESTAMPTZ,
    delivered_at                TIMESTAMPTZ,
    opened_at                   TIMESTAMPTZ,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'SCHEDULED',
    retry_count                 INTEGER         NOT NULL DEFAULT 0,
    error_message               TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_rem_type CHECK (
        reminder_type IN (
            'INITIAL_NOTIFICATION', 'GENTLE_REMINDER', 'DEADLINE_APPROACHING',
            'OVERDUE_NOTICE', 'ESCALATION', 'FINAL_WARNING',
            'REVISION_REQUEST', 'APPROVAL_NOTIFICATION', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_rem_channel CHECK (
        delivery_channel IN ('EMAIL', 'SLACK', 'TEAMS', 'WEBHOOK', 'IN_APP', 'SMS')
    ),
    CONSTRAINT chk_p049_rem_status CHECK (
        status IN ('SCHEDULED', 'SENT', 'DELIVERED', 'OPENED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p049_rem_retry CHECK (retry_count >= 0 AND retry_count <= 10)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_rem_tenant         ON ghg_multisite.gl_ms_reminders(tenant_id);
CREATE INDEX idx_p049_rem_config         ON ghg_multisite.gl_ms_reminders(config_id);
CREATE INDEX idx_p049_rem_period         ON ghg_multisite.gl_ms_reminders(period_id);
CREATE INDEX idx_p049_rem_site           ON ghg_multisite.gl_ms_reminders(site_id);
CREATE INDEX idx_p049_rem_type           ON ghg_multisite.gl_ms_reminders(reminder_type);
CREATE INDEX idx_p049_rem_status         ON ghg_multisite.gl_ms_reminders(status);
CREATE INDEX idx_p049_rem_scheduled      ON ghg_multisite.gl_ms_reminders(scheduled_at)
    WHERE status = 'SCHEDULED';
CREATE INDEX idx_p049_rem_pending        ON ghg_multisite.gl_ms_reminders(status)
    WHERE status IN ('SCHEDULED', 'FAILED');
CREATE INDEX idx_p049_rem_sent           ON ghg_multisite.gl_ms_reminders(sent_at)
    WHERE sent_at IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_reminders ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_rem_tenant_isolation ON ghg_multisite.gl_ms_reminders
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 5: ghg_multisite.gl_ms_deadlines
-- =============================================================================
-- Deadline configuration per site or group. Defines milestone deadlines
-- for data submission, review, and approval with escalation rules.

CREATE TABLE ghg_multisite.gl_ms_deadlines (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_reporting_periods(id) ON DELETE CASCADE,
    site_id                     UUID            REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    group_id                    UUID            REFERENCES ghg_multisite.gl_ms_site_groups(id) ON DELETE CASCADE,
    deadline_type               VARCHAR(30)     NOT NULL,
    deadline_name               VARCHAR(255)    NOT NULL,
    deadline_date               DATE            NOT NULL,
    reminder_days_before        INTEGER         NOT NULL DEFAULT 7,
    escalation_days_after       INTEGER         NOT NULL DEFAULT 3,
    escalation_recipient        VARCHAR(255),
    is_mandatory                BOOLEAN         NOT NULL DEFAULT true,
    grace_period_days           INTEGER         NOT NULL DEFAULT 0,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    completed_at                TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_dl_type CHECK (
        deadline_type IN (
            'DATA_SUBMISSION', 'REVIEW_COMPLETION', 'APPROVAL',
            'CONSOLIDATION', 'REPORTING', 'AUDIT_READINESS',
            'BASE_YEAR_UPDATE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_dl_status CHECK (
        status IN ('PENDING', 'MET', 'MISSED', 'EXTENDED', 'WAIVED', 'CANCELLED')
    ),
    CONSTRAINT chk_p049_dl_reminder CHECK (
        reminder_days_before >= 0 AND reminder_days_before <= 90
    ),
    CONSTRAINT chk_p049_dl_escalation CHECK (
        escalation_days_after >= 0 AND escalation_days_after <= 90
    ),
    CONSTRAINT chk_p049_dl_grace CHECK (
        grace_period_days >= 0 AND grace_period_days <= 90
    ),
    CONSTRAINT chk_p049_dl_scope CHECK (
        site_id IS NOT NULL OR group_id IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_dl_tenant          ON ghg_multisite.gl_ms_deadlines(tenant_id);
CREATE INDEX idx_p049_dl_config          ON ghg_multisite.gl_ms_deadlines(config_id);
CREATE INDEX idx_p049_dl_period          ON ghg_multisite.gl_ms_deadlines(period_id);
CREATE INDEX idx_p049_dl_site            ON ghg_multisite.gl_ms_deadlines(site_id)
    WHERE site_id IS NOT NULL;
CREATE INDEX idx_p049_dl_group           ON ghg_multisite.gl_ms_deadlines(group_id)
    WHERE group_id IS NOT NULL;
CREATE INDEX idx_p049_dl_type            ON ghg_multisite.gl_ms_deadlines(deadline_type);
CREATE INDEX idx_p049_dl_date            ON ghg_multisite.gl_ms_deadlines(deadline_date);
CREATE INDEX idx_p049_dl_status          ON ghg_multisite.gl_ms_deadlines(status);
CREATE INDEX idx_p049_dl_pending         ON ghg_multisite.gl_ms_deadlines(period_id, deadline_date)
    WHERE status = 'PENDING';
CREATE INDEX idx_p049_dl_missed          ON ghg_multisite.gl_ms_deadlines(period_id)
    WHERE status = 'MISSED';
CREATE INDEX idx_p049_dl_mandatory       ON ghg_multisite.gl_ms_deadlines(period_id)
    WHERE is_mandatory = true AND status = 'PENDING';

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_deadlines ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_dl_tenant_isolation ON ghg_multisite.gl_ms_deadlines
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_completion_status IS
    'PACK-049: Site completion status with traffic light (4 levels), scope flags, and source/month/category counts.';
COMMENT ON TABLE ghg_multisite.gl_ms_submission_tracker IS
    'PACK-049: Submission activity log (14 actions) with actor, timestamps, and completion delta.';
COMMENT ON TABLE ghg_multisite.gl_ms_gap_detections IS
    'PACK-049: Data gap detections (9 types, 4 severities) with estimation options and resolution tracking.';
COMMENT ON TABLE ghg_multisite.gl_ms_reminders IS
    'PACK-049: Reminder records (9 types, 6 channels) with delivery tracking and retry logic.';
COMMENT ON TABLE ghg_multisite.gl_ms_deadlines IS
    'PACK-049: Deadline configuration (8 types) with reminder, escalation, grace period, and status tracking.';
