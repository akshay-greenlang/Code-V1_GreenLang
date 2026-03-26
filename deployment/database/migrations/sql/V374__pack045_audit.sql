-- =============================================================================
-- V374: PACK-045 Base Year Management Pack - Audit Trail & Approvals
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates audit trail, approvals, and verification tables for complete base
-- year management governance. The audit trail logs every significant event
-- (creation, modification, recalculation, approval) with before/after values
-- and SHA-256 provenance hashes. Approvals track the workflow for base year
-- changes. Verifications record third-party assurance opinions.
--
-- Tables (3):
--   1. ghg_base_year.gl_by_audit_trail
--   2. ghg_base_year.gl_by_approvals
--   3. ghg_base_year.gl_by_verifications
--
-- Also includes: deferred trigger from V369, indexes, RLS, comments.
-- Previous: V373__pack045_targets.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_audit_trail
-- =============================================================================
-- Immutable audit log for all base year management events. Each entry records
-- the event type, actor, description, before/after values, evidence references,
-- and a SHA-256 provenance hash. Entries are append-only (no UPDATE/DELETE
-- allowed for audit integrity).

CREATE TABLE ghg_base_year.gl_by_audit_trail (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    base_year_id                UUID            REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE SET NULL,
    event_type                  VARCHAR(60)     NOT NULL,
    event_subtype               VARCHAR(60),
    actor                       VARCHAR(255)    NOT NULL,
    actor_role                  VARCHAR(60),
    description                 TEXT            NOT NULL,
    before_value                TEXT,
    after_value                 TEXT,
    change_detail_json          JSONB           DEFAULT '{}',
    affected_table              VARCHAR(100),
    affected_record_id          UUID,
    evidence_refs               TEXT[],
    ip_address                  VARCHAR(45),
    user_agent                  VARCHAR(500),
    session_id                  VARCHAR(100),
    provenance_hash             VARCHAR(64)     NOT NULL,
    timestamp                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_at_event_type CHECK (
        event_type IN (
            'BASE_YEAR_CREATED', 'BASE_YEAR_MODIFIED', 'BASE_YEAR_ACTIVATED',
            'BASE_YEAR_SUPERSEDED', 'BASE_YEAR_ARCHIVED',
            'INVENTORY_ADDED', 'INVENTORY_MODIFIED', 'INVENTORY_DELETED',
            'POLICY_CREATED', 'POLICY_MODIFIED', 'POLICY_APPROVED',
            'TRIGGER_DETECTED', 'TRIGGER_STATUS_CHANGE', 'TRIGGER_RESOLVED',
            'ASSESSMENT_CREATED', 'ASSESSMENT_COMPLETED',
            'ADJUSTMENT_CREATED', 'ADJUSTMENT_SUBMITTED', 'ADJUSTMENT_APPROVED',
            'ADJUSTMENT_REJECTED', 'ADJUSTMENT_APPLIED',
            'TARGET_CREATED', 'TARGET_MODIFIED', 'TARGET_APPROVED',
            'PROGRESS_RECORDED', 'PROGRESS_STATUS_CHANGE',
            'CONFIGURATION_CHANGED', 'VERIFICATION_COMPLETED',
            'APPROVAL_REQUESTED', 'APPROVAL_GRANTED', 'APPROVAL_DENIED',
            'TIME_SERIES_UPDATED', 'CONSISTENCY_FINDING', 'DATA_EXPORT',
            'SYSTEM_RECALCULATION', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_at_tenant          ON ghg_base_year.gl_by_audit_trail(tenant_id);
CREATE INDEX idx_p045_at_org             ON ghg_base_year.gl_by_audit_trail(org_id);
CREATE INDEX idx_p045_at_base_year       ON ghg_base_year.gl_by_audit_trail(base_year_id);
CREATE INDEX idx_p045_at_event_type      ON ghg_base_year.gl_by_audit_trail(event_type);
CREATE INDEX idx_p045_at_actor           ON ghg_base_year.gl_by_audit_trail(actor);
CREATE INDEX idx_p045_at_timestamp       ON ghg_base_year.gl_by_audit_trail(timestamp DESC);
CREATE INDEX idx_p045_at_affected_table  ON ghg_base_year.gl_by_audit_trail(affected_table);
CREATE INDEX idx_p045_at_affected_record ON ghg_base_year.gl_by_audit_trail(affected_record_id);
CREATE INDEX idx_p045_at_provenance      ON ghg_base_year.gl_by_audit_trail(provenance_hash);
CREATE INDEX idx_p045_at_created         ON ghg_base_year.gl_by_audit_trail(created_at DESC);
CREATE INDEX idx_p045_at_change_detail   ON ghg_base_year.gl_by_audit_trail USING GIN(change_detail_json);

-- Composite: org + event type for filtered audit queries
CREATE INDEX idx_p045_at_org_event       ON ghg_base_year.gl_by_audit_trail(org_id, event_type, timestamp DESC);

-- Composite: base_year + timestamp for base-year-specific audit
CREATE INDEX idx_p045_at_by_timestamp    ON ghg_base_year.gl_by_audit_trail(base_year_id, timestamp DESC);

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_approvals
-- =============================================================================
-- Approval workflow records for base year management actions. Tracks who
-- requested what, who approved/denied, decision rationale, and timestamps.
-- Supports multi-level approvals with escalation.

CREATE TABLE ghg_base_year.gl_by_approvals (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    subject_type                VARCHAR(40)     NOT NULL,
    subject_id                  UUID            NOT NULL,
    subject_description         TEXT            NOT NULL,
    approval_level              INTEGER         NOT NULL DEFAULT 1,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    requested_by                UUID            NOT NULL,
    requested_by_name           VARCHAR(255)    NOT NULL,
    requested_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approver                    UUID,
    approver_name               VARCHAR(255),
    approver_role               VARCHAR(60),
    decision_date               TIMESTAMPTZ,
    decision_rationale          TEXT,
    conditions                  TEXT,
    escalated                   BOOLEAN         NOT NULL DEFAULT false,
    escalation_date             TIMESTAMPTZ,
    escalation_reason           TEXT,
    due_date                    DATE,
    reminder_sent               BOOLEAN         NOT NULL DEFAULT false,
    evidence_refs               TEXT[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_appr_subject CHECK (
        subject_type IN (
            'BASE_YEAR', 'ADJUSTMENT_PACKAGE', 'POLICY_CHANGE', 'TARGET',
            'RECALCULATION', 'CONFIGURATION_CHANGE', 'TRIGGER_DISMISSAL'
        )
    ),
    CONSTRAINT chk_p045_appr_level CHECK (
        approval_level >= 1 AND approval_level <= 5
    ),
    CONSTRAINT chk_p045_appr_status CHECK (
        status IN ('PENDING', 'APPROVED', 'DENIED', 'ESCALATED', 'WITHDRAWN', 'EXPIRED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_appr_tenant        ON ghg_base_year.gl_by_approvals(tenant_id);
CREATE INDEX idx_p045_appr_org           ON ghg_base_year.gl_by_approvals(org_id);
CREATE INDEX idx_p045_appr_subject_type  ON ghg_base_year.gl_by_approvals(subject_type);
CREATE INDEX idx_p045_appr_subject_id    ON ghg_base_year.gl_by_approvals(subject_id);
CREATE INDEX idx_p045_appr_status        ON ghg_base_year.gl_by_approvals(status);
CREATE INDEX idx_p045_appr_requested_by  ON ghg_base_year.gl_by_approvals(requested_by);
CREATE INDEX idx_p045_appr_approver      ON ghg_base_year.gl_by_approvals(approver);
CREATE INDEX idx_p045_appr_decision_date ON ghg_base_year.gl_by_approvals(decision_date);
CREATE INDEX idx_p045_appr_due_date      ON ghg_base_year.gl_by_approvals(due_date);
CREATE INDEX idx_p045_appr_created       ON ghg_base_year.gl_by_approvals(created_at DESC);

-- Composite: pending approvals for dashboard
CREATE INDEX idx_p045_appr_pending       ON ghg_base_year.gl_by_approvals(org_id, subject_type, requested_date DESC)
    WHERE status = 'PENDING';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_appr_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_approvals
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_base_year.gl_by_verifications
-- =============================================================================
-- Third-party verification records for base year data and recalculation
-- packages. Tracks the verifier, assurance level (limited/reasonable),
-- opinion, scope of verification, and any findings or recommendations.

CREATE TABLE ghg_base_year.gl_by_verifications (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    base_year_id                UUID            REFERENCES ghg_base_year.gl_by_base_years(id) ON DELETE SET NULL,
    package_id                  UUID            REFERENCES ghg_base_year.gl_by_adjustment_packages(id) ON DELETE SET NULL,
    verification_type           VARCHAR(30)     NOT NULL DEFAULT 'BASE_YEAR',
    assurance_level             VARCHAR(30)     NOT NULL,
    verifier_name               VARCHAR(255)    NOT NULL,
    verifier_organization       VARCHAR(255),
    verifier_accreditation      VARCHAR(100),
    verification_standard       VARCHAR(100),
    engagement_start_date       DATE,
    engagement_end_date         DATE,
    verification_date           DATE            NOT NULL,
    opinion                     VARCHAR(30)     NOT NULL,
    scope_of_verification       TEXT,
    materiality_threshold_pct   NUMERIC(5,2),
    findings_json               JSONB           DEFAULT '{}',
    recommendations_json        JSONB           DEFAULT '{}',
    non_conformities            INTEGER         DEFAULT 0,
    observations                INTEGER         DEFAULT 0,
    statement_ref               VARCHAR(255),
    certificate_ref             VARCHAR(255),
    evidence_refs               TEXT[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_ver_type CHECK (
        verification_type IN ('BASE_YEAR', 'RECALCULATION', 'ANNUAL_INVENTORY', 'TARGET_PROGRESS')
    ),
    CONSTRAINT chk_p045_ver_assurance CHECK (
        assurance_level IN ('LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE', 'AGREED_UPON_PROCEDURES')
    ),
    CONSTRAINT chk_p045_ver_opinion CHECK (
        opinion IN (
            'UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER',
            'CLEAN', 'MODIFIED', 'NOT_ISSUED'
        )
    ),
    CONSTRAINT chk_p045_ver_materiality CHECK (
        materiality_threshold_pct IS NULL OR (materiality_threshold_pct > 0 AND materiality_threshold_pct <= 100)
    ),
    CONSTRAINT chk_p045_ver_dates CHECK (
        engagement_start_date IS NULL OR engagement_end_date IS NULL OR
        engagement_start_date <= engagement_end_date
    ),
    CONSTRAINT chk_p045_ver_non_conf CHECK (
        non_conformities >= 0
    ),
    CONSTRAINT chk_p045_ver_observations CHECK (
        observations >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_ver_tenant         ON ghg_base_year.gl_by_verifications(tenant_id);
CREATE INDEX idx_p045_ver_org            ON ghg_base_year.gl_by_verifications(org_id);
CREATE INDEX idx_p045_ver_base_year      ON ghg_base_year.gl_by_verifications(base_year_id);
CREATE INDEX idx_p045_ver_package        ON ghg_base_year.gl_by_verifications(package_id);
CREATE INDEX idx_p045_ver_type           ON ghg_base_year.gl_by_verifications(verification_type);
CREATE INDEX idx_p045_ver_assurance      ON ghg_base_year.gl_by_verifications(assurance_level);
CREATE INDEX idx_p045_ver_opinion        ON ghg_base_year.gl_by_verifications(opinion);
CREATE INDEX idx_p045_ver_date           ON ghg_base_year.gl_by_verifications(verification_date);
CREATE INDEX idx_p045_ver_verifier       ON ghg_base_year.gl_by_verifications(verifier_name);
CREATE INDEX idx_p045_ver_created        ON ghg_base_year.gl_by_verifications(created_at DESC);
CREATE INDEX idx_p045_ver_findings       ON ghg_base_year.gl_by_verifications USING GIN(findings_json);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_ver_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_verifications
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Deferred trigger from V369: Now that gl_by_audit_trail exists, create the
-- DB trigger that logs trigger status changes to the audit trail.
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_trigger_status_audit
    AFTER UPDATE ON ghg_base_year.gl_by_triggers
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_trigger_status_audit();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_verifications ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_at_tenant_isolation
    ON ghg_base_year.gl_by_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_at_service_bypass
    ON ghg_base_year.gl_by_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_appr_tenant_isolation
    ON ghg_base_year.gl_by_approvals
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_appr_service_bypass
    ON ghg_base_year.gl_by_approvals
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_ver_tenant_isolation
    ON ghg_base_year.gl_by_verifications
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_ver_service_bypass
    ON ghg_base_year.gl_by_verifications
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON ghg_base_year.gl_by_audit_trail TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_approvals TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_verifications TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_audit_trail IS
    'Append-only audit log for all base year management events with before/after values, actors, and SHA-256 provenance hashes.';
COMMENT ON TABLE ghg_base_year.gl_by_approvals IS
    'Approval workflow records for base year changes, recalculation packages, policy modifications, and target approvals.';
COMMENT ON TABLE ghg_base_year.gl_by_verifications IS
    'Third-party verification records with assurance level, opinion, findings, and recommendations per ISO 14064-3.';

COMMENT ON COLUMN ghg_base_year.gl_by_audit_trail.provenance_hash IS 'SHA-256 hash for tamper-evident audit integrity. Links to source data.';
COMMENT ON COLUMN ghg_base_year.gl_by_audit_trail.event_type IS 'Categorised event type for filtering and reporting (e.g., ADJUSTMENT_APPROVED, TRIGGER_DETECTED).';
COMMENT ON COLUMN ghg_base_year.gl_by_approvals.subject_type IS 'Type of item requiring approval: BASE_YEAR, ADJUSTMENT_PACKAGE, POLICY_CHANGE, TARGET, etc.';
COMMENT ON COLUMN ghg_base_year.gl_by_verifications.assurance_level IS 'Level of assurance: LIMITED_ASSURANCE, REASONABLE_ASSURANCE, or AGREED_UPON_PROCEDURES.';
COMMENT ON COLUMN ghg_base_year.gl_by_verifications.opinion IS 'Verifier opinion: UNQUALIFIED (clean), QUALIFIED, ADVERSE, DISCLAIMER, etc.';
