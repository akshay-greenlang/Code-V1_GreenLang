-- =============================================================================
-- V360: PACK-044 GHG Inventory Management - Review & Approval Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Multi-level review and approval workflow tables for GHG inventory data.
-- Supports configurable review levels (1-N), reviewer assignment by role,
-- sequential or parallel review, and formal sign-off. Review comments
-- provide a threaded discussion capability. Approval records capture the
-- final decision and any conditions imposed.
--
-- Tables (3):
--   1. ghg_inventory.gl_inv_review_requests
--   2. ghg_inventory.gl_inv_review_comments
--   3. ghg_inventory.gl_inv_approval_records
--
-- Previous: V359__pack044_change_management.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_review_requests
-- =============================================================================
-- A request for review of inventory data at a specific level. Each review
-- request targets a scope (period, facility, source category) and assigns
-- a reviewer. Tracks the review lifecycle from REQUESTED through to
-- COMPLETED or RETURNED.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_review_requests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    review_level                INTEGER         NOT NULL DEFAULT 1,
    review_scope                VARCHAR(30)     NOT NULL DEFAULT 'PERIOD',
    facility_id                 UUID,
    source_category             VARCHAR(60),
    reviewer_user_id            UUID,
    reviewer_name               VARCHAR(255)    NOT NULL,
    reviewer_role               VARCHAR(100),
    requested_by_user_id        UUID,
    requested_by_name           VARCHAR(255),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'REQUESTED',
    requested_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    due_date                    DATE,
    started_at                  TIMESTAMPTZ,
    completed_at                TIMESTAMPTZ,
    outcome                     VARCHAR(30),
    outcome_notes               TEXT,
    total_comments              INTEGER         NOT NULL DEFAULT 0,
    total_issues_raised         INTEGER         NOT NULL DEFAULT 0,
    total_issues_resolved       INTEGER         NOT NULL DEFAULT 0,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_rr_level CHECK (
        review_level >= 1 AND review_level <= 10
    ),
    CONSTRAINT chk_p044_rr_scope CHECK (
        review_scope IN ('PERIOD', 'FACILITY', 'SOURCE_CATEGORY', 'ENTITY', 'CAMPAIGN')
    ),
    CONSTRAINT chk_p044_rr_status CHECK (
        status IN (
            'REQUESTED', 'ASSIGNED', 'IN_PROGRESS', 'COMPLETED',
            'RETURNED', 'CANCELLED', 'OVERDUE'
        )
    ),
    CONSTRAINT chk_p044_rr_outcome CHECK (
        outcome IS NULL OR outcome IN (
            'APPROVED', 'APPROVED_WITH_COMMENTS', 'RETURNED_FOR_REVISION',
            'REJECTED', 'ESCALATED'
        )
    ),
    CONSTRAINT chk_p044_rr_issues CHECK (
        total_issues_raised >= 0 AND total_issues_resolved >= 0 AND
        total_issues_resolved <= total_issues_raised
    ),
    CONSTRAINT chk_p044_rr_comments CHECK (
        total_comments >= 0
    ),
    CONSTRAINT chk_p044_rr_times CHECK (
        started_at IS NULL OR completed_at IS NULL OR started_at <= completed_at
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_rr_tenant          ON ghg_inventory.gl_inv_review_requests(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_rr_period          ON ghg_inventory.gl_inv_review_requests(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_rr_level           ON ghg_inventory.gl_inv_review_requests(review_level);
CREATE INDEX IF NOT EXISTS idx_p044_rr_scope           ON ghg_inventory.gl_inv_review_requests(review_scope);
CREATE INDEX IF NOT EXISTS idx_p044_rr_facility        ON ghg_inventory.gl_inv_review_requests(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_rr_reviewer        ON ghg_inventory.gl_inv_review_requests(reviewer_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_rr_status          ON ghg_inventory.gl_inv_review_requests(status);
CREATE INDEX IF NOT EXISTS idx_p044_rr_outcome         ON ghg_inventory.gl_inv_review_requests(outcome);
CREATE INDEX IF NOT EXISTS idx_p044_rr_due             ON ghg_inventory.gl_inv_review_requests(due_date);
CREATE INDEX IF NOT EXISTS idx_p044_rr_created         ON ghg_inventory.gl_inv_review_requests(created_at DESC);

-- Composite: period + open reviews
CREATE INDEX IF NOT EXISTS idx_p044_rr_period_open     ON ghg_inventory.gl_inv_review_requests(period_id, review_level)
    WHERE status IN ('REQUESTED', 'ASSIGNED', 'IN_PROGRESS');

-- Composite: reviewer + pending reviews
CREATE INDEX IF NOT EXISTS idx_p044_rr_reviewer_open   ON ghg_inventory.gl_inv_review_requests(reviewer_user_id, due_date)
    WHERE status IN ('REQUESTED', 'ASSIGNED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_rr_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_review_requests
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_review_comments
-- =============================================================================
-- Threaded comments on review requests. Reviewers can add comments requesting
-- clarification, flagging issues, or providing guidance. Data owners can
-- respond. Comments may be linked to specific facilities, source categories,
-- or data submissions.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_review_comments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    review_request_id           UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_review_requests(id) ON DELETE CASCADE,
    parent_comment_id           UUID            REFERENCES ghg_inventory.gl_inv_review_comments(id) ON DELETE SET NULL,
    author_user_id              UUID,
    author_name                 VARCHAR(255)    NOT NULL,
    author_role                 VARCHAR(100),
    comment_type                VARCHAR(30)     NOT NULL DEFAULT 'GENERAL',
    comment_text                TEXT            NOT NULL,
    facility_id                 UUID,
    source_category             VARCHAR(60),
    submission_id               UUID,
    is_issue                    BOOLEAN         NOT NULL DEFAULT false,
    issue_resolved              BOOLEAN         NOT NULL DEFAULT false,
    issue_resolved_at           TIMESTAMPTZ,
    issue_resolved_by           VARCHAR(255),
    attachments                 UUID[],
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_rc_type CHECK (
        comment_type IN (
            'GENERAL', 'CLARIFICATION', 'ISSUE', 'RESPONSE',
            'APPROVAL_NOTE', 'REJECTION_REASON', 'GUIDANCE'
        )
    ),
    CONSTRAINT chk_p044_rc_issue_resolved CHECK (
        is_issue = true OR issue_resolved = false
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_rc_tenant          ON ghg_inventory.gl_inv_review_comments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_rc_review          ON ghg_inventory.gl_inv_review_comments(review_request_id);
CREATE INDEX IF NOT EXISTS idx_p044_rc_parent          ON ghg_inventory.gl_inv_review_comments(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_p044_rc_author          ON ghg_inventory.gl_inv_review_comments(author_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_rc_type            ON ghg_inventory.gl_inv_review_comments(comment_type);
CREATE INDEX IF NOT EXISTS idx_p044_rc_facility        ON ghg_inventory.gl_inv_review_comments(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_rc_issue           ON ghg_inventory.gl_inv_review_comments(is_issue) WHERE is_issue = true;
CREATE INDEX IF NOT EXISTS idx_p044_rc_unresolved      ON ghg_inventory.gl_inv_review_comments(review_request_id)
    WHERE is_issue = true AND issue_resolved = false;
CREATE INDEX IF NOT EXISTS idx_p044_rc_created         ON ghg_inventory.gl_inv_review_comments(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_rc_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_review_comments
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_approval_records
-- =============================================================================
-- Formal approval records for inventory periods. Captures the final sign-off
-- decision for each approval level, the approver's identity, conditions
-- imposed, and the digital signature or attestation reference.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_approval_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    approval_level              INTEGER         NOT NULL DEFAULT 1,
    approval_type               VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    approver_user_id            UUID,
    approver_name               VARCHAR(255)    NOT NULL,
    approver_role               VARCHAR(100),
    approver_title              VARCHAR(200),
    decision                    VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    decision_date               TIMESTAMPTZ,
    conditions                  TEXT,
    conditions_met              BOOLEAN,
    conditions_met_date         TIMESTAMPTZ,
    attestation_text            TEXT,
    digital_signature_ref       VARCHAR(200),
    total_scope1_tco2e          NUMERIC(14,3),
    total_scope2_tco2e          NUMERIC(14,3),
    total_scope3_tco2e          NUMERIC(14,3),
    total_tco2e                 NUMERIC(14,3),
    is_final_approval           BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ar_level CHECK (
        approval_level >= 1 AND approval_level <= 10
    ),
    CONSTRAINT chk_p044_ar_type CHECK (
        approval_type IN ('STANDARD', 'EXPEDITED', 'DELEGATED', 'BOARD_LEVEL', 'EXTERNAL_VERIFIER')
    ),
    CONSTRAINT chk_p044_ar_decision CHECK (
        decision IN ('PENDING', 'APPROVED', 'APPROVED_WITH_CONDITIONS', 'REJECTED', 'DEFERRED')
    ),
    CONSTRAINT chk_p044_ar_totals CHECK (
        total_scope1_tco2e IS NULL OR total_scope1_tco2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ar_tenant          ON ghg_inventory.gl_inv_approval_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ar_period          ON ghg_inventory.gl_inv_approval_records(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_ar_level           ON ghg_inventory.gl_inv_approval_records(approval_level);
CREATE INDEX IF NOT EXISTS idx_p044_ar_type            ON ghg_inventory.gl_inv_approval_records(approval_type);
CREATE INDEX IF NOT EXISTS idx_p044_ar_approver        ON ghg_inventory.gl_inv_approval_records(approver_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_ar_decision        ON ghg_inventory.gl_inv_approval_records(decision);
CREATE INDEX IF NOT EXISTS idx_p044_ar_final           ON ghg_inventory.gl_inv_approval_records(is_final_approval) WHERE is_final_approval = true;
CREATE INDEX IF NOT EXISTS idx_p044_ar_created         ON ghg_inventory.gl_inv_approval_records(created_at DESC);

-- Composite: period + pending approvals
CREATE INDEX IF NOT EXISTS idx_p044_ar_period_pending  ON ghg_inventory.gl_inv_approval_records(period_id, approval_level)
    WHERE decision = 'PENDING';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ar_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_approval_records
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_review_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_review_comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_approval_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_rr_tenant_isolation
    ON ghg_inventory.gl_inv_review_requests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_rr_service_bypass
    ON ghg_inventory.gl_inv_review_requests
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_rc_tenant_isolation
    ON ghg_inventory.gl_inv_review_comments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_rc_service_bypass
    ON ghg_inventory.gl_inv_review_comments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ar_tenant_isolation
    ON ghg_inventory.gl_inv_approval_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ar_service_bypass
    ON ghg_inventory.gl_inv_approval_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_review_requests TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_review_comments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_approval_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_review_requests IS
    'Multi-level review requests for GHG inventory data with configurable review scopes and reviewer assignment.';
COMMENT ON TABLE ghg_inventory.gl_inv_review_comments IS
    'Threaded comments on review requests with issue tracking and resolution.';
COMMENT ON TABLE ghg_inventory.gl_inv_approval_records IS
    'Formal approval records for inventory periods with digital attestation and sign-off.';

COMMENT ON COLUMN ghg_inventory.gl_inv_review_requests.review_level IS 'Review hierarchy level (1=first reviewer, 2=second reviewer, etc.).';
COMMENT ON COLUMN ghg_inventory.gl_inv_review_requests.review_scope IS 'What is being reviewed: PERIOD (full period), FACILITY (single site), SOURCE_CATEGORY, ENTITY, CAMPAIGN.';
COMMENT ON COLUMN ghg_inventory.gl_inv_review_requests.outcome IS 'Review outcome: APPROVED, APPROVED_WITH_COMMENTS, RETURNED_FOR_REVISION, REJECTED, ESCALATED.';
COMMENT ON COLUMN ghg_inventory.gl_inv_review_comments.is_issue IS 'Whether this comment raises a formal issue requiring resolution before approval.';
COMMENT ON COLUMN ghg_inventory.gl_inv_approval_records.digital_signature_ref IS 'Reference to digital signature or e-signature service attestation.';
COMMENT ON COLUMN ghg_inventory.gl_inv_approval_records.is_final_approval IS 'Whether this is the final approval level that triggers period status advancement.';
