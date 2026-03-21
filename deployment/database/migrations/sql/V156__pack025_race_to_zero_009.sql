-- =============================================================================
-- V156: PACK-025 Race to Zero - Campaign Submissions
-- =============================================================================
-- Pack:         PACK-025 (Race to Zero Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Campaign submission packages with pledge letters, starting line proof,
-- action plan summaries, and approval tracking. Verification schedule
-- management with due dates, verifier assignments, and completion tracking.
--
-- Tables (2):
--   1. pack025_race_to_zero.campaign_submissions
--   2. pack025_race_to_zero.verification_schedules
--
-- Previous: V155__pack025_race_to_zero_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack025_race_to_zero.campaign_submissions
-- =============================================================================
-- Campaign submission packages containing pledge letters, starting line
-- proof, action plan summaries, and verification schedules.

CREATE TABLE pack025_race_to_zero.campaign_submissions (
    submission_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    pledge_id               UUID            NOT NULL REFERENCES pack025_race_to_zero.pledges(pledge_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    submission_date         DATE            NOT NULL,
    -- Package contents
    pledge_letter           TEXT,
    pledge_letter_url       TEXT,
    starting_line_proof     TEXT,
    starting_line_proof_url TEXT,
    action_plan_summary     TEXT,
    action_plan_url         TEXT,
    verification_schedule   TEXT,
    supporting_documents    JSONB           DEFAULT '[]',
    package_completeness    DECIMAL(6,2),
    -- Approval workflow
    approval_status         VARCHAR(30)     NOT NULL DEFAULT 'draft',
    reviewer_id             UUID,
    review_date             DATE,
    review_notes            TEXT,
    approved_date           DATE,
    approved_by             UUID,
    rejection_reason        TEXT,
    -- Badge
    verification_badge      VARCHAR(50),
    badge_valid_from        DATE,
    badge_valid_until       DATE,
    -- Submission channel
    submission_channel      VARCHAR(100),
    partner_initiative      VARCHAR(100),
    reporting_year          INTEGER,
    acknowledgment_received BOOLEAN         DEFAULT FALSE,
    acknowledgment_date     DATE,
    feedback                TEXT,
    warnings                TEXT[]          DEFAULT '{}',
    errors                  TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_cs_approval CHECK (
        approval_status IN ('draft', 'submitted', 'under_review', 'approved', 'rejected',
                            'resubmission_required', 'withdrawn')
    ),
    CONSTRAINT chk_p025_cs_badge CHECK (
        verification_badge IS NULL OR verification_badge IN (
            'RACE_TO_ZERO_PARTICIPANT', 'RACE_TO_ZERO_LEADER',
            'RACE_TO_ZERO_ACCELERATOR', 'NONE'
        )
    ),
    CONSTRAINT chk_p025_cs_completeness CHECK (
        package_completeness IS NULL OR (package_completeness >= 0 AND package_completeness <= 100)
    ),
    CONSTRAINT chk_p025_cs_badge_dates CHECK (
        badge_valid_from IS NULL OR badge_valid_until IS NULL OR badge_valid_from <= badge_valid_until
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for campaign_submissions
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_cs_org             ON pack025_race_to_zero.campaign_submissions(org_id);
CREATE INDEX idx_p025_cs_pledge          ON pack025_race_to_zero.campaign_submissions(pledge_id);
CREATE INDEX idx_p025_cs_tenant          ON pack025_race_to_zero.campaign_submissions(tenant_id);
CREATE INDEX idx_p025_cs_date            ON pack025_race_to_zero.campaign_submissions(submission_date);
CREATE INDEX idx_p025_cs_approval        ON pack025_race_to_zero.campaign_submissions(approval_status);
CREATE INDEX idx_p025_cs_badge           ON pack025_race_to_zero.campaign_submissions(verification_badge);
CREATE INDEX idx_p025_cs_approved_date   ON pack025_race_to_zero.campaign_submissions(approved_date);
CREATE INDEX idx_p025_cs_partner         ON pack025_race_to_zero.campaign_submissions(partner_initiative);
CREATE INDEX idx_p025_cs_year            ON pack025_race_to_zero.campaign_submissions(reporting_year);
CREATE INDEX idx_p025_cs_created         ON pack025_race_to_zero.campaign_submissions(created_at DESC);
CREATE INDEX idx_p025_cs_docs            ON pack025_race_to_zero.campaign_submissions USING GIN(supporting_documents);
CREATE INDEX idx_p025_cs_metadata        ON pack025_race_to_zero.campaign_submissions USING GIN(metadata);

-- =============================================================================
-- Table 2: pack025_race_to_zero.verification_schedules
-- =============================================================================
-- Verification schedule management with annual due dates, verifier
-- assignments, and completion tracking.

CREATE TABLE pack025_race_to_zero.verification_schedules (
    schedule_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES pack025_race_to_zero.organization_profiles(org_id) ON DELETE CASCADE,
    submission_id           UUID            REFERENCES pack025_race_to_zero.campaign_submissions(submission_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    verification_year       INTEGER         NOT NULL,
    due_date                DATE            NOT NULL,
    -- Verifier details
    verifier_assigned       VARCHAR(255),
    verifier_accreditation  VARCHAR(255),
    verifier_contact        VARCHAR(255),
    assurance_level         VARCHAR(30)     DEFAULT 'LIMITED',
    -- Status tracking
    status                  VARCHAR(30)     NOT NULL DEFAULT 'scheduled',
    engagement_date         DATE,
    fieldwork_start         DATE,
    fieldwork_end           DATE,
    completion_date         DATE,
    report_received_date    DATE,
    verification_opinion    VARCHAR(30),
    -- Findings
    findings_summary        TEXT,
    material_findings       JSONB           DEFAULT '[]',
    non_conformities        INTEGER         DEFAULT 0,
    observations            INTEGER         DEFAULT 0,
    corrective_actions      JSONB           DEFAULT '[]',
    -- Costs
    estimated_cost_usd      DECIMAL(18,2),
    actual_cost_usd         DECIMAL(18,2),
    -- Documents
    verification_report_url TEXT,
    statement_url           TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p025_vs_year CHECK (
        verification_year >= 2020 AND verification_year <= 2100
    ),
    CONSTRAINT chk_p025_vs_status CHECK (
        status IN ('scheduled', 'engaged', 'in_progress', 'completed', 'cancelled', 'overdue')
    ),
    CONSTRAINT chk_p025_vs_assurance CHECK (
        assurance_level IN ('LIMITED', 'REASONABLE', 'NONE')
    ),
    CONSTRAINT chk_p025_vs_opinion CHECK (
        verification_opinion IS NULL OR verification_opinion IN (
            'UNQUALIFIED', 'QUALIFIED', 'ADVERSE', 'DISCLAIMER'
        )
    ),
    CONSTRAINT chk_p025_vs_cost_non_neg CHECK (
        (estimated_cost_usd IS NULL OR estimated_cost_usd >= 0) AND
        (actual_cost_usd IS NULL OR actual_cost_usd >= 0)
    ),
    CONSTRAINT uq_p025_vs_org_year UNIQUE (org_id, verification_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for verification_schedules
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p025_vs_org             ON pack025_race_to_zero.verification_schedules(org_id);
CREATE INDEX idx_p025_vs_submission      ON pack025_race_to_zero.verification_schedules(submission_id);
CREATE INDEX idx_p025_vs_tenant          ON pack025_race_to_zero.verification_schedules(tenant_id);
CREATE INDEX idx_p025_vs_year            ON pack025_race_to_zero.verification_schedules(verification_year);
CREATE INDEX idx_p025_vs_due             ON pack025_race_to_zero.verification_schedules(due_date);
CREATE INDEX idx_p025_vs_status          ON pack025_race_to_zero.verification_schedules(status);
CREATE INDEX idx_p025_vs_verifier        ON pack025_race_to_zero.verification_schedules(verifier_assigned);
CREATE INDEX idx_p025_vs_opinion         ON pack025_race_to_zero.verification_schedules(verification_opinion);
CREATE INDEX idx_p025_vs_created         ON pack025_race_to_zero.verification_schedules(created_at DESC);
CREATE INDEX idx_p025_vs_findings        ON pack025_race_to_zero.verification_schedules USING GIN(material_findings);
CREATE INDEX idx_p025_vs_corrective      ON pack025_race_to_zero.verification_schedules USING GIN(corrective_actions);
CREATE INDEX idx_p025_vs_metadata        ON pack025_race_to_zero.verification_schedules USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p025_campaign_submissions_updated
    BEFORE UPDATE ON pack025_race_to_zero.campaign_submissions
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

CREATE TRIGGER trg_p025_verification_schedules_updated
    BEFORE UPDATE ON pack025_race_to_zero.verification_schedules
    FOR EACH ROW EXECUTE FUNCTION pack025_race_to_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack025_race_to_zero.campaign_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack025_race_to_zero.verification_schedules ENABLE ROW LEVEL SECURITY;

CREATE POLICY p025_cs_tenant_isolation
    ON pack025_race_to_zero.campaign_submissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_cs_service_bypass
    ON pack025_race_to_zero.campaign_submissions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p025_vs_tenant_isolation
    ON pack025_race_to_zero.verification_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p025_vs_service_bypass
    ON pack025_race_to_zero.verification_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.campaign_submissions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack025_race_to_zero.verification_schedules TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack025_race_to_zero.campaign_submissions IS
    'Campaign submission packages with pledge letters, starting line proof, action plan summaries, and approval workflow.';
COMMENT ON TABLE pack025_race_to_zero.verification_schedules IS
    'Verification schedule management with annual due dates, verifier assignments, and completion tracking.';

COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.submission_id IS 'Unique submission package identifier.';
COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.pledge_letter IS 'Text of the pledge commitment letter.';
COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.starting_line_proof IS 'Evidence of Starting Line Criteria compliance.';
COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.action_plan_summary IS 'Summary of the climate action plan.';
COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.verification_badge IS 'Awarded verification badge type.';
COMMENT ON COLUMN pack025_race_to_zero.campaign_submissions.approved_date IS 'Date the submission was approved.';
COMMENT ON COLUMN pack025_race_to_zero.verification_schedules.schedule_id IS 'Unique verification schedule identifier.';
COMMENT ON COLUMN pack025_race_to_zero.verification_schedules.due_date IS 'Deadline for verification completion.';
COMMENT ON COLUMN pack025_race_to_zero.verification_schedules.verifier_assigned IS 'Assigned verification body name.';
COMMENT ON COLUMN pack025_race_to_zero.verification_schedules.completion_date IS 'Actual verification completion date.';
