-- =============================================================================
-- V209: PACK-029 Interim Targets Pack - SBTi Submissions
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    014 of 015
-- Date:         March 2026
--
-- SBTi submission tracking with submission dates, interim target details,
-- SBTi validation status (pending/approved/rejected), feedback management,
-- and resubmission workflow for science-based target validation.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_sbti_submissions
--
-- Previous: V208__PACK029_trend_forecasts.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_sbti_submissions
-- =============================================================================
-- SBTi submission records with target details, submission workflow,
-- validation outcomes, feedback tracking, and resubmission management
-- for science-based target initiative validation process.

CREATE TABLE pack029_interim_targets.gl_sbti_submissions (
    submission_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Submission context
    submission_date             DATE            NOT NULL,
    submission_type             VARCHAR(30)     NOT NULL DEFAULT 'INITIAL',
    submission_version          INTEGER         DEFAULT 1,
    submission_reference        VARCHAR(50),
    -- Target details
    interim_target_year         INTEGER         NOT NULL,
    scope                       VARCHAR(20)     NOT NULL,
    reduction_pct               DECIMAL(8,4)    NOT NULL,
    base_year                   INTEGER         NOT NULL,
    base_year_emissions_tco2e   DECIMAL(18,4)   NOT NULL,
    target_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    -- SBTi specifics
    sbti_pathway                VARCHAR(20)     NOT NULL DEFAULT '1_5C',
    sbti_method                 VARCHAR(30)     NOT NULL DEFAULT 'ABSOLUTE_CONTRACTION',
    sbti_sector                 VARCHAR(50),
    near_term_target            BOOLEAN         DEFAULT TRUE,
    long_term_target            BOOLEAN         DEFAULT FALSE,
    net_zero_target             BOOLEAN         DEFAULT FALSE,
    -- Coverage
    scope1_coverage_pct         DECIMAL(5,2),
    scope2_coverage_pct         DECIMAL(5,2),
    scope3_coverage_pct         DECIMAL(5,2),
    scope3_categories_included  INTEGER[],
    -- Validation status
    sbti_validation_status      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    validation_date             DATE,
    validator_name              VARCHAR(255),
    validation_reference        VARCHAR(50),
    -- Feedback
    sbti_feedback               TEXT,
    feedback_date               DATE,
    feedback_category           VARCHAR(30),
    feedback_items              JSONB           DEFAULT '[]',
    -- Resubmission
    resubmission_required       BOOLEAN         DEFAULT FALSE,
    resubmission_deadline       DATE,
    resubmission_reason         TEXT,
    resubmission_count          INTEGER         DEFAULT 0,
    prior_submission_id         UUID,
    -- Response tracking
    response_submitted          BOOLEAN         DEFAULT FALSE,
    response_date               DATE,
    response_details            TEXT,
    -- Commitment
    commitment_letter_signed    BOOLEAN         DEFAULT FALSE,
    commitment_date             DATE,
    commitment_reference        VARCHAR(50),
    -- Publication
    target_published            BOOLEAN         DEFAULT FALSE,
    publication_date            DATE,
    sbti_website_listed         BOOLEAN         DEFAULT FALSE,
    -- Timeline
    expected_review_weeks       INTEGER         DEFAULT 8,
    actual_review_weeks         INTEGER,
    target_approval_date        DATE,
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_latest                   BOOLEAN         DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_ss_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_ss_reduction_pct CHECK (
        reduction_pct >= 0 AND reduction_pct <= 100
    ),
    CONSTRAINT chk_p029_ss_interim_year CHECK (
        interim_target_year >= 2025 AND interim_target_year <= 2100
    ),
    CONSTRAINT chk_p029_ss_base_year CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_p029_ss_target_after_base CHECK (
        interim_target_year > base_year
    ),
    CONSTRAINT chk_p029_ss_base_emissions CHECK (
        base_year_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ss_target_emissions CHECK (
        target_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ss_submission_type CHECK (
        submission_type IN ('INITIAL', 'RESUBMISSION', 'UPDATE', 'RECALCULATION', 'EXTENSION')
    ),
    CONSTRAINT chk_p029_ss_sbti_pathway CHECK (
        sbti_pathway IN ('1_5C', 'WB2C', '2C', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_ss_sbti_method CHECK (
        sbti_method IN ('ABSOLUTE_CONTRACTION', 'SECTORAL_DECARBONIZATION',
                        'ECONOMIC_INTENSITY', 'PHYSICAL_INTENSITY', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_ss_validation_status CHECK (
        sbti_validation_status IN ('DRAFT', 'PREPARING', 'SUBMITTED', 'UNDER_REVIEW',
                                   'ADDITIONAL_INFO_REQUESTED', 'APPROVED', 'REJECTED',
                                   'WITHDRAWN', 'EXPIRED')
    ),
    CONSTRAINT chk_p029_ss_feedback_category CHECK (
        feedback_category IS NULL OR feedback_category IN (
            'AMBITION', 'SCOPE_COVERAGE', 'METHODOLOGY', 'BASE_YEAR',
            'BOUNDARY', 'DOCUMENTATION', 'CALCULATION', 'TIMEFRAME', 'OTHER'
        )
    ),
    CONSTRAINT chk_p029_ss_coverage CHECK (
        (scope1_coverage_pct IS NULL OR (scope1_coverage_pct >= 0 AND scope1_coverage_pct <= 100))
        AND (scope2_coverage_pct IS NULL OR (scope2_coverage_pct >= 0 AND scope2_coverage_pct <= 100))
        AND (scope3_coverage_pct IS NULL OR (scope3_coverage_pct >= 0 AND scope3_coverage_pct <= 100))
    ),
    CONSTRAINT chk_p029_ss_resubmission_count CHECK (
        resubmission_count >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ss_tenant             ON pack029_interim_targets.gl_sbti_submissions(tenant_id);
CREATE INDEX idx_p029_ss_org                ON pack029_interim_targets.gl_sbti_submissions(organization_id);
CREATE INDEX idx_p029_ss_target             ON pack029_interim_targets.gl_sbti_submissions(target_id);
CREATE INDEX idx_p029_ss_org_submission     ON pack029_interim_targets.gl_sbti_submissions(organization_id, submission_date DESC);
CREATE INDEX idx_p029_ss_validation_status  ON pack029_interim_targets.gl_sbti_submissions(sbti_validation_status);
CREATE INDEX idx_p029_ss_pending            ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE sbti_validation_status IN ('SUBMITTED', 'UNDER_REVIEW');
CREATE INDEX idx_p029_ss_approved           ON pack029_interim_targets.gl_sbti_submissions(organization_id, validation_date DESC) WHERE sbti_validation_status = 'APPROVED';
CREATE INDEX idx_p029_ss_rejected           ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE sbti_validation_status = 'REJECTED';
CREATE INDEX idx_p029_ss_resubmission       ON pack029_interim_targets.gl_sbti_submissions(organization_id, resubmission_deadline) WHERE resubmission_required = TRUE;
CREATE INDEX idx_p029_ss_info_requested     ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE sbti_validation_status = 'ADDITIONAL_INFO_REQUESTED';
CREATE INDEX idx_p029_ss_submission_type    ON pack029_interim_targets.gl_sbti_submissions(submission_type);
CREATE INDEX idx_p029_ss_sbti_pathway       ON pack029_interim_targets.gl_sbti_submissions(sbti_pathway);
CREATE INDEX idx_p029_ss_sbti_method        ON pack029_interim_targets.gl_sbti_submissions(sbti_method);
CREATE INDEX idx_p029_ss_latest             ON pack029_interim_targets.gl_sbti_submissions(organization_id, scope) WHERE is_latest = TRUE;
CREATE INDEX idx_p029_ss_near_term          ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE near_term_target = TRUE;
CREATE INDEX idx_p029_ss_long_term          ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE long_term_target = TRUE;
CREATE INDEX idx_p029_ss_prior_submission   ON pack029_interim_targets.gl_sbti_submissions(prior_submission_id) WHERE prior_submission_id IS NOT NULL;
CREATE INDEX idx_p029_ss_active             ON pack029_interim_targets.gl_sbti_submissions(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_ss_created            ON pack029_interim_targets.gl_sbti_submissions(created_at DESC);
CREATE INDEX idx_p029_ss_feedback_items     ON pack029_interim_targets.gl_sbti_submissions USING GIN(feedback_items);
CREATE INDEX idx_p029_ss_s3_categories      ON pack029_interim_targets.gl_sbti_submissions USING GIN(scope3_categories_included);
CREATE INDEX idx_p029_ss_metadata           ON pack029_interim_targets.gl_sbti_submissions USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_sbti_submissions_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_sbti_submissions
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_sbti_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_ss_tenant_isolation
    ON pack029_interim_targets.gl_sbti_submissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_ss_service_bypass
    ON pack029_interim_targets.gl_sbti_submissions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_sbti_submissions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_sbti_submissions IS
    'SBTi submission tracking with submission workflow, validation outcomes (pending/approved/rejected), feedback management, resubmission workflow, and commitment/publication tracking for science-based target validation.';

COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.submission_id IS 'Unique SBTi submission identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.organization_id IS 'Reference to the submitting organization.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.submission_date IS 'Date when the submission was made to SBTi.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.interim_target_year IS 'The interim target year being submitted for validation.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.reduction_pct IS 'Reduction percentage from baseline submitted for validation.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.sbti_pathway IS 'SBTi temperature pathway: 1_5C, WB2C, 2C, CUSTOM.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.sbti_validation_status IS 'SBTi validation status: DRAFT through APPROVED/REJECTED/EXPIRED.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.sbti_feedback IS 'Text feedback received from SBTi.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.resubmission_required IS 'Whether resubmission is required based on SBTi feedback.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.commitment_letter_signed IS 'Whether the SBTi commitment letter has been signed.';
COMMENT ON COLUMN pack029_interim_targets.gl_sbti_submissions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
