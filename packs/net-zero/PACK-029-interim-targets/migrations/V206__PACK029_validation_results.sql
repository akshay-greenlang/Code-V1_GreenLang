-- =============================================================================
-- V206: PACK-029 Interim Targets Pack - Validation Results
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    011 of 015
-- Date:         March 2026
--
-- SBTi validation results for the 21 criteria of the SBTi Corporate Standard
-- with per-criterion pass/fail status, warning flags, validation messages,
-- and interim target year association.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_validation_results
--
-- Previous: V205__PACK029_reporting_periods.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_validation_results
-- =============================================================================
-- SBTi validation results per criterion (21 criteria per SBTi Corporate
-- Standard) with pass/fail status, warning flags, detailed validation
-- messages, and interim target year association.

CREATE TABLE pack029_interim_targets.gl_validation_results (
    result_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Validation context
    validation_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    validation_run_id           UUID,
    validation_version          VARCHAR(20)     DEFAULT '2.1',
    validation_type             VARCHAR(30)     DEFAULT 'AUTOMATED',
    -- SBTi criterion
    sbti_criteria_id            INTEGER         NOT NULL,
    criterion_name              VARCHAR(200)    NOT NULL,
    criterion_category          VARCHAR(50),
    criterion_description       TEXT,
    criterion_reference         VARCHAR(100),
    -- Target context
    interim_target_year         INTEGER,
    scope                       VARCHAR(20),
    -- Result
    pass_fail_status            VARCHAR(10)     NOT NULL,
    warning_flag                BOOLEAN         DEFAULT FALSE,
    warning_message             TEXT,
    validation_message          TEXT            NOT NULL,
    -- Details
    expected_value              VARCHAR(200),
    actual_value                VARCHAR(200),
    threshold_value             VARCHAR(200),
    gap_value                   VARCHAR(200),
    -- Severity
    severity                    VARCHAR(20)     DEFAULT 'STANDARD',
    is_mandatory                BOOLEAN         DEFAULT TRUE,
    is_blocking                 BOOLEAN         DEFAULT FALSE,
    -- Remediation
    remediation_required        BOOLEAN         DEFAULT FALSE,
    remediation_guidance        TEXT,
    remediation_deadline        DATE,
    remediation_status          VARCHAR(20)     DEFAULT 'NOT_REQUIRED',
    -- Historical comparison
    prior_validation_result     VARCHAR(10),
    result_changed              BOOLEAN         DEFAULT FALSE,
    improvement                 BOOLEAN,
    -- Confidence
    confidence_score            DECIMAL(5,2),
    automated_check             BOOLEAN         DEFAULT TRUE,
    manual_override             BOOLEAN         DEFAULT FALSE,
    override_by                 VARCHAR(255),
    override_reason             TEXT,
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_vr_criteria_id CHECK (
        sbti_criteria_id >= 1 AND sbti_criteria_id <= 21
    ),
    CONSTRAINT chk_p029_vr_pass_fail CHECK (
        pass_fail_status IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE', 'PENDING')
    ),
    CONSTRAINT chk_p029_vr_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_vr_target_year CHECK (
        interim_target_year IS NULL OR (interim_target_year >= 2025 AND interim_target_year <= 2100)
    ),
    CONSTRAINT chk_p029_vr_severity CHECK (
        severity IN ('CRITICAL', 'MAJOR', 'STANDARD', 'MINOR', 'INFORMATIONAL')
    ),
    CONSTRAINT chk_p029_vr_validation_type CHECK (
        validation_type IN ('AUTOMATED', 'MANUAL', 'SBTI_OFFICIAL', 'PRE_SUBMISSION', 'HYBRID')
    ),
    CONSTRAINT chk_p029_vr_criterion_category CHECK (
        criterion_category IS NULL OR criterion_category IN (
            'TIMEFRAME', 'AMBITION', 'SCOPE_COVERAGE', 'BOUNDARY',
            'BASE_YEAR', 'TARGET_YEAR', 'METHODOLOGY', 'DISCLOSURE',
            'RECALCULATION', 'REPORTING', 'GENERAL'
        )
    ),
    CONSTRAINT chk_p029_vr_remediation_status CHECK (
        remediation_status IN ('NOT_REQUIRED', 'REQUIRED', 'IN_PROGRESS', 'COMPLETED', 'WAIVED')
    ),
    CONSTRAINT chk_p029_vr_confidence CHECK (
        confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 100)
    ),
    CONSTRAINT chk_p029_vr_prior_result CHECK (
        prior_validation_result IS NULL OR prior_validation_result IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_vr_tenant             ON pack029_interim_targets.gl_validation_results(tenant_id);
CREATE INDEX idx_p029_vr_org                ON pack029_interim_targets.gl_validation_results(organization_id);
CREATE INDEX idx_p029_vr_target             ON pack029_interim_targets.gl_validation_results(target_id);
CREATE INDEX idx_p029_vr_org_date           ON pack029_interim_targets.gl_validation_results(organization_id, validation_date DESC);
CREATE INDEX idx_p029_vr_org_criteria       ON pack029_interim_targets.gl_validation_results(organization_id, sbti_criteria_id);
CREATE INDEX idx_p029_vr_pass_fail          ON pack029_interim_targets.gl_validation_results(pass_fail_status);
CREATE INDEX idx_p029_vr_failed             ON pack029_interim_targets.gl_validation_results(organization_id, validation_date) WHERE pass_fail_status = 'FAIL';
CREATE INDEX idx_p029_vr_warnings           ON pack029_interim_targets.gl_validation_results(organization_id) WHERE warning_flag = TRUE;
CREATE INDEX idx_p029_vr_run_id             ON pack029_interim_targets.gl_validation_results(validation_run_id) WHERE validation_run_id IS NOT NULL;
CREATE INDEX idx_p029_vr_criteria_category  ON pack029_interim_targets.gl_validation_results(criterion_category);
CREATE INDEX idx_p029_vr_remediation        ON pack029_interim_targets.gl_validation_results(organization_id, remediation_status) WHERE remediation_required = TRUE;
CREATE INDEX idx_p029_vr_remediation_due    ON pack029_interim_targets.gl_validation_results(remediation_deadline) WHERE remediation_status IN ('REQUIRED', 'IN_PROGRESS');
CREATE INDEX idx_p029_vr_blocking           ON pack029_interim_targets.gl_validation_results(organization_id) WHERE is_blocking = TRUE AND pass_fail_status = 'FAIL';
CREATE INDEX idx_p029_vr_overridden         ON pack029_interim_targets.gl_validation_results(organization_id) WHERE manual_override = TRUE;
CREATE INDEX idx_p029_vr_active             ON pack029_interim_targets.gl_validation_results(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_vr_created            ON pack029_interim_targets.gl_validation_results(created_at DESC);
CREATE INDEX idx_p029_vr_metadata           ON pack029_interim_targets.gl_validation_results USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_validation_results_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_validation_results
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_validation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_vr_tenant_isolation
    ON pack029_interim_targets.gl_validation_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_vr_service_bypass
    ON pack029_interim_targets.gl_validation_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_validation_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_validation_results IS
    'SBTi validation results for the 21 criteria of the SBTi Corporate Standard with per-criterion pass/fail status, warning flags, remediation tracking, and historical comparison for interim target validation.';

COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.result_id IS 'Unique validation result identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.organization_id IS 'Reference to the organization being validated.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.validation_date IS 'Date when the validation was performed.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.sbti_criteria_id IS 'SBTi criterion number (1-21 per Corporate Standard).';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.criterion_name IS 'Human-readable name of the SBTi criterion.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.pass_fail_status IS 'Validation result: PASS, FAIL, WARNING, NOT_APPLICABLE, PENDING.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.warning_flag IS 'Whether a warning was raised during validation.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.validation_message IS 'Detailed validation message with explanation.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.remediation_required IS 'Whether corrective action is needed to pass this criterion.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.is_blocking IS 'Whether failure of this criterion blocks SBTi submission.';
COMMENT ON COLUMN pack029_interim_targets.gl_validation_results.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
