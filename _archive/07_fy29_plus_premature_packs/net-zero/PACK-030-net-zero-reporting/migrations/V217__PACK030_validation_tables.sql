-- =============================================================================
-- V217: PACK-030 Net Zero Reporting Pack - Validation Results
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    007 of 015
-- Date:         March 2026
--
-- Validation results from schema validation, completeness checks,
-- cross-framework consistency validation, and quality scoring.
--
-- Tables (1):
--   1. pack030_nz_reporting.gl_nz_validation_results
--
-- Previous: V216__PACK030_xbrl_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_validation_results
-- =============================================================================
-- Validation results from automated checks including schema validation,
-- completeness assessment, cross-framework consistency, XBRL taxonomy
-- compliance, and quality scoring with resolution tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_validation_results (
    validation_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    -- Validation run
    validation_run_id           UUID            NOT NULL,
    validation_run_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Validator identification
    validator                   VARCHAR(100)    NOT NULL,
    validator_version           VARCHAR(30),
    -- Validation classification
    validation_type             VARCHAR(50)     NOT NULL,
    validation_category         VARCHAR(50)     NOT NULL,
    -- Result
    severity                    VARCHAR(20)     NOT NULL,
    result_status               VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    -- Issue details
    message                     TEXT            NOT NULL,
    message_code                VARCHAR(50),
    field_path                  VARCHAR(500),
    expected_value              TEXT,
    actual_value                TEXT,
    -- Context
    framework                   VARCHAR(50),
    section_type                VARCHAR(100),
    metric_name                 VARCHAR(200),
    -- Cross-framework
    related_framework           VARCHAR(50),
    related_report_id           UUID,
    inconsistency_description   TEXT,
    -- Resolution
    resolved                    BOOLEAN         NOT NULL DEFAULT FALSE,
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 UUID,
    resolution_method           VARCHAR(50),
    resolution_notes            TEXT,
    -- Auto-fix
    auto_fixable                BOOLEAN         NOT NULL DEFAULT FALSE,
    auto_fix_applied            BOOLEAN         NOT NULL DEFAULT FALSE,
    auto_fix_description        TEXT,
    -- Impact
    blocking                    BOOLEAN         NOT NULL DEFAULT FALSE,
    quality_impact              DECIMAL(5,2),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_vr_validator CHECK (
        validator IN (
            'SCHEMA_VALIDATOR', 'COMPLETENESS_CHECKER', 'CONSISTENCY_CHECKER',
            'XBRL_VALIDATOR', 'METRIC_RECONCILER', 'NARRATIVE_ANALYZER',
            'DEADLINE_CHECKER', 'DATA_QUALITY_CHECKER', 'CROSS_FRAMEWORK_CHECKER',
            'FORMAT_VALIDATOR', 'CITATION_CHECKER', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p030_vr_validation_type CHECK (
        validation_type IN ('ERROR', 'WARNING', 'INFO', 'SUGGESTION')
    ),
    CONSTRAINT chk_p030_vr_validation_category CHECK (
        validation_category IN (
            'SCHEMA', 'COMPLETENESS', 'CONSISTENCY', 'XBRL', 'RECONCILIATION',
            'NARRATIVE', 'DEADLINE', 'DATA_QUALITY', 'FORMAT', 'CITATION',
            'CROSS_FRAMEWORK', 'METRIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p030_vr_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')
    ),
    CONSTRAINT chk_p030_vr_result_status CHECK (
        result_status IN ('OPEN', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'WONT_FIX', 'FALSE_POSITIVE')
    ),
    CONSTRAINT chk_p030_vr_resolution_method CHECK (
        resolution_method IS NULL OR resolution_method IN (
            'MANUAL_FIX', 'AUTO_FIX', 'DATA_UPDATE', 'OVERRIDE', 'ACCEPTED', 'FALSE_POSITIVE'
        )
    ),
    CONSTRAINT chk_p030_vr_framework CHECK (
        framework IS NULL OR framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'MULTI_FRAMEWORK', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_vr_quality_impact CHECK (
        quality_impact IS NULL OR (quality_impact >= 0 AND quality_impact <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_validation_results
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_vr_tenant               ON pack030_nz_reporting.gl_nz_validation_results(tenant_id);
CREATE INDEX idx_p030_vr_org                  ON pack030_nz_reporting.gl_nz_validation_results(organization_id);
CREATE INDEX idx_p030_vr_report               ON pack030_nz_reporting.gl_nz_validation_results(report_id);
CREATE INDEX idx_p030_vr_run                  ON pack030_nz_reporting.gl_nz_validation_results(validation_run_id);
CREATE INDEX idx_p030_vr_report_severity      ON pack030_nz_reporting.gl_nz_validation_results(report_id, severity);
CREATE INDEX idx_p030_vr_report_category      ON pack030_nz_reporting.gl_nz_validation_results(report_id, validation_category);
CREATE INDEX idx_p030_vr_validator            ON pack030_nz_reporting.gl_nz_validation_results(validator);
CREATE INDEX idx_p030_vr_severity             ON pack030_nz_reporting.gl_nz_validation_results(severity);
CREATE INDEX idx_p030_vr_validation_type      ON pack030_nz_reporting.gl_nz_validation_results(validation_type);
CREATE INDEX idx_p030_vr_category             ON pack030_nz_reporting.gl_nz_validation_results(validation_category);
CREATE INDEX idx_p030_vr_result_status        ON pack030_nz_reporting.gl_nz_validation_results(result_status);
CREATE INDEX idx_p030_vr_framework            ON pack030_nz_reporting.gl_nz_validation_results(framework);
CREATE INDEX idx_p030_vr_unresolved           ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE resolved = FALSE;
CREATE INDEX idx_p030_vr_critical_unresolved  ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE severity = 'CRITICAL' AND resolved = FALSE;
CREATE INDEX idx_p030_vr_blocking             ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE blocking = TRUE AND resolved = FALSE;
CREATE INDEX idx_p030_vr_auto_fixable         ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE auto_fixable = TRUE AND auto_fix_applied = FALSE;
CREATE INDEX idx_p030_vr_cross_framework      ON pack030_nz_reporting.gl_nz_validation_results(report_id) WHERE validation_category = 'CROSS_FRAMEWORK';
CREATE INDEX idx_p030_vr_message_code         ON pack030_nz_reporting.gl_nz_validation_results(message_code);
CREATE INDEX idx_p030_vr_created              ON pack030_nz_reporting.gl_nz_validation_results(created_at DESC);
CREATE INDEX idx_p030_vr_run_at               ON pack030_nz_reporting.gl_nz_validation_results(validation_run_at DESC);
CREATE INDEX idx_p030_vr_metadata             ON pack030_nz_reporting.gl_nz_validation_results USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_validation_results
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_validation_results_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_validation_results
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_validation_results
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_validation_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_vr_tenant_isolation
    ON pack030_nz_reporting.gl_nz_validation_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_vr_service_bypass
    ON pack030_nz_reporting.gl_nz_validation_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_validation_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_validation_results IS
    'Validation results from automated checks including schema validation, completeness assessment, cross-framework consistency, XBRL taxonomy compliance, metric reconciliation, narrative analysis, and quality scoring with resolution workflow tracking.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.validation_id IS 'Unique validation result identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.validator IS 'Validator engine: SCHEMA_VALIDATOR, COMPLETENESS_CHECKER, CONSISTENCY_CHECKER, XBRL_VALIDATOR, etc.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.severity IS 'Issue severity: CRITICAL, HIGH, MEDIUM, LOW, INFO.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.blocking IS 'Whether this issue blocks report publication.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.auto_fixable IS 'Whether this issue can be automatically resolved.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_validation_results.quality_impact IS 'Impact on overall report quality score (0-100 point deduction).';
