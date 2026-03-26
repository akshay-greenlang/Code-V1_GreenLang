-- =============================================================================
-- V384: PACK-046 Intensity Metrics Pack - Framework Disclosures
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for multi-framework disclosure management. Tracks configured
-- frameworks (ESRS E1, CDP, SEC, SBTi, ISO 14064, TCFD, GRI, IFRS S2) per
-- organisation with mandatory/optional status and deadlines. Disclosure
-- mappings link individual framework fields to calculated intensity metrics
-- with validation status. Disclosure packages track generated output files
-- (MD, HTML, PDF, JSON, XBRL) with completeness and approval workflows.
--
-- Tables (3):
--   1. ghg_intensity.gl_im_disclosure_frameworks
--   2. ghg_intensity.gl_im_disclosure_mappings
--   3. ghg_intensity.gl_im_disclosure_packages
--
-- Also includes: indexes, RLS, comments.
-- Previous: V383__pack046_uncertainty.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_disclosure_frameworks
-- =============================================================================
-- Configured disclosure frameworks per organisation and reporting year.
-- Tracks which frameworks require intensity metric disclosures, whether
-- they are mandatory (regulatory) or voluntary, the reporting year, and
-- the current completion status. Deadline tracking supports compliance
-- calendar management.

CREATE TABLE ghg_intensity.gl_im_disclosure_frameworks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    framework                   VARCHAR(30)     NOT NULL,
    framework_version           VARCHAR(30),
    is_mandatory                BOOLEAN         NOT NULL DEFAULT false,
    reporting_year              INTEGER         NOT NULL,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    deadline                    DATE,
    submission_date             DATE,
    total_fields                INTEGER         NOT NULL DEFAULT 0,
    mandatory_fields            INTEGER         NOT NULL DEFAULT 0,
    populated_fields            INTEGER         NOT NULL DEFAULT 0,
    completeness_pct            NUMERIC(10,6)   DEFAULT 0.0,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_df_framework CHECK (
        framework IN (
            'ESRS_E1', 'CDP', 'SEC', 'SBTI', 'ISO_14064',
            'TCFD', 'GRI', 'IFRS_S2', 'PCAF', 'TPI',
            'GRESB', 'CRREM', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_df_year CHECK (
        reporting_year >= 2020 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p046_df_status CHECK (
        status IN (
            'PENDING', 'IN_PROGRESS', 'COMPLETE', 'SUBMITTED',
            'VERIFIED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p046_df_fields CHECK (
        total_fields >= 0 AND mandatory_fields >= 0 AND populated_fields >= 0
    ),
    CONSTRAINT chk_p046_df_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p046_df_submission CHECK (
        submission_date IS NULL OR deadline IS NULL OR submission_date >= '2020-01-01'::DATE
    ),
    CONSTRAINT uq_p046_df_org_framework_year UNIQUE (org_id, config_id, framework, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_df_tenant            ON ghg_intensity.gl_im_disclosure_frameworks(tenant_id);
CREATE INDEX idx_p046_df_org               ON ghg_intensity.gl_im_disclosure_frameworks(org_id);
CREATE INDEX idx_p046_df_config            ON ghg_intensity.gl_im_disclosure_frameworks(config_id);
CREATE INDEX idx_p046_df_framework         ON ghg_intensity.gl_im_disclosure_frameworks(framework);
CREATE INDEX idx_p046_df_year              ON ghg_intensity.gl_im_disclosure_frameworks(reporting_year);
CREATE INDEX idx_p046_df_status            ON ghg_intensity.gl_im_disclosure_frameworks(status);
CREATE INDEX idx_p046_df_mandatory         ON ghg_intensity.gl_im_disclosure_frameworks(is_mandatory) WHERE is_mandatory = true;
CREATE INDEX idx_p046_df_deadline          ON ghg_intensity.gl_im_disclosure_frameworks(deadline);
CREATE INDEX idx_p046_df_created           ON ghg_intensity.gl_im_disclosure_frameworks(created_at DESC);

-- Composite: org + framework + year for lookup
CREATE INDEX idx_p046_df_org_fw_year       ON ghg_intensity.gl_im_disclosure_frameworks(org_id, framework, reporting_year);

-- Composite: pending/incomplete for dashboard alerts
CREATE INDEX idx_p046_df_pending           ON ghg_intensity.gl_im_disclosure_frameworks(deadline)
    WHERE status IN ('PENDING', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_df_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_disclosure_frameworks
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_disclosure_mappings
-- =============================================================================
-- Individual disclosure field mappings linking framework-specific field
-- requirements to calculated intensity metrics. Each mapping identifies
-- the framework field code (e.g., ESRS_E1_6_REVENUE_INTENSITY), whether
-- it is mandatory, the linked calculation result, and the validation
-- status. Supports manual override values and review notes for auditors.

CREATE TABLE ghg_intensity.gl_im_disclosure_mappings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_framework_id     UUID            NOT NULL REFERENCES ghg_intensity.gl_im_disclosure_frameworks(id) ON DELETE CASCADE,
    field_code                  VARCHAR(100)    NOT NULL,
    field_name                  VARCHAR(255)    NOT NULL,
    field_section               VARCHAR(100),
    is_mandatory                BOOLEAN         NOT NULL DEFAULT false,
    calculation_id              UUID            REFERENCES ghg_intensity.gl_im_calculations(id),
    field_value                 TEXT,
    field_numeric_value         NUMERIC(20,10),
    field_unit                  VARCHAR(100),
    data_quality_score          INTEGER,
    validation_status           VARCHAR(30)     DEFAULT 'PENDING',
    validation_notes            TEXT,
    manual_override             BOOLEAN         NOT NULL DEFAULT false,
    override_reason             TEXT,
    reviewer                    UUID,
    reviewed_at                 TIMESTAMPTZ,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dm_validation CHECK (
        validation_status IN (
            'PENDING', 'VALID', 'INVALID', 'MANUAL_REVIEW',
            'OVERRIDDEN', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p046_dm_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_dm_override CHECK (
        (manual_override = false) OR (manual_override = true AND override_reason IS NOT NULL)
    ),
    CONSTRAINT uq_p046_dm_framework_field UNIQUE (disclosure_framework_id, field_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dm_framework         ON ghg_intensity.gl_im_disclosure_mappings(disclosure_framework_id);
CREATE INDEX idx_p046_dm_field_code        ON ghg_intensity.gl_im_disclosure_mappings(field_code);
CREATE INDEX idx_p046_dm_calculation       ON ghg_intensity.gl_im_disclosure_mappings(calculation_id);
CREATE INDEX idx_p046_dm_mandatory         ON ghg_intensity.gl_im_disclosure_mappings(is_mandatory) WHERE is_mandatory = true;
CREATE INDEX idx_p046_dm_validation        ON ghg_intensity.gl_im_disclosure_mappings(validation_status);
CREATE INDEX idx_p046_dm_override          ON ghg_intensity.gl_im_disclosure_mappings(manual_override) WHERE manual_override = true;
CREATE INDEX idx_p046_dm_created           ON ghg_intensity.gl_im_disclosure_mappings(created_at DESC);

-- Composite: framework + mandatory for completeness checks
CREATE INDEX idx_p046_dm_fw_mandatory      ON ghg_intensity.gl_im_disclosure_mappings(disclosure_framework_id, is_mandatory);

-- Composite: framework + validation for review queues
CREATE INDEX idx_p046_dm_fw_validation     ON ghg_intensity.gl_im_disclosure_mappings(disclosure_framework_id, validation_status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_dm_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_disclosure_mappings
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_intensity.gl_im_disclosure_packages
-- =============================================================================
-- Generated disclosure packages (output files) per framework. Each package
-- represents a complete or partial disclosure in a specific format (MD,
-- HTML, PDF, JSON, XBRL). Tracks content hash for integrity, completeness
-- percentage, mandatory field coverage, and approval workflow. Supports
-- version history via multiple packages per framework.

CREATE TABLE ghg_intensity.gl_im_disclosure_packages (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    disclosure_framework_id     UUID            NOT NULL REFERENCES ghg_intensity.gl_im_disclosure_frameworks(id) ON DELETE CASCADE,
    package_version             INTEGER         NOT NULL DEFAULT 1,
    package_format              VARCHAR(20)     NOT NULL,
    content_hash                VARCHAR(64)     NOT NULL,
    file_size_bytes             BIGINT,
    storage_path                VARCHAR(500),
    completeness_pct            NUMERIC(10,6)   NOT NULL,
    mandatory_fields_complete   BOOLEAN         NOT NULL,
    total_fields                INTEGER         NOT NULL,
    populated_fields            INTEGER         NOT NULL,
    missing_fields              JSONB           DEFAULT '[]',
    warnings                    JSONB           DEFAULT '[]',
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by                UUID,
    approved_at                 TIMESTAMPTZ,
    approved_by                 UUID,
    approval_notes              TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dp_format CHECK (
        package_format IN ('MD', 'HTML', 'PDF', 'JSON', 'XBRL', 'CSV', 'XLSX')
    ),
    CONSTRAINT chk_p046_dp_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p046_dp_fields CHECK (
        total_fields >= 0 AND populated_fields >= 0 AND populated_fields <= total_fields
    ),
    CONSTRAINT chk_p046_dp_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_p046_dp_version CHECK (
        package_version >= 1
    ),
    CONSTRAINT uq_p046_dp_fw_format_version UNIQUE (disclosure_framework_id, package_format, package_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dp_tenant            ON ghg_intensity.gl_im_disclosure_packages(tenant_id);
CREATE INDEX idx_p046_dp_org               ON ghg_intensity.gl_im_disclosure_packages(org_id);
CREATE INDEX idx_p046_dp_framework         ON ghg_intensity.gl_im_disclosure_packages(disclosure_framework_id);
CREATE INDEX idx_p046_dp_format            ON ghg_intensity.gl_im_disclosure_packages(package_format);
CREATE INDEX idx_p046_dp_content_hash      ON ghg_intensity.gl_im_disclosure_packages(content_hash);
CREATE INDEX idx_p046_dp_completeness      ON ghg_intensity.gl_im_disclosure_packages(completeness_pct);
CREATE INDEX idx_p046_dp_generated         ON ghg_intensity.gl_im_disclosure_packages(generated_at DESC);
CREATE INDEX idx_p046_dp_approved          ON ghg_intensity.gl_im_disclosure_packages(approved_at DESC);
CREATE INDEX idx_p046_dp_created           ON ghg_intensity.gl_im_disclosure_packages(created_at DESC);
CREATE INDEX idx_p046_dp_missing           ON ghg_intensity.gl_im_disclosure_packages USING GIN(missing_fields);

-- Composite: framework + format for latest version lookup
CREATE INDEX idx_p046_dp_fw_format         ON ghg_intensity.gl_im_disclosure_packages(disclosure_framework_id, package_format, package_version DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_disclosure_frameworks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_disclosure_mappings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_disclosure_packages ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_df_tenant_isolation
    ON ghg_intensity.gl_im_disclosure_frameworks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_df_service_bypass
    ON ghg_intensity.gl_im_disclosure_frameworks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Mappings inherit access via framework FK
CREATE POLICY p046_dm_service_bypass
    ON ghg_intensity.gl_im_disclosure_mappings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_dp_tenant_isolation
    ON ghg_intensity.gl_im_disclosure_packages
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_dp_service_bypass
    ON ghg_intensity.gl_im_disclosure_packages
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_disclosure_frameworks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_disclosure_mappings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_disclosure_packages TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_frameworks IS
    'Configured disclosure frameworks (ESRS E1, CDP, SEC, SBTi, etc.) per organisation with mandatory/optional status, deadlines, and completion tracking.';
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_mappings IS
    'Individual field mappings linking framework disclosure requirements to calculated intensity metrics with validation and manual override support.';
COMMENT ON TABLE ghg_intensity.gl_im_disclosure_packages IS
    'Generated disclosure output packages (MD, HTML, PDF, JSON, XBRL) with completeness tracking, content hashing, and approval workflow.';

COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_frameworks.framework IS 'ESRS_E1, CDP, SEC, SBTI, ISO_14064, TCFD, GRI, IFRS_S2, PCAF, TPI, GRESB, CRREM, or CUSTOM.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_frameworks.status IS 'PENDING (not started), IN_PROGRESS, COMPLETE (all fields populated), SUBMITTED, VERIFIED, ARCHIVED.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_frameworks.completeness_pct IS 'Percentage of total fields populated (0-100). Auto-updated when mappings change.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_mappings.field_code IS 'Framework-specific field code, e.g. ESRS_E1_6_REVENUE_INTENSITY, CDP_C6.10_INTENSITY_METRIC_1.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_mappings.validation_status IS 'PENDING, VALID (auto-validated), INVALID (failed checks), MANUAL_REVIEW, OVERRIDDEN, NOT_APPLICABLE.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_mappings.manual_override IS 'True when the field value was manually set instead of auto-populated from calculations. Requires override_reason.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_packages.content_hash IS 'SHA-256 hash of the generated content for integrity verification and change detection.';
COMMENT ON COLUMN ghg_intensity.gl_im_disclosure_packages.missing_fields IS 'JSON array of field codes that are required but not yet populated.';
