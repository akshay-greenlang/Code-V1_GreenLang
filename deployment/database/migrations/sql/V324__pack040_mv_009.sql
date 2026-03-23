-- =============================================================================
-- V324: PACK-040 M&V Pack - Reports and Compliance
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for M&V report generation and compliance tracking including
-- report configuration, report outputs, compliance checks, compliance
-- findings, and report scheduling. Supports automated report generation
-- compliant with IPMVP, ASHRAE 14, ISO 50015, FEMP, and EU EED.
--
-- Tables (5):
--   1. pack040_mv.mv_report_configs
--   2. pack040_mv.mv_report_outputs
--   3. pack040_mv.mv_compliance_checks
--   4. pack040_mv.mv_compliance_findings
--   5. pack040_mv.mv_report_schedules
--
-- Previous: V323__pack040_mv_008.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_report_configs
-- =============================================================================
-- Report configuration definitions specifying what content to include,
-- formatting preferences, compliance framework requirements, and distribution
-- settings for each report type within an M&V project.

CREATE TABLE pack040_mv.mv_report_configs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    config_name                 VARCHAR(255)    NOT NULL,
    report_type                 VARCHAR(50)     NOT NULL DEFAULT 'SAVINGS_REPORT',
    -- Format settings
    output_format               VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    template_id                 VARCHAR(100),
    language                    VARCHAR(10)     NOT NULL DEFAULT 'en',
    -- Content settings
    include_executive_summary   BOOLEAN         NOT NULL DEFAULT true,
    include_methodology         BOOLEAN         NOT NULL DEFAULT true,
    include_baseline_details    BOOLEAN         NOT NULL DEFAULT true,
    include_regression_stats    BOOLEAN         NOT NULL DEFAULT true,
    include_residual_plots      BOOLEAN         NOT NULL DEFAULT true,
    include_adjustment_details  BOOLEAN         NOT NULL DEFAULT true,
    include_uncertainty         BOOLEAN         NOT NULL DEFAULT true,
    include_cusum_chart         BOOLEAN         NOT NULL DEFAULT true,
    include_persistence         BOOLEAN         NOT NULL DEFAULT false,
    include_financial_analysis  BOOLEAN         NOT NULL DEFAULT true,
    include_compliance_status   BOOLEAN         NOT NULL DEFAULT true,
    include_recommendations     BOOLEAN         NOT NULL DEFAULT true,
    include_appendices          BOOLEAN         NOT NULL DEFAULT true,
    custom_sections             JSONB           DEFAULT '[]',
    -- Compliance frameworks to check
    check_ipmvp                 BOOLEAN         NOT NULL DEFAULT true,
    check_ashrae_14             BOOLEAN         NOT NULL DEFAULT true,
    check_iso_50015             BOOLEAN         NOT NULL DEFAULT false,
    check_femp                  BOOLEAN         NOT NULL DEFAULT false,
    check_eu_eed                BOOLEAN         NOT NULL DEFAULT false,
    -- Branding
    company_logo_ref            VARCHAR(255),
    header_text                 VARCHAR(255),
    footer_text                 VARCHAR(255),
    cover_page_title            VARCHAR(255),
    -- Distribution
    distribution_list           JSONB           DEFAULT '[]',
    auto_distribute             BOOLEAN         NOT NULL DEFAULT false,
    distribution_method         VARCHAR(20)     DEFAULT 'EMAIL',
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_rc_type CHECK (
        report_type IN (
            'MV_PLAN', 'BASELINE_REPORT', 'SAVINGS_REPORT',
            'UNCERTAINTY_REPORT', 'ANNUAL_MV_REPORT', 'OPTION_COMPARISON',
            'METERING_PLAN', 'PERSISTENCE_REPORT', 'EXECUTIVE_SUMMARY',
            'COMPLIANCE_REPORT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_rc_format CHECK (
        output_format IN ('PDF', 'HTML', 'MARKDOWN', 'JSON', 'EXCEL', 'DOCX')
    ),
    CONSTRAINT chk_p040_rc_language CHECK (
        language IN ('en', 'de', 'fr', 'es', 'it', 'nl', 'pt', 'ja', 'zh')
    ),
    CONSTRAINT chk_p040_rc_dist_method CHECK (
        distribution_method IS NULL OR distribution_method IN (
            'EMAIL', 'PORTAL', 'API', 'S3', 'SHAREPOINT'
        )
    ),
    CONSTRAINT uq_p040_rc_project_type UNIQUE (project_id, config_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_rc_tenant            ON pack040_mv.mv_report_configs(tenant_id);
CREATE INDEX idx_p040_rc_project           ON pack040_mv.mv_report_configs(project_id);
CREATE INDEX idx_p040_rc_type              ON pack040_mv.mv_report_configs(report_type);
CREATE INDEX idx_p040_rc_format            ON pack040_mv.mv_report_configs(output_format);
CREATE INDEX idx_p040_rc_active            ON pack040_mv.mv_report_configs(is_active) WHERE is_active = true;
CREATE INDEX idx_p040_rc_created           ON pack040_mv.mv_report_configs(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_rc_updated
    BEFORE UPDATE ON pack040_mv.mv_report_configs
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_report_outputs
-- =============================================================================
-- Generated report outputs with file references, generation metadata,
-- content hashes, and distribution status. Each row represents one
-- generated report instance.

CREATE TABLE pack040_mv.mv_report_outputs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    report_config_id            UUID            NOT NULL REFERENCES pack040_mv.mv_report_configs(id) ON DELETE CASCADE,
    -- Report identification
    report_title                VARCHAR(255)    NOT NULL,
    report_type                 VARCHAR(50)     NOT NULL,
    report_version              INTEGER         NOT NULL DEFAULT 1,
    -- Period covered
    report_period_start         DATE,
    report_period_end           DATE,
    reporting_year              INTEGER,
    -- Generation
    generated_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generation_duration_ms      INTEGER,
    generated_by                VARCHAR(255)    NOT NULL DEFAULT 'SYSTEM',
    generation_status           VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    generation_errors           JSONB           DEFAULT '[]',
    -- Output file
    output_format               VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    file_path                   VARCHAR(500),
    file_size_bytes             BIGINT,
    content_hash_sha256         VARCHAR(64),
    page_count                  INTEGER,
    -- Key metrics included
    total_savings_kwh           NUMERIC(18,3),
    total_cost_savings          NUMERIC(18,2),
    fsu_pct                     NUMERIC(8,4),
    passes_compliance           BOOLEAN,
    -- Distribution
    distribution_status         VARCHAR(20)     NOT NULL DEFAULT 'NOT_DISTRIBUTED',
    distributed_at              TIMESTAMPTZ,
    distributed_to              JSONB           DEFAULT '[]',
    -- Approval
    review_status               VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ro_type CHECK (
        report_type IN (
            'MV_PLAN', 'BASELINE_REPORT', 'SAVINGS_REPORT',
            'UNCERTAINTY_REPORT', 'ANNUAL_MV_REPORT', 'OPTION_COMPARISON',
            'METERING_PLAN', 'PERSISTENCE_REPORT', 'EXECUTIVE_SUMMARY',
            'COMPLIANCE_REPORT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_ro_format CHECK (
        output_format IN ('PDF', 'HTML', 'MARKDOWN', 'JSON', 'EXCEL', 'DOCX')
    ),
    CONSTRAINT chk_p040_ro_gen_status CHECK (
        generation_status IN ('COMPLETED', 'FAILED', 'PARTIAL', 'IN_PROGRESS')
    ),
    CONSTRAINT chk_p040_ro_dist_status CHECK (
        distribution_status IN (
            'NOT_DISTRIBUTED', 'QUEUED', 'DISTRIBUTED', 'FAILED', 'RECALLED'
        )
    ),
    CONSTRAINT chk_p040_ro_review CHECK (
        review_status IN ('DRAFT', 'REVIEWED', 'APPROVED', 'REJECTED', 'FINAL')
    ),
    CONSTRAINT chk_p040_ro_version CHECK (
        report_version >= 1
    ),
    CONSTRAINT chk_p040_ro_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_p040_ro_pages CHECK (
        page_count IS NULL OR page_count >= 0
    ),
    CONSTRAINT chk_p040_ro_duration CHECK (
        generation_duration_ms IS NULL OR generation_duration_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ro_tenant            ON pack040_mv.mv_report_outputs(tenant_id);
CREATE INDEX idx_p040_ro_project           ON pack040_mv.mv_report_outputs(project_id);
CREATE INDEX idx_p040_ro_config            ON pack040_mv.mv_report_outputs(report_config_id);
CREATE INDEX idx_p040_ro_type              ON pack040_mv.mv_report_outputs(report_type);
CREATE INDEX idx_p040_ro_generated         ON pack040_mv.mv_report_outputs(generated_at DESC);
CREATE INDEX idx_p040_ro_gen_status        ON pack040_mv.mv_report_outputs(generation_status);
CREATE INDEX idx_p040_ro_review            ON pack040_mv.mv_report_outputs(review_status);
CREATE INDEX idx_p040_ro_year              ON pack040_mv.mv_report_outputs(reporting_year);
CREATE INDEX idx_p040_ro_created           ON pack040_mv.mv_report_outputs(created_at DESC);

-- Composite: project + approved reports
CREATE INDEX idx_p040_ro_project_approved  ON pack040_mv.mv_report_outputs(project_id, generated_at DESC)
    WHERE review_status IN ('APPROVED', 'FINAL');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ro_updated
    BEFORE UPDATE ON pack040_mv.mv_report_outputs
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_compliance_checks
-- =============================================================================
-- Compliance check results against M&V standards (IPMVP, ASHRAE 14,
-- ISO 50015, FEMP, EU EED). Each check evaluates the M&V project against
-- specific requirements of the selected compliance framework.

CREATE TABLE pack040_mv.mv_compliance_checks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    report_output_id            UUID            REFERENCES pack040_mv.mv_report_outputs(id) ON DELETE SET NULL,
    -- Check identification
    check_date                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    compliance_framework        VARCHAR(50)     NOT NULL,
    framework_version           VARCHAR(20),
    check_scope                 VARCHAR(50)     NOT NULL DEFAULT 'FULL',
    -- Results
    total_requirements          INTEGER         NOT NULL,
    requirements_met            INTEGER         NOT NULL,
    requirements_not_met        INTEGER         NOT NULL,
    requirements_not_applicable INTEGER         NOT NULL DEFAULT 0,
    requirements_with_warnings  INTEGER         NOT NULL DEFAULT 0,
    compliance_pct              NUMERIC(5,2)    NOT NULL,
    overall_status              VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    -- Key metrics checked
    cvrmse_check                VARCHAR(10),
    nmbe_check                  VARCHAR(10),
    r_squared_check             VARCHAR(10),
    fsu_check                   VARCHAR(10),
    data_completeness_check     VARCHAR(10),
    documentation_check         VARCHAR(10),
    -- Detailed results
    check_details               JSONB           NOT NULL DEFAULT '[]',
    -- Approval
    checked_by                  VARCHAR(255)    NOT NULL DEFAULT 'SYSTEM',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cc_framework CHECK (
        compliance_framework IN (
            'IPMVP', 'ASHRAE_14', 'ISO_50015', 'ISO_50001',
            'FEMP_4_0', 'EU_EED', 'EU_EPC', 'BPA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_cc_scope CHECK (
        check_scope IN ('FULL', 'BASELINE_ONLY', 'SAVINGS_ONLY',
                        'UNCERTAINTY_ONLY', 'DOCUMENTATION_ONLY', 'PARTIAL')
    ),
    CONSTRAINT chk_p040_cc_counts CHECK (
        total_requirements >= 0 AND requirements_met >= 0 AND
        requirements_not_met >= 0 AND requirements_not_applicable >= 0
    ),
    CONSTRAINT chk_p040_cc_total CHECK (
        requirements_met + requirements_not_met + requirements_not_applicable <= total_requirements
    ),
    CONSTRAINT chk_p040_cc_pct CHECK (
        compliance_pct >= 0 AND compliance_pct <= 100
    ),
    CONSTRAINT chk_p040_cc_status CHECK (
        overall_status IN (
            'PENDING', 'COMPLIANT', 'NON_COMPLIANT', 'PARTIAL',
            'CONDITIONAL', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p040_cc_metric_check CHECK (
        cvrmse_check IS NULL OR cvrmse_check IN ('PASS', 'FAIL', 'N_A', 'WARNING')
    ),
    CONSTRAINT chk_p040_cc_nmbe_check CHECK (
        nmbe_check IS NULL OR nmbe_check IN ('PASS', 'FAIL', 'N_A', 'WARNING')
    ),
    CONSTRAINT chk_p040_cc_r2_check CHECK (
        r_squared_check IS NULL OR r_squared_check IN ('PASS', 'FAIL', 'N_A', 'WARNING')
    ),
    CONSTRAINT chk_p040_cc_fsu_check CHECK (
        fsu_check IS NULL OR fsu_check IN ('PASS', 'FAIL', 'N_A', 'WARNING')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cc_tenant            ON pack040_mv.mv_compliance_checks(tenant_id);
CREATE INDEX idx_p040_cc_project           ON pack040_mv.mv_compliance_checks(project_id);
CREATE INDEX idx_p040_cc_report            ON pack040_mv.mv_compliance_checks(report_output_id);
CREATE INDEX idx_p040_cc_framework         ON pack040_mv.mv_compliance_checks(compliance_framework);
CREATE INDEX idx_p040_cc_status            ON pack040_mv.mv_compliance_checks(overall_status);
CREATE INDEX idx_p040_cc_date              ON pack040_mv.mv_compliance_checks(check_date DESC);
CREATE INDEX idx_p040_cc_created           ON pack040_mv.mv_compliance_checks(created_at DESC);

-- Composite: project + non-compliant checks
CREATE INDEX idx_p040_cc_project_noncomp   ON pack040_mv.mv_compliance_checks(project_id, compliance_framework)
    WHERE overall_status = 'NON_COMPLIANT';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cc_updated
    BEFORE UPDATE ON pack040_mv.mv_compliance_checks
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_compliance_findings
-- =============================================================================
-- Individual compliance findings from compliance checks. Each finding
-- represents a specific requirement that was checked, the result, and
-- any corrective actions needed for non-compliant items.

CREATE TABLE pack040_mv.mv_compliance_findings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    compliance_check_id         UUID            NOT NULL REFERENCES pack040_mv.mv_compliance_checks(id) ON DELETE CASCADE,
    -- Finding details
    requirement_id              VARCHAR(50)     NOT NULL,
    requirement_name            VARCHAR(255)    NOT NULL,
    requirement_category        VARCHAR(50)     NOT NULL,
    requirement_description     TEXT,
    -- Result
    finding_status              VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    finding_description         TEXT,
    evidence_summary            TEXT,
    -- Values (if applicable)
    required_value              VARCHAR(100),
    actual_value                VARCHAR(100),
    threshold_met               BOOLEAN,
    -- Corrective action
    corrective_action_required  BOOLEAN         NOT NULL DEFAULT false,
    corrective_action           TEXT,
    corrective_action_deadline  DATE,
    corrective_action_status    VARCHAR(20),
    corrective_action_completed DATE,
    -- Severity
    finding_severity            VARCHAR(20)     NOT NULL DEFAULT 'MINOR',
    impact_description          TEXT,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cf_category CHECK (
        requirement_category IN (
            'BASELINE', 'MODEL_VALIDATION', 'ADJUSTMENTS', 'SAVINGS',
            'UNCERTAINTY', 'METERING', 'DOCUMENTATION', 'REPORTING',
            'DATA_QUALITY', 'METHODOLOGY', 'GENERAL'
        )
    ),
    CONSTRAINT chk_p040_cf_status CHECK (
        finding_status IN (
            'PENDING', 'COMPLIANT', 'NON_COMPLIANT', 'PARTIAL',
            'NOT_APPLICABLE', 'WARNING', 'EXEMPTED'
        )
    ),
    CONSTRAINT chk_p040_cf_severity CHECK (
        finding_severity IN ('INFORMATIONAL', 'MINOR', 'MAJOR', 'CRITICAL')
    ),
    CONSTRAINT chk_p040_cf_ca_status CHECK (
        corrective_action_status IS NULL OR corrective_action_status IN (
            'IDENTIFIED', 'PLANNED', 'IN_PROGRESS', 'COMPLETED',
            'VERIFIED', 'DEFERRED', 'CANCELLED'
        )
    ),
    CONSTRAINT uq_p040_cf_check_req UNIQUE (compliance_check_id, requirement_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cf_tenant            ON pack040_mv.mv_compliance_findings(tenant_id);
CREATE INDEX idx_p040_cf_check             ON pack040_mv.mv_compliance_findings(compliance_check_id);
CREATE INDEX idx_p040_cf_category          ON pack040_mv.mv_compliance_findings(requirement_category);
CREATE INDEX idx_p040_cf_status            ON pack040_mv.mv_compliance_findings(finding_status);
CREATE INDEX idx_p040_cf_severity          ON pack040_mv.mv_compliance_findings(finding_severity);
CREATE INDEX idx_p040_cf_ca_required       ON pack040_mv.mv_compliance_findings(corrective_action_required) WHERE corrective_action_required = true;
CREATE INDEX idx_p040_cf_ca_status         ON pack040_mv.mv_compliance_findings(corrective_action_status);
CREATE INDEX idx_p040_cf_created           ON pack040_mv.mv_compliance_findings(created_at DESC);

-- Composite: check + non-compliant findings
CREATE INDEX idx_p040_cf_check_noncomp     ON pack040_mv.mv_compliance_findings(compliance_check_id, finding_severity DESC)
    WHERE finding_status = 'NON_COMPLIANT';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cf_updated
    BEFORE UPDATE ON pack040_mv.mv_compliance_findings
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_report_schedules
-- =============================================================================
-- Automated report generation schedules defining when reports are generated,
-- who receives them, and what triggers generation (time-based or event-based).

CREATE TABLE pack040_mv.mv_report_schedules (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    report_config_id            UUID            NOT NULL REFERENCES pack040_mv.mv_report_configs(id) ON DELETE CASCADE,
    schedule_name               VARCHAR(255)    NOT NULL,
    -- Schedule settings
    schedule_type               VARCHAR(20)     NOT NULL DEFAULT 'RECURRING',
    frequency                   VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    day_of_month                INTEGER,
    day_of_week                 INTEGER,
    time_of_day                 TIME            DEFAULT '08:00:00',
    timezone                    VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    -- Event triggers
    trigger_on_period_close     BOOLEAN         NOT NULL DEFAULT false,
    trigger_on_data_complete    BOOLEAN         NOT NULL DEFAULT false,
    trigger_on_compliance_fail  BOOLEAN         NOT NULL DEFAULT false,
    trigger_on_persistence_alert BOOLEAN        NOT NULL DEFAULT false,
    -- Dates
    start_date                  DATE            NOT NULL,
    end_date                    DATE,
    next_run_date               DATE,
    last_run_date               DATE,
    last_run_status             VARCHAR(20),
    -- Distribution
    auto_distribute             BOOLEAN         NOT NULL DEFAULT true,
    recipients                  JSONB           NOT NULL DEFAULT '[]',
    cc_recipients               JSONB           DEFAULT '[]',
    -- Status
    is_enabled                  BOOLEAN         NOT NULL DEFAULT true,
    run_count                   INTEGER         NOT NULL DEFAULT 0,
    failure_count               INTEGER         NOT NULL DEFAULT 0,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_rs_schedule_type CHECK (
        schedule_type IN ('RECURRING', 'ONE_TIME', 'EVENT_TRIGGERED')
    ),
    CONSTRAINT chk_p040_rs_frequency CHECK (
        frequency IN (
            'WEEKLY', 'BIWEEKLY', 'MONTHLY', 'QUARTERLY',
            'SEMI_ANNUALLY', 'ANNUALLY', 'ON_DEMAND'
        )
    ),
    CONSTRAINT chk_p040_rs_day_month CHECK (
        day_of_month IS NULL OR (day_of_month >= 1 AND day_of_month <= 28)
    ),
    CONSTRAINT chk_p040_rs_day_week CHECK (
        day_of_week IS NULL OR (day_of_week >= 0 AND day_of_week <= 6)
    ),
    CONSTRAINT chk_p040_rs_dates CHECK (
        end_date IS NULL OR start_date <= end_date
    ),
    CONSTRAINT chk_p040_rs_last_run CHECK (
        last_run_status IS NULL OR last_run_status IN (
            'SUCCESS', 'FAILED', 'PARTIAL', 'SKIPPED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p040_rs_counts CHECK (
        run_count >= 0 AND failure_count >= 0
    ),
    CONSTRAINT uq_p040_rs_project_name UNIQUE (project_id, schedule_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_rs_tenant            ON pack040_mv.mv_report_schedules(tenant_id);
CREATE INDEX idx_p040_rs_project           ON pack040_mv.mv_report_schedules(project_id);
CREATE INDEX idx_p040_rs_config            ON pack040_mv.mv_report_schedules(report_config_id);
CREATE INDEX idx_p040_rs_frequency         ON pack040_mv.mv_report_schedules(frequency);
CREATE INDEX idx_p040_rs_next_run          ON pack040_mv.mv_report_schedules(next_run_date);
CREATE INDEX idx_p040_rs_enabled           ON pack040_mv.mv_report_schedules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p040_rs_created           ON pack040_mv.mv_report_schedules(created_at DESC);

-- Composite: enabled schedules due for run
CREATE INDEX idx_p040_rs_due               ON pack040_mv.mv_report_schedules(next_run_date, frequency)
    WHERE is_enabled = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_rs_updated
    BEFORE UPDATE ON pack040_mv.mv_report_schedules
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_report_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_report_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_compliance_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_compliance_findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_report_schedules ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_rc_tenant_isolation
    ON pack040_mv.mv_report_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_rc_service_bypass
    ON pack040_mv.mv_report_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ro_tenant_isolation
    ON pack040_mv.mv_report_outputs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ro_service_bypass
    ON pack040_mv.mv_report_outputs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cc_tenant_isolation
    ON pack040_mv.mv_compliance_checks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cc_service_bypass
    ON pack040_mv.mv_compliance_checks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cf_tenant_isolation
    ON pack040_mv.mv_compliance_findings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cf_service_bypass
    ON pack040_mv.mv_compliance_findings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_rs_tenant_isolation
    ON pack040_mv.mv_report_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_rs_service_bypass
    ON pack040_mv.mv_report_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_report_configs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_report_outputs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_compliance_checks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_compliance_findings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_report_schedules TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_report_configs IS
    'Report configuration defining content, format, compliance frameworks, branding, and distribution for M&V reports.';
COMMENT ON TABLE pack040_mv.mv_report_outputs IS
    'Generated report instances with file references, content hashes, key metrics, and distribution status.';
COMMENT ON TABLE pack040_mv.mv_compliance_checks IS
    'Compliance check results against IPMVP, ASHRAE 14, ISO 50015, FEMP, and EU EED requirements.';
COMMENT ON TABLE pack040_mv.mv_compliance_findings IS
    'Individual compliance findings with requirement details, evidence, corrective actions, and severity.';
COMMENT ON TABLE pack040_mv.mv_report_schedules IS
    'Automated report generation schedules with frequency, triggers, recipients, and execution tracking.';

COMMENT ON COLUMN pack040_mv.mv_report_configs.report_type IS 'Report type: MV_PLAN, BASELINE_REPORT, SAVINGS_REPORT, UNCERTAINTY_REPORT, ANNUAL_MV_REPORT, etc.';
COMMENT ON COLUMN pack040_mv.mv_report_configs.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_report_outputs.content_hash_sha256 IS 'SHA-256 hash of report file content for integrity verification.';
COMMENT ON COLUMN pack040_mv.mv_report_outputs.passes_compliance IS 'Whether the M&V project passes all checked compliance frameworks.';

COMMENT ON COLUMN pack040_mv.mv_compliance_checks.compliance_pct IS 'Percentage of applicable requirements that are met (0-100).';
COMMENT ON COLUMN pack040_mv.mv_compliance_checks.cvrmse_check IS 'ASHRAE 14 CVRMSE validation result: PASS, FAIL, N_A, WARNING.';

COMMENT ON COLUMN pack040_mv.mv_compliance_findings.finding_severity IS 'Severity: INFORMATIONAL, MINOR (documentation gap), MAJOR (methodology gap), CRITICAL (invalid results).';
COMMENT ON COLUMN pack040_mv.mv_compliance_findings.corrective_action IS 'Required corrective action to achieve compliance for non-compliant findings.';
