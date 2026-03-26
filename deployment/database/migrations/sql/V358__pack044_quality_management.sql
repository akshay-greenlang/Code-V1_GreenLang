-- =============================================================================
-- V358: PACK-044 GHG Inventory Management - Quality Management Tables
-- =============================================================================
-- Pack:         PACK-044 (GHG Inventory Management)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Quality assurance and quality control (QA/QC) tables for GHG inventory
-- data. Implements GHG Protocol Chapter 8 guidance on data quality management.
-- QA/QC runs execute automated checks against submitted data. Individual
-- checks detect issues (range violations, year-on-year anomalies, missing
-- data). Quality issues are tracked to resolution with improvement actions.
--
-- Tables (4):
--   1. ghg_inventory.gl_inv_qaqc_runs
--   2. ghg_inventory.gl_inv_qaqc_checks
--   3. ghg_inventory.gl_inv_quality_issues
--   4. ghg_inventory.gl_inv_improvement_actions
--
-- Previous: V357__pack044_data_collection.sql
-- =============================================================================

SET search_path TO ghg_inventory, public;

-- =============================================================================
-- Table 1: ghg_inventory.gl_inv_qaqc_runs
-- =============================================================================
-- A QA/QC execution run against an inventory period or campaign. Runs may be
-- triggered automatically (on data submission) or manually (before review).
-- Tracks overall pass/fail counts and the data quality score achieved.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_qaqc_runs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    campaign_id                 UUID,
    run_name                    VARCHAR(300)    NOT NULL,
    run_type                    VARCHAR(30)     NOT NULL DEFAULT 'AUTOMATED',
    triggered_by                UUID,
    triggered_by_name           VARCHAR(255),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    started_at                  TIMESTAMPTZ,
    completed_at                TIMESTAMPTZ,
    total_checks                INTEGER         NOT NULL DEFAULT 0,
    passed_checks               INTEGER         NOT NULL DEFAULT 0,
    failed_checks               INTEGER         NOT NULL DEFAULT 0,
    warning_checks              INTEGER         NOT NULL DEFAULT 0,
    skipped_checks              INTEGER         NOT NULL DEFAULT 0,
    overall_score               NUMERIC(5,2),
    overall_result              VARCHAR(30),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_qr_run_type CHECK (
        run_type IN ('AUTOMATED', 'MANUAL', 'SCHEDULED', 'PRE_REVIEW', 'PRE_PUBLICATION')
    ),
    CONSTRAINT chk_p044_qr_status CHECK (
        status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    ),
    CONSTRAINT chk_p044_qr_checks CHECK (
        total_checks >= 0 AND passed_checks >= 0 AND failed_checks >= 0 AND
        warning_checks >= 0 AND skipped_checks >= 0
    ),
    CONSTRAINT chk_p044_qr_score CHECK (
        overall_score IS NULL OR (overall_score >= 0 AND overall_score <= 100)
    ),
    CONSTRAINT chk_p044_qr_result CHECK (
        overall_result IS NULL OR overall_result IN ('PASS', 'PASS_WITH_WARNINGS', 'FAIL', 'INCONCLUSIVE')
    ),
    CONSTRAINT chk_p044_qr_times CHECK (
        started_at IS NULL OR completed_at IS NULL OR started_at <= completed_at
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_qr_tenant          ON ghg_inventory.gl_inv_qaqc_runs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_qr_period          ON ghg_inventory.gl_inv_qaqc_runs(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_qr_campaign        ON ghg_inventory.gl_inv_qaqc_runs(campaign_id);
CREATE INDEX IF NOT EXISTS idx_p044_qr_type            ON ghg_inventory.gl_inv_qaqc_runs(run_type);
CREATE INDEX IF NOT EXISTS idx_p044_qr_status          ON ghg_inventory.gl_inv_qaqc_runs(status);
CREATE INDEX IF NOT EXISTS idx_p044_qr_result          ON ghg_inventory.gl_inv_qaqc_runs(overall_result);
CREATE INDEX IF NOT EXISTS idx_p044_qr_created         ON ghg_inventory.gl_inv_qaqc_runs(created_at DESC);

-- Composite: period + latest completed run
CREATE INDEX IF NOT EXISTS idx_p044_qr_period_latest   ON ghg_inventory.gl_inv_qaqc_runs(period_id, completed_at DESC)
    WHERE status = 'COMPLETED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_qr_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_qaqc_runs
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_inventory.gl_inv_qaqc_checks
-- =============================================================================
-- Individual QA/QC check results within a run. Each check tests a specific
-- data quality criterion (completeness, range, variance, unit consistency,
-- mass balance, outlier detection). Records the check logic, expected vs
-- actual values, and the pass/fail/warning outcome.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_qaqc_checks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    run_id                      UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_qaqc_runs(id) ON DELETE CASCADE,
    check_code                  VARCHAR(60)     NOT NULL,
    check_name                  VARCHAR(300)    NOT NULL,
    check_category              VARCHAR(50)     NOT NULL,
    facility_id                 UUID,
    source_category             VARCHAR(60),
    submission_id               UUID,
    expected_value              NUMERIC(18,6),
    actual_value                NUMERIC(18,6),
    threshold_value             NUMERIC(18,6),
    deviation_pct               NUMERIC(8,3),
    result                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    message                     TEXT,
    recommendation              TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_qc_category CHECK (
        check_category IN (
            'COMPLETENESS', 'RANGE_CHECK', 'YEAR_ON_YEAR_VARIANCE',
            'MONTH_ON_MONTH_VARIANCE', 'SEASONAL_VARIANCE', 'UNIT_CONSISTENCY',
            'MASS_BALANCE', 'PRODUCTION_CORRELATION', 'OUTLIER_DETECTION',
            'CROSS_SOURCE', 'DUPLICATE_DETECTION', 'TIMELINESS', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p044_qc_result CHECK (
        result IN ('PENDING', 'PASS', 'WARNING', 'FAIL', 'SKIPPED', 'ERROR')
    ),
    CONSTRAINT chk_p044_qc_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_qc_tenant          ON ghg_inventory.gl_inv_qaqc_checks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_qc_run             ON ghg_inventory.gl_inv_qaqc_checks(run_id);
CREATE INDEX IF NOT EXISTS idx_p044_qc_code            ON ghg_inventory.gl_inv_qaqc_checks(check_code);
CREATE INDEX IF NOT EXISTS idx_p044_qc_category        ON ghg_inventory.gl_inv_qaqc_checks(check_category);
CREATE INDEX IF NOT EXISTS idx_p044_qc_facility        ON ghg_inventory.gl_inv_qaqc_checks(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_qc_result          ON ghg_inventory.gl_inv_qaqc_checks(result);
CREATE INDEX IF NOT EXISTS idx_p044_qc_severity        ON ghg_inventory.gl_inv_qaqc_checks(severity);
CREATE INDEX IF NOT EXISTS idx_p044_qc_created         ON ghg_inventory.gl_inv_qaqc_checks(created_at DESC);

-- Composite: run + failed/warning checks
CREATE INDEX IF NOT EXISTS idx_p044_qc_run_issues      ON ghg_inventory.gl_inv_qaqc_checks(run_id, severity)
    WHERE result IN ('FAIL', 'WARNING');

-- =============================================================================
-- Table 3: ghg_inventory.gl_inv_quality_issues
-- =============================================================================
-- Quality issues identified by QA/QC checks or manual review. Issues are
-- tracked to resolution with assigned owners, target dates, and root cause
-- analysis. Provides a register of all data quality problems for audit.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_quality_issues (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    period_id                   UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_inventory_periods(id) ON DELETE CASCADE,
    check_id                    UUID            REFERENCES ghg_inventory.gl_inv_qaqc_checks(id) ON DELETE SET NULL,
    issue_code                  VARCHAR(60)     NOT NULL,
    issue_title                 VARCHAR(300)    NOT NULL,
    issue_description           TEXT            NOT NULL,
    facility_id                 UUID,
    source_category             VARCHAR(60),
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    assigned_to_user_id         UUID,
    assigned_to_name            VARCHAR(255),
    target_resolution_date      DATE,
    resolved_at                 TIMESTAMPTZ,
    resolved_by                 VARCHAR(255),
    resolution_description      TEXT,
    root_cause                  VARCHAR(50),
    impact_tco2e                NUMERIC(12,3),
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_qi_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')
    ),
    CONSTRAINT chk_p044_qi_status CHECK (
        status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'CLOSED', 'WONT_FIX', 'DEFERRED')
    ),
    CONSTRAINT chk_p044_qi_root_cause CHECK (
        root_cause IS NULL OR root_cause IN (
            'DATA_ENTRY_ERROR', 'MEASUREMENT_ERROR', 'ESTIMATION_ERROR',
            'METHODOLOGY_CHANGE', 'STRUCTURAL_CHANGE', 'SYSTEM_ERROR',
            'MISSING_DATA', 'UNIT_MISMATCH', 'TIMING_DIFFERENCE', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_qi_tenant          ON ghg_inventory.gl_inv_quality_issues(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_qi_period          ON ghg_inventory.gl_inv_quality_issues(period_id);
CREATE INDEX IF NOT EXISTS idx_p044_qi_check           ON ghg_inventory.gl_inv_quality_issues(check_id);
CREATE INDEX IF NOT EXISTS idx_p044_qi_facility        ON ghg_inventory.gl_inv_quality_issues(facility_id);
CREATE INDEX IF NOT EXISTS idx_p044_qi_severity        ON ghg_inventory.gl_inv_quality_issues(severity);
CREATE INDEX IF NOT EXISTS idx_p044_qi_status          ON ghg_inventory.gl_inv_quality_issues(status);
CREATE INDEX IF NOT EXISTS idx_p044_qi_assigned        ON ghg_inventory.gl_inv_quality_issues(assigned_to_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_qi_target          ON ghg_inventory.gl_inv_quality_issues(target_resolution_date);
CREATE INDEX IF NOT EXISTS idx_p044_qi_created         ON ghg_inventory.gl_inv_quality_issues(created_at DESC);

-- Composite: period + open issues
CREATE INDEX IF NOT EXISTS idx_p044_qi_period_open     ON ghg_inventory.gl_inv_quality_issues(period_id, severity)
    WHERE status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_qi_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_quality_issues
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_inventory.gl_inv_improvement_actions
-- =============================================================================
-- Corrective and preventive actions arising from quality issues. Tracks
-- action plans, implementation status, and effectiveness verification.
-- Links back to the originating quality issue for traceability.

CREATE TABLE IF NOT EXISTS ghg_inventory.gl_inv_improvement_actions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    issue_id                    UUID            NOT NULL REFERENCES ghg_inventory.gl_inv_quality_issues(id) ON DELETE CASCADE,
    action_type                 VARCHAR(30)     NOT NULL DEFAULT 'CORRECTIVE',
    action_title                VARCHAR(300)    NOT NULL,
    action_description          TEXT            NOT NULL,
    assigned_to_user_id         UUID,
    assigned_to_name            VARCHAR(255),
    priority                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PLANNED',
    target_date                 DATE,
    completed_at                TIMESTAMPTZ,
    effectiveness_verified      BOOLEAN         NOT NULL DEFAULT false,
    effectiveness_notes         TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p044_ia_type CHECK (
        action_type IN ('CORRECTIVE', 'PREVENTIVE', 'IMPROVEMENT', 'TRAINING', 'PROCESS_CHANGE')
    ),
    CONSTRAINT chk_p044_ia_priority CHECK (
        priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p044_ia_status CHECK (
        status IN ('PLANNED', 'IN_PROGRESS', 'COMPLETED', 'VERIFIED', 'CANCELLED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_p044_ia_tenant          ON ghg_inventory.gl_inv_improvement_actions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_p044_ia_issue           ON ghg_inventory.gl_inv_improvement_actions(issue_id);
CREATE INDEX IF NOT EXISTS idx_p044_ia_type            ON ghg_inventory.gl_inv_improvement_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_p044_ia_priority        ON ghg_inventory.gl_inv_improvement_actions(priority);
CREATE INDEX IF NOT EXISTS idx_p044_ia_status          ON ghg_inventory.gl_inv_improvement_actions(status);
CREATE INDEX IF NOT EXISTS idx_p044_ia_assigned        ON ghg_inventory.gl_inv_improvement_actions(assigned_to_user_id);
CREATE INDEX IF NOT EXISTS idx_p044_ia_target          ON ghg_inventory.gl_inv_improvement_actions(target_date);
CREATE INDEX IF NOT EXISTS idx_p044_ia_created         ON ghg_inventory.gl_inv_improvement_actions(created_at DESC);

-- Composite: issue + open actions
CREATE INDEX IF NOT EXISTS idx_p044_ia_issue_open      ON ghg_inventory.gl_inv_improvement_actions(issue_id, priority)
    WHERE status IN ('PLANNED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p044_ia_updated
    BEFORE UPDATE ON ghg_inventory.gl_inv_improvement_actions
    FOR EACH ROW EXECUTE FUNCTION ghg_inventory.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_inventory.gl_inv_qaqc_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_qaqc_checks ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_quality_issues ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_inventory.gl_inv_improvement_actions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p044_qr_tenant_isolation
    ON ghg_inventory.gl_inv_qaqc_runs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_qr_service_bypass
    ON ghg_inventory.gl_inv_qaqc_runs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_qc_tenant_isolation
    ON ghg_inventory.gl_inv_qaqc_checks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_qc_service_bypass
    ON ghg_inventory.gl_inv_qaqc_checks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_qi_tenant_isolation
    ON ghg_inventory.gl_inv_quality_issues
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_qi_service_bypass
    ON ghg_inventory.gl_inv_quality_issues
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p044_ia_tenant_isolation
    ON ghg_inventory.gl_inv_improvement_actions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p044_ia_service_bypass
    ON ghg_inventory.gl_inv_improvement_actions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_qaqc_runs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_qaqc_checks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_quality_issues TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_inventory.gl_inv_improvement_actions TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_inventory.gl_inv_qaqc_runs IS
    'QA/QC execution runs against inventory periods implementing GHG Protocol Chapter 8 data quality management.';
COMMENT ON TABLE ghg_inventory.gl_inv_qaqc_checks IS
    'Individual QA/QC check results within a run testing specific data quality criteria.';
COMMENT ON TABLE ghg_inventory.gl_inv_quality_issues IS
    'Quality issues identified by QA/QC checks or manual review, tracked to resolution with root cause analysis.';
COMMENT ON TABLE ghg_inventory.gl_inv_improvement_actions IS
    'Corrective and preventive actions arising from quality issues with implementation tracking.';

COMMENT ON COLUMN ghg_inventory.gl_inv_qaqc_runs.run_type IS 'How the run was triggered: AUTOMATED, MANUAL, SCHEDULED, PRE_REVIEW, PRE_PUBLICATION.';
COMMENT ON COLUMN ghg_inventory.gl_inv_qaqc_runs.overall_result IS 'Aggregate outcome: PASS, PASS_WITH_WARNINGS, FAIL, INCONCLUSIVE.';
COMMENT ON COLUMN ghg_inventory.gl_inv_qaqc_checks.check_category IS 'Category of QA/QC check: COMPLETENESS, RANGE_CHECK, YEAR_ON_YEAR_VARIANCE, OUTLIER_DETECTION, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_qaqc_checks.deviation_pct IS 'Percentage deviation from expected value (positive = above, negative = below).';
COMMENT ON COLUMN ghg_inventory.gl_inv_quality_issues.root_cause IS 'Root cause category: DATA_ENTRY_ERROR, MEASUREMENT_ERROR, METHODOLOGY_CHANGE, etc.';
COMMENT ON COLUMN ghg_inventory.gl_inv_improvement_actions.action_type IS 'Type of action: CORRECTIVE, PREVENTIVE, IMPROVEMENT, TRAINING, PROCESS_CHANGE.';
