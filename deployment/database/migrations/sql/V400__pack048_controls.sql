-- =============================================================================
-- PACK-048 GHG Assurance Prep Pack
-- Migration: V400 - Internal Controls
-- =============================================================================
-- Pack:         PACK-048 (GHG Assurance Prep Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates internal control framework tables for GHG reporting. Controls are
-- categorised (data collection, calculation, review, reporting, IT general),
-- typed (preventive, detective, corrective), maturity-levelled, and
-- testable. Control tests evaluate design and operating effectiveness with
-- sample-based exception tracking. Deficiencies are classified by severity
-- (deficiency, significant deficiency, material weakness) with remediation
-- lifecycle management.
--
-- Tables (3):
--   1. ghg_assurance.gl_ap_controls
--   2. ghg_assurance.gl_ap_control_tests
--   3. ghg_assurance.gl_ap_deficiencies
--
-- Also includes: indexes, RLS, comments.
-- Previous: V399__pack048_provenance.sql
-- =============================================================================

SET search_path TO ghg_assurance, public;

-- =============================================================================
-- Table 1: ghg_assurance.gl_ap_controls
-- =============================================================================
-- Internal control definitions for GHG reporting processes. Each control
-- has a unique code per tenant, category (data collection through IT
-- general), type (preventive/detective/corrective), owner role, frequency,
-- key control flag, and CMMI maturity level (1-5).

CREATE TABLE ghg_assurance.gl_ap_controls (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_configurations(id) ON DELETE CASCADE,
    control_code                TEXT            NOT NULL,
    control_name                VARCHAR(255)    NOT NULL,
    control_description         TEXT,
    category                    VARCHAR(30)     NOT NULL,
    control_type                VARCHAR(20)     NOT NULL,
    owner_role                  VARCHAR(100),
    frequency                   VARCHAR(50),
    is_key_control              BOOLEAN         NOT NULL DEFAULT false,
    maturity_level              VARCHAR(10)     NOT NULL DEFAULT 'LEVEL_1',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p048_ctrl_category CHECK (
        category IN (
            'DATA_COLLECTION', 'CALCULATION', 'REVIEW',
            'REPORTING', 'IT_GENERAL'
        )
    ),
    CONSTRAINT chk_p048_ctrl_type CHECK (
        control_type IN ('PREVENTIVE', 'DETECTIVE', 'CORRECTIVE')
    ),
    CONSTRAINT chk_p048_ctrl_maturity CHECK (
        maturity_level IN (
            'LEVEL_1', 'LEVEL_2', 'LEVEL_3', 'LEVEL_4', 'LEVEL_5'
        )
    ),
    CONSTRAINT uq_p048_ctrl_tenant_code UNIQUE (tenant_id, control_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ctrl_tenant          ON ghg_assurance.gl_ap_controls(tenant_id);
CREATE INDEX idx_p048_ctrl_config          ON ghg_assurance.gl_ap_controls(config_id);
CREATE INDEX idx_p048_ctrl_code            ON ghg_assurance.gl_ap_controls(control_code);
CREATE INDEX idx_p048_ctrl_category        ON ghg_assurance.gl_ap_controls(category);
CREATE INDEX idx_p048_ctrl_type            ON ghg_assurance.gl_ap_controls(control_type);
CREATE INDEX idx_p048_ctrl_maturity        ON ghg_assurance.gl_ap_controls(maturity_level);
CREATE INDEX idx_p048_ctrl_key             ON ghg_assurance.gl_ap_controls(is_key_control) WHERE is_key_control = true;
CREATE INDEX idx_p048_ctrl_active          ON ghg_assurance.gl_ap_controls(is_active) WHERE is_active = true;
CREATE INDEX idx_p048_ctrl_owner           ON ghg_assurance.gl_ap_controls(owner_role);
CREATE INDEX idx_p048_ctrl_created         ON ghg_assurance.gl_ap_controls(created_at DESC);
CREATE INDEX idx_p048_ctrl_metadata        ON ghg_assurance.gl_ap_controls USING GIN(metadata);

-- Composite: config + category for category-filtered listing
CREATE INDEX idx_p048_ctrl_config_cat      ON ghg_assurance.gl_ap_controls(config_id, category);

-- Composite: tenant + config for scoped listing
CREATE INDEX idx_p048_ctrl_tenant_config   ON ghg_assurance.gl_ap_controls(tenant_id, config_id);

-- Composite: category + type for control matrix queries
CREATE INDEX idx_p048_ctrl_cat_type        ON ghg_assurance.gl_ap_controls(category, control_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ctrl_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_controls
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_assurance.gl_ap_control_tests
-- =============================================================================
-- Control testing records evaluating design and operating effectiveness.
-- Each test covers a time period, uses sample-based testing with exception
-- tracking, and produces effectiveness ratings. Tests are linked to controls
-- and include provenance for audit trail.

CREATE TABLE ghg_assurance.gl_ap_control_tests (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    control_id                  UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_controls(id) ON DELETE CASCADE,
    test_period_start           DATE            NOT NULL,
    test_period_end             DATE            NOT NULL,
    design_effective            VARCHAR(20)     NOT NULL DEFAULT 'NOT_TESTED',
    operating_effective         VARCHAR(20)     NOT NULL DEFAULT 'NOT_TESTED',
    sample_size                 INTEGER,
    exceptions_found            INTEGER         NOT NULL DEFAULT 0,
    test_description            TEXT,
    tested_by                   UUID,
    tested_at                   TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_ct_dates CHECK (
        test_period_end >= test_period_start
    ),
    CONSTRAINT chk_p048_ct_design CHECK (
        design_effective IN (
            'EFFECTIVE', 'PARTIALLY_EFFECTIVE',
            'INEFFECTIVE', 'NOT_TESTED'
        )
    ),
    CONSTRAINT chk_p048_ct_operating CHECK (
        operating_effective IN (
            'EFFECTIVE', 'PARTIALLY_EFFECTIVE',
            'INEFFECTIVE', 'NOT_TESTED'
        )
    ),
    CONSTRAINT chk_p048_ct_sample CHECK (
        sample_size IS NULL OR sample_size >= 0
    ),
    CONSTRAINT chk_p048_ct_exceptions CHECK (
        exceptions_found >= 0
    ),
    CONSTRAINT chk_p048_ct_exceptions_le_sample CHECK (
        sample_size IS NULL OR exceptions_found <= sample_size
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_ct_tenant            ON ghg_assurance.gl_ap_control_tests(tenant_id);
CREATE INDEX idx_p048_ct_control           ON ghg_assurance.gl_ap_control_tests(control_id);
CREATE INDEX idx_p048_ct_period_start      ON ghg_assurance.gl_ap_control_tests(test_period_start);
CREATE INDEX idx_p048_ct_period_end        ON ghg_assurance.gl_ap_control_tests(test_period_end);
CREATE INDEX idx_p048_ct_design            ON ghg_assurance.gl_ap_control_tests(design_effective);
CREATE INDEX idx_p048_ct_operating         ON ghg_assurance.gl_ap_control_tests(operating_effective);
CREATE INDEX idx_p048_ct_tested_by         ON ghg_assurance.gl_ap_control_tests(tested_by);
CREATE INDEX idx_p048_ct_tested_at         ON ghg_assurance.gl_ap_control_tests(tested_at DESC);
CREATE INDEX idx_p048_ct_created           ON ghg_assurance.gl_ap_control_tests(created_at DESC);
CREATE INDEX idx_p048_ct_provenance        ON ghg_assurance.gl_ap_control_tests(provenance_hash);
CREATE INDEX idx_p048_ct_metadata          ON ghg_assurance.gl_ap_control_tests USING GIN(metadata);

-- Composite: control + period for time series
CREATE INDEX idx_p048_ct_ctrl_period       ON ghg_assurance.gl_ap_control_tests(control_id, test_period_start DESC);

-- Composite: design + operating for effectiveness matrix
CREATE INDEX idx_p048_ct_effectiveness     ON ghg_assurance.gl_ap_control_tests(design_effective, operating_effective);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_ct_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_control_tests
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_assurance.gl_ap_deficiencies
-- =============================================================================
-- Control deficiencies identified through testing. Classified by severity
-- (deficiency, significant deficiency, material weakness per COSO/PCAOB
-- framework), with root cause analysis, remediation planning, and
-- lifecycle management through to verified closure.

CREATE TABLE ghg_assurance.gl_ap_deficiencies (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    control_test_id             UUID            NOT NULL REFERENCES ghg_assurance.gl_ap_control_tests(id) ON DELETE CASCADE,
    deficiency_type             VARCHAR(30)     NOT NULL,
    description                 TEXT            NOT NULL,
    root_cause                  TEXT,
    remediation_plan            TEXT,
    remediation_owner           UUID,
    remediation_deadline        DATE,
    status                      VARCHAR(20)     NOT NULL DEFAULT 'OPEN',
    verified_by                 UUID,
    verified_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p048_def_type CHECK (
        deficiency_type IN (
            'DEFICIENCY', 'SIGNIFICANT_DEFICIENCY', 'MATERIAL_WEAKNESS'
        )
    ),
    CONSTRAINT chk_p048_def_status CHECK (
        status IN (
            'OPEN', 'IN_PROGRESS', 'REMEDIATED', 'VERIFIED_CLOSED'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p048_def_tenant           ON ghg_assurance.gl_ap_deficiencies(tenant_id);
CREATE INDEX idx_p048_def_test             ON ghg_assurance.gl_ap_deficiencies(control_test_id);
CREATE INDEX idx_p048_def_type             ON ghg_assurance.gl_ap_deficiencies(deficiency_type);
CREATE INDEX idx_p048_def_status           ON ghg_assurance.gl_ap_deficiencies(status);
CREATE INDEX idx_p048_def_owner            ON ghg_assurance.gl_ap_deficiencies(remediation_owner);
CREATE INDEX idx_p048_def_deadline         ON ghg_assurance.gl_ap_deficiencies(remediation_deadline);
CREATE INDEX idx_p048_def_verified_by      ON ghg_assurance.gl_ap_deficiencies(verified_by);
CREATE INDEX idx_p048_def_created          ON ghg_assurance.gl_ap_deficiencies(created_at DESC);
CREATE INDEX idx_p048_def_metadata         ON ghg_assurance.gl_ap_deficiencies USING GIN(metadata);

-- Composite: type + status for severity-based tracking
CREATE INDEX idx_p048_def_type_status      ON ghg_assurance.gl_ap_deficiencies(deficiency_type, status);

-- Composite: status + deadline for overdue tracking
CREATE INDEX idx_p048_def_status_deadline  ON ghg_assurance.gl_ap_deficiencies(status, remediation_deadline);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p048_def_updated
    BEFORE UPDATE ON ghg_assurance.gl_ap_deficiencies
    FOR EACH ROW EXECUTE FUNCTION ghg_assurance.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_assurance.gl_ap_controls ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_control_tests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_assurance.gl_ap_deficiencies ENABLE ROW LEVEL SECURITY;

CREATE POLICY p048_ctrl_tenant_isolation
    ON ghg_assurance.gl_ap_controls
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ctrl_service_bypass
    ON ghg_assurance.gl_ap_controls
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_ct_tenant_isolation
    ON ghg_assurance.gl_ap_control_tests
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_ct_service_bypass
    ON ghg_assurance.gl_ap_control_tests
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p048_def_tenant_isolation
    ON ghg_assurance.gl_ap_deficiencies
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p048_def_service_bypass
    ON ghg_assurance.gl_ap_deficiencies
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_controls TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_control_tests TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_assurance.gl_ap_deficiencies TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_assurance.gl_ap_controls IS
    'Internal control definitions for GHG reporting with category, type, maturity level, and key control designation.';
COMMENT ON TABLE ghg_assurance.gl_ap_control_tests IS
    'Control testing records evaluating design and operating effectiveness with sample-based exception tracking.';
COMMENT ON TABLE ghg_assurance.gl_ap_deficiencies IS
    'Control deficiencies with COSO/PCAOB severity classification, root cause analysis, and remediation lifecycle.';

COMMENT ON COLUMN ghg_assurance.gl_ap_controls.control_code IS 'Unique control code per tenant, e.g. DC-01, CALC-03, REV-02, RPT-01, IT-04.';
COMMENT ON COLUMN ghg_assurance.gl_ap_controls.category IS 'Control category: DATA_COLLECTION, CALCULATION, REVIEW, REPORTING, IT_GENERAL.';
COMMENT ON COLUMN ghg_assurance.gl_ap_controls.control_type IS 'PREVENTIVE (stops errors), DETECTIVE (identifies errors), CORRECTIVE (fixes errors).';
COMMENT ON COLUMN ghg_assurance.gl_ap_controls.maturity_level IS 'CMMI maturity: LEVEL_1 (initial), LEVEL_2 (managed), LEVEL_3 (defined), LEVEL_4 (quantitatively managed), LEVEL_5 (optimising).';
COMMENT ON COLUMN ghg_assurance.gl_ap_controls.is_key_control IS 'True for controls that are critical to preventing/detecting material misstatement.';
COMMENT ON COLUMN ghg_assurance.gl_ap_control_tests.design_effective IS 'Design effectiveness: EFFECTIVE, PARTIALLY_EFFECTIVE, INEFFECTIVE, NOT_TESTED.';
COMMENT ON COLUMN ghg_assurance.gl_ap_control_tests.operating_effective IS 'Operating effectiveness: EFFECTIVE, PARTIALLY_EFFECTIVE, INEFFECTIVE, NOT_TESTED.';
COMMENT ON COLUMN ghg_assurance.gl_ap_control_tests.exceptions_found IS 'Number of exceptions found during sample testing.';
COMMENT ON COLUMN ghg_assurance.gl_ap_deficiencies.deficiency_type IS 'DEFICIENCY (minor control gap), SIGNIFICANT_DEFICIENCY (more than remote likelihood), MATERIAL_WEAKNESS (reasonable possibility of material misstatement).';
COMMENT ON COLUMN ghg_assurance.gl_ap_deficiencies.status IS 'Lifecycle: OPEN (identified), IN_PROGRESS (remediation underway), REMEDIATED (fixed), VERIFIED_CLOSED (independently verified).';
