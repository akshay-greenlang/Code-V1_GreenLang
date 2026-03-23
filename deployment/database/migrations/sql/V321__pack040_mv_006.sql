-- =============================================================================
-- V321: PACK-040 M&V Pack - IPMVP Option Records
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for IPMVP option implementation including option definitions,
-- option evaluations for each ECM, option selection decisions, measurement
-- boundary definitions, and stipulated values for Option A.
--
-- Tables (5):
--   1. pack040_mv.mv_ipmvp_options
--   2. pack040_mv.mv_option_evaluations
--   3. pack040_mv.mv_option_selections
--   4. pack040_mv.mv_boundary_definitions
--   5. pack040_mv.mv_stipulated_values
--
-- Previous: V320__pack040_mv_005.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_ipmvp_options
-- =============================================================================
-- Master table of IPMVP option assignments per project-ECM combination.
-- Records which IPMVP option (A, B, C, or D) is applied to each ECM and
-- the rationale for the selection. This is the primary option tracking table.

CREATE TABLE pack040_mv.mv_ipmvp_options (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            NOT NULL REFERENCES pack040_mv.mv_ecms(id) ON DELETE CASCADE,
    project_ecm_map_id          UUID            REFERENCES pack040_mv.mv_project_ecm_map(id) ON DELETE SET NULL,
    -- Option assignment
    ipmvp_option                VARCHAR(10)     NOT NULL,
    option_name                 VARCHAR(100)    NOT NULL,
    option_description          TEXT,
    -- Option characteristics
    measurement_boundary        VARCHAR(50)     NOT NULL DEFAULT 'RETROFIT_ISOLATION',
    parameter_measurement       VARCHAR(50)     NOT NULL DEFAULT 'ALL_PARAMETERS',
    uses_stipulated_values      BOOLEAN         NOT NULL DEFAULT false,
    uses_simulation             BOOLEAN         NOT NULL DEFAULT false,
    requires_sub_metering       BOOLEAN         NOT NULL DEFAULT false,
    requires_calibrated_model   BOOLEAN         NOT NULL DEFAULT false,
    -- Expected performance
    expected_accuracy           VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    estimated_mv_cost           NUMERIC(18,2),
    estimated_mv_cost_pct_savings NUMERIC(8,4),
    estimated_uncertainty_pct   NUMERIC(8,4),
    -- Compliance
    compliance_standard         VARCHAR(50),
    compliance_requirements     JSONB           DEFAULT '[]',
    -- Status
    option_status               VARCHAR(20)     NOT NULL DEFAULT 'PROPOSED',
    selection_date              DATE,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    selection_rationale         TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_io_option CHECK (
        ipmvp_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p040_io_boundary CHECK (
        measurement_boundary IN (
            'RETROFIT_ISOLATION', 'WHOLE_FACILITY', 'SIMULATED'
        )
    ),
    CONSTRAINT chk_p040_io_parameter CHECK (
        parameter_measurement IN (
            'KEY_PARAMETER', 'ALL_PARAMETERS', 'UTILITY_METERS',
            'CALIBRATED_SIMULATION'
        )
    ),
    CONSTRAINT chk_p040_io_accuracy CHECK (
        expected_accuracy IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p040_io_status CHECK (
        option_status IN (
            'PROPOSED', 'EVALUATED', 'SELECTED', 'APPROVED',
            'ACTIVE', 'COMPLETED', 'CHANGED'
        )
    ),
    CONSTRAINT chk_p040_io_cost_pct CHECK (
        estimated_mv_cost_pct_savings IS NULL OR
        (estimated_mv_cost_pct_savings >= 0 AND estimated_mv_cost_pct_savings <= 100)
    ),
    CONSTRAINT chk_p040_io_unc CHECK (
        estimated_uncertainty_pct IS NULL OR
        (estimated_uncertainty_pct >= 0 AND estimated_uncertainty_pct <= 200)
    ),
    CONSTRAINT uq_p040_io_project_ecm UNIQUE (project_id, ecm_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_io_tenant            ON pack040_mv.mv_ipmvp_options(tenant_id);
CREATE INDEX idx_p040_io_project           ON pack040_mv.mv_ipmvp_options(project_id);
CREATE INDEX idx_p040_io_ecm               ON pack040_mv.mv_ipmvp_options(ecm_id);
CREATE INDEX idx_p040_io_option            ON pack040_mv.mv_ipmvp_options(ipmvp_option);
CREATE INDEX idx_p040_io_boundary          ON pack040_mv.mv_ipmvp_options(measurement_boundary);
CREATE INDEX idx_p040_io_status            ON pack040_mv.mv_ipmvp_options(option_status);
CREATE INDEX idx_p040_io_created           ON pack040_mv.mv_ipmvp_options(created_at DESC);

-- Composite: project + active options
CREATE INDEX idx_p040_io_project_active    ON pack040_mv.mv_ipmvp_options(project_id, ipmvp_option)
    WHERE option_status IN ('SELECTED', 'APPROVED', 'ACTIVE');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_io_updated
    BEFORE UPDATE ON pack040_mv.mv_ipmvp_options
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_option_evaluations
-- =============================================================================
-- Detailed evaluation of each IPMVP option for a given ECM. Scores each
-- option on multiple criteria (suitability, accuracy, cost, complexity)
-- to support systematic option selection decisions.

CREATE TABLE pack040_mv.mv_option_evaluations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            NOT NULL REFERENCES pack040_mv.mv_ecms(id) ON DELETE CASCADE,
    ipmvp_option                VARCHAR(10)     NOT NULL,
    evaluation_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    evaluator                   VARCHAR(255),
    -- Suitability scoring (0-100)
    suitability_score           NUMERIC(5,2)    NOT NULL,
    suitability_rationale       TEXT,
    -- Accuracy scoring
    accuracy_score              NUMERIC(5,2)    NOT NULL,
    estimated_accuracy_pct      NUMERIC(8,4),
    accuracy_rationale          TEXT,
    -- Cost scoring
    cost_score                  NUMERIC(5,2)    NOT NULL,
    estimated_cost              NUMERIC(18,2),
    cost_pct_of_annual_savings  NUMERIC(8,4),
    cost_rationale              TEXT,
    -- Complexity scoring
    complexity_score            NUMERIC(5,2)    NOT NULL,
    implementation_difficulty   VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    complexity_rationale        TEXT,
    -- Risk scoring
    risk_score                  NUMERIC(5,2)    NOT NULL DEFAULT 50.0,
    key_risks                   JSONB           DEFAULT '[]',
    risk_rationale              TEXT,
    -- Overall
    overall_score               NUMERIC(5,2)    NOT NULL,
    scoring_weights             JSONB           DEFAULT '{"suitability":0.30,"accuracy":0.25,"cost":0.20,"complexity":0.15,"risk":0.10}',
    recommendation              VARCHAR(20)     NOT NULL DEFAULT 'NEUTRAL',
    is_recommended              BOOLEAN         NOT NULL DEFAULT false,
    -- Evaluation criteria
    ecm_pct_of_facility_energy  NUMERIC(8,4),
    isolation_feasible          BOOLEAN,
    sub_metering_available      BOOLEAN,
    baseline_data_available     BOOLEAN,
    interactive_effects_exist   BOOLEAN,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_oe_option CHECK (
        ipmvp_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p040_oe_suit_score CHECK (
        suitability_score >= 0 AND suitability_score <= 100
    ),
    CONSTRAINT chk_p040_oe_acc_score CHECK (
        accuracy_score >= 0 AND accuracy_score <= 100
    ),
    CONSTRAINT chk_p040_oe_cost_score CHECK (
        cost_score >= 0 AND cost_score <= 100
    ),
    CONSTRAINT chk_p040_oe_comp_score CHECK (
        complexity_score >= 0 AND complexity_score <= 100
    ),
    CONSTRAINT chk_p040_oe_risk_score CHECK (
        risk_score >= 0 AND risk_score <= 100
    ),
    CONSTRAINT chk_p040_oe_overall CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_p040_oe_difficulty CHECK (
        implementation_difficulty IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p040_oe_recommendation CHECK (
        recommendation IN (
            'STRONGLY_RECOMMENDED', 'RECOMMENDED', 'NEUTRAL',
            'NOT_RECOMMENDED', 'UNSUITABLE'
        )
    ),
    CONSTRAINT chk_p040_oe_ecm_pct CHECK (
        ecm_pct_of_facility_energy IS NULL OR
        (ecm_pct_of_facility_energy >= 0 AND ecm_pct_of_facility_energy <= 100)
    ),
    CONSTRAINT uq_p040_oe_project_ecm_option UNIQUE (project_id, ecm_id, ipmvp_option)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_oe_tenant            ON pack040_mv.mv_option_evaluations(tenant_id);
CREATE INDEX idx_p040_oe_project           ON pack040_mv.mv_option_evaluations(project_id);
CREATE INDEX idx_p040_oe_ecm               ON pack040_mv.mv_option_evaluations(ecm_id);
CREATE INDEX idx_p040_oe_option            ON pack040_mv.mv_option_evaluations(ipmvp_option);
CREATE INDEX idx_p040_oe_recommended       ON pack040_mv.mv_option_evaluations(is_recommended) WHERE is_recommended = true;
CREATE INDEX idx_p040_oe_score             ON pack040_mv.mv_option_evaluations(overall_score DESC);
CREATE INDEX idx_p040_oe_created           ON pack040_mv.mv_option_evaluations(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_oe_updated
    BEFORE UPDATE ON pack040_mv.mv_option_evaluations
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_option_selections
-- =============================================================================
-- Formal option selection records documenting the decision to use a specific
-- IPMVP option for each ECM. Captures the decision rationale, approval chain,
-- and any conditions or caveats associated with the selection.

CREATE TABLE pack040_mv.mv_option_selections (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            NOT NULL REFERENCES pack040_mv.mv_ecms(id) ON DELETE CASCADE,
    ipmvp_option_id             UUID            NOT NULL REFERENCES pack040_mv.mv_ipmvp_options(id) ON DELETE CASCADE,
    evaluation_id               UUID            REFERENCES pack040_mv.mv_option_evaluations(id) ON DELETE SET NULL,
    selected_option             VARCHAR(10)     NOT NULL,
    selection_date              DATE            NOT NULL,
    -- Decision details
    primary_reason              VARCHAR(100)    NOT NULL,
    secondary_reasons           JSONB           DEFAULT '[]',
    decision_rationale          TEXT            NOT NULL,
    alternatives_considered     JSONB           DEFAULT '[]',
    -- Conditions
    conditions                  JSONB           DEFAULT '[]',
    prerequisites               JSONB           DEFAULT '[]',
    limitations                 TEXT,
    -- Approval
    proposed_by                 VARCHAR(255)    NOT NULL,
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    selection_status            VARCHAR(20)     NOT NULL DEFAULT 'PROPOSED',
    -- Comparison summary
    option_a_score              NUMERIC(5,2),
    option_b_score              NUMERIC(5,2),
    option_c_score              NUMERIC(5,2),
    option_d_score              NUMERIC(5,2),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_os_option CHECK (
        selected_option IN ('OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D')
    ),
    CONSTRAINT chk_p040_os_status CHECK (
        selection_status IN (
            'PROPOSED', 'REVIEWED', 'APPROVED', 'REJECTED', 'SUPERSEDED'
        )
    ),
    CONSTRAINT chk_p040_os_a_score CHECK (
        option_a_score IS NULL OR (option_a_score >= 0 AND option_a_score <= 100)
    ),
    CONSTRAINT chk_p040_os_b_score CHECK (
        option_b_score IS NULL OR (option_b_score >= 0 AND option_b_score <= 100)
    ),
    CONSTRAINT chk_p040_os_c_score CHECK (
        option_c_score IS NULL OR (option_c_score >= 0 AND option_c_score <= 100)
    ),
    CONSTRAINT chk_p040_os_d_score CHECK (
        option_d_score IS NULL OR (option_d_score >= 0 AND option_d_score <= 100)
    ),
    CONSTRAINT uq_p040_os_project_ecm UNIQUE (project_id, ecm_id, selection_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_os_tenant            ON pack040_mv.mv_option_selections(tenant_id);
CREATE INDEX idx_p040_os_project           ON pack040_mv.mv_option_selections(project_id);
CREATE INDEX idx_p040_os_ecm               ON pack040_mv.mv_option_selections(ecm_id);
CREATE INDEX idx_p040_os_option_id         ON pack040_mv.mv_option_selections(ipmvp_option_id);
CREATE INDEX idx_p040_os_selected          ON pack040_mv.mv_option_selections(selected_option);
CREATE INDEX idx_p040_os_status            ON pack040_mv.mv_option_selections(selection_status);
CREATE INDEX idx_p040_os_created           ON pack040_mv.mv_option_selections(created_at DESC);

-- Composite: project + approved selections
CREATE INDEX idx_p040_os_project_approved  ON pack040_mv.mv_option_selections(project_id, ecm_id)
    WHERE selection_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_os_updated
    BEFORE UPDATE ON pack040_mv.mv_option_selections
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_boundary_definitions
-- =============================================================================
-- Detailed measurement boundary specifications for IPMVP options. Defines
-- exact energy flows crossing the boundary, metering points, and what is
-- included vs. excluded within the boundary for savings calculation.

CREATE TABLE pack040_mv.mv_boundary_definitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ipmvp_option_id             UUID            NOT NULL REFERENCES pack040_mv.mv_ipmvp_options(id) ON DELETE CASCADE,
    boundary_name               VARCHAR(255)    NOT NULL,
    boundary_level              VARCHAR(30)     NOT NULL DEFAULT 'SYSTEM',
    -- Energy flows
    energy_inflows              JSONB           NOT NULL DEFAULT '[]',
    energy_outflows             JSONB           NOT NULL DEFAULT '[]',
    -- Included/excluded
    included_equipment          JSONB           DEFAULT '[]',
    excluded_equipment          JSONB           DEFAULT '[]',
    included_energy_types       VARCHAR(50)[]   DEFAULT '{ELECTRICITY}',
    -- Metering
    metering_points             JSONB           DEFAULT '[]',
    meter_ids                   UUID[]          DEFAULT '{}',
    metering_approach           VARCHAR(50)     NOT NULL DEFAULT 'DIRECT',
    -- Physical definition
    physical_description        TEXT,
    diagram_reference           VARCHAR(255),
    floor_area_m2               NUMERIC(12,2),
    building_section            VARCHAR(100),
    -- Variables within boundary
    independent_variables       JSONB           DEFAULT '[]',
    static_factors              JSONB           DEFAULT '[]',
    -- Validation
    boundary_validated          BOOLEAN         NOT NULL DEFAULT false,
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_bdef_level CHECK (
        boundary_level IN (
            'COMPONENT', 'SYSTEM', 'ZONE', 'FLOOR', 'BUILDING',
            'CAMPUS', 'WHOLE_FACILITY'
        )
    ),
    CONSTRAINT chk_p040_bdef_metering CHECK (
        metering_approach IN (
            'DIRECT', 'SUB_METERED', 'CALCULATED', 'ESTIMATED',
            'STIPULATED', 'SIMULATED'
        )
    ),
    CONSTRAINT chk_p040_bdef_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT uq_p040_bdef_option_name UNIQUE (ipmvp_option_id, boundary_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_bdef_tenant          ON pack040_mv.mv_boundary_definitions(tenant_id);
CREATE INDEX idx_p040_bdef_project         ON pack040_mv.mv_boundary_definitions(project_id);
CREATE INDEX idx_p040_bdef_option          ON pack040_mv.mv_boundary_definitions(ipmvp_option_id);
CREATE INDEX idx_p040_bdef_level           ON pack040_mv.mv_boundary_definitions(boundary_level);
CREATE INDEX idx_p040_bdef_active          ON pack040_mv.mv_boundary_definitions(is_active) WHERE is_active = true;
CREATE INDEX idx_p040_bdef_meters          ON pack040_mv.mv_boundary_definitions USING GIN(meter_ids);
CREATE INDEX idx_p040_bdef_created         ON pack040_mv.mv_boundary_definitions(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_bdef_updated
    BEFORE UPDATE ON pack040_mv.mv_boundary_definitions
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_stipulated_values
-- =============================================================================
-- Stipulated parameter values for IPMVP Option A where not all parameters
-- are measured directly. Records the stipulated value, its source/basis,
-- associated uncertainty, and conditions under which the stipulation is valid.

CREATE TABLE pack040_mv.mv_stipulated_values (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            NOT NULL REFERENCES pack040_mv.mv_ecms(id) ON DELETE CASCADE,
    ipmvp_option_id             UUID            NOT NULL REFERENCES pack040_mv.mv_ipmvp_options(id) ON DELETE CASCADE,
    -- Stipulated parameter
    parameter_name              VARCHAR(100)    NOT NULL,
    parameter_description       TEXT,
    parameter_category          VARCHAR(50)     NOT NULL DEFAULT 'OPERATING',
    stipulated_value            NUMERIC(18,6)   NOT NULL,
    parameter_unit              VARCHAR(50)     NOT NULL,
    -- Basis for stipulation
    stipulation_basis           VARCHAR(50)     NOT NULL DEFAULT 'MANUFACTURER_DATA',
    basis_description           TEXT            NOT NULL,
    source_document             VARCHAR(255),
    source_measurement_date     DATE,
    -- Uncertainty
    uncertainty_pct             NUMERIC(8,4),
    confidence_level_pct        NUMERIC(5,2)    DEFAULT 90.0,
    value_lower_bound           NUMERIC(18,6),
    value_upper_bound           NUMERIC(18,6),
    -- Conditions
    valid_conditions            TEXT,
    condition_parameters        JSONB           DEFAULT '{}',
    sensitivity_to_savings_pct  NUMERIC(8,4),
    -- Measurement alternative
    measurement_cost_estimate   NUMERIC(18,2),
    measurement_would_reduce_unc BOOLEAN        DEFAULT true,
    measurement_reduction_pct   NUMERIC(8,4),
    -- Verification
    verification_frequency      VARCHAR(30),
    last_verified_date          DATE,
    verification_result         VARCHAR(20),
    -- Status
    stipulation_status          VARCHAR(20)     NOT NULL DEFAULT 'PROPOSED',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_sv_category CHECK (
        parameter_category IN (
            'OPERATING', 'EQUIPMENT', 'ENVIRONMENTAL', 'EFFICIENCY',
            'LOAD_FACTOR', 'SCHEDULE', 'AREA', 'OCCUPANCY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_sv_basis CHECK (
        stipulation_basis IN (
            'MANUFACTURER_DATA', 'ENGINEERING_REFERENCE', 'SPOT_MEASUREMENT',
            'SHORT_TERM_MEASUREMENT', 'HISTORICAL_DATA', 'BUILDING_RECORDS',
            'NAMEPLATE', 'INDUSTRY_STANDARD', 'EXPERT_JUDGMENT'
        )
    ),
    CONSTRAINT chk_p040_sv_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    ),
    CONSTRAINT chk_p040_sv_bounds CHECK (
        value_lower_bound IS NULL OR value_upper_bound IS NULL OR
        value_lower_bound <= value_upper_bound
    ),
    CONSTRAINT chk_p040_sv_sensitivity CHECK (
        sensitivity_to_savings_pct IS NULL OR
        (sensitivity_to_savings_pct >= 0 AND sensitivity_to_savings_pct <= 100)
    ),
    CONSTRAINT chk_p040_sv_verif_freq CHECK (
        verification_frequency IS NULL OR verification_frequency IN (
            'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'ANNUALLY', 'ONCE'
        )
    ),
    CONSTRAINT chk_p040_sv_verif_result CHECK (
        verification_result IS NULL OR verification_result IN (
            'CONFIRMED', 'REVISED', 'INVALIDATED', 'PENDING'
        )
    ),
    CONSTRAINT chk_p040_sv_status CHECK (
        stipulation_status IN (
            'PROPOSED', 'REVIEWED', 'APPROVED', 'REJECTED',
            'REVISED', 'VERIFIED', 'EXPIRED'
        )
    ),
    CONSTRAINT uq_p040_sv_ecm_param UNIQUE (ecm_id, ipmvp_option_id, parameter_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_sv_tenant            ON pack040_mv.mv_stipulated_values(tenant_id);
CREATE INDEX idx_p040_sv_project           ON pack040_mv.mv_stipulated_values(project_id);
CREATE INDEX idx_p040_sv_ecm               ON pack040_mv.mv_stipulated_values(ecm_id);
CREATE INDEX idx_p040_sv_option            ON pack040_mv.mv_stipulated_values(ipmvp_option_id);
CREATE INDEX idx_p040_sv_category          ON pack040_mv.mv_stipulated_values(parameter_category);
CREATE INDEX idx_p040_sv_basis             ON pack040_mv.mv_stipulated_values(stipulation_basis);
CREATE INDEX idx_p040_sv_status            ON pack040_mv.mv_stipulated_values(stipulation_status);
CREATE INDEX idx_p040_sv_created           ON pack040_mv.mv_stipulated_values(created_at DESC);

-- Composite: ECM + approved stipulations
CREATE INDEX idx_p040_sv_ecm_approved      ON pack040_mv.mv_stipulated_values(ecm_id, parameter_name)
    WHERE stipulation_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_sv_updated
    BEFORE UPDATE ON pack040_mv.mv_stipulated_values
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_ipmvp_options ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_option_evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_option_selections ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_boundary_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_stipulated_values ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_io_tenant_isolation
    ON pack040_mv.mv_ipmvp_options
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_io_service_bypass
    ON pack040_mv.mv_ipmvp_options
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_oe_tenant_isolation
    ON pack040_mv.mv_option_evaluations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_oe_service_bypass
    ON pack040_mv.mv_option_evaluations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_os_tenant_isolation
    ON pack040_mv.mv_option_selections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_os_service_bypass
    ON pack040_mv.mv_option_selections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_bdef_tenant_isolation
    ON pack040_mv.mv_boundary_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_bdef_service_bypass
    ON pack040_mv.mv_boundary_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_sv_tenant_isolation
    ON pack040_mv.mv_stipulated_values
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_sv_service_bypass
    ON pack040_mv.mv_stipulated_values
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_ipmvp_options TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_option_evaluations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_option_selections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_boundary_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_stipulated_values TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_ipmvp_options IS
    'IPMVP option assignments per project-ECM with measurement boundary, parameter measurement approach, and expected accuracy.';
COMMENT ON TABLE pack040_mv.mv_option_evaluations IS
    'Multi-criteria evaluation of each IPMVP option per ECM: suitability, accuracy, cost, complexity, and risk scoring.';
COMMENT ON TABLE pack040_mv.mv_option_selections IS
    'Formal option selection decisions with rationale, approval chain, comparison scores, and conditions.';
COMMENT ON TABLE pack040_mv.mv_boundary_definitions IS
    'Detailed measurement boundary specifications: energy flows, metering points, included/excluded equipment, and physical description.';
COMMENT ON TABLE pack040_mv.mv_stipulated_values IS
    'IPMVP Option A stipulated parameter values with basis, uncertainty, sensitivity analysis, and verification schedule.';

COMMENT ON COLUMN pack040_mv.mv_ipmvp_options.ipmvp_option IS 'IPMVP option: OPTION_A (key parameter), OPTION_B (all parameter), OPTION_C (whole facility), OPTION_D (simulation).';
COMMENT ON COLUMN pack040_mv.mv_ipmvp_options.uses_stipulated_values IS 'Whether the option uses stipulated (non-measured) values for some parameters (Option A).';
COMMENT ON COLUMN pack040_mv.mv_ipmvp_options.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_option_evaluations.overall_score IS 'Weighted composite score (0-100) for option comparison and ranking.';
COMMENT ON COLUMN pack040_mv.mv_option_evaluations.ecm_pct_of_facility_energy IS 'ECM energy share of whole facility - key factor for Option C viability.';

COMMENT ON COLUMN pack040_mv.mv_stipulated_values.stipulation_basis IS 'Source of stipulated value: MANUFACTURER_DATA, SPOT_MEASUREMENT, ENGINEERING_REFERENCE, etc.';
COMMENT ON COLUMN pack040_mv.mv_stipulated_values.sensitivity_to_savings_pct IS 'Percentage change in savings for a 10% change in stipulated value (sensitivity analysis).';
