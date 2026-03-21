-- =============================================================================
-- V258: PACK-034 ISO 50001 Energy Management System - Baseline Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    003 of 010
-- Date:         March 2026
--
-- Creates energy baseline tables for establishing reference energy performance,
-- regression models, and relevant variables. Baselines are critical for
-- measuring energy performance improvement per ISO 50001 Clause 6.5.
--
-- Tables (3):
--   1. pack034_iso50001.energy_baselines
--   2. pack034_iso50001.baseline_models
--   3. pack034_iso50001.relevant_variables
--
-- Previous: V257__pack034_iso50001_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.energy_baselines
-- =============================================================================
-- Energy baselines (EnBs) as defined by ISO 50001. Quantitative references
-- for comparing energy performance, with statistical model quality metrics
-- (R-squared, CV(RMSE), p-value).

CREATE TABLE pack034_iso50001.energy_baselines (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    baseline_name               VARCHAR(500)    NOT NULL,
    baseline_period_start       DATE            NOT NULL,
    baseline_period_end         DATE            NOT NULL,
    energy_type                 VARCHAR(100)    NOT NULL,
    total_energy_kwh            DECIMAL(18,4),
    model_type                  VARCHAR(30)     NOT NULL DEFAULT 'simple_mean',
    r_squared                   DECIMAL(8,6),
    cv_rmse                     DECIMAL(8,4),
    p_value                     DECIMAL(10,8),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'draft',
    approved_by                 UUID,
    approved_date               DATE,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_bl_model_type CHECK (
        model_type IN ('simple_mean', 'single_variable', 'multi_variable')
    ),
    CONSTRAINT chk_p034_bl_period CHECK (
        baseline_period_start < baseline_period_end
    ),
    CONSTRAINT chk_p034_bl_energy CHECK (
        total_energy_kwh IS NULL OR total_energy_kwh >= 0
    ),
    CONSTRAINT chk_p034_bl_r_squared CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p034_bl_cv_rmse CHECK (
        cv_rmse IS NULL OR cv_rmse >= 0
    ),
    CONSTRAINT chk_p034_bl_p_value CHECK (
        p_value IS NULL OR (p_value >= 0 AND p_value <= 1)
    ),
    CONSTRAINT chk_p034_bl_status CHECK (
        status IN ('draft', 'approved', 'adjusted', 'superseded')
    ),
    CONSTRAINT chk_p034_bl_approved CHECK (
        (status != 'approved') OR (approved_by IS NOT NULL AND approved_date IS NOT NULL)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_bl_enms            ON pack034_iso50001.energy_baselines(enms_id);
CREATE INDEX idx_p034_bl_energy_type     ON pack034_iso50001.energy_baselines(energy_type);
CREATE INDEX idx_p034_bl_model_type      ON pack034_iso50001.energy_baselines(model_type);
CREATE INDEX idx_p034_bl_status          ON pack034_iso50001.energy_baselines(status);
CREATE INDEX idx_p034_bl_period          ON pack034_iso50001.energy_baselines(baseline_period_start, baseline_period_end);
CREATE INDEX idx_p034_bl_r_squared       ON pack034_iso50001.energy_baselines(r_squared DESC);
CREATE INDEX idx_p034_bl_created         ON pack034_iso50001.energy_baselines(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_bl_updated
    BEFORE UPDATE ON pack034_iso50001.energy_baselines
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.baseline_models
-- =============================================================================
-- Regression model coefficients for energy baselines. Each row represents
-- one independent variable in the regression equation:
-- Energy = intercept + (coefficient_1 * variable_1) + (coefficient_2 * variable_2) + ...

CREATE TABLE pack034_iso50001.baseline_models (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id                 UUID            NOT NULL REFERENCES pack034_iso50001.energy_baselines(id) ON DELETE CASCADE,
    variable_name               VARCHAR(255)    NOT NULL,
    coefficient                 DECIMAL(18,8)   NOT NULL,
    intercept                   DECIMAL(18,8),
    standard_error              DECIMAL(12,8),
    t_statistic                 DECIMAL(10,6),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_bm_std_err CHECK (
        standard_error IS NULL OR standard_error >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_bm_baseline        ON pack034_iso50001.baseline_models(baseline_id);
CREATE INDEX idx_p034_bm_variable        ON pack034_iso50001.baseline_models(variable_name);
CREATE INDEX idx_p034_bm_created         ON pack034_iso50001.baseline_models(created_at DESC);
CREATE UNIQUE INDEX idx_p034_bm_bl_var   ON pack034_iso50001.baseline_models(baseline_id, variable_name);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_bm_updated
    BEFORE UPDATE ON pack034_iso50001.baseline_models
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.relevant_variables
-- =============================================================================
-- Static factors and relevant variables that affect energy performance,
-- used in baseline normalization and adjustment per ISO 50001 Clause 6.5.

CREATE TABLE pack034_iso50001.relevant_variables (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id                 UUID            NOT NULL REFERENCES pack034_iso50001.energy_baselines(id) ON DELETE CASCADE,
    variable_name               VARCHAR(255)    NOT NULL,
    variable_type               VARCHAR(30)     NOT NULL,
    unit                        VARCHAR(50),
    description                 TEXT,
    data_source                 VARCHAR(255),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_rv_type CHECK (
        variable_type IN ('static_factor', 'relevant_variable')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_rv_baseline        ON pack034_iso50001.relevant_variables(baseline_id);
CREATE INDEX idx_p034_rv_type            ON pack034_iso50001.relevant_variables(variable_type);
CREATE INDEX idx_p034_rv_name            ON pack034_iso50001.relevant_variables(variable_name);
CREATE INDEX idx_p034_rv_created         ON pack034_iso50001.relevant_variables(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_rv_updated
    BEFORE UPDATE ON pack034_iso50001.relevant_variables
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.energy_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.baseline_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.relevant_variables ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_bl_tenant_isolation
    ON pack034_iso50001.energy_baselines
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_bl_service_bypass
    ON pack034_iso50001.energy_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_bm_tenant_isolation
    ON pack034_iso50001.baseline_models
    USING (baseline_id IN (
        SELECT id FROM pack034_iso50001.energy_baselines
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_bm_service_bypass
    ON pack034_iso50001.baseline_models
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_rv_tenant_isolation
    ON pack034_iso50001.relevant_variables
    USING (baseline_id IN (
        SELECT id FROM pack034_iso50001.energy_baselines
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_rv_service_bypass
    ON pack034_iso50001.relevant_variables
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.baseline_models TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.relevant_variables TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.energy_baselines IS
    'Energy baselines (EnBs) per ISO 50001 Clause 6.5 - quantitative references for comparing energy performance with regression model quality metrics.';

COMMENT ON TABLE pack034_iso50001.baseline_models IS
    'Regression model coefficients for energy baselines, representing terms in the energy consumption model equation.';

COMMENT ON TABLE pack034_iso50001.relevant_variables IS
    'Static factors and relevant variables affecting energy performance, used in baseline normalization and adjustment.';

COMMENT ON COLUMN pack034_iso50001.energy_baselines.model_type IS
    'Regression model type: simple_mean (no regression), single_variable, or multi_variable.';
COMMENT ON COLUMN pack034_iso50001.energy_baselines.r_squared IS
    'Coefficient of determination (0-1). Values >= 0.75 typically indicate an adequate model.';
COMMENT ON COLUMN pack034_iso50001.energy_baselines.cv_rmse IS
    'Coefficient of Variation of Root Mean Square Error (%). Values <= 25% typically indicate adequate model fit.';
COMMENT ON COLUMN pack034_iso50001.energy_baselines.p_value IS
    'Statistical significance of the model. Values <= 0.05 indicate the model is statistically significant.';
COMMENT ON COLUMN pack034_iso50001.energy_baselines.status IS
    'Baseline lifecycle: draft, approved (locked for use), adjusted (modified for static factor changes), superseded (replaced by newer baseline).';
COMMENT ON COLUMN pack034_iso50001.baseline_models.coefficient IS
    'Regression coefficient for this variable in the energy model equation.';
COMMENT ON COLUMN pack034_iso50001.baseline_models.intercept IS
    'Y-intercept of the regression equation (base energy consumption).';
COMMENT ON COLUMN pack034_iso50001.baseline_models.t_statistic IS
    'T-statistic for the coefficient, indicating statistical significance of the variable.';
COMMENT ON COLUMN pack034_iso50001.relevant_variables.variable_type IS
    'Type: static_factor (e.g., building area, equipment count) or relevant_variable (e.g., production volume, weather).';
COMMENT ON COLUMN pack034_iso50001.relevant_variables.data_source IS
    'Source of variable data (e.g., BMS, weather station, ERP system).';
