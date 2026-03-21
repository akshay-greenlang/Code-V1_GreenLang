-- =============================================================================
-- V259: PACK-034 ISO 50001 Energy Management System - EnPI Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Creates Energy Performance Indicator (EnPI) tables for tracking, measuring,
-- and targeting energy performance improvement per ISO 50001 Clause 6.4.
-- Supports absolute, intensity, regression, and proportion EnPI types.
--
-- Tables (3):
--   1. pack034_iso50001.energy_performance_indicators
--   2. pack034_iso50001.enpi_values
--   3. pack034_iso50001.enpi_targets
--
-- Previous: V258__pack034_iso50001_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.energy_performance_indicators
-- =============================================================================
-- Energy Performance Indicators (EnPIs) as defined by ISO 50001. Quantitative
-- values or measures of energy performance, used to demonstrate improvement
-- relative to the energy baseline.

CREATE TABLE pack034_iso50001.energy_performance_indicators (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    enpi_name                   VARCHAR(500)    NOT NULL,
    enpi_type                   VARCHAR(30)     NOT NULL,
    energy_type                 VARCHAR(100),
    numerator_unit              VARCHAR(50),
    denominator_unit            VARCHAR(50),
    target_value                DECIMAL(18,6),
    baseline_value              DECIMAL(18,6),
    current_value               DECIMAL(18,6),
    improvement_pct             DECIMAL(8,4),
    methodology_ref             TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_enpi_type CHECK (
        enpi_type IN ('absolute', 'intensity', 'regression', 'proportion')
    ),
    CONSTRAINT chk_p034_enpi_denominator CHECK (
        (enpi_type != 'intensity') OR (denominator_unit IS NOT NULL)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_enpi_enms          ON pack034_iso50001.energy_performance_indicators(enms_id);
CREATE INDEX idx_p034_enpi_type          ON pack034_iso50001.energy_performance_indicators(enpi_type);
CREATE INDEX idx_p034_enpi_energy_type   ON pack034_iso50001.energy_performance_indicators(energy_type);
CREATE INDEX idx_p034_enpi_improvement   ON pack034_iso50001.energy_performance_indicators(improvement_pct DESC);
CREATE INDEX idx_p034_enpi_created       ON pack034_iso50001.energy_performance_indicators(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_enpi_updated
    BEFORE UPDATE ON pack034_iso50001.energy_performance_indicators
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.enpi_values
-- =============================================================================
-- Time-series EnPI measurements with measured, normalized, and expected values
-- for variance analysis and performance trending.

CREATE TABLE pack034_iso50001.enpi_values (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enpi_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_performance_indicators(id) ON DELETE CASCADE,
    period_start                DATE            NOT NULL,
    period_end                  DATE            NOT NULL,
    measured_value              DECIMAL(18,6),
    normalized_value            DECIMAL(18,6),
    expected_value              DECIMAL(18,6),
    variance_pct                DECIMAL(8,4),
    data_quality_score          DECIMAL(4,2),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ev_period CHECK (
        period_start < period_end
    ),
    CONSTRAINT chk_p034_ev_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 10)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ev_enpi            ON pack034_iso50001.enpi_values(enpi_id);
CREATE INDEX idx_p034_ev_period          ON pack034_iso50001.enpi_values(period_start, period_end);
CREATE INDEX idx_p034_ev_variance        ON pack034_iso50001.enpi_values(variance_pct);
CREATE INDEX idx_p034_ev_quality         ON pack034_iso50001.enpi_values(data_quality_score);
CREATE INDEX idx_p034_ev_created         ON pack034_iso50001.enpi_values(created_at DESC);
CREATE UNIQUE INDEX idx_p034_ev_enpi_period
    ON pack034_iso50001.enpi_values(enpi_id, period_start, period_end);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ev_updated
    BEFORE UPDATE ON pack034_iso50001.enpi_values
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.enpi_targets
-- =============================================================================
-- Annual and interim EnPI targets with target basis documentation and
-- milestone tracking for energy performance objectives.

CREATE TABLE pack034_iso50001.enpi_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enpi_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_performance_indicators(id) ON DELETE CASCADE,
    target_year                 INTEGER         NOT NULL,
    target_value                DECIMAL(18,6)   NOT NULL,
    target_basis                TEXT,
    interim_targets_json        JSONB           DEFAULT '[]',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_et_year CHECK (
        target_year >= 2020 AND target_year <= 2100
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_et_enpi            ON pack034_iso50001.enpi_targets(enpi_id);
CREATE INDEX idx_p034_et_year            ON pack034_iso50001.enpi_targets(target_year);
CREATE INDEX idx_p034_et_created         ON pack034_iso50001.enpi_targets(created_at DESC);
CREATE UNIQUE INDEX idx_p034_et_enpi_year
    ON pack034_iso50001.enpi_targets(enpi_id, target_year);
CREATE INDEX idx_p034_et_interim         ON pack034_iso50001.enpi_targets USING GIN(interim_targets_json);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_et_updated
    BEFORE UPDATE ON pack034_iso50001.enpi_targets
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.energy_performance_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.enpi_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.enpi_targets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_enpi_tenant_isolation
    ON pack034_iso50001.energy_performance_indicators
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_enpi_service_bypass
    ON pack034_iso50001.energy_performance_indicators
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ev_tenant_isolation
    ON pack034_iso50001.enpi_values
    USING (enpi_id IN (
        SELECT id FROM pack034_iso50001.energy_performance_indicators
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_ev_service_bypass
    ON pack034_iso50001.enpi_values
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_et_tenant_isolation
    ON pack034_iso50001.enpi_targets
    USING (enpi_id IN (
        SELECT id FROM pack034_iso50001.energy_performance_indicators
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_et_service_bypass
    ON pack034_iso50001.enpi_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_performance_indicators TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.enpi_values TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.enpi_targets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.energy_performance_indicators IS
    'Energy Performance Indicators (EnPIs) per ISO 50001 Clause 6.4 - quantitative measures of energy performance improvement.';

COMMENT ON TABLE pack034_iso50001.enpi_values IS
    'Time-series EnPI measurements with measured, normalized, and expected values for variance analysis and trending.';

COMMENT ON TABLE pack034_iso50001.enpi_targets IS
    'Annual and interim EnPI targets with target basis documentation for energy performance objectives.';

COMMENT ON COLUMN pack034_iso50001.energy_performance_indicators.enpi_type IS
    'EnPI methodology type: absolute (total kWh), intensity (kWh/unit), regression (modeled), proportion (% of total).';
COMMENT ON COLUMN pack034_iso50001.energy_performance_indicators.improvement_pct IS
    'Percentage improvement from baseline to current value. Positive = improvement.';
COMMENT ON COLUMN pack034_iso50001.energy_performance_indicators.methodology_ref IS
    'Reference to the methodology document or standard used for this EnPI calculation.';
COMMENT ON COLUMN pack034_iso50001.enpi_values.measured_value IS
    'Raw measured EnPI value for the period.';
COMMENT ON COLUMN pack034_iso50001.enpi_values.normalized_value IS
    'EnPI value normalized for relevant variables (e.g., weather, production).';
COMMENT ON COLUMN pack034_iso50001.enpi_values.expected_value IS
    'Expected EnPI value from the baseline model for the same conditions.';
COMMENT ON COLUMN pack034_iso50001.enpi_values.variance_pct IS
    'Percentage variance between normalized and expected values. Negative = better than expected.';
COMMENT ON COLUMN pack034_iso50001.enpi_values.data_quality_score IS
    'Data quality score (0-10) indicating reliability of the measurement.';
COMMENT ON COLUMN pack034_iso50001.enpi_targets.target_basis IS
    'Documentation of how the target was derived (e.g., engineering analysis, benchmark comparison).';
COMMENT ON COLUMN pack034_iso50001.enpi_targets.interim_targets_json IS
    'JSON array of interim milestone targets (e.g., [{"quarter": "Q1", "value": 150.0}]).';
