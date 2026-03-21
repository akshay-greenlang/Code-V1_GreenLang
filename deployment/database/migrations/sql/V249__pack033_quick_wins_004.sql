-- =============================================================================
-- V249: PACK-033 Quick Wins Identifier - Carbon Reduction Tracking
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Creates carbon impact tracking tables for quick-win actions including scope
-- attribution, emission factor sourcing, cumulative reduction tracking, and
-- SBTi alignment flagging. Also includes an emission factor cache for
-- regional grid and fuel factors.
--
-- Tables (2):
--   1. pack033_quick_wins.carbon_impacts
--   2. pack033_quick_wins.emission_factors_cache
--
-- Previous: V248__pack033_quick_wins_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.carbon_impacts
-- =============================================================================
-- Carbon reduction impact per action, attributed to GHG Protocol scope with
-- emission factor sourcing and SBTi alignment flagging.

CREATE TABLE pack033_quick_wins.carbon_impacts (
    impact_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id                 UUID            NOT NULL REFERENCES pack033_quick_wins.quick_wins_scans(scan_id) ON DELETE CASCADE,
    action_id               UUID,
    scope                   VARCHAR(20)     NOT NULL,
    calculation_method      VARCHAR(50)     NOT NULL,
    emission_factor         NUMERIC(12,6)   NOT NULL,
    emission_factor_source  VARCHAR(255)    NOT NULL,
    annual_co2e_reduction   NUMERIC(14,4)   NOT NULL,
    cumulative_co2e_reduction NUMERIC(16,4),
    reduction_year_start    INTEGER         NOT NULL,
    reduction_year_end      INTEGER         NOT NULL,
    sbti_aligned            BOOLEAN         DEFAULT FALSE,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_ci_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3')
    ),
    CONSTRAINT chk_p033_ci_calc_method CHECK (
        calculation_method IN ('LOCATION_BASED', 'MARKET_BASED', 'DIRECT_MEASUREMENT',
                                'EMISSION_FACTOR', 'HYBRID', 'DEEMED')
    ),
    CONSTRAINT chk_p033_ci_ef CHECK (
        emission_factor > 0
    ),
    CONSTRAINT chk_p033_ci_annual CHECK (
        annual_co2e_reduction >= 0
    ),
    CONSTRAINT chk_p033_ci_cumulative CHECK (
        cumulative_co2e_reduction IS NULL OR cumulative_co2e_reduction >= 0
    ),
    CONSTRAINT chk_p033_ci_year_start CHECK (
        reduction_year_start >= 2020 AND reduction_year_start <= 2100
    ),
    CONSTRAINT chk_p033_ci_year_end CHECK (
        reduction_year_end >= 2020 AND reduction_year_end <= 2100
    ),
    CONSTRAINT chk_p033_ci_year_order CHECK (
        reduction_year_end >= reduction_year_start
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_ci_scan          ON pack033_quick_wins.carbon_impacts(scan_id);
CREATE INDEX idx_p033_ci_action        ON pack033_quick_wins.carbon_impacts(action_id);
CREATE INDEX idx_p033_ci_scope         ON pack033_quick_wins.carbon_impacts(scope);
CREATE INDEX idx_p033_ci_method        ON pack033_quick_wins.carbon_impacts(calculation_method);
CREATE INDEX idx_p033_ci_sbti          ON pack033_quick_wins.carbon_impacts(sbti_aligned);
CREATE INDEX idx_p033_ci_years         ON pack033_quick_wins.carbon_impacts(reduction_year_start, reduction_year_end);
CREATE INDEX idx_p033_ci_created       ON pack033_quick_wins.carbon_impacts(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_ci_updated
    BEFORE UPDATE ON pack033_quick_wins.carbon_impacts
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.emission_factors_cache
-- =============================================================================
-- Cached emission factors by region, grid operator, and fuel type for
-- consistent carbon calculations across scans. Time-bounded validity.

CREATE TABLE pack033_quick_wins.emission_factors_cache (
    factor_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    region                  VARCHAR(100)    NOT NULL,
    grid_operator           VARCHAR(255),
    fuel_type               VARCHAR(50),
    factor_type             VARCHAR(50)     NOT NULL,
    factor_value            NUMERIC(12,6)   NOT NULL,
    factor_unit             VARCHAR(30)     NOT NULL,
    source                  VARCHAR(255)    NOT NULL,
    valid_from              DATE            NOT NULL,
    valid_to                DATE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_efc_factor_type CHECK (
        factor_type IN ('GRID_AVERAGE', 'GRID_MARGINAL', 'RESIDUAL_MIX',
                         'FUEL_COMBUSTION', 'UPSTREAM', 'LIFECYCLE', 'CUSTOM')
    ),
    CONSTRAINT chk_p033_efc_value CHECK (
        factor_value > 0
    ),
    CONSTRAINT chk_p033_efc_unit CHECK (
        factor_unit IN ('kgCO2e/kWh', 'kgCO2e/MWh', 'kgCO2e/GJ', 'kgCO2e/therm',
                          'kgCO2e/litre', 'kgCO2e/m3', 'tCO2e/MWh', 'tCO2e/TJ')
    ),
    CONSTRAINT chk_p033_efc_validity CHECK (
        valid_to IS NULL OR valid_to >= valid_from
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_efc_region       ON pack033_quick_wins.emission_factors_cache(region);
CREATE INDEX idx_p033_efc_grid         ON pack033_quick_wins.emission_factors_cache(grid_operator);
CREATE INDEX idx_p033_efc_fuel         ON pack033_quick_wins.emission_factors_cache(fuel_type);
CREATE INDEX idx_p033_efc_type         ON pack033_quick_wins.emission_factors_cache(factor_type);
CREATE INDEX idx_p033_efc_validity     ON pack033_quick_wins.emission_factors_cache(valid_from, valid_to);
CREATE INDEX idx_p033_efc_source       ON pack033_quick_wins.emission_factors_cache(source);
CREATE UNIQUE INDEX idx_p033_efc_unique_factor
    ON pack033_quick_wins.emission_factors_cache(region, COALESCE(grid_operator, ''), COALESCE(fuel_type, ''), factor_type, valid_from);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.carbon_impacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.emission_factors_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_ci_tenant_isolation
    ON pack033_quick_wins.carbon_impacts
    USING (scan_id IN (
        SELECT scan_id FROM pack033_quick_wins.quick_wins_scans
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_ci_service_bypass
    ON pack033_quick_wins.carbon_impacts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Emission factors cache is shared reference data -- read access for all
CREATE POLICY p033_efc_read_all
    ON pack033_quick_wins.emission_factors_cache
    FOR SELECT
    USING (TRUE);
CREATE POLICY p033_efc_service_bypass
    ON pack033_quick_wins.emission_factors_cache
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.carbon_impacts TO PUBLIC;
GRANT SELECT ON pack033_quick_wins.emission_factors_cache TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.emission_factors_cache TO greenlang_service;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.carbon_impacts IS
    'Carbon reduction impact per action attributed to GHG Protocol scope with emission factor sourcing and SBTi alignment.';

COMMENT ON TABLE pack033_quick_wins.emission_factors_cache IS
    'Cached emission factors by region, grid operator, and fuel type for consistent carbon calculations across scans.';

COMMENT ON COLUMN pack033_quick_wins.carbon_impacts.scope IS
    'GHG Protocol scope attribution (SCOPE_1, SCOPE_2_LOCATION, SCOPE_2_MARKET, SCOPE_3).';
COMMENT ON COLUMN pack033_quick_wins.carbon_impacts.emission_factor_source IS
    'Source of the emission factor (e.g., IEA 2025, DEFRA 2025, eGRID 2024, AIB Residual Mix).';
COMMENT ON COLUMN pack033_quick_wins.carbon_impacts.sbti_aligned IS
    'Whether the reduction methodology is aligned with SBTi requirements.';
COMMENT ON COLUMN pack033_quick_wins.carbon_impacts.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack033_quick_wins.emission_factors_cache.factor_type IS
    'Type of emission factor: GRID_AVERAGE, GRID_MARGINAL, RESIDUAL_MIX, FUEL_COMBUSTION, UPSTREAM, LIFECYCLE, CUSTOM.';
COMMENT ON COLUMN pack033_quick_wins.emission_factors_cache.factor_unit IS
    'Unit of the emission factor (e.g., kgCO2e/kWh, tCO2e/MWh).';
COMMENT ON COLUMN pack033_quick_wins.emission_factors_cache.valid_from IS
    'Start date of the emission factor validity period.';
COMMENT ON COLUMN pack033_quick_wins.emission_factors_cache.valid_to IS
    'End date of the emission factor validity period (NULL = still valid).';
