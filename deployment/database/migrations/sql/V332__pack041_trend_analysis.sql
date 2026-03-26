-- =============================================================================
-- V332: PACK-041 Scope 1-2 Complete Pack - Trend Analysis
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates trend analysis tables for tracking emissions over time, intensity
-- metrics, and decomposition analysis. Yearly emissions enable year-over-year
-- comparison and target tracking. Intensity metrics support per-revenue,
-- per-employee, per-area, and per-production unit intensity ratios.
-- Decomposition analysis (LMDI, Kaya) identifies the drivers of emission
-- changes (activity, structure, efficiency, fuel mix, emission factor).
--
-- Tables (3):
--   1. ghg_scope12.yearly_emissions
--   2. ghg_scope12.intensity_metrics
--   3. ghg_scope12.decomposition_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V331__pack041_base_year.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.yearly_emissions
-- =============================================================================
-- Summarized annual emissions by organization for time-series trending.
-- Consolidates Scope 1, Scope 2 (both methods), and combined totals with
-- year-over-year change calculations. Supports dashboard visualization,
-- target tracking, and KPI reporting.

CREATE TABLE ghg_scope12.yearly_emissions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    -- Scope 1
    scope1_total                DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope1_stationary           DECIMAL(12,3)   DEFAULT 0,
    scope1_mobile               DECIMAL(12,3)   DEFAULT 0,
    scope1_process              DECIMAL(12,3)   DEFAULT 0,
    scope1_fugitive             DECIMAL(12,3)   DEFAULT 0,
    scope1_refrigerant          DECIMAL(12,3)   DEFAULT 0,
    scope1_land_use             DECIMAL(12,3)   DEFAULT 0,
    scope1_waste                DECIMAL(12,3)   DEFAULT 0,
    scope1_agricultural         DECIMAL(12,3)   DEFAULT 0,
    scope1_biogenic_co2         DECIMAL(12,3)   DEFAULT 0,
    -- Scope 2
    scope2_location             DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope2_market               DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope2_electricity_mwh      DECIMAL(15,3),
    scope2_steam_gj             DECIMAL(12,3),
    scope2_cooling_gj           DECIMAL(12,3),
    scope2_heating_gj           DECIMAL(12,3),
    -- Combined totals
    total_co2e                  DECIMAL(15,3)   GENERATED ALWAYS AS (scope1_total + scope2_location) STORED,
    total_co2e_market           DECIMAL(15,3)   GENERATED ALWAYS AS (scope1_total + scope2_market) STORED,
    -- Year-over-year changes (populated by application logic)
    yoy_scope1_change_pct       DECIMAL(8,4),
    yoy_scope2_loc_change_pct   DECIMAL(8,4),
    yoy_scope2_mkt_change_pct   DECIMAL(8,4),
    yoy_total_change_pct        DECIMAL(8,4),
    -- Base year comparison
    base_year_scope1_change_pct DECIMAL(8,4),
    base_year_scope2_change_pct DECIMAL(8,4),
    base_year_total_change_pct  DECIMAL(8,4),
    -- Intensity denominators (stored for ratio calculation)
    annual_revenue              DECIMAL(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    employee_count              INTEGER,
    production_output           DECIMAL(18,3),
    production_unit             VARCHAR(50),
    floor_area_m2               DECIMAL(14,2),
    -- Data quality
    data_status                 VARCHAR(30)     NOT NULL DEFAULT 'PRELIMINARY',
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    data_completeness_pct       DECIMAL(5,2),
    is_restated                 BOOLEAN         NOT NULL DEFAULT false,
    restatement_reason          TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_ye_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p041_ye_scope1 CHECK (
        scope1_total >= 0
    ),
    CONSTRAINT chk_p041_ye_scope2_loc CHECK (
        scope2_location >= 0
    ),
    CONSTRAINT chk_p041_ye_scope2_mkt CHECK (
        scope2_market >= 0
    ),
    CONSTRAINT chk_p041_ye_data_status CHECK (
        data_status IN ('PRELIMINARY', 'FINAL', 'VERIFIED', 'RESTATED', 'ESTIMATED')
    ),
    CONSTRAINT chk_p041_ye_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_ye_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p041_ye_revenue CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_p041_ye_employees CHECK (
        employee_count IS NULL OR employee_count >= 0
    ),
    CONSTRAINT chk_p041_ye_production CHECK (
        production_output IS NULL OR production_output >= 0
    ),
    CONSTRAINT chk_p041_ye_floor_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT uq_p041_ye_tenant_org_year UNIQUE (tenant_id, organization_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ye_tenant             ON ghg_scope12.yearly_emissions(tenant_id);
CREATE INDEX idx_p041_ye_org               ON ghg_scope12.yearly_emissions(organization_id);
CREATE INDEX idx_p041_ye_year              ON ghg_scope12.yearly_emissions(year DESC);
CREATE INDEX idx_p041_ye_scope1            ON ghg_scope12.yearly_emissions(scope1_total DESC);
CREATE INDEX idx_p041_ye_scope2_loc        ON ghg_scope12.yearly_emissions(scope2_location DESC);
CREATE INDEX idx_p041_ye_data_status       ON ghg_scope12.yearly_emissions(data_status);
CREATE INDEX idx_p041_ye_restated          ON ghg_scope12.yearly_emissions(is_restated) WHERE is_restated = true;
CREATE INDEX idx_p041_ye_created           ON ghg_scope12.yearly_emissions(created_at DESC);
CREATE INDEX idx_p041_ye_metadata          ON ghg_scope12.yearly_emissions USING GIN(metadata);

-- Composite: org + year for time-series queries
CREATE INDEX idx_p041_ye_org_year          ON ghg_scope12.yearly_emissions(organization_id, year DESC);

-- Composite: tenant + year for multi-org dashboards
CREATE INDEX idx_p041_ye_tenant_year       ON ghg_scope12.yearly_emissions(tenant_id, year DESC, scope1_total DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ye_updated
    BEFORE UPDATE ON ghg_scope12.yearly_emissions
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.intensity_metrics
-- =============================================================================
-- Emission intensity ratios (emissions per unit of business activity) per
-- GHG Protocol Chapter 9. Supports multiple denominator types: revenue,
-- employees (FTE), production output, floor area, energy consumed. Intensity
-- metrics enable meaningful comparison across years and peer organizations
-- by normalizing for changes in organizational size.

CREATE TABLE ghg_scope12.intensity_metrics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    metric_type                 VARCHAR(50)     NOT NULL,
    metric_label                VARCHAR(200),
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    -- Numerator (emissions)
    numerator_tco2e             DECIMAL(12,3)   NOT NULL,
    numerator_scope             VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2_LOCATION',
    -- Denominator
    denominator_value           DECIMAL(15,3)   NOT NULL,
    denominator_unit            VARCHAR(50)     NOT NULL,
    denominator_source          VARCHAR(100),
    -- Result
    intensity_value             DECIMAL(15,6)   NOT NULL,
    intensity_unit              VARCHAR(100),
    -- Year-over-year
    yoy_change_pct              DECIMAL(8,4),
    yoy_change_absolute         DECIMAL(15,6),
    base_year_change_pct        DECIMAL(8,4),
    -- Target tracking
    target_intensity            DECIMAL(15,6),
    target_year                 INTEGER,
    on_track_for_target         BOOLEAN,
    required_annual_reduction   DECIMAL(8,4),
    -- Quality
    data_quality_indicator      VARCHAR(20),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_im_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p041_im_metric_type CHECK (
        metric_type IN (
            'REVENUE', 'EMPLOYEE_FTE', 'PRODUCTION_OUTPUT', 'FLOOR_AREA',
            'ENERGY_CONSUMED', 'UNITS_SOLD', 'CUSTOMERS_SERVED',
            'BEDS_OCCUPIED', 'PASSENGERS_KM', 'TONNE_KM', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_im_scope CHECK (
        scope_coverage IN (
            'SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET',
            'SCOPE_1_2', 'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET'
        )
    ),
    CONSTRAINT chk_p041_im_numerator CHECK (
        numerator_tco2e >= 0
    ),
    CONSTRAINT chk_p041_im_denominator CHECK (
        denominator_value > 0
    ),
    CONSTRAINT chk_p041_im_intensity CHECK (
        intensity_value >= 0
    ),
    CONSTRAINT chk_p041_im_target_intensity CHECK (
        target_intensity IS NULL OR target_intensity >= 0
    ),
    CONSTRAINT chk_p041_im_target_year CHECK (
        target_year IS NULL OR (target_year >= 1990 AND target_year <= 2100)
    ),
    CONSTRAINT chk_p041_im_quality CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'ESTIMATED', 'DEFAULT'
        )
    ),
    CONSTRAINT uq_p041_im_org_year_type_scope UNIQUE (organization_id, year, metric_type, scope_coverage)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_im_tenant             ON ghg_scope12.intensity_metrics(tenant_id);
CREATE INDEX idx_p041_im_org               ON ghg_scope12.intensity_metrics(organization_id);
CREATE INDEX idx_p041_im_year              ON ghg_scope12.intensity_metrics(year DESC);
CREATE INDEX idx_p041_im_metric_type       ON ghg_scope12.intensity_metrics(metric_type);
CREATE INDEX idx_p041_im_scope             ON ghg_scope12.intensity_metrics(scope_coverage);
CREATE INDEX idx_p041_im_intensity         ON ghg_scope12.intensity_metrics(intensity_value);
CREATE INDEX idx_p041_im_on_track          ON ghg_scope12.intensity_metrics(on_track_for_target) WHERE on_track_for_target IS NOT NULL;
CREATE INDEX idx_p041_im_created           ON ghg_scope12.intensity_metrics(created_at DESC);
CREATE INDEX idx_p041_im_metadata          ON ghg_scope12.intensity_metrics USING GIN(metadata);

-- Composite: org + metric + year for trend line
CREATE INDEX idx_p041_im_org_metric_year   ON ghg_scope12.intensity_metrics(organization_id, metric_type, year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_im_updated
    BEFORE UPDATE ON ghg_scope12.intensity_metrics
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.decomposition_results
-- =============================================================================
-- Decomposition analysis results identifying the drivers of emission changes
-- between two years. Uses LMDI (Logarithmic Mean Divisia Index) or Kaya
-- Identity methodology to decompose total change into contributing factors:
-- activity effect (growth), structure effect (sectoral shift), intensity
-- effect (efficiency), fuel mix effect, and emission factor effect.

CREATE TABLE ghg_scope12.decomposition_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    year_from                   INTEGER         NOT NULL,
    year_to                     INTEGER         NOT NULL,
    methodology                 VARCHAR(30)     NOT NULL DEFAULT 'LMDI_ADDITIVE',
    scope_coverage              VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    -- Total change
    total_change_tco2e          DECIMAL(12,3)   NOT NULL,
    total_change_pct            DECIMAL(8,4)    NOT NULL,
    -- Factor decomposition
    factor                      VARCHAR(50)     NOT NULL,
    factor_label                VARCHAR(200),
    contribution_tco2e          DECIMAL(12,3)   NOT NULL,
    contribution_pct            DECIMAL(8,4)    NOT NULL,
    contribution_share_pct      DECIMAL(8,4),
    -- Context
    factor_from_value           DECIMAL(18,6),
    factor_to_value             DECIMAL(18,6),
    factor_change_pct           DECIMAL(8,4),
    factor_unit                 VARCHAR(50),
    -- Decomposition details
    decomposition_details       JSONB           DEFAULT '{}',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_dr_year_from CHECK (
        year_from >= 1990 AND year_from <= 2100
    ),
    CONSTRAINT chk_p041_dr_year_to CHECK (
        year_to >= 1990 AND year_to <= 2100
    ),
    CONSTRAINT chk_p041_dr_year_order CHECK (
        year_from < year_to
    ),
    CONSTRAINT chk_p041_dr_methodology CHECK (
        methodology IN (
            'LMDI_ADDITIVE', 'LMDI_MULTIPLICATIVE', 'KAYA_IDENTITY',
            'SDA', 'IDA', 'STRUCTURAL_DECOMPOSITION', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p041_dr_scope CHECK (
        scope_coverage IN (
            'SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET',
            'SCOPE_1_2', 'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET'
        )
    ),
    CONSTRAINT chk_p041_dr_factor CHECK (
        factor IN (
            'ACTIVITY_EFFECT', 'STRUCTURE_EFFECT', 'INTENSITY_EFFECT',
            'FUEL_MIX_EFFECT', 'EMISSION_FACTOR_EFFECT', 'ENERGY_MIX_EFFECT',
            'RENEWABLE_EFFECT', 'GRID_DECARBONIZATION', 'BOUNDARY_CHANGE',
            'RESIDUAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_dr_contribution_share CHECK (
        contribution_share_pct IS NULL OR (contribution_share_pct >= -100 AND contribution_share_pct <= 100)
    ),
    CONSTRAINT uq_p041_dr_org_years_method_factor UNIQUE (organization_id, year_from, year_to, methodology, scope_coverage, factor)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_dr_tenant             ON ghg_scope12.decomposition_results(tenant_id);
CREATE INDEX idx_p041_dr_org               ON ghg_scope12.decomposition_results(organization_id);
CREATE INDEX idx_p041_dr_year_from         ON ghg_scope12.decomposition_results(year_from);
CREATE INDEX idx_p041_dr_year_to           ON ghg_scope12.decomposition_results(year_to);
CREATE INDEX idx_p041_dr_methodology       ON ghg_scope12.decomposition_results(methodology);
CREATE INDEX idx_p041_dr_scope             ON ghg_scope12.decomposition_results(scope_coverage);
CREATE INDEX idx_p041_dr_factor            ON ghg_scope12.decomposition_results(factor);
CREATE INDEX idx_p041_dr_contribution      ON ghg_scope12.decomposition_results(contribution_tco2e DESC);
CREATE INDEX idx_p041_dr_created           ON ghg_scope12.decomposition_results(created_at DESC);
CREATE INDEX idx_p041_dr_metadata          ON ghg_scope12.decomposition_results USING GIN(metadata);
CREATE INDEX idx_p041_dr_details           ON ghg_scope12.decomposition_results USING GIN(decomposition_details);

-- Composite: org + year pair + methodology for full decomposition
CREATE INDEX idx_p041_dr_org_years_method  ON ghg_scope12.decomposition_results(organization_id, year_from, year_to, methodology);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_dr_updated
    BEFORE UPDATE ON ghg_scope12.decomposition_results
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.yearly_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.intensity_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.decomposition_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_ye_tenant_isolation
    ON ghg_scope12.yearly_emissions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ye_service_bypass
    ON ghg_scope12.yearly_emissions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_im_tenant_isolation
    ON ghg_scope12.intensity_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_im_service_bypass
    ON ghg_scope12.intensity_metrics
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_dr_tenant_isolation
    ON ghg_scope12.decomposition_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_dr_service_bypass
    ON ghg_scope12.decomposition_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.yearly_emissions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.intensity_metrics TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.decomposition_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.yearly_emissions IS
    'Annual emission summaries per organization with Scope 1, Scope 2 (dual method), and combined totals for time-series trending and KPI dashboards.';
COMMENT ON TABLE ghg_scope12.intensity_metrics IS
    'Emission intensity ratios (tCO2e per revenue, FTE, production, area) per GHG Protocol Chapter 9 with target tracking.';
COMMENT ON TABLE ghg_scope12.decomposition_results IS
    'LMDI/Kaya decomposition of emission changes into activity, structure, intensity, fuel mix, and emission factor effects.';

COMMENT ON COLUMN ghg_scope12.yearly_emissions.total_co2e IS 'Auto-calculated Scope 1 + Scope 2 location-based total (tCO2e).';
COMMENT ON COLUMN ghg_scope12.yearly_emissions.total_co2e_market IS 'Auto-calculated Scope 1 + Scope 2 market-based total (tCO2e).';
COMMENT ON COLUMN ghg_scope12.yearly_emissions.yoy_scope1_change_pct IS 'Year-over-year percentage change in Scope 1 emissions.';
COMMENT ON COLUMN ghg_scope12.yearly_emissions.base_year_total_change_pct IS 'Percentage change from base year combined total.';
COMMENT ON COLUMN ghg_scope12.yearly_emissions.data_status IS 'Data maturity: PRELIMINARY, FINAL, VERIFIED, RESTATED, ESTIMATED.';
COMMENT ON COLUMN ghg_scope12.yearly_emissions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN ghg_scope12.intensity_metrics.metric_type IS 'Denominator type: REVENUE, EMPLOYEE_FTE, PRODUCTION_OUTPUT, FLOOR_AREA, ENERGY_CONSUMED, etc.';
COMMENT ON COLUMN ghg_scope12.intensity_metrics.intensity_value IS 'Calculated intensity ratio (numerator_tco2e / denominator_value).';
COMMENT ON COLUMN ghg_scope12.intensity_metrics.on_track_for_target IS 'Whether current trajectory achieves the target intensity by target_year.';
COMMENT ON COLUMN ghg_scope12.intensity_metrics.required_annual_reduction IS 'Required annual intensity reduction (%) to reach target from current year.';

COMMENT ON COLUMN ghg_scope12.decomposition_results.methodology IS 'Decomposition method: LMDI_ADDITIVE, LMDI_MULTIPLICATIVE, KAYA_IDENTITY, SDA, IDA.';
COMMENT ON COLUMN ghg_scope12.decomposition_results.factor IS 'Decomposition factor: ACTIVITY_EFFECT, STRUCTURE_EFFECT, INTENSITY_EFFECT, FUEL_MIX_EFFECT, EMISSION_FACTOR_EFFECT, etc.';
COMMENT ON COLUMN ghg_scope12.decomposition_results.contribution_tco2e IS 'Absolute contribution of this factor to total emission change (tCO2e).';
COMMENT ON COLUMN ghg_scope12.decomposition_results.contribution_share_pct IS 'Share of total change attributable to this factor (sum of all factors = 100%).';
