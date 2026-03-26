-- =============================================================================
-- V331: PACK-041 Scope 1-2 Complete Pack - Base Year Management
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates base year management tables implementing GHG Protocol Chapter 5
-- (Tracking Emissions Over Time). The base year serves as the reference point
-- against which emission reductions are measured. Tracks base year totals by
-- scope and category, supports versioned recalculations triggered by structural
-- changes (M&A, divestiture, methodology changes, error corrections), and
-- implements the significance threshold test per GHG Protocol guidance.
--
-- Tables (3):
--   1. ghg_scope12.base_years
--   2. ghg_scope12.base_year_categories
--   3. ghg_scope12.base_year_recalculations
--
-- Also includes: indexes, RLS, comments.
-- Previous: V330__pack041_uncertainty.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.base_years
-- =============================================================================
-- Base year definition for an organization. Per GHG Protocol, organizations
-- must select a base year for which verifiable emissions data exist and
-- recalculate if significant structural changes occur. Supports fixed base
-- year and rolling base year approaches. Multiple versions track the history
-- of recalculations.

CREATE TABLE ghg_scope12.base_years (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    base_year                   INTEGER         NOT NULL,
    base_year_type              VARCHAR(30)     NOT NULL DEFAULT 'FIXED',
    -- Scope 1 totals
    scope1_total                DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope1_co2                  DECIMAL(15,3)   DEFAULT 0,
    scope1_ch4                  DECIMAL(12,6)   DEFAULT 0,
    scope1_n2o                  DECIMAL(12,6)   DEFAULT 0,
    scope1_fluorinated          DECIMAL(12,6)   DEFAULT 0,
    scope1_biogenic_co2         DECIMAL(12,3)   DEFAULT 0,
    -- Scope 2 totals
    scope2_location_total       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope2_market_total         DECIMAL(15,3)   NOT NULL DEFAULT 0,
    scope2_consumption_mwh      DECIMAL(15,3),
    -- Combined
    combined_scope12_location   DECIMAL(15,3)   GENERATED ALWAYS AS (scope1_total + scope2_location_total) STORED,
    combined_scope12_market     DECIMAL(15,3)   GENERATED ALWAYS AS (scope1_total + scope2_market_total) STORED,
    -- Intensity denominators
    revenue_base                DECIMAL(18,2),
    revenue_currency            VARCHAR(3)      DEFAULT 'USD',
    employee_count_base         INTEGER,
    production_output_base      DECIMAL(18,3),
    production_unit             VARCHAR(50),
    floor_area_m2_base          DECIMAL(14,2),
    -- Version management
    version                     INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    previous_version_id         UUID            REFERENCES ghg_scope12.base_years(id) ON DELETE SET NULL,
    -- GWP and methodology
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    consolidation_approach      VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL_CONTROL',
    boundary_id                 UUID            REFERENCES ghg_scope12.organizational_boundaries(id) ON DELETE SET NULL,
    -- Significance threshold
    significance_threshold_pct  DECIMAL(5,2)    NOT NULL DEFAULT 5.00,
    -- Workflow
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    established_date            DATE,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    verified_by                 VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    notes                       TEXT,
    methodology_notes           TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_by_year CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_p041_by_type CHECK (
        base_year_type IN ('FIXED', 'ROLLING_AVERAGE', 'ROLLING_3YR', 'ROLLING_5YR')
    ),
    CONSTRAINT chk_p041_by_scope1 CHECK (
        scope1_total >= 0
    ),
    CONSTRAINT chk_p041_by_scope2_loc CHECK (
        scope2_location_total >= 0
    ),
    CONSTRAINT chk_p041_by_scope2_mkt CHECK (
        scope2_market_total >= 0
    ),
    CONSTRAINT chk_p041_by_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p041_by_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_by_consolidation CHECK (
        consolidation_approach IN (
            'EQUITY_SHARE', 'OPERATIONAL_CONTROL', 'FINANCIAL_CONTROL'
        )
    ),
    CONSTRAINT chk_p041_by_status CHECK (
        status IN ('DRAFT', 'ESTABLISHED', 'APPROVED', 'VERIFIED', 'RECALCULATED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p041_by_threshold CHECK (
        significance_threshold_pct > 0 AND significance_threshold_pct <= 20
    ),
    CONSTRAINT chk_p041_by_revenue CHECK (
        revenue_base IS NULL OR revenue_base >= 0
    ),
    CONSTRAINT chk_p041_by_employees CHECK (
        employee_count_base IS NULL OR employee_count_base >= 0
    ),
    CONSTRAINT chk_p041_by_production CHECK (
        production_output_base IS NULL OR production_output_base >= 0
    ),
    CONSTRAINT chk_p041_by_floor_area CHECK (
        floor_area_m2_base IS NULL OR floor_area_m2_base > 0
    ),
    CONSTRAINT uq_p041_by_org_year_version UNIQUE (organization_id, base_year, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_by_tenant             ON ghg_scope12.base_years(tenant_id);
CREATE INDEX idx_p041_by_org               ON ghg_scope12.base_years(organization_id);
CREATE INDEX idx_p041_by_year              ON ghg_scope12.base_years(base_year);
CREATE INDEX idx_p041_by_type              ON ghg_scope12.base_years(base_year_type);
CREATE INDEX idx_p041_by_status            ON ghg_scope12.base_years(status);
CREATE INDEX idx_p041_by_current           ON ghg_scope12.base_years(is_current) WHERE is_current = true;
CREATE INDEX idx_p041_by_version           ON ghg_scope12.base_years(version);
CREATE INDEX idx_p041_by_created           ON ghg_scope12.base_years(created_at DESC);
CREATE INDEX idx_p041_by_metadata          ON ghg_scope12.base_years USING GIN(metadata);

-- Composite: org + current base year
CREATE INDEX idx_p041_by_org_current       ON ghg_scope12.base_years(organization_id, base_year)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_by_updated
    BEFORE UPDATE ON ghg_scope12.base_years
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.base_year_categories
-- =============================================================================
-- Detailed per-category breakdown of the base year inventory. Each row
-- represents one scope/category combination with its base year emissions.
-- This granular data supports accurate recalculation when specific categories
-- are affected by structural changes.

CREATE TABLE ghg_scope12.base_year_categories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_scope12.base_years(id) ON DELETE CASCADE,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(60)     NOT NULL,
    sub_category                VARCHAR(60),
    -- Emissions
    emissions_tco2e             DECIMAL(12,3)   NOT NULL DEFAULT 0,
    emissions_co2               DECIMAL(12,3)   DEFAULT 0,
    emissions_ch4               DECIMAL(12,6)   DEFAULT 0,
    emissions_n2o               DECIMAL(12,6)   DEFAULT 0,
    emissions_fluorinated       DECIMAL(12,6)   DEFAULT 0,
    -- Proportion
    pct_of_scope_total          DECIMAL(8,4),
    pct_of_inventory_total      DECIMAL(8,4),
    -- Methodology
    methodology_tier            VARCHAR(20),
    emission_factor_source      VARCHAR(100),
    -- Facility count
    facilities_contributing     INTEGER         DEFAULT 0,
    -- Quality
    data_quality_score          NUMERIC(5,2),
    uncertainty_pct             DECIMAL(8,4),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_byc_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET')
    ),
    CONSTRAINT chk_p041_byc_emissions CHECK (
        emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p041_byc_pct_scope CHECK (
        pct_of_scope_total IS NULL OR (pct_of_scope_total >= 0 AND pct_of_scope_total <= 100)
    ),
    CONSTRAINT chk_p041_byc_pct_inv CHECK (
        pct_of_inventory_total IS NULL OR (pct_of_inventory_total >= 0 AND pct_of_inventory_total <= 100)
    ),
    CONSTRAINT chk_p041_byc_tier CHECK (
        methodology_tier IS NULL OR methodology_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'DIRECT_MEASUREMENT', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p041_byc_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p041_byc_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 500)
    ),
    CONSTRAINT chk_p041_byc_facilities CHECK (
        facilities_contributing IS NULL OR facilities_contributing >= 0
    ),
    CONSTRAINT uq_p041_byc_baseyear_scope_cat UNIQUE (base_year_id, scope, category, sub_category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_byc_tenant            ON ghg_scope12.base_year_categories(tenant_id);
CREATE INDEX idx_p041_byc_base_year         ON ghg_scope12.base_year_categories(base_year_id);
CREATE INDEX idx_p041_byc_scope             ON ghg_scope12.base_year_categories(scope);
CREATE INDEX idx_p041_byc_category          ON ghg_scope12.base_year_categories(category);
CREATE INDEX idx_p041_byc_emissions         ON ghg_scope12.base_year_categories(emissions_tco2e DESC);
CREATE INDEX idx_p041_byc_created           ON ghg_scope12.base_year_categories(created_at DESC);

-- Composite: base year + scope for aggregation
CREATE INDEX idx_p041_byc_by_scope         ON ghg_scope12.base_year_categories(base_year_id, scope, emissions_tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_byc_updated
    BEFORE UPDATE ON ghg_scope12.base_year_categories
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.base_year_recalculations
-- =============================================================================
-- Tracks base year recalculation events triggered by significant structural
-- changes per GHG Protocol Chapter 5. Each recalculation documents the
-- trigger type, affected entities, original vs. recalculated totals, the
-- significance test result, and approval workflow. Recalculation triggers
-- include: acquisitions, divestitures, mergers, methodology changes, error
-- corrections, and changes in organizational boundary.

CREATE TABLE ghg_scope12.base_year_recalculations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    base_year_id                UUID            NOT NULL REFERENCES ghg_scope12.base_years(id) ON DELETE CASCADE,
    recalculation_version       INTEGER         NOT NULL DEFAULT 1,
    -- Trigger
    trigger_type                VARCHAR(30)     NOT NULL,
    trigger_description         TEXT            NOT NULL,
    trigger_date                DATE            NOT NULL,
    affected_entities           TEXT[],
    affected_facilities         TEXT[],
    affected_categories         TEXT[],
    -- Original values
    original_scope1_total       DECIMAL(15,3)   NOT NULL,
    original_scope2_loc_total   DECIMAL(15,3)   NOT NULL,
    original_scope2_mkt_total   DECIMAL(15,3)   NOT NULL,
    original_combined_total     DECIMAL(15,3)   NOT NULL,
    -- Recalculated values
    recalculated_scope1_total   DECIMAL(15,3)   NOT NULL,
    recalculated_scope2_loc     DECIMAL(15,3)   NOT NULL,
    recalculated_scope2_mkt     DECIMAL(15,3)   NOT NULL,
    recalculated_combined_total DECIMAL(15,3)   NOT NULL,
    -- Adjustment analysis
    adjustment_scope1           DECIMAL(12,3)   GENERATED ALWAYS AS (recalculated_scope1_total - original_scope1_total) STORED,
    adjustment_scope2_loc       DECIMAL(12,3)   GENERATED ALWAYS AS (recalculated_scope2_loc - original_scope2_loc_total) STORED,
    adjustment_scope2_mkt       DECIMAL(12,3)   GENERATED ALWAYS AS (recalculated_scope2_mkt - original_scope2_mkt_total) STORED,
    adjustment_combined         DECIMAL(12,3)   GENERATED ALWAYS AS (recalculated_combined_total - original_combined_total) STORED,
    -- Significance test
    significance_pct            DECIMAL(8,4)    NOT NULL,
    significance_threshold_pct  DECIMAL(5,2)    NOT NULL DEFAULT 5.00,
    is_significant              BOOLEAN         NOT NULL,
    significance_rationale      TEXT,
    -- Methodology
    recalculation_methodology   TEXT,
    data_sources                TEXT,
    assumptions                 TEXT,
    -- Workflow
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    prepared_by                 VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    effective_date              DATE,
    new_base_year_version_id    UUID            REFERENCES ghg_scope12.base_years(id) ON DELETE SET NULL,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_byr_trigger CHECK (
        trigger_type IN (
            'ACQUISITION', 'DIVESTITURE', 'MERGER', 'OUTSOURCING', 'INSOURCING',
            'METHODOLOGY_CHANGE', 'ERROR_CORRECTION', 'BOUNDARY_CHANGE',
            'GWP_UPDATE', 'EMISSION_FACTOR_UPDATE', 'REGULATORY_REQUIREMENT',
            'STRUCTURAL_CHANGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_byr_significance_pct CHECK (
        significance_pct >= 0
    ),
    CONSTRAINT chk_p041_byr_threshold CHECK (
        significance_threshold_pct > 0 AND significance_threshold_pct <= 20
    ),
    CONSTRAINT chk_p041_byr_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'APPLIED', 'REJECTED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p041_byr_version CHECK (
        recalculation_version >= 1
    ),
    CONSTRAINT chk_p041_byr_original CHECK (
        original_combined_total >= 0
    ),
    CONSTRAINT chk_p041_byr_recalculated CHECK (
        recalculated_combined_total >= 0
    ),
    CONSTRAINT uq_p041_byr_baseyear_version UNIQUE (base_year_id, recalculation_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_byr_tenant            ON ghg_scope12.base_year_recalculations(tenant_id);
CREATE INDEX idx_p041_byr_base_year         ON ghg_scope12.base_year_recalculations(base_year_id);
CREATE INDEX idx_p041_byr_trigger           ON ghg_scope12.base_year_recalculations(trigger_type);
CREATE INDEX idx_p041_byr_trigger_date      ON ghg_scope12.base_year_recalculations(trigger_date);
CREATE INDEX idx_p041_byr_significant       ON ghg_scope12.base_year_recalculations(is_significant);
CREATE INDEX idx_p041_byr_status            ON ghg_scope12.base_year_recalculations(status);
CREATE INDEX idx_p041_byr_effective         ON ghg_scope12.base_year_recalculations(effective_date);
CREATE INDEX idx_p041_byr_created           ON ghg_scope12.base_year_recalculations(created_at DESC);
CREATE INDEX idx_p041_byr_metadata          ON ghg_scope12.base_year_recalculations USING GIN(metadata);
CREATE INDEX idx_p041_byr_entities          ON ghg_scope12.base_year_recalculations USING GIN(affected_entities);

-- Composite: base year + significant recalculations
CREATE INDEX idx_p041_byr_by_sig           ON ghg_scope12.base_year_recalculations(base_year_id, trigger_date DESC)
    WHERE is_significant = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_byr_updated
    BEFORE UPDATE ON ghg_scope12.base_year_recalculations
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.base_years ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.base_year_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.base_year_recalculations ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_by_tenant_isolation
    ON ghg_scope12.base_years
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_by_service_bypass
    ON ghg_scope12.base_years
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_byc_tenant_isolation
    ON ghg_scope12.base_year_categories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_byc_service_bypass
    ON ghg_scope12.base_year_categories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_byr_tenant_isolation
    ON ghg_scope12.base_year_recalculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_byr_service_bypass
    ON ghg_scope12.base_year_recalculations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.base_years TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.base_year_categories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.base_year_recalculations TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.base_years IS
    'Base year definitions with Scope 1 and Scope 2 totals, intensity denominators, and version management per GHG Protocol Chapter 5.';
COMMENT ON TABLE ghg_scope12.base_year_categories IS
    'Per-scope, per-category base year emission breakdown for granular recalculation support.';
COMMENT ON TABLE ghg_scope12.base_year_recalculations IS
    'Base year recalculation tracking with trigger analysis, significance testing, and approval workflow per GHG Protocol guidance.';

COMMENT ON COLUMN ghg_scope12.base_years.base_year_type IS 'Base year approach: FIXED (single year) or ROLLING_AVERAGE (3-year or 5-year rolling average).';
COMMENT ON COLUMN ghg_scope12.base_years.version IS 'Version number incremented with each recalculation. Version 1 = original base year.';
COMMENT ON COLUMN ghg_scope12.base_years.is_current IS 'Whether this is the current active base year version (only one per org should be true).';
COMMENT ON COLUMN ghg_scope12.base_years.significance_threshold_pct IS 'Threshold for significant change requiring recalculation (typically 5% per GHG Protocol).';
COMMENT ON COLUMN ghg_scope12.base_years.combined_scope12_location IS 'Auto-calculated Scope 1 + Scope 2 location-based total.';
COMMENT ON COLUMN ghg_scope12.base_years.combined_scope12_market IS 'Auto-calculated Scope 1 + Scope 2 market-based total.';
COMMENT ON COLUMN ghg_scope12.base_years.provenance_hash IS 'SHA-256 hash for base year data integrity.';

COMMENT ON COLUMN ghg_scope12.base_year_categories.pct_of_scope_total IS 'Percentage contribution of this category to its scope total.';
COMMENT ON COLUMN ghg_scope12.base_year_categories.pct_of_inventory_total IS 'Percentage contribution of this category to the full inventory (all scopes).';

COMMENT ON COLUMN ghg_scope12.base_year_recalculations.trigger_type IS 'Recalculation trigger per GHG Protocol: ACQUISITION, DIVESTITURE, MERGER, METHODOLOGY_CHANGE, ERROR_CORRECTION, etc.';
COMMENT ON COLUMN ghg_scope12.base_year_recalculations.is_significant IS 'Whether the recalculation exceeds the significance threshold and requires base year update.';
COMMENT ON COLUMN ghg_scope12.base_year_recalculations.significance_pct IS 'Absolute percentage change from original to recalculated combined total.';
COMMENT ON COLUMN ghg_scope12.base_year_recalculations.adjustment_combined IS 'Auto-calculated difference between recalculated and original combined totals (tCO2e).';
