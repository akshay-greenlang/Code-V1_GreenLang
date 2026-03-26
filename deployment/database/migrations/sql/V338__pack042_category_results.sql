-- =============================================================================
-- V338: PACK-042 Scope 3 Starter Pack - Category Calculation Results
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates per-category emission calculation result tables. Stores aggregated
-- category totals, per-gas breakdowns, emission factor usage audit trail,
-- and sub-category detail. These tables hold the core calculation outputs
-- from each of the 15 Scope 3 category agents, providing full transparency
-- into how emissions were calculated for audit and verification purposes.
--
-- Tables (4):
--   1. ghg_accounting_scope3.category_results
--   2. ghg_accounting_scope3.category_gas_breakdown
--   3. ghg_accounting_scope3.emission_factor_usage
--   4. ghg_accounting_scope3.category_sub_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V337__pack042_spend_classification.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.category_results
-- =============================================================================
-- Aggregated per-category emission totals for a Scope 3 inventory. Each
-- record represents the total calculated emissions for one of the 15
-- categories within a reporting period. Tracks methodology tier used,
-- data quality rating, and calculation timestamp for reproducibility.

CREATE TABLE ghg_accounting_scope3.category_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Emission totals
    total_tco2e                 DECIMAL(15,3)   NOT NULL DEFAULT 0,
    total_co2_tonnes            DECIMAL(15,3)   DEFAULT 0,
    total_ch4_tonnes            DECIMAL(12,6)   DEFAULT 0,
    total_n2o_tonnes            DECIMAL(12,6)   DEFAULT 0,
    total_other_ghg_tonnes      DECIMAL(12,6)   DEFAULT 0,
    biogenic_co2_tonnes         DECIMAL(15,3)   DEFAULT 0,
    -- Methodology
    methodology_tier            ghg_accounting_scope3.methodology_tier_type NOT NULL DEFAULT 'SPEND_BASED',
    calculation_approach        VARCHAR(50)     NOT NULL DEFAULT 'SPEND_BASED',
    -- Data quality
    data_quality_rating         VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    primary_data_pct            DECIMAL(5,2)    DEFAULT 0,
    data_completeness_pct       DECIMAL(5,2)    DEFAULT 100,
    -- Calculation metadata
    calculation_timestamp       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculation_version         VARCHAR(20)     DEFAULT '1.0',
    calculation_engine          VARCHAR(100)    DEFAULT 'GREENLANG_SCOPE3',
    -- Activity data summary
    activity_data_count         INTEGER         DEFAULT 0,
    activity_data_total_spend   NUMERIC(18,2),
    activity_data_unit          VARCHAR(50),
    -- Significance
    pct_of_total_scope3         DECIMAL(5,2),
    is_material                 BOOLEAN         DEFAULT true,
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'CALCULATED',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Notes
    methodology_notes           TEXT,
    exclusion_notes             TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p042_catr_total CHECK (
        total_tco2e >= 0
    ),
    CONSTRAINT chk_p042_catr_co2 CHECK (
        total_co2_tonnes IS NULL OR total_co2_tonnes >= 0
    ),
    CONSTRAINT chk_p042_catr_ch4 CHECK (
        total_ch4_tonnes IS NULL OR total_ch4_tonnes >= 0
    ),
    CONSTRAINT chk_p042_catr_n2o CHECK (
        total_n2o_tonnes IS NULL OR total_n2o_tonnes >= 0
    ),
    CONSTRAINT chk_p042_catr_other CHECK (
        total_other_ghg_tonnes IS NULL OR total_other_ghg_tonnes >= 0
    ),
    CONSTRAINT chk_p042_catr_biogenic CHECK (
        biogenic_co2_tonnes IS NULL OR biogenic_co2_tonnes >= 0
    ),
    CONSTRAINT chk_p042_catr_approach CHECK (
        calculation_approach IN (
            'SPEND_BASED', 'AVERAGE_DATA', 'SUPPLIER_SPECIFIC',
            'HYBRID', 'DISTANCE_BASED', 'FUEL_BASED', 'ASSET_SPECIFIC',
            'SITE_SPECIFIC', 'WASTE_TYPE_SPECIFIC', 'INVESTMENT_SPECIFIC'
        )
    ),
    CONSTRAINT chk_p042_catr_quality CHECK (
        data_quality_rating IN (
            'VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW', 'ESTIMATED'
        )
    ),
    CONSTRAINT chk_p042_catr_primary_data CHECK (
        primary_data_pct IS NULL OR (primary_data_pct >= 0 AND primary_data_pct <= 100)
    ),
    CONSTRAINT chk_p042_catr_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p042_catr_pct_total CHECK (
        pct_of_total_scope3 IS NULL OR (pct_of_total_scope3 >= 0 AND pct_of_total_scope3 <= 100)
    ),
    CONSTRAINT chk_p042_catr_status CHECK (
        status IN (
            'DRAFT', 'CALCULATED', 'REVIEWED', 'VERIFIED', 'FINALIZED', 'ERROR'
        )
    ),
    CONSTRAINT chk_p042_catr_activity_count CHECK (
        activity_data_count IS NULL OR activity_data_count >= 0
    ),
    CONSTRAINT uq_p042_catr_inventory_category UNIQUE (inventory_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_catr_tenant           ON ghg_accounting_scope3.category_results(tenant_id);
CREATE INDEX idx_p042_catr_inventory        ON ghg_accounting_scope3.category_results(inventory_id);
CREATE INDEX idx_p042_catr_category         ON ghg_accounting_scope3.category_results(category);
CREATE INDEX idx_p042_catr_tier             ON ghg_accounting_scope3.category_results(methodology_tier);
CREATE INDEX idx_p042_catr_quality          ON ghg_accounting_scope3.category_results(data_quality_rating);
CREATE INDEX idx_p042_catr_total            ON ghg_accounting_scope3.category_results(total_tco2e DESC);
CREATE INDEX idx_p042_catr_status           ON ghg_accounting_scope3.category_results(status);
CREATE INDEX idx_p042_catr_material         ON ghg_accounting_scope3.category_results(is_material) WHERE is_material = true;
CREATE INDEX idx_p042_catr_calc_time        ON ghg_accounting_scope3.category_results(calculation_timestamp DESC);
CREATE INDEX idx_p042_catr_created          ON ghg_accounting_scope3.category_results(created_at DESC);
CREATE INDEX idx_p042_catr_metadata         ON ghg_accounting_scope3.category_results USING GIN(metadata);

-- Composite: inventory + category + status for result lookup
CREATE INDEX idx_p042_catr_inv_cat_status   ON ghg_accounting_scope3.category_results(inventory_id, category, status);

-- Composite: tenant + inventory for dashboard queries
CREATE INDEX idx_p042_catr_tenant_inv       ON ghg_accounting_scope3.category_results(tenant_id, inventory_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_catr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.category_gas_breakdown
-- =============================================================================
-- Per-gas emission detail within a category result. Separates the aggregate
-- category total into individual greenhouse gas components (CO2, CH4, N2O,
-- HFCs, PFCs, SF6, NF3) with their respective GWP values and sources.
-- Enables gas-specific reporting required by GHG Protocol and CSRD.

CREATE TABLE ghg_accounting_scope3.category_gas_breakdown (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    result_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3.category_results(id) ON DELETE CASCADE,
    -- Gas identification
    gas_type                    VARCHAR(30)     NOT NULL,
    gas_formula                 VARCHAR(30),
    -- Mass
    mass_kg                     DECIMAL(15,3)   NOT NULL DEFAULT 0,
    mass_tonnes                 DECIMAL(12,6)   GENERATED ALWAYS AS (mass_kg / 1000.0) STORED,
    -- GWP conversion
    gwp_value                   DECIMAL(10,2)   NOT NULL DEFAULT 1,
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    tco2e                       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    -- Contribution
    pct_of_category             DECIMAL(5,2),
    -- Source
    emission_source             VARCHAR(200),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_cgb_gas_type CHECK (
        gas_type IN (
            'CO2', 'CH4', 'N2O', 'SF6', 'NF3',
            'HFC', 'PFC', 'OTHER_FLUORINATED',
            'CO2_BIOGENIC', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_cgb_mass CHECK (
        mass_kg >= 0
    ),
    CONSTRAINT chk_p042_cgb_gwp CHECK (
        gwp_value >= 0
    ),
    CONSTRAINT chk_p042_cgb_gwp_source CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p042_cgb_tco2e CHECK (
        tco2e >= 0
    ),
    CONSTRAINT chk_p042_cgb_pct CHECK (
        pct_of_category IS NULL OR (pct_of_category >= 0 AND pct_of_category <= 100)
    ),
    CONSTRAINT uq_p042_cgb_result_gas UNIQUE (result_id, gas_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_cgb_tenant            ON ghg_accounting_scope3.category_gas_breakdown(tenant_id);
CREATE INDEX idx_p042_cgb_result            ON ghg_accounting_scope3.category_gas_breakdown(result_id);
CREATE INDEX idx_p042_cgb_gas_type          ON ghg_accounting_scope3.category_gas_breakdown(gas_type);
CREATE INDEX idx_p042_cgb_tco2e             ON ghg_accounting_scope3.category_gas_breakdown(tco2e DESC);
CREATE INDEX idx_p042_cgb_created           ON ghg_accounting_scope3.category_gas_breakdown(created_at DESC);

-- Composite: result + gas type for detailed breakdown
CREATE INDEX idx_p042_cgb_result_gas_tco2e  ON ghg_accounting_scope3.category_gas_breakdown(result_id, gas_type, tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_cgb_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_gas_breakdown
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.emission_factor_usage
-- =============================================================================
-- Audit trail for emission factors used in each category calculation. Records
-- the exact factor source, value, unit, and year applied to each activity
-- data point. This table is critical for reproducibility and verification:
-- an auditor can trace every tCO2e value back to the specific factor used.

CREATE TABLE ghg_accounting_scope3.emission_factor_usage (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    result_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3.category_results(id) ON DELETE CASCADE,
    -- Factor details
    factor_source               VARCHAR(100)    NOT NULL,
    factor_source_detail        VARCHAR(500),
    factor_value                DECIMAL(12,6)   NOT NULL,
    factor_unit                 VARCHAR(50)     NOT NULL,
    factor_year                 INTEGER         NOT NULL,
    factor_geography            VARCHAR(100)    DEFAULT 'GLOBAL',
    -- Activity data
    activity_data_value         DECIMAL(18,6)   NOT NULL,
    activity_data_unit          VARCHAR(50)     NOT NULL,
    activity_data_description   TEXT,
    -- Calculated output
    calculated_tco2e            DECIMAL(12,3)   NOT NULL,
    -- Factor provenance
    factor_tier                 VARCHAR(20)     DEFAULT 'TIER_1',
    factor_id                   UUID,
    factor_database             VARCHAR(100),
    data_quality_score          NUMERIC(5,2),
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_efu_factor_value CHECK (
        factor_value >= 0
    ),
    CONSTRAINT chk_p042_efu_factor_year CHECK (
        factor_year >= 1990 AND factor_year <= 2100
    ),
    CONSTRAINT chk_p042_efu_activity CHECK (
        activity_data_value >= 0
    ),
    CONSTRAINT chk_p042_efu_tco2e CHECK (
        calculated_tco2e >= 0
    ),
    CONSTRAINT chk_p042_efu_factor_source CHECK (
        factor_source IN (
            'EEIO_EXIOBASE', 'EEIO_USEEIO', 'EEIO_CEDA',
            'EPA', 'DEFRA', 'IPCC', 'IEA', 'ECOINVENT',
            'GHG_PROTOCOL', 'SUPPLIER_SPECIFIC', 'INDUSTRY_AVG',
            'NATIONAL_INVENTORY', 'VERIFIED_CUSTOM', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_efu_tier CHECK (
        factor_tier IS NULL OR factor_tier IN (
            'TIER_1', 'TIER_2', 'TIER_3', 'SUPPLIER_SPECIFIC', 'HYBRID'
        )
    ),
    CONSTRAINT chk_p042_efu_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_efu_tenant            ON ghg_accounting_scope3.emission_factor_usage(tenant_id);
CREATE INDEX idx_p042_efu_result            ON ghg_accounting_scope3.emission_factor_usage(result_id);
CREATE INDEX idx_p042_efu_source            ON ghg_accounting_scope3.emission_factor_usage(factor_source);
CREATE INDEX idx_p042_efu_year              ON ghg_accounting_scope3.emission_factor_usage(factor_year);
CREATE INDEX idx_p042_efu_geography         ON ghg_accounting_scope3.emission_factor_usage(factor_geography);
CREATE INDEX idx_p042_efu_tier              ON ghg_accounting_scope3.emission_factor_usage(factor_tier);
CREATE INDEX idx_p042_efu_tco2e             ON ghg_accounting_scope3.emission_factor_usage(calculated_tco2e DESC);
CREATE INDEX idx_p042_efu_factor_id         ON ghg_accounting_scope3.emission_factor_usage(factor_id);
CREATE INDEX idx_p042_efu_created           ON ghg_accounting_scope3.emission_factor_usage(created_at DESC);

-- Composite: result + source for factor audit
CREATE INDEX idx_p042_efu_result_source     ON ghg_accounting_scope3.emission_factor_usage(result_id, factor_source);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_efu_updated
    BEFORE UPDATE ON ghg_accounting_scope3.emission_factor_usage
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.category_sub_results
-- =============================================================================
-- Sub-category detail within a category result. Breaks down category totals
-- into finer-grained groupings (e.g., Cat 1 by commodity type, Cat 4 by
-- transport mode, Cat 6 by travel type). Provides the granularity needed
-- for hotspot identification and targeted reduction strategies.

CREATE TABLE ghg_accounting_scope3.category_sub_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    result_id                   UUID            NOT NULL REFERENCES ghg_accounting_scope3.category_results(id) ON DELETE CASCADE,
    -- Sub-category
    sub_category                VARCHAR(200)    NOT NULL,
    sub_category_code           VARCHAR(50),
    sub_category_level          INTEGER         DEFAULT 1,
    -- Emissions
    tco2e                       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    pct_of_category             DECIMAL(5,2),
    -- Data source
    data_source                 VARCHAR(200),
    data_source_type            VARCHAR(30)     DEFAULT 'SECONDARY',
    record_count                INTEGER         DEFAULT 0,
    total_spend                 NUMERIC(18,2),
    -- Methodology
    methodology_note            TEXT,
    emission_factor_source      VARCHAR(100),
    emission_factor_value       DECIMAL(12,6),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_csr_tco2e CHECK (
        tco2e >= 0
    ),
    CONSTRAINT chk_p042_csr_pct CHECK (
        pct_of_category IS NULL OR (pct_of_category >= 0 AND pct_of_category <= 100)
    ),
    CONSTRAINT chk_p042_csr_level CHECK (
        sub_category_level >= 1 AND sub_category_level <= 5
    ),
    CONSTRAINT chk_p042_csr_data_source_type CHECK (
        data_source_type IS NULL OR data_source_type IN (
            'PRIMARY', 'SECONDARY', 'ESTIMATED', 'MODELED', 'EXTRAPOLATED'
        )
    ),
    CONSTRAINT chk_p042_csr_record_count CHECK (
        record_count IS NULL OR record_count >= 0
    ),
    CONSTRAINT chk_p042_csr_spend CHECK (
        total_spend IS NULL OR total_spend >= 0
    ),
    CONSTRAINT chk_p042_csr_ef_value CHECK (
        emission_factor_value IS NULL OR emission_factor_value >= 0
    ),
    CONSTRAINT uq_p042_csr_result_sub UNIQUE (result_id, sub_category, sub_category_level)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_csr_tenant            ON ghg_accounting_scope3.category_sub_results(tenant_id);
CREATE INDEX idx_p042_csr_result            ON ghg_accounting_scope3.category_sub_results(result_id);
CREATE INDEX idx_p042_csr_sub_category      ON ghg_accounting_scope3.category_sub_results(sub_category);
CREATE INDEX idx_p042_csr_code              ON ghg_accounting_scope3.category_sub_results(sub_category_code);
CREATE INDEX idx_p042_csr_tco2e             ON ghg_accounting_scope3.category_sub_results(tco2e DESC);
CREATE INDEX idx_p042_csr_pct               ON ghg_accounting_scope3.category_sub_results(pct_of_category DESC);
CREATE INDEX idx_p042_csr_data_source       ON ghg_accounting_scope3.category_sub_results(data_source_type);
CREATE INDEX idx_p042_csr_created           ON ghg_accounting_scope3.category_sub_results(created_at DESC);
CREATE INDEX idx_p042_csr_metadata          ON ghg_accounting_scope3.category_sub_results USING GIN(metadata);

-- Composite: result + sub-category tco2e for top sub-categories
CREATE INDEX idx_p042_csr_result_top        ON ghg_accounting_scope3.category_sub_results(result_id, tco2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_csr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.category_sub_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.category_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.category_gas_breakdown ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.emission_factor_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.category_sub_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_catr_tenant_isolation
    ON ghg_accounting_scope3.category_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_catr_service_bypass
    ON ghg_accounting_scope3.category_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_cgb_tenant_isolation
    ON ghg_accounting_scope3.category_gas_breakdown
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_cgb_service_bypass
    ON ghg_accounting_scope3.category_gas_breakdown
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_efu_tenant_isolation
    ON ghg_accounting_scope3.emission_factor_usage
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_efu_service_bypass
    ON ghg_accounting_scope3.emission_factor_usage
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_csr_tenant_isolation
    ON ghg_accounting_scope3.category_sub_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_csr_service_bypass
    ON ghg_accounting_scope3.category_sub_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_gas_breakdown TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.emission_factor_usage TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.category_sub_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.category_results IS
    'Per-category aggregated emission totals with methodology tier, data quality rating, and calculation metadata for audit and verification.';
COMMENT ON TABLE ghg_accounting_scope3.category_gas_breakdown IS
    'Per-gas emission breakdown within a category result (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3) with GWP values and CO2e conversion.';
COMMENT ON TABLE ghg_accounting_scope3.emission_factor_usage IS
    'Emission factor audit trail recording the exact factor, source, value, unit, and activity data applied in each calculation.';
COMMENT ON TABLE ghg_accounting_scope3.category_sub_results IS
    'Sub-category detail within a category result for granular hotspot identification (e.g., by commodity, transport mode, travel type).';

COMMENT ON COLUMN ghg_accounting_scope3.category_results.total_tco2e IS 'Total emissions in tonnes CO2-equivalent for this category.';
COMMENT ON COLUMN ghg_accounting_scope3.category_results.methodology_tier IS 'Methodology tier: SPEND_BASED (Tier 1), AVERAGE_DATA (Tier 2), SUPPLIER_SPECIFIC (Tier 3), or HYBRID.';
COMMENT ON COLUMN ghg_accounting_scope3.category_results.data_quality_rating IS 'Data quality rating: VERY_HIGH, HIGH, MODERATE, LOW, VERY_LOW, ESTIMATED.';
COMMENT ON COLUMN ghg_accounting_scope3.category_results.primary_data_pct IS 'Percentage of emissions calculated using primary (supplier-specific) data.';
COMMENT ON COLUMN ghg_accounting_scope3.category_results.pct_of_total_scope3 IS 'This category as percentage of total Scope 3 emissions.';
COMMENT ON COLUMN ghg_accounting_scope3.category_results.provenance_hash IS 'SHA-256 hash of all inputs for calculation reproducibility.';

COMMENT ON COLUMN ghg_accounting_scope3.category_gas_breakdown.mass_kg IS 'Mass of individual gas in kilograms.';
COMMENT ON COLUMN ghg_accounting_scope3.category_gas_breakdown.mass_tonnes IS 'Generated column: mass in tonnes (mass_kg / 1000).';
COMMENT ON COLUMN ghg_accounting_scope3.category_gas_breakdown.gwp_value IS 'Global Warming Potential value used for CO2e conversion.';

COMMENT ON COLUMN ghg_accounting_scope3.emission_factor_usage.factor_source IS 'Source of the emission factor: EEIO_EXIOBASE, EPA, DEFRA, SUPPLIER_SPECIFIC, etc.';
COMMENT ON COLUMN ghg_accounting_scope3.emission_factor_usage.activity_data_value IS 'Quantity of activity data to which the factor was applied.';
COMMENT ON COLUMN ghg_accounting_scope3.emission_factor_usage.calculated_tco2e IS 'Result of activity_data_value x factor_value in tCO2e.';

COMMENT ON COLUMN ghg_accounting_scope3.category_sub_results.sub_category IS 'Sub-category name (e.g., Electronics, Air Travel Domestic, Landfill Waste).';
COMMENT ON COLUMN ghg_accounting_scope3.category_sub_results.data_source_type IS 'Data origin: PRIMARY (supplier), SECONDARY (database), ESTIMATED, MODELED, EXTRAPOLATED.';
