-- =============================================================================
-- V329: PACK-041 Scope 1-2 Complete Pack - Scope 2 Consolidation & Results
-- =============================================================================
-- Pack:         PACK-041 (Scope 1-2 Complete Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates Scope 2 inventory consolidation tables implementing GHG Protocol
-- Scope 2 Guidance dual reporting requirements. Tracks both location-based
-- and market-based results per facility, contractual instruments (RECs, PPAs,
-- GOs, supplier-specific factors), and dual reporting reconciliation with
-- variance analysis between the two methods.
--
-- Tables (4):
--   1. ghg_scope12.scope2_inventories
--   2. ghg_scope12.scope2_facility_results
--   3. ghg_scope12.contractual_instruments
--   4. ghg_scope12.dual_reporting_reconciliation
--
-- Also includes: indexes, RLS, comments.
-- Previous: V328__pack041_scope1_consolidation.sql
-- =============================================================================

SET search_path TO ghg_scope12, public;

-- =============================================================================
-- Table 1: ghg_scope12.scope2_inventories
-- =============================================================================
-- Top-level Scope 2 inventory for an organization and reporting year. Per
-- GHG Protocol Scope 2 Guidance (2015), organizations must report both
-- location-based and market-based totals. The inventory tracks both method
-- totals and the workflow status through to verification.

CREATE TABLE ghg_scope12.scope2_inventories (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    boundary_id                 UUID            NOT NULL REFERENCES ghg_scope12.organizational_boundaries(id) ON DELETE RESTRICT,
    reporting_year              INTEGER         NOT NULL,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    -- Location-based totals
    location_based_co2          DECIMAL(15,3)   DEFAULT 0,
    location_based_ch4          DECIMAL(12,6)   DEFAULT 0,
    location_based_n2o          DECIMAL(12,6)   DEFAULT 0,
    location_based_total        DECIMAL(15,3)   NOT NULL DEFAULT 0,
    -- Market-based totals
    market_based_co2            DECIMAL(15,3)   DEFAULT 0,
    market_based_ch4            DECIMAL(12,6)   DEFAULT 0,
    market_based_n2o            DECIMAL(12,6)   DEFAULT 0,
    market_based_total          DECIMAL(15,3)   NOT NULL DEFAULT 0,
    -- Method variance
    method_variance_tco2e       DECIMAL(15,3)   DEFAULT 0,
    method_variance_pct         DECIMAL(8,4),
    -- Quality metrics
    gwp_source                  VARCHAR(20)     NOT NULL DEFAULT 'AR5',
    data_completeness_pct       DECIMAL(5,2),
    weighted_data_quality       DECIMAL(5,2),
    facilities_reported         INTEGER         DEFAULT 0,
    facilities_total            INTEGER         DEFAULT 0,
    total_consumption_mwh       DECIMAL(15,3),
    renewable_pct               DECIMAL(5,2),
    -- Workflow
    prepared_by                 VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    approved_by                 VARCHAR(255),
    notes                       TEXT,
    methodology_notes           TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    finalized_at                TIMESTAMPTZ,
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_s2inv_year CHECK (
        reporting_year >= 1990 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p041_s2inv_status CHECK (
        status IN (
            'DRAFT', 'DATA_COLLECTION', 'CALCULATION', 'REVIEW',
            'APPROVED', 'FINALIZED', 'VERIFIED', 'RESTATED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p041_s2inv_gwp CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'SAR', 'TAR')
    ),
    CONSTRAINT chk_p041_s2inv_loc_total CHECK (
        location_based_total >= 0
    ),
    CONSTRAINT chk_p041_s2inv_mkt_total CHECK (
        market_based_total >= 0
    ),
    CONSTRAINT chk_p041_s2inv_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p041_s2inv_quality CHECK (
        weighted_data_quality IS NULL OR (weighted_data_quality >= 0 AND weighted_data_quality <= 100)
    ),
    CONSTRAINT chk_p041_s2inv_renewable CHECK (
        renewable_pct IS NULL OR (renewable_pct >= 0 AND renewable_pct <= 100)
    ),
    CONSTRAINT chk_p041_s2inv_dates CHECK (
        reporting_period_start IS NULL OR reporting_period_end IS NULL OR
        reporting_period_start <= reporting_period_end
    ),
    CONSTRAINT uq_p041_s2inv_org_year UNIQUE (organization_id, reporting_year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_s2inv_tenant          ON ghg_scope12.scope2_inventories(tenant_id);
CREATE INDEX idx_p041_s2inv_org             ON ghg_scope12.scope2_inventories(organization_id);
CREATE INDEX idx_p041_s2inv_boundary        ON ghg_scope12.scope2_inventories(boundary_id);
CREATE INDEX idx_p041_s2inv_year            ON ghg_scope12.scope2_inventories(reporting_year);
CREATE INDEX idx_p041_s2inv_status          ON ghg_scope12.scope2_inventories(status);
CREATE INDEX idx_p041_s2inv_loc_total       ON ghg_scope12.scope2_inventories(location_based_total DESC);
CREATE INDEX idx_p041_s2inv_mkt_total       ON ghg_scope12.scope2_inventories(market_based_total DESC);
CREATE INDEX idx_p041_s2inv_created         ON ghg_scope12.scope2_inventories(created_at DESC);
CREATE INDEX idx_p041_s2inv_metadata        ON ghg_scope12.scope2_inventories USING GIN(metadata);

-- Composite: tenant + year for dashboard queries
CREATE INDEX idx_p041_s2inv_tenant_year     ON ghg_scope12.scope2_inventories(tenant_id, reporting_year DESC);

-- Composite: org + finalized for reporting
CREATE INDEX idx_p041_s2inv_org_final       ON ghg_scope12.scope2_inventories(organization_id, reporting_year DESC)
    WHERE status IN ('FINALIZED', 'VERIFIED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_s2inv_updated
    BEFORE UPDATE ON ghg_scope12.scope2_inventories
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_scope12.scope2_facility_results
-- =============================================================================
-- Per-facility Scope 2 emission results with both location-based and
-- market-based calculations. Each row represents one energy type at one
-- facility. Location-based uses grid average emission factors; market-based
-- uses contractual instrument factors or residual mix.

CREATE TABLE ghg_scope12.scope2_facility_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_scope12.scope2_inventories(id) ON DELETE CASCADE,
    facility_id                 UUID            NOT NULL REFERENCES ghg_scope12.facilities(id) ON DELETE RESTRICT,
    energy_type                 VARCHAR(50)     NOT NULL,
    energy_sub_type             VARCHAR(50),
    -- Consumption data
    consumption_kwh             DECIMAL(15,3)   NOT NULL DEFAULT 0,
    consumption_mwh             DECIMAL(12,3)   GENERATED ALWAYS AS (consumption_kwh / 1000.0) STORED,
    consumption_gj              DECIMAL(12,3),
    -- Location-based calculation
    location_ef                 DECIMAL(10,6)   NOT NULL DEFAULT 0,
    location_ef_source          VARCHAR(100),
    location_ef_year            INTEGER,
    location_ef_region          VARCHAR(100),
    location_co2_tonnes         DECIMAL(12,3)   DEFAULT 0,
    location_ch4_tonnes         DECIMAL(12,6)   DEFAULT 0,
    location_n2o_tonnes         DECIMAL(12,6)   DEFAULT 0,
    location_co2e               DECIMAL(12,3)   NOT NULL DEFAULT 0,
    -- Market-based calculation
    market_ef                   DECIMAL(10,6)   NOT NULL DEFAULT 0,
    market_ef_source            VARCHAR(100),
    market_ef_type              VARCHAR(30),
    market_co2_tonnes           DECIMAL(12,3)   DEFAULT 0,
    market_ch4_tonnes           DECIMAL(12,6)   DEFAULT 0,
    market_n2o_tonnes           DECIMAL(12,6)   DEFAULT 0,
    market_co2e                 DECIMAL(12,3)   NOT NULL DEFAULT 0,
    -- Contractual instruments applied
    instrument_volume_mwh       DECIMAL(12,3)   DEFAULT 0,
    residual_volume_mwh         DECIMAL(12,3)   DEFAULT 0,
    residual_ef                 DECIMAL(10,6),
    -- Quality
    data_quality_indicator      VARCHAR(20),
    method_notes                TEXT,
    -- Provenance
    calculation_hash            VARCHAR(64),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_s2fr_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'STEAM', 'COOLING', 'HEATING',
            'COMBINED_HEAT_POWER', 'DISTRICT_ENERGY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_s2fr_market_ef_type CHECK (
        market_ef_type IS NULL OR market_ef_type IN (
            'SUPPLIER_SPECIFIC', 'REC_GO', 'PPA', 'GREEN_TARIFF',
            'RESIDUAL_MIX', 'GRID_DEFAULT', 'DIRECT_CONTRACT'
        )
    ),
    CONSTRAINT chk_p041_s2fr_consumption CHECK (
        consumption_kwh >= 0
    ),
    CONSTRAINT chk_p041_s2fr_loc_ef CHECK (
        location_ef >= 0
    ),
    CONSTRAINT chk_p041_s2fr_loc_co2e CHECK (
        location_co2e >= 0
    ),
    CONSTRAINT chk_p041_s2fr_mkt_ef CHECK (
        market_ef >= 0
    ),
    CONSTRAINT chk_p041_s2fr_mkt_co2e CHECK (
        market_co2e >= 0
    ),
    CONSTRAINT chk_p041_s2fr_instrument_vol CHECK (
        instrument_volume_mwh IS NULL OR instrument_volume_mwh >= 0
    ),
    CONSTRAINT chk_p041_s2fr_residual_vol CHECK (
        residual_volume_mwh IS NULL OR residual_volume_mwh >= 0
    ),
    CONSTRAINT chk_p041_s2fr_quality_ind CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'HIGH', 'MEDIUM', 'LOW', 'ESTIMATED', 'DEFAULT'
        )
    ),
    CONSTRAINT uq_p041_s2fr_inv_fac_energy UNIQUE (inventory_id, facility_id, energy_type, energy_sub_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_s2fr_tenant           ON ghg_scope12.scope2_facility_results(tenant_id);
CREATE INDEX idx_p041_s2fr_inventory        ON ghg_scope12.scope2_facility_results(inventory_id);
CREATE INDEX idx_p041_s2fr_facility         ON ghg_scope12.scope2_facility_results(facility_id);
CREATE INDEX idx_p041_s2fr_energy_type      ON ghg_scope12.scope2_facility_results(energy_type);
CREATE INDEX idx_p041_s2fr_loc_co2e         ON ghg_scope12.scope2_facility_results(location_co2e DESC);
CREATE INDEX idx_p041_s2fr_mkt_co2e         ON ghg_scope12.scope2_facility_results(market_co2e DESC);
CREATE INDEX idx_p041_s2fr_consumption      ON ghg_scope12.scope2_facility_results(consumption_kwh DESC);
CREATE INDEX idx_p041_s2fr_calc_hash        ON ghg_scope12.scope2_facility_results(calculation_hash);
CREATE INDEX idx_p041_s2fr_created          ON ghg_scope12.scope2_facility_results(created_at DESC);
CREATE INDEX idx_p041_s2fr_metadata         ON ghg_scope12.scope2_facility_results USING GIN(metadata);

-- Composite: inventory + energy type for aggregation
CREATE INDEX idx_p041_s2fr_inv_energy       ON ghg_scope12.scope2_facility_results(inventory_id, energy_type);

-- Composite: facility + energy type for facility reporting
CREATE INDEX idx_p041_s2fr_fac_energy       ON ghg_scope12.scope2_facility_results(facility_id, energy_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_s2fr_updated
    BEFORE UPDATE ON ghg_scope12.scope2_facility_results
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_scope12.contractual_instruments
-- =============================================================================
-- Contractual instruments that convey emission factor claims for market-based
-- accounting per GHG Protocol Scope 2 Guidance Quality Criteria. Includes
-- Renewable Energy Certificates (RECs), Guarantees of Origin (GOs), Power
-- Purchase Agreements (PPAs), green tariffs, and supplier-specific factors.
-- Each instrument is tracked with vintage, volume, and allocation to facilities.

CREATE TABLE ghg_scope12.contractual_instruments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL REFERENCES ghg_scope12.organizations(id) ON DELETE CASCADE,
    instrument_type             VARCHAR(30)     NOT NULL,
    instrument_name             VARCHAR(255)    NOT NULL,
    issuer                      VARCHAR(255)    NOT NULL,
    issuer_country              VARCHAR(3),
    registry                    VARCHAR(100),
    certificate_id              VARCHAR(200),
    technology                  VARCHAR(50),
    generation_source           VARCHAR(100),
    volume_mwh                  DECIMAL(12,3)   NOT NULL,
    remaining_volume_mwh        DECIMAL(12,3)   NOT NULL,
    emission_factor             DECIMAL(10,6)   NOT NULL DEFAULT 0,
    emission_factor_unit        VARCHAR(30)     DEFAULT 'tCO2e/MWh',
    vintage_year                INTEGER         NOT NULL,
    vintage_month               INTEGER,
    contract_start_date         DATE,
    contract_end_date           DATE,
    delivery_date               DATE,
    cancellation_date           DATE,
    cancellation_reference      VARCHAR(200),
    allocated_to_facility_id    UUID            REFERENCES ghg_scope12.facilities(id) ON DELETE SET NULL,
    allocation_pct              DECIMAL(5,2)    DEFAULT 100.00,
    allocation_mwh              DECIMAL(12,3),
    meets_quality_criteria      BOOLEAN         NOT NULL DEFAULT true,
    quality_criteria_notes      TEXT,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    cost_per_mwh                DECIMAL(10,4),
    total_cost                  DECIMAL(18,2),
    currency                    VARCHAR(3)      DEFAULT 'USD',
    supporting_document         VARCHAR(500),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p041_ci_type CHECK (
        instrument_type IN (
            'REC', 'GO', 'I_REC', 'PPA', 'VPPA', 'GREEN_TARIFF',
            'DIRECT_CONTRACT', 'SUPPLIER_EMISSION_FACTOR',
            'UTILITY_SPECIFIC', 'BUNDLED_REC', 'UNBUNDLED_REC', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_ci_technology CHECK (
        technology IS NULL OR technology IN (
            'SOLAR', 'WIND_ONSHORE', 'WIND_OFFSHORE', 'HYDRO', 'BIOMASS',
            'BIOGAS', 'GEOTHERMAL', 'NUCLEAR', 'TIDAL', 'WAVE',
            'MIXED_RENEWABLE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_ci_volume CHECK (
        volume_mwh > 0
    ),
    CONSTRAINT chk_p041_ci_remaining CHECK (
        remaining_volume_mwh >= 0 AND remaining_volume_mwh <= volume_mwh
    ),
    CONSTRAINT chk_p041_ci_ef CHECK (
        emission_factor >= 0
    ),
    CONSTRAINT chk_p041_ci_vintage_year CHECK (
        vintage_year >= 1990 AND vintage_year <= 2100
    ),
    CONSTRAINT chk_p041_ci_vintage_month CHECK (
        vintage_month IS NULL OR (vintage_month >= 1 AND vintage_month <= 12)
    ),
    CONSTRAINT chk_p041_ci_allocation_pct CHECK (
        allocation_pct IS NULL OR (allocation_pct >= 0 AND allocation_pct <= 100)
    ),
    CONSTRAINT chk_p041_ci_allocation_mwh CHECK (
        allocation_mwh IS NULL OR (allocation_mwh >= 0 AND allocation_mwh <= volume_mwh)
    ),
    CONSTRAINT chk_p041_ci_status CHECK (
        status IN ('ACTIVE', 'ALLOCATED', 'CANCELLED', 'EXPIRED', 'RETIRED', 'PENDING')
    ),
    CONSTRAINT chk_p041_ci_contract_dates CHECK (
        contract_start_date IS NULL OR contract_end_date IS NULL OR
        contract_start_date <= contract_end_date
    ),
    CONSTRAINT chk_p041_ci_cost CHECK (
        cost_per_mwh IS NULL OR cost_per_mwh >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_ci_tenant             ON ghg_scope12.contractual_instruments(tenant_id);
CREATE INDEX idx_p041_ci_org               ON ghg_scope12.contractual_instruments(organization_id);
CREATE INDEX idx_p041_ci_type              ON ghg_scope12.contractual_instruments(instrument_type);
CREATE INDEX idx_p041_ci_issuer            ON ghg_scope12.contractual_instruments(issuer);
CREATE INDEX idx_p041_ci_vintage           ON ghg_scope12.contractual_instruments(vintage_year);
CREATE INDEX idx_p041_ci_technology        ON ghg_scope12.contractual_instruments(technology);
CREATE INDEX idx_p041_ci_facility          ON ghg_scope12.contractual_instruments(allocated_to_facility_id);
CREATE INDEX idx_p041_ci_status            ON ghg_scope12.contractual_instruments(status);
CREATE INDEX idx_p041_ci_certificate       ON ghg_scope12.contractual_instruments(certificate_id);
CREATE INDEX idx_p041_ci_registry          ON ghg_scope12.contractual_instruments(registry);
CREATE INDEX idx_p041_ci_created           ON ghg_scope12.contractual_instruments(created_at DESC);
CREATE INDEX idx_p041_ci_metadata          ON ghg_scope12.contractual_instruments USING GIN(metadata);

-- Composite: org + vintage + active for allocation
CREATE INDEX idx_p041_ci_org_vintage_active ON ghg_scope12.contractual_instruments(organization_id, vintage_year)
    WHERE status IN ('ACTIVE', 'ALLOCATED');

-- Composite: facility + active for market-based calculation
CREATE INDEX idx_p041_ci_fac_active        ON ghg_scope12.contractual_instruments(allocated_to_facility_id, instrument_type)
    WHERE status IN ('ACTIVE', 'ALLOCATED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_ci_updated
    BEFORE UPDATE ON ghg_scope12.contractual_instruments
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_scope12.dual_reporting_reconciliation
-- =============================================================================
-- Reconciliation between location-based and market-based Scope 2 results per
-- facility. Tracks the variance between methods, identifies the primary
-- drivers of variance (RECs, PPAs, residual mix differences, grid vs supplier
-- factors), and provides transparency for dual reporting disclosures.

CREATE TABLE ghg_scope12.dual_reporting_reconciliation (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_scope12.scope2_inventories(id) ON DELETE CASCADE,
    facility_id                 UUID            NOT NULL REFERENCES ghg_scope12.facilities(id) ON DELETE RESTRICT,
    energy_type                 VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    -- Location-based
    location_co2e               DECIMAL(12,3)   NOT NULL DEFAULT 0,
    location_ef_used            DECIMAL(10,6),
    location_ef_source          VARCHAR(100),
    -- Market-based
    market_co2e                 DECIMAL(12,3)   NOT NULL DEFAULT 0,
    market_ef_used              DECIMAL(10,6),
    market_ef_source            VARCHAR(100),
    -- Variance
    variance_co2e               DECIMAL(12,3)   NOT NULL DEFAULT 0,
    variance_pct                DECIMAL(8,4),
    absolute_variance_co2e      DECIMAL(12,3)   GENERATED ALWAYS AS (ABS(variance_co2e)) STORED,
    -- Drivers
    driver                      TEXT,
    driver_category             VARCHAR(30),
    instrument_volume_mwh       DECIMAL(12,3)   DEFAULT 0,
    residual_mix_volume_mwh     DECIMAL(12,3)   DEFAULT 0,
    grid_default_volume_mwh     DECIMAL(12,3)   DEFAULT 0,
    -- Analysis
    recommendation              TEXT,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p041_drr_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'STEAM', 'COOLING', 'HEATING',
            'COMBINED_HEAT_POWER', 'DISTRICT_ENERGY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p041_drr_loc_co2e CHECK (
        location_co2e >= 0
    ),
    CONSTRAINT chk_p041_drr_mkt_co2e CHECK (
        market_co2e >= 0
    ),
    CONSTRAINT chk_p041_drr_driver_cat CHECK (
        driver_category IS NULL OR driver_category IN (
            'RENEWABLE_PROCUREMENT', 'RESIDUAL_MIX_DIFF', 'SUPPLIER_FACTOR',
            'GRID_REGION_DIFF', 'TEMPORAL_MISMATCH', 'METHODOLOGY_DIFF',
            'MULTIPLE_FACTORS', 'NONE'
        )
    ),
    CONSTRAINT chk_p041_drr_instrument_vol CHECK (
        instrument_volume_mwh IS NULL OR instrument_volume_mwh >= 0
    ),
    CONSTRAINT chk_p041_drr_residual_vol CHECK (
        residual_mix_volume_mwh IS NULL OR residual_mix_volume_mwh >= 0
    ),
    CONSTRAINT uq_p041_drr_inv_fac_energy UNIQUE (inventory_id, facility_id, energy_type)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p041_drr_tenant            ON ghg_scope12.dual_reporting_reconciliation(tenant_id);
CREATE INDEX idx_p041_drr_inventory         ON ghg_scope12.dual_reporting_reconciliation(inventory_id);
CREATE INDEX idx_p041_drr_facility          ON ghg_scope12.dual_reporting_reconciliation(facility_id);
CREATE INDEX idx_p041_drr_energy_type       ON ghg_scope12.dual_reporting_reconciliation(energy_type);
CREATE INDEX idx_p041_drr_variance          ON ghg_scope12.dual_reporting_reconciliation(variance_co2e);
CREATE INDEX idx_p041_drr_driver_cat        ON ghg_scope12.dual_reporting_reconciliation(driver_category);
CREATE INDEX idx_p041_drr_created           ON ghg_scope12.dual_reporting_reconciliation(created_at DESC);

-- Composite: inventory + significant variances
CREATE INDEX idx_p041_drr_inv_sig_var       ON ghg_scope12.dual_reporting_reconciliation(inventory_id, absolute_variance_co2e DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p041_drr_updated
    BEFORE UPDATE ON ghg_scope12.dual_reporting_reconciliation
    FOR EACH ROW EXECUTE FUNCTION ghg_scope12.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_scope12.scope2_inventories ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.scope2_facility_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.contractual_instruments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_scope12.dual_reporting_reconciliation ENABLE ROW LEVEL SECURITY;

CREATE POLICY p041_s2inv_tenant_isolation
    ON ghg_scope12.scope2_inventories
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_s2inv_service_bypass
    ON ghg_scope12.scope2_inventories
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_s2fr_tenant_isolation
    ON ghg_scope12.scope2_facility_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_s2fr_service_bypass
    ON ghg_scope12.scope2_facility_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_ci_tenant_isolation
    ON ghg_scope12.contractual_instruments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_ci_service_bypass
    ON ghg_scope12.contractual_instruments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p041_drr_tenant_isolation
    ON ghg_scope12.dual_reporting_reconciliation
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p041_drr_service_bypass
    ON ghg_scope12.dual_reporting_reconciliation
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.scope2_inventories TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.scope2_facility_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.contractual_instruments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_scope12.dual_reporting_reconciliation TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_scope12.scope2_inventories IS
    'Top-level Scope 2 inventory per organization and year with dual reporting totals (location-based and market-based) per GHG Protocol Scope 2 Guidance.';
COMMENT ON TABLE ghg_scope12.scope2_facility_results IS
    'Per-facility Scope 2 results with both location-based and market-based calculations, consumption data, and emission factors used.';
COMMENT ON TABLE ghg_scope12.contractual_instruments IS
    'Contractual instruments (RECs, GOs, PPAs, green tariffs, supplier factors) for market-based Scope 2 accounting with volume, vintage, and allocation tracking.';
COMMENT ON TABLE ghg_scope12.dual_reporting_reconciliation IS
    'Reconciliation between location-based and market-based Scope 2 results per facility with variance analysis and driver identification.';

COMMENT ON COLUMN ghg_scope12.scope2_inventories.location_based_total IS 'Total Scope 2 emissions using grid-average emission factors (tCO2e).';
COMMENT ON COLUMN ghg_scope12.scope2_inventories.market_based_total IS 'Total Scope 2 emissions using contractual instruments and residual mix (tCO2e).';
COMMENT ON COLUMN ghg_scope12.scope2_inventories.method_variance_tco2e IS 'Difference between location-based and market-based totals (location - market).';
COMMENT ON COLUMN ghg_scope12.scope2_inventories.renewable_pct IS 'Percentage of total electricity consumption covered by renewable instruments.';
COMMENT ON COLUMN ghg_scope12.scope2_inventories.provenance_hash IS 'SHA-256 hash of all facility result hashes for inventory-level provenance.';

COMMENT ON COLUMN ghg_scope12.scope2_facility_results.consumption_mwh IS 'Auto-calculated MWh from kWh consumption for convenience.';
COMMENT ON COLUMN ghg_scope12.scope2_facility_results.location_ef IS 'Grid average emission factor used for location-based calculation (tCO2e/kWh).';
COMMENT ON COLUMN ghg_scope12.scope2_facility_results.market_ef IS 'Market-based emission factor (contractual or residual mix) (tCO2e/kWh).';
COMMENT ON COLUMN ghg_scope12.scope2_facility_results.instrument_volume_mwh IS 'Volume covered by contractual instruments (RECs, PPAs, etc.) in MWh.';
COMMENT ON COLUMN ghg_scope12.scope2_facility_results.residual_volume_mwh IS 'Volume covered by residual mix factor (no contractual instrument) in MWh.';

COMMENT ON COLUMN ghg_scope12.contractual_instruments.instrument_type IS 'Type: REC, GO, I_REC, PPA, VPPA, GREEN_TARIFF, DIRECT_CONTRACT, SUPPLIER_EMISSION_FACTOR, etc.';
COMMENT ON COLUMN ghg_scope12.contractual_instruments.meets_quality_criteria IS 'Whether instrument meets GHG Protocol Scope 2 Quality Criteria (conveyance, geographic, temporal, unique claim).';
COMMENT ON COLUMN ghg_scope12.contractual_instruments.remaining_volume_mwh IS 'Unallocated volume remaining for further facility allocation.';

COMMENT ON COLUMN ghg_scope12.dual_reporting_reconciliation.variance_co2e IS 'Location minus market difference in tCO2e. Positive = location higher than market.';
COMMENT ON COLUMN ghg_scope12.dual_reporting_reconciliation.driver_category IS 'Primary driver category: RENEWABLE_PROCUREMENT, RESIDUAL_MIX_DIFF, SUPPLIER_FACTOR, etc.';
