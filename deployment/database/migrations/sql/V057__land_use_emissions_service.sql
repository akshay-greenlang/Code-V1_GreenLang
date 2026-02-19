-- =============================================================================
-- V057: Land Use Emissions Service Schema
-- =============================================================================
-- Component: AGENT-MRV-006 (GL-MRV-SCOPE1-006)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Land Use Emissions Agent (GL-MRV-SCOPE1-006) with capabilities for
-- land parcel registry management (parcel identification with area in
-- hectares, climate zone, soil type, land category classification
-- including forest_land, cropland, grassland, wetlands, settlements,
-- other_land, with geographic coordinates, elevation, country codes,
-- management practices, input levels, and peatland status tracking),
-- IPCC default carbon stock factor registry (per land category, climate
-- zone, soil type, and carbon pool including above_ground_biomass,
-- below_ground_biomass, dead_wood, litter, soil_organic_carbon with
-- stock values in tC/ha, growth rates, root-shoot ratios, dead wood
-- fractions, litter stocks, tiered sourcing from IPCC/GPG/UNFCCC/
-- NATIONAL/CUSTOM, and confidence percentages with validity date
-- ranges), emission factor database (land-use conversion from/to
-- category pairs by climate zone and gas CO2/CH4/N2O with methodology
-- references), land-use transition records (parcel-level from/to
-- category transitions with transition dates, areas, REMAINING/
-- CONVERSION types, disturbance types, deforestation flags, and
-- peatland change flags), carbon stock snapshots (point-in-time
-- per-pool stock measurements in tC/ha and total tC with measurement
-- methods, tiers, sources, and uncertainty percentages), land use
-- emission calculations (parcel-level from/to category calculations
-- with area, climate zone, soil type, tier 1/2/3 methods, GWP
-- sources AR4/AR5/AR6, total emissions/removals/net in tCO2e,
-- SHA-256 calculation hashes, and step-by-step trace JSONB), per-gas
-- per-pool calculation detail breakdowns (carbon pool x gas emission
-- values in tC and tCO2e with removal flags, factor values, formulas,
-- and metadata), soil organic carbon (SOC) assessments (IPCC Tier 1
-- SOC = SOCref x FLU x FMG x FI with management practice and input
-- level stock change factors, reference SOC values, current/previous
-- SOC stocks, annual delta SOC, configurable depth and transition
-- period), regulatory compliance records (GHG Protocol/IPCC/UNFCCC/
-- EU LULUCF/ISO 14064 framework checks with total requirements,
-- passed/failed counts, and findings JSONB), and step-by-step audit
-- trail entries (entity-level action trace with prev_hash/entry_hash
-- chaining for tamper-evident provenance and actor attribution).
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. lu_land_parcels              - Land parcel registry (area, climate, soil, category)
--   2. lu_carbon_stock_factors      - IPCC default carbon stock values by pool
--   3. lu_emission_factors          - Emission factors by land-use conversion type
--   4. lu_land_use_transitions      - Land-use change records (REMAINING/CONVERSION)
--   5. lu_carbon_stock_snapshots    - Point-in-time carbon stock measurements
--   6. lu_calculations              - Emission calculation results (emissions/removals/net)
--   7. lu_calculation_details       - Per-gas per-pool breakdown (tC and tCO2e)
--   8. lu_soc_assessments           - Soil organic carbon assessments (SOCref x FLU x FMG x FI)
--   9. lu_compliance_records        - Regulatory compliance checks (GHG Protocol/IPCC/UNFCCC/EU LULUCF/ISO 14064)
--  10. lu_audit_entries             - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  11. lu_calculation_events        - Calculation event time-series (hypertable on event_time)
--  12. lu_transition_events         - Transition event time-series (hypertable on event_time)
--  13. lu_compliance_events         - Compliance event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. lu_hourly_calculation_stats  - Hourly count/sum(emissions)/sum(removals) by land_category and method
--   2. lu_daily_emission_totals     - Daily count/sum(emissions) by land_category and carbon_pool
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-006.
-- Previous: V056__flaring_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS land_use_emissions_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION land_use_emissions_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: land_use_emissions_service.lu_land_parcels
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_land_parcels (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    name                VARCHAR(500)    NOT NULL,
    description         TEXT,
    area_ha             NUMERIC(15,4)   NOT NULL,
    climate_zone        VARCHAR(50)     NOT NULL,
    soil_type           VARCHAR(50)     NOT NULL,
    land_category       VARCHAR(50)     NOT NULL,
    latitude            NUMERIC(10,7),
    longitude           NUMERIC(11,7),
    elevation_m         NUMERIC(8,2),
    country_code        CHAR(3),
    management_practice VARCHAR(50),
    input_level         VARCHAR(50),
    peatland_status     VARCHAR(50),
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(200),
    updated_by          VARCHAR(200)
);

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_area_ha_positive CHECK (area_ha > 0);

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_climate_zone_not_empty CHECK (LENGTH(TRIM(climate_zone)) > 0);

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_climate_zone CHECK (climate_zone IN (
        'tropical_wet', 'tropical_moist', 'tropical_dry', 'tropical_montane',
        'warm_temperate_moist', 'warm_temperate_dry',
        'cool_temperate_moist', 'cool_temperate_dry',
        'boreal_moist', 'boreal_dry',
        'polar_moist', 'polar_dry'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_soil_type_not_empty CHECK (LENGTH(TRIM(soil_type)) > 0);

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_soil_type CHECK (soil_type IN (
        'high_activity_clay', 'low_activity_clay', 'sandy',
        'spodic', 'volcanic', 'wetland', 'organic', 'other'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_land_category CHECK (land_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_latitude_range CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_longitude_range CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_management_practice CHECK (management_practice IS NULL OR management_practice IN (
        'full_tillage', 'reduced_tillage', 'no_tillage',
        'improved', 'unimproved', 'degraded',
        'native', 'managed', 'plantation',
        'other'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_input_level CHECK (input_level IS NULL OR input_level IN (
        'low', 'medium', 'high', 'high_without_manure', 'high_with_manure'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_peatland_status CHECK (peatland_status IS NULL OR peatland_status IN (
        'not_peatland', 'intact', 'drained', 'rewetted', 'extracted'
    ));

ALTER TABLE land_use_emissions_service.lu_land_parcels
    ADD CONSTRAINT chk_lp_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_lp_updated_at
    BEFORE UPDATE ON land_use_emissions_service.lu_land_parcels
    FOR EACH ROW EXECUTE FUNCTION land_use_emissions_service.set_updated_at();

-- =============================================================================
-- Table 2: land_use_emissions_service.lu_carbon_stock_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_carbon_stock_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    land_category       VARCHAR(50)     NOT NULL,
    climate_zone        VARCHAR(50)     NOT NULL,
    soil_type           VARCHAR(50)     NOT NULL,
    carbon_pool         VARCHAR(50)     NOT NULL,
    stock_tc_ha         NUMERIC(12,4)   NOT NULL,
    growth_rate_tc_ha_yr NUMERIC(10,6),
    root_shoot_ratio    NUMERIC(8,6),
    dead_wood_fraction  NUMERIC(8,6),
    litter_stock_tc_ha  NUMERIC(10,4),
    source              VARCHAR(50)     NOT NULL,
    tier                INTEGER         NOT NULL,
    confidence_pct      NUMERIC(5,2),
    valid_from          DATE,
    valid_to            DATE,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_lu_csf_category_zone_soil_pool_source
        UNIQUE (land_category, climate_zone, soil_type, carbon_pool, source)
);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_land_category CHECK (land_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_climate_zone CHECK (climate_zone IN (
        'tropical_wet', 'tropical_moist', 'tropical_dry', 'tropical_montane',
        'warm_temperate_moist', 'warm_temperate_dry',
        'cool_temperate_moist', 'cool_temperate_dry',
        'boreal_moist', 'boreal_dry',
        'polar_moist', 'polar_dry'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_soil_type CHECK (soil_type IN (
        'high_activity_clay', 'low_activity_clay', 'sandy',
        'spodic', 'volcanic', 'wetland', 'organic', 'other'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_carbon_pool CHECK (carbon_pool IN (
        'above_ground_biomass', 'below_ground_biomass', 'dead_wood',
        'litter', 'soil_organic_carbon'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_stock_tc_ha_non_negative CHECK (stock_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_growth_rate_non_negative CHECK (growth_rate_tc_ha_yr IS NULL OR growth_rate_tc_ha_yr >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_root_shoot_ratio_non_negative CHECK (root_shoot_ratio IS NULL OR root_shoot_ratio >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_dead_wood_fraction_range CHECK (dead_wood_fraction IS NULL OR (dead_wood_fraction >= 0 AND dead_wood_fraction <= 1));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_litter_stock_non_negative CHECK (litter_stock_tc_ha IS NULL OR litter_stock_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_source CHECK (source IN (
        'IPCC', 'GPG', 'UNFCCC', 'NATIONAL', 'CUSTOM'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_tier CHECK (tier IN (1, 2, 3));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_confidence_pct_range CHECK (confidence_pct IS NULL OR (confidence_pct >= 0 AND confidence_pct <= 100));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors
    ADD CONSTRAINT chk_csf_date_order CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from);

CREATE TRIGGER trg_csf_updated_at
    BEFORE UPDATE ON land_use_emissions_service.lu_carbon_stock_factors
    FOR EACH ROW EXECUTE FUNCTION land_use_emissions_service.set_updated_at();

-- =============================================================================
-- Table 3: land_use_emissions_service.lu_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    from_category       VARCHAR(50)     NOT NULL,
    to_category         VARCHAR(50)     NOT NULL,
    climate_zone        VARCHAR(50)     NOT NULL,
    gas                 VARCHAR(10)     NOT NULL,
    ef_value            NUMERIC(15,8)   NOT NULL,
    ef_unit             VARCHAR(50)     NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    methodology         TEXT,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_lu_ef_from_to_zone_gas_source
        UNIQUE (from_category, to_category, climate_zone, gas, source)
);

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_from_category CHECK (from_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_to_category CHECK (to_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_climate_zone CHECK (climate_zone IN (
        'tropical_wet', 'tropical_moist', 'tropical_dry', 'tropical_montane',
        'warm_temperate_moist', 'warm_temperate_dry',
        'cool_temperate_moist', 'cool_temperate_dry',
        'boreal_moist', 'boreal_dry',
        'polar_moist', 'polar_dry'
    ));

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O'
    ));

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_ef_value_non_negative CHECK (ef_value >= 0);

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_ef_unit_not_empty CHECK (LENGTH(TRIM(ef_unit)) > 0);

ALTER TABLE land_use_emissions_service.lu_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'IPCC', 'GPG', 'UNFCCC', 'NATIONAL', 'CUSTOM'
    ));

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON land_use_emissions_service.lu_emission_factors
    FOR EACH ROW EXECUTE FUNCTION land_use_emissions_service.set_updated_at();

-- =============================================================================
-- Table 4: land_use_emissions_service.lu_land_use_transitions
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_land_use_transitions (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    parcel_id           UUID            NOT NULL REFERENCES land_use_emissions_service.lu_land_parcels(id) ON DELETE CASCADE,
    from_category       VARCHAR(50)     NOT NULL,
    to_category         VARCHAR(50)     NOT NULL,
    transition_date     DATE            NOT NULL,
    area_ha             NUMERIC(15,4)   NOT NULL,
    transition_type     VARCHAR(20)     NOT NULL,
    disturbance_type    VARCHAR(50),
    is_deforestation    BOOLEAN         DEFAULT FALSE,
    is_peatland_change  BOOLEAN         DEFAULT FALSE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by          VARCHAR(200)
);

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_from_category CHECK (from_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_to_category CHECK (to_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_area_ha_positive CHECK (area_ha > 0);

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_transition_type CHECK (transition_type IN (
        'REMAINING', 'CONVERSION'
    ));

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_disturbance_type CHECK (disturbance_type IS NULL OR disturbance_type IN (
        'fire', 'storm', 'insect', 'drought', 'flood', 'logging',
        'clearing', 'cultivation', 'urbanization', 'mining',
        'drainage', 'rewetting', 'other'
    ));

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_deforestation_logic CHECK (
        is_deforestation = FALSE
        OR (from_category = 'forest_land' AND to_category != 'forest_land')
    );

ALTER TABLE land_use_emissions_service.lu_land_use_transitions
    ADD CONSTRAINT chk_lut_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 5: land_use_emissions_service.lu_carbon_stock_snapshots
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_carbon_stock_snapshots (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    parcel_id           UUID            NOT NULL REFERENCES land_use_emissions_service.lu_land_parcels(id) ON DELETE CASCADE,
    carbon_pool         VARCHAR(50)     NOT NULL,
    stock_tc_ha         NUMERIC(12,4)   NOT NULL,
    stock_total_tc      NUMERIC(15,4)   NOT NULL,
    measurement_date    DATE            NOT NULL,
    measurement_method  VARCHAR(100),
    tier                INTEGER         NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    uncertainty_pct     NUMERIC(8,4),
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_carbon_pool CHECK (carbon_pool IN (
        'above_ground_biomass', 'below_ground_biomass', 'dead_wood',
        'litter', 'soil_organic_carbon'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_stock_tc_ha_non_negative CHECK (stock_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_stock_total_tc_non_negative CHECK (stock_total_tc >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_tier CHECK (tier IN (1, 2, 3));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_source CHECK (source IN (
        'IPCC', 'GPG', 'UNFCCC', 'NATIONAL', 'FIELD_MEASUREMENT', 'REMOTE_SENSING', 'CUSTOM'
    ));

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_uncertainty_pct_non_negative CHECK (uncertainty_pct IS NULL OR uncertainty_pct >= 0);

ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots
    ADD CONSTRAINT chk_css_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 6: land_use_emissions_service.lu_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    parcel_id               UUID            REFERENCES land_use_emissions_service.lu_land_parcels(id),
    from_category           VARCHAR(50)     NOT NULL,
    to_category             VARCHAR(50)     NOT NULL,
    area_ha                 NUMERIC(15,4)   NOT NULL,
    climate_zone            VARCHAR(50)     NOT NULL,
    soil_type               VARCHAR(50)     NOT NULL,
    tier                    INTEGER         NOT NULL,
    method                  VARCHAR(50)     NOT NULL,
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    total_emissions_tco2e   NUMERIC(18,6)   NOT NULL DEFAULT 0,
    total_removals_tco2e    NUMERIC(18,6)   NOT NULL DEFAULT 0,
    net_emissions_tco2e     NUMERIC(18,6)   NOT NULL,
    calculation_hash        VARCHAR(64)     NOT NULL,
    trace_json              JSONB           DEFAULT '[]'::jsonb,
    reporting_period        VARCHAR(30),
    uncertainty_pct         NUMERIC(10,4),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    calculated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculated_by           VARCHAR(200),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_from_category CHECK (from_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_to_category CHECK (to_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_area_ha_positive CHECK (area_ha > 0);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_climate_zone CHECK (climate_zone IN (
        'tropical_wet', 'tropical_moist', 'tropical_dry', 'tropical_montane',
        'warm_temperate_moist', 'warm_temperate_dry',
        'cool_temperate_moist', 'cool_temperate_dry',
        'boreal_moist', 'boreal_dry',
        'polar_moist', 'polar_dry'
    ));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_soil_type CHECK (soil_type IN (
        'high_activity_clay', 'low_activity_clay', 'sandy',
        'spodic', 'volcanic', 'wetland', 'organic', 'other'
    ));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_tier CHECK (tier IN (1, 2, 3));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_method CHECK (method IN (
        'stock_difference', 'gain_loss', 'default_factor',
        'biomass_expansion', 'allometric', 'direct_measurement'
    ));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_total_emissions_non_negative CHECK (total_emissions_tco2e >= 0);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_total_removals_non_negative CHECK (total_removals_tco2e >= 0);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_calculation_hash_not_empty CHECK (LENGTH(TRIM(calculation_hash)) > 0);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_uncertainty_pct_non_negative CHECK (uncertainty_pct IS NULL OR uncertainty_pct >= 0);

ALTER TABLE land_use_emissions_service.lu_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON land_use_emissions_service.lu_calculations
    FOR EACH ROW EXECUTE FUNCTION land_use_emissions_service.set_updated_at();

-- =============================================================================
-- Table 7: land_use_emissions_service.lu_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_calculation_details (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id      UUID            NOT NULL REFERENCES land_use_emissions_service.lu_calculations(id) ON DELETE CASCADE,
    carbon_pool         VARCHAR(50)     NOT NULL,
    gas                 VARCHAR(10)     NOT NULL,
    emission_tc         NUMERIC(18,8)   NOT NULL,
    emission_tco2e      NUMERIC(18,6)   NOT NULL,
    is_removal          BOOLEAN         DEFAULT FALSE,
    factor_used         NUMERIC(15,8),
    formula             TEXT,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_calculation_details
    ADD CONSTRAINT chk_cd_carbon_pool CHECK (carbon_pool IN (
        'above_ground_biomass', 'below_ground_biomass', 'dead_wood',
        'litter', 'soil_organic_carbon'
    ));

ALTER TABLE land_use_emissions_service.lu_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O'
    ));

ALTER TABLE land_use_emissions_service.lu_calculation_details
    ADD CONSTRAINT chk_cd_factor_used_non_negative CHECK (factor_used IS NULL OR factor_used >= 0);

-- =============================================================================
-- Table 8: land_use_emissions_service.lu_soc_assessments
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_soc_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    parcel_id               UUID            NOT NULL REFERENCES land_use_emissions_service.lu_land_parcels(id) ON DELETE CASCADE,
    climate_zone            VARCHAR(50)     NOT NULL,
    soil_type               VARCHAR(50)     NOT NULL,
    land_category           VARCHAR(50)     NOT NULL,
    management_practice     VARCHAR(50),
    input_level             VARCHAR(50),
    depth_cm                INTEGER         NOT NULL DEFAULT 30,
    soc_ref_tc_ha           NUMERIC(12,4)   NOT NULL,
    f_lu                    NUMERIC(8,6)    NOT NULL,
    f_mg                    NUMERIC(8,6)    NOT NULL,
    f_i                     NUMERIC(8,6)    NOT NULL,
    soc_current_tc_ha       NUMERIC(12,4)   NOT NULL,
    soc_previous_tc_ha      NUMERIC(12,4),
    delta_soc_annual_tc_ha  NUMERIC(12,6),
    transition_years        INTEGER         NOT NULL DEFAULT 20,
    source                  VARCHAR(50)     NOT NULL DEFAULT 'IPCC',
    uncertainty_pct         NUMERIC(8,4),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    assessed_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessed_by             VARCHAR(200),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_climate_zone CHECK (climate_zone IN (
        'tropical_wet', 'tropical_moist', 'tropical_dry', 'tropical_montane',
        'warm_temperate_moist', 'warm_temperate_dry',
        'cool_temperate_moist', 'cool_temperate_dry',
        'boreal_moist', 'boreal_dry',
        'polar_moist', 'polar_dry'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_soil_type CHECK (soil_type IN (
        'high_activity_clay', 'low_activity_clay', 'sandy',
        'spodic', 'volcanic', 'wetland', 'organic', 'other'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_land_category CHECK (land_category IN (
        'forest_land', 'cropland', 'grassland', 'wetlands',
        'settlements', 'other_land'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_management_practice CHECK (management_practice IS NULL OR management_practice IN (
        'full_tillage', 'reduced_tillage', 'no_tillage',
        'improved', 'unimproved', 'degraded',
        'native', 'managed', 'plantation',
        'other'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_input_level CHECK (input_level IS NULL OR input_level IN (
        'low', 'medium', 'high', 'high_without_manure', 'high_with_manure'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_depth_cm_positive CHECK (depth_cm > 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_soc_ref_tc_ha_non_negative CHECK (soc_ref_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_f_lu_positive CHECK (f_lu > 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_f_mg_positive CHECK (f_mg > 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_f_i_positive CHECK (f_i > 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_soc_current_tc_ha_non_negative CHECK (soc_current_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_soc_previous_tc_ha_non_negative CHECK (soc_previous_tc_ha IS NULL OR soc_previous_tc_ha >= 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_transition_years_positive CHECK (transition_years > 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_source CHECK (source IN (
        'IPCC', 'GPG', 'UNFCCC', 'NATIONAL', 'FIELD_MEASUREMENT', 'CUSTOM'
    ));

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_uncertainty_pct_non_negative CHECK (uncertainty_pct IS NULL OR uncertainty_pct >= 0);

ALTER TABLE land_use_emissions_service.lu_soc_assessments
    ADD CONSTRAINT chk_soc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_soc_updated_at
    BEFORE UPDATE ON land_use_emissions_service.lu_soc_assessments
    FOR EACH ROW EXECUTE FUNCTION land_use_emissions_service.set_updated_at();

-- =============================================================================
-- Table 9: land_use_emissions_service.lu_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            REFERENCES land_use_emissions_service.lu_calculations(id),
    framework           VARCHAR(100)    NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    total_requirements  INTEGER         NOT NULL DEFAULT 0,
    passed              INTEGER         NOT NULL DEFAULT 0,
    failed              INTEGER         NOT NULL DEFAULT 0,
    findings            JSONB           DEFAULT '[]'::jsonb,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    checked_by          VARCHAR(200),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'IPCC', 'UNFCCC', 'EU_LULUCF', 'ISO_14064', 'CUSTOM'
    ));

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_total_requirements_non_negative CHECK (total_requirements >= 0);

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_passed_non_negative CHECK (passed >= 0);

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_failed_non_negative CHECK (failed >= 0);

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_passed_plus_failed CHECK (passed + failed <= total_requirements);

ALTER TABLE land_use_emissions_service.lu_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 10: land_use_emissions_service.lu_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           UUID            NOT NULL,
    action              VARCHAR(50)     NOT NULL,
    actor               VARCHAR(200),
    details             JSONB           DEFAULT '{}'::jsonb,
    prev_hash           VARCHAR(64),
    entry_hash          VARCHAR(64)     NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE land_use_emissions_service.lu_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE land_use_emissions_service.lu_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT',
        'TRANSITION', 'SNAPSHOT', 'SOC_ASSESS'
    ));

ALTER TABLE land_use_emissions_service.lu_audit_entries
    ADD CONSTRAINT chk_ae_entry_hash_not_empty CHECK (LENGTH(TRIM(entry_hash)) > 0);

ALTER TABLE land_use_emissions_service.lu_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: land_use_emissions_service.lu_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_calculation_events (
    time                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               UUID            NOT NULL,
    event_type              VARCHAR(50),
    calculation_id          UUID,
    land_category           VARCHAR(50),
    method                  VARCHAR(50),
    emissions_tco2e         NUMERIC(18,6),
    removals_tco2e          NUMERIC(18,6),
    duration_ms             NUMERIC(12,2),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'land_use_emissions_service.lu_calculation_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE land_use_emissions_service.lu_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0);

ALTER TABLE land_use_emissions_service.lu_calculation_events
    ADD CONSTRAINT chk_cae_removals_non_negative CHECK (removals_tco2e IS NULL OR removals_tco2e >= 0);

ALTER TABLE land_use_emissions_service.lu_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE land_use_emissions_service.lu_calculation_events
    ADD CONSTRAINT chk_cae_method CHECK (
        method IS NULL OR method IN (
            'stock_difference', 'gain_loss', 'default_factor',
            'biomass_expansion', 'allometric', 'direct_measurement'
        )
    );

ALTER TABLE land_use_emissions_service.lu_calculation_events
    ADD CONSTRAINT chk_cae_land_category CHECK (
        land_category IS NULL OR land_category IN (
            'forest_land', 'cropland', 'grassland', 'wetlands',
            'settlements', 'other_land'
        )
    );

-- =============================================================================
-- Table 12: land_use_emissions_service.lu_transition_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_transition_events (
    time                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               UUID            NOT NULL,
    event_type              VARCHAR(50),
    transition_id           UUID,
    from_category           VARCHAR(50),
    to_category             VARCHAR(50),
    area_ha                 NUMERIC(15,4),
    is_deforestation        BOOLEAN,
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'land_use_emissions_service.lu_transition_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE land_use_emissions_service.lu_transition_events
    ADD CONSTRAINT chk_tre_from_category CHECK (
        from_category IS NULL OR from_category IN (
            'forest_land', 'cropland', 'grassland', 'wetlands',
            'settlements', 'other_land'
        )
    );

ALTER TABLE land_use_emissions_service.lu_transition_events
    ADD CONSTRAINT chk_tre_to_category CHECK (
        to_category IS NULL OR to_category IN (
            'forest_land', 'cropland', 'grassland', 'wetlands',
            'settlements', 'other_land'
        )
    );

ALTER TABLE land_use_emissions_service.lu_transition_events
    ADD CONSTRAINT chk_tre_area_ha_non_negative CHECK (area_ha IS NULL OR area_ha >= 0);

-- =============================================================================
-- Table 13: land_use_emissions_service.lu_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS land_use_emissions_service.lu_compliance_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    compliance_id       UUID,
    framework           VARCHAR(100),
    status              VARCHAR(20),
    check_count         INTEGER,
    pass_count          INTEGER,
    fail_count          INTEGER,
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'land_use_emissions_service.lu_compliance_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE land_use_emissions_service.lu_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('GHG_PROTOCOL', 'IPCC', 'UNFCCC', 'EU_LULUCF', 'ISO_14064', 'CUSTOM')
    );

ALTER TABLE land_use_emissions_service.lu_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE land_use_emissions_service.lu_compliance_events
    ADD CONSTRAINT chk_coe_check_count_non_negative CHECK (check_count IS NULL OR check_count >= 0);

ALTER TABLE land_use_emissions_service.lu_compliance_events
    ADD CONSTRAINT chk_coe_pass_count_non_negative CHECK (pass_count IS NULL OR pass_count >= 0);

ALTER TABLE land_use_emissions_service.lu_compliance_events
    ADD CONSTRAINT chk_coe_fail_count_non_negative CHECK (fail_count IS NULL OR fail_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- lu_hourly_calculation_stats: hourly count/sum(emissions)/sum(removals) by land_category and method
CREATE MATERIALIZED VIEW land_use_emissions_service.lu_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    land_category,
    method,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_tco2e)            AS sum_emissions_tco2e,
    SUM(removals_tco2e)             AS sum_removals_tco2e,
    AVG(duration_ms)                AS avg_duration_ms,
    MAX(duration_ms)                AS max_duration_ms
FROM land_use_emissions_service.lu_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, land_category, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'land_use_emissions_service.lu_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- lu_daily_emission_totals: daily count/sum(emissions) by land_category and carbon_pool
CREATE MATERIALIZED VIEW land_use_emissions_service.lu_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    land_category,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_tco2e)            AS sum_emissions_tco2e,
    SUM(removals_tco2e)             AS sum_removals_tco2e
FROM land_use_emissions_service.lu_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, land_category
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'land_use_emissions_service.lu_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- lu_land_parcels indexes
CREATE INDEX IF NOT EXISTS idx_lu_lp_tenant_id              ON land_use_emissions_service.lu_land_parcels(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_lp_name                   ON land_use_emissions_service.lu_land_parcels(name);
CREATE INDEX IF NOT EXISTS idx_lu_lp_land_category          ON land_use_emissions_service.lu_land_parcels(land_category);
CREATE INDEX IF NOT EXISTS idx_lu_lp_climate_zone           ON land_use_emissions_service.lu_land_parcels(climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_lp_soil_type              ON land_use_emissions_service.lu_land_parcels(soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_lp_country_code           ON land_use_emissions_service.lu_land_parcels(country_code);
CREATE INDEX IF NOT EXISTS idx_lu_lp_management_practice    ON land_use_emissions_service.lu_land_parcels(management_practice);
CREATE INDEX IF NOT EXISTS idx_lu_lp_input_level            ON land_use_emissions_service.lu_land_parcels(input_level);
CREATE INDEX IF NOT EXISTS idx_lu_lp_peatland_status        ON land_use_emissions_service.lu_land_parcels(peatland_status);
CREATE INDEX IF NOT EXISTS idx_lu_lp_is_active              ON land_use_emissions_service.lu_land_parcels(is_active);
CREATE INDEX IF NOT EXISTS idx_lu_lp_tenant_category        ON land_use_emissions_service.lu_land_parcels(tenant_id, land_category);
CREATE INDEX IF NOT EXISTS idx_lu_lp_tenant_climate         ON land_use_emissions_service.lu_land_parcels(tenant_id, climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_lp_tenant_active          ON land_use_emissions_service.lu_land_parcels(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_lu_lp_category_climate       ON land_use_emissions_service.lu_land_parcels(land_category, climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_lp_category_climate_soil  ON land_use_emissions_service.lu_land_parcels(land_category, climate_zone, soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_lp_tenant_country         ON land_use_emissions_service.lu_land_parcels(tenant_id, country_code);
CREATE INDEX IF NOT EXISTS idx_lu_lp_created_at             ON land_use_emissions_service.lu_land_parcels(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_lp_updated_at             ON land_use_emissions_service.lu_land_parcels(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_lp_metadata               ON land_use_emissions_service.lu_land_parcels USING GIN (metadata);

-- Partial index: active parcels only
CREATE INDEX IF NOT EXISTS idx_lu_lp_active_parcels         ON land_use_emissions_service.lu_land_parcels(tenant_id, land_category)
    WHERE is_active = TRUE;

-- lu_carbon_stock_factors indexes
CREATE INDEX IF NOT EXISTS idx_lu_csf_land_category         ON land_use_emissions_service.lu_carbon_stock_factors(land_category);
CREATE INDEX IF NOT EXISTS idx_lu_csf_climate_zone          ON land_use_emissions_service.lu_carbon_stock_factors(climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_csf_soil_type             ON land_use_emissions_service.lu_carbon_stock_factors(soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_csf_carbon_pool           ON land_use_emissions_service.lu_carbon_stock_factors(carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_csf_source                ON land_use_emissions_service.lu_carbon_stock_factors(source);
CREATE INDEX IF NOT EXISTS idx_lu_csf_tier                  ON land_use_emissions_service.lu_carbon_stock_factors(tier);
CREATE INDEX IF NOT EXISTS idx_lu_csf_is_active             ON land_use_emissions_service.lu_carbon_stock_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_lu_csf_category_zone         ON land_use_emissions_service.lu_carbon_stock_factors(land_category, climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_csf_category_zone_soil    ON land_use_emissions_service.lu_carbon_stock_factors(land_category, climate_zone, soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_csf_category_pool         ON land_use_emissions_service.lu_carbon_stock_factors(land_category, carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_csf_category_zone_pool    ON land_use_emissions_service.lu_carbon_stock_factors(land_category, climate_zone, carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_csf_valid_from            ON land_use_emissions_service.lu_carbon_stock_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_lu_csf_valid_to              ON land_use_emissions_service.lu_carbon_stock_factors(valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_lu_csf_created_at            ON land_use_emissions_service.lu_carbon_stock_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_csf_updated_at            ON land_use_emissions_service.lu_carbon_stock_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_csf_metadata              ON land_use_emissions_service.lu_carbon_stock_factors USING GIN (metadata);

-- Partial index: active factors only
CREATE INDEX IF NOT EXISTS idx_lu_csf_active_factors        ON land_use_emissions_service.lu_carbon_stock_factors(land_category, climate_zone, soil_type, carbon_pool)
    WHERE is_active = TRUE;

-- lu_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_lu_ef_from_category          ON land_use_emissions_service.lu_emission_factors(from_category);
CREATE INDEX IF NOT EXISTS idx_lu_ef_to_category            ON land_use_emissions_service.lu_emission_factors(to_category);
CREATE INDEX IF NOT EXISTS idx_lu_ef_climate_zone           ON land_use_emissions_service.lu_emission_factors(climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_ef_gas                    ON land_use_emissions_service.lu_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_lu_ef_source                 ON land_use_emissions_service.lu_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_lu_ef_is_active              ON land_use_emissions_service.lu_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_lu_ef_from_to                ON land_use_emissions_service.lu_emission_factors(from_category, to_category);
CREATE INDEX IF NOT EXISTS idx_lu_ef_from_to_zone           ON land_use_emissions_service.lu_emission_factors(from_category, to_category, climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_ef_from_to_zone_gas       ON land_use_emissions_service.lu_emission_factors(from_category, to_category, climate_zone, gas);
CREATE INDEX IF NOT EXISTS idx_lu_ef_created_at             ON land_use_emissions_service.lu_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_ef_updated_at             ON land_use_emissions_service.lu_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_ef_metadata               ON land_use_emissions_service.lu_emission_factors USING GIN (metadata);

-- Partial index: active factors for conversion lookups
CREATE INDEX IF NOT EXISTS idx_lu_ef_active_conversions     ON land_use_emissions_service.lu_emission_factors(from_category, to_category, climate_zone, gas)
    WHERE is_active = TRUE;

-- lu_land_use_transitions indexes
CREATE INDEX IF NOT EXISTS idx_lu_lut_tenant_id             ON land_use_emissions_service.lu_land_use_transitions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_lut_parcel_id             ON land_use_emissions_service.lu_land_use_transitions(parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_lut_from_category         ON land_use_emissions_service.lu_land_use_transitions(from_category);
CREATE INDEX IF NOT EXISTS idx_lu_lut_to_category           ON land_use_emissions_service.lu_land_use_transitions(to_category);
CREATE INDEX IF NOT EXISTS idx_lu_lut_transition_date       ON land_use_emissions_service.lu_land_use_transitions(transition_date DESC);
CREATE INDEX IF NOT EXISTS idx_lu_lut_transition_type       ON land_use_emissions_service.lu_land_use_transitions(transition_type);
CREATE INDEX IF NOT EXISTS idx_lu_lut_is_deforestation      ON land_use_emissions_service.lu_land_use_transitions(is_deforestation);
CREATE INDEX IF NOT EXISTS idx_lu_lut_is_peatland_change    ON land_use_emissions_service.lu_land_use_transitions(is_peatland_change);
CREATE INDEX IF NOT EXISTS idx_lu_lut_tenant_parcel         ON land_use_emissions_service.lu_land_use_transitions(tenant_id, parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_lut_tenant_from_to        ON land_use_emissions_service.lu_land_use_transitions(tenant_id, from_category, to_category);
CREATE INDEX IF NOT EXISTS idx_lu_lut_tenant_type           ON land_use_emissions_service.lu_land_use_transitions(tenant_id, transition_type);
CREATE INDEX IF NOT EXISTS idx_lu_lut_parcel_date           ON land_use_emissions_service.lu_land_use_transitions(parcel_id, transition_date DESC);
CREATE INDEX IF NOT EXISTS idx_lu_lut_created_at            ON land_use_emissions_service.lu_land_use_transitions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_lut_metadata              ON land_use_emissions_service.lu_land_use_transitions USING GIN (metadata);

-- Partial index: deforestation transitions
CREATE INDEX IF NOT EXISTS idx_lu_lut_deforestation         ON land_use_emissions_service.lu_land_use_transitions(tenant_id, transition_date DESC)
    WHERE is_deforestation = TRUE;

-- Partial index: conversion transitions
CREATE INDEX IF NOT EXISTS idx_lu_lut_conversions           ON land_use_emissions_service.lu_land_use_transitions(tenant_id, from_category, to_category)
    WHERE transition_type = 'CONVERSION';

-- lu_carbon_stock_snapshots indexes
CREATE INDEX IF NOT EXISTS idx_lu_css_tenant_id             ON land_use_emissions_service.lu_carbon_stock_snapshots(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_css_parcel_id             ON land_use_emissions_service.lu_carbon_stock_snapshots(parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_css_carbon_pool           ON land_use_emissions_service.lu_carbon_stock_snapshots(carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_css_measurement_date      ON land_use_emissions_service.lu_carbon_stock_snapshots(measurement_date DESC);
CREATE INDEX IF NOT EXISTS idx_lu_css_tier                  ON land_use_emissions_service.lu_carbon_stock_snapshots(tier);
CREATE INDEX IF NOT EXISTS idx_lu_css_source                ON land_use_emissions_service.lu_carbon_stock_snapshots(source);
CREATE INDEX IF NOT EXISTS idx_lu_css_tenant_parcel         ON land_use_emissions_service.lu_carbon_stock_snapshots(tenant_id, parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_css_parcel_pool           ON land_use_emissions_service.lu_carbon_stock_snapshots(parcel_id, carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_css_parcel_date           ON land_use_emissions_service.lu_carbon_stock_snapshots(parcel_id, measurement_date DESC);
CREATE INDEX IF NOT EXISTS idx_lu_css_tenant_pool           ON land_use_emissions_service.lu_carbon_stock_snapshots(tenant_id, carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_css_created_at            ON land_use_emissions_service.lu_carbon_stock_snapshots(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_css_metadata              ON land_use_emissions_service.lu_carbon_stock_snapshots USING GIN (metadata);

-- lu_calculations indexes
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_id            ON land_use_emissions_service.lu_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_calc_parcel_id            ON land_use_emissions_service.lu_calculations(parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_calc_from_category        ON land_use_emissions_service.lu_calculations(from_category);
CREATE INDEX IF NOT EXISTS idx_lu_calc_to_category          ON land_use_emissions_service.lu_calculations(to_category);
CREATE INDEX IF NOT EXISTS idx_lu_calc_climate_zone         ON land_use_emissions_service.lu_calculations(climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_calc_soil_type            ON land_use_emissions_service.lu_calculations(soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tier                 ON land_use_emissions_service.lu_calculations(tier);
CREATE INDEX IF NOT EXISTS idx_lu_calc_method               ON land_use_emissions_service.lu_calculations(method);
CREATE INDEX IF NOT EXISTS idx_lu_calc_gwp_source           ON land_use_emissions_service.lu_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_lu_calc_net_emissions        ON land_use_emissions_service.lu_calculations(net_emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_lu_calc_calculation_hash     ON land_use_emissions_service.lu_calculations(calculation_hash);
CREATE INDEX IF NOT EXISTS idx_lu_calc_reporting_period     ON land_use_emissions_service.lu_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_parcel        ON land_use_emissions_service.lu_calculations(tenant_id, parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_from_to       ON land_use_emissions_service.lu_calculations(tenant_id, from_category, to_category);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_method        ON land_use_emissions_service.lu_calculations(tenant_id, method);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_period        ON land_use_emissions_service.lu_calculations(tenant_id, reporting_period);
CREATE INDEX IF NOT EXISTS idx_lu_calc_tenant_climate_soil  ON land_use_emissions_service.lu_calculations(tenant_id, climate_zone, soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_calc_calculated_at        ON land_use_emissions_service.lu_calculations(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_calc_created_at           ON land_use_emissions_service.lu_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_calc_updated_at           ON land_use_emissions_service.lu_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_calc_trace_json           ON land_use_emissions_service.lu_calculations USING GIN (trace_json);
CREATE INDEX IF NOT EXISTS idx_lu_calc_metadata             ON land_use_emissions_service.lu_calculations USING GIN (metadata);

-- lu_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_lu_cd_calculation_id         ON land_use_emissions_service.lu_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_lu_cd_carbon_pool            ON land_use_emissions_service.lu_calculation_details(carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_cd_gas                    ON land_use_emissions_service.lu_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_lu_cd_is_removal             ON land_use_emissions_service.lu_calculation_details(is_removal);
CREATE INDEX IF NOT EXISTS idx_lu_cd_calc_pool              ON land_use_emissions_service.lu_calculation_details(calculation_id, carbon_pool);
CREATE INDEX IF NOT EXISTS idx_lu_cd_calc_gas               ON land_use_emissions_service.lu_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_lu_cd_calc_pool_gas          ON land_use_emissions_service.lu_calculation_details(calculation_id, carbon_pool, gas);
CREATE INDEX IF NOT EXISTS idx_lu_cd_created_at             ON land_use_emissions_service.lu_calculation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cd_metadata               ON land_use_emissions_service.lu_calculation_details USING GIN (metadata);

-- Partial index: removals only
CREATE INDEX IF NOT EXISTS idx_lu_cd_removals               ON land_use_emissions_service.lu_calculation_details(calculation_id, carbon_pool)
    WHERE is_removal = TRUE;

-- lu_soc_assessments indexes
CREATE INDEX IF NOT EXISTS idx_lu_soc_tenant_id             ON land_use_emissions_service.lu_soc_assessments(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_soc_parcel_id             ON land_use_emissions_service.lu_soc_assessments(parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_soc_climate_zone          ON land_use_emissions_service.lu_soc_assessments(climate_zone);
CREATE INDEX IF NOT EXISTS idx_lu_soc_soil_type             ON land_use_emissions_service.lu_soc_assessments(soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_soc_land_category         ON land_use_emissions_service.lu_soc_assessments(land_category);
CREATE INDEX IF NOT EXISTS idx_lu_soc_management_practice   ON land_use_emissions_service.lu_soc_assessments(management_practice);
CREATE INDEX IF NOT EXISTS idx_lu_soc_input_level           ON land_use_emissions_service.lu_soc_assessments(input_level);
CREATE INDEX IF NOT EXISTS idx_lu_soc_source                ON land_use_emissions_service.lu_soc_assessments(source);
CREATE INDEX IF NOT EXISTS idx_lu_soc_tenant_parcel         ON land_use_emissions_service.lu_soc_assessments(tenant_id, parcel_id);
CREATE INDEX IF NOT EXISTS idx_lu_soc_tenant_category       ON land_use_emissions_service.lu_soc_assessments(tenant_id, land_category);
CREATE INDEX IF NOT EXISTS idx_lu_soc_parcel_date           ON land_use_emissions_service.lu_soc_assessments(parcel_id, assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_soc_climate_soil          ON land_use_emissions_service.lu_soc_assessments(climate_zone, soil_type);
CREATE INDEX IF NOT EXISTS idx_lu_soc_assessed_at           ON land_use_emissions_service.lu_soc_assessments(assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_soc_created_at            ON land_use_emissions_service.lu_soc_assessments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_soc_updated_at            ON land_use_emissions_service.lu_soc_assessments(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_soc_metadata              ON land_use_emissions_service.lu_soc_assessments USING GIN (metadata);

-- lu_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_lu_cr_tenant_id              ON land_use_emissions_service.lu_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_cr_calculation_id         ON land_use_emissions_service.lu_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_lu_cr_framework              ON land_use_emissions_service.lu_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_lu_cr_status                 ON land_use_emissions_service.lu_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_lu_cr_checked_at             ON land_use_emissions_service.lu_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cr_tenant_framework       ON land_use_emissions_service.lu_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_lu_cr_tenant_status          ON land_use_emissions_service.lu_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_lu_cr_framework_status       ON land_use_emissions_service.lu_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_lu_cr_tenant_calculation     ON land_use_emissions_service.lu_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_lu_cr_created_at             ON land_use_emissions_service.lu_compliance_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cr_findings               ON land_use_emissions_service.lu_compliance_records USING GIN (findings);
CREATE INDEX IF NOT EXISTS idx_lu_cr_metadata               ON land_use_emissions_service.lu_compliance_records USING GIN (metadata);

-- lu_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_lu_ae_tenant_id              ON land_use_emissions_service.lu_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lu_ae_entity_type            ON land_use_emissions_service.lu_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_lu_ae_entity_id              ON land_use_emissions_service.lu_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_lu_ae_action                 ON land_use_emissions_service.lu_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_lu_ae_actor                  ON land_use_emissions_service.lu_audit_entries(actor);
CREATE INDEX IF NOT EXISTS idx_lu_ae_prev_hash              ON land_use_emissions_service.lu_audit_entries(prev_hash);
CREATE INDEX IF NOT EXISTS idx_lu_ae_entry_hash             ON land_use_emissions_service.lu_audit_entries(entry_hash);
CREATE INDEX IF NOT EXISTS idx_lu_ae_tenant_entity          ON land_use_emissions_service.lu_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_lu_ae_tenant_action          ON land_use_emissions_service.lu_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_lu_ae_created_at             ON land_use_emissions_service.lu_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lu_ae_details                ON land_use_emissions_service.lu_audit_entries USING GIN (details);

-- lu_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_lu_cae_tenant_id             ON land_use_emissions_service.lu_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_event_type            ON land_use_emissions_service.lu_calculation_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_land_category         ON land_use_emissions_service.lu_calculation_events(land_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_method                ON land_use_emissions_service.lu_calculation_events(method, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_tenant_category       ON land_use_emissions_service.lu_calculation_events(tenant_id, land_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_tenant_method         ON land_use_emissions_service.lu_calculation_events(tenant_id, method, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_cae_metadata              ON land_use_emissions_service.lu_calculation_events USING GIN (metadata);

-- lu_transition_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_lu_tre_tenant_id             ON land_use_emissions_service.lu_transition_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_tre_event_type            ON land_use_emissions_service.lu_transition_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_tre_from_category         ON land_use_emissions_service.lu_transition_events(from_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_tre_to_category           ON land_use_emissions_service.lu_transition_events(to_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_tre_tenant_from_to        ON land_use_emissions_service.lu_transition_events(tenant_id, from_category, to_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_tre_deforestation         ON land_use_emissions_service.lu_transition_events(tenant_id, time DESC)
    WHERE is_deforestation = TRUE;
CREATE INDEX IF NOT EXISTS idx_lu_tre_metadata              ON land_use_emissions_service.lu_transition_events USING GIN (metadata);

-- lu_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_lu_coe_tenant_id             ON land_use_emissions_service.lu_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_coe_framework             ON land_use_emissions_service.lu_compliance_events(framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_coe_status                ON land_use_emissions_service.lu_compliance_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_coe_tenant_framework      ON land_use_emissions_service.lu_compliance_events(tenant_id, framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_coe_tenant_status         ON land_use_emissions_service.lu_compliance_events(tenant_id, status, time DESC);
CREATE INDEX IF NOT EXISTS idx_lu_coe_metadata              ON land_use_emissions_service.lu_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- lu_land_parcels: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_land_parcels ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_lp_read  ON land_use_emissions_service.lu_land_parcels FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_lp_write ON land_use_emissions_service.lu_land_parcels FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_carbon_stock_factors: shared reference data (open read, admin write)
ALTER TABLE land_use_emissions_service.lu_carbon_stock_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_csf_read  ON land_use_emissions_service.lu_carbon_stock_factors FOR SELECT USING (TRUE);
CREATE POLICY lu_csf_write ON land_use_emissions_service.lu_carbon_stock_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- lu_emission_factors: shared reference data (open read, admin write)
ALTER TABLE land_use_emissions_service.lu_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_ef_read  ON land_use_emissions_service.lu_emission_factors FOR SELECT USING (TRUE);
CREATE POLICY lu_ef_write ON land_use_emissions_service.lu_emission_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- lu_land_use_transitions: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_land_use_transitions ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_lut_read  ON land_use_emissions_service.lu_land_use_transitions FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_lut_write ON land_use_emissions_service.lu_land_use_transitions FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_carbon_stock_snapshots: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_carbon_stock_snapshots ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_css_read  ON land_use_emissions_service.lu_carbon_stock_snapshots FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_css_write ON land_use_emissions_service.lu_carbon_stock_snapshots FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_calculations: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_calc_read  ON land_use_emissions_service.lu_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_calc_write ON land_use_emissions_service.lu_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_calculation_details: open read/write (linked via FK to tenant-isolated lu_calculations)
ALTER TABLE land_use_emissions_service.lu_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_cd_read  ON land_use_emissions_service.lu_calculation_details FOR SELECT USING (TRUE);
CREATE POLICY lu_cd_write ON land_use_emissions_service.lu_calculation_details FOR ALL   USING (TRUE);

-- lu_soc_assessments: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_soc_assessments ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_soc_read  ON land_use_emissions_service.lu_soc_assessments FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_soc_write ON land_use_emissions_service.lu_soc_assessments FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_compliance_records: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_cr_read  ON land_use_emissions_service.lu_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_cr_write ON land_use_emissions_service.lu_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_audit_entries: tenant-isolated
ALTER TABLE land_use_emissions_service.lu_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_ae_read  ON land_use_emissions_service.lu_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY lu_ae_write ON land_use_emissions_service.lu_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- lu_calculation_events: open read/write (time-series telemetry)
ALTER TABLE land_use_emissions_service.lu_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_cae_read  ON land_use_emissions_service.lu_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY lu_cae_write ON land_use_emissions_service.lu_calculation_events FOR ALL   USING (TRUE);

-- lu_transition_events: open read/write (time-series telemetry)
ALTER TABLE land_use_emissions_service.lu_transition_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_tre_read  ON land_use_emissions_service.lu_transition_events FOR SELECT USING (TRUE);
CREATE POLICY lu_tre_write ON land_use_emissions_service.lu_transition_events FOR ALL   USING (TRUE);

-- lu_compliance_events: open read/write (time-series telemetry)
ALTER TABLE land_use_emissions_service.lu_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY lu_coe_read  ON land_use_emissions_service.lu_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY lu_coe_write ON land_use_emissions_service.lu_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA land_use_emissions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA land_use_emissions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA land_use_emissions_service TO greenlang_app;
GRANT SELECT ON land_use_emissions_service.lu_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON land_use_emissions_service.lu_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA land_use_emissions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA land_use_emissions_service TO greenlang_readonly;
GRANT SELECT ON land_use_emissions_service.lu_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON land_use_emissions_service.lu_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA land_use_emissions_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA land_use_emissions_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA land_use_emissions_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'land-use-emissions:read',                    'land-use-emissions', 'read',                    'View all land use emissions service data including parcels, carbon stocks, transitions, calculations, SOC assessments, and compliance records'),
    (gen_random_uuid(), 'land-use-emissions:write',                   'land-use-emissions', 'write',                   'Create, update, and manage all land use emissions service data'),
    (gen_random_uuid(), 'land-use-emissions:execute',                 'land-use-emissions', 'execute',                 'Execute land use emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'land-use-emissions:parcels:read',            'land-use-emissions', 'parcels_read',            'View land parcel registry with area, climate zone, soil type, land category (forest_land/cropland/grassland/wetlands/settlements/other_land), coordinates, management practices, input levels, and peatland status'),
    (gen_random_uuid(), 'land-use-emissions:parcels:write',           'land-use-emissions', 'parcels_write',           'Create, update, and manage land parcel registry entries'),
    (gen_random_uuid(), 'land-use-emissions:carbon-stocks:read',      'land-use-emissions', 'carbon_stocks_read',      'View IPCC default carbon stock factors by land category, climate zone, soil type, and carbon pool (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon) with stock values, growth rates, root-shoot ratios, and tiered sourcing from IPCC/GPG/UNFCCC/NATIONAL/CUSTOM'),
    (gen_random_uuid(), 'land-use-emissions:carbon-stocks:write',     'land-use-emissions', 'carbon_stocks_write',     'Create, update, and manage carbon stock factor entries'),
    (gen_random_uuid(), 'land-use-emissions:factors:read',            'land-use-emissions', 'factors_read',            'View emission factors by land-use conversion type (from/to category pairs), climate zone, gas (CO2/CH4/N2O), and source (IPCC/GPG/UNFCCC/NATIONAL/CUSTOM) with methodology references'),
    (gen_random_uuid(), 'land-use-emissions:factors:write',           'land-use-emissions', 'factors_write',           'Create, update, and manage emission factor entries for land-use conversions'),
    (gen_random_uuid(), 'land-use-emissions:transitions:read',        'land-use-emissions', 'transitions_read',        'View land-use transition records with parcel-level from/to category changes, transition dates, areas, REMAINING/CONVERSION types, disturbance types, deforestation flags, and peatland change flags'),
    (gen_random_uuid(), 'land-use-emissions:transitions:write',       'land-use-emissions', 'transitions_write',       'Create, update, and manage land-use transition records'),
    (gen_random_uuid(), 'land-use-emissions:snapshots:read',          'land-use-emissions', 'snapshots_read',          'View carbon stock snapshot measurements with per-pool stock values (tC/ha and total tC), measurement dates and methods, tier levels, sources, and uncertainty percentages'),
    (gen_random_uuid(), 'land-use-emissions:snapshots:write',         'land-use-emissions', 'snapshots_write',         'Create and manage carbon stock snapshot measurement records'),
    (gen_random_uuid(), 'land-use-emissions:calculations:read',       'land-use-emissions', 'calculations_read',       'View land use emission calculation results with from/to categories, emissions/removals/net tCO2e, per-gas per-pool breakdowns, calculation hashes, and step-by-step trace data'),
    (gen_random_uuid(), 'land-use-emissions:calculations:write',      'land-use-emissions', 'calculations_write',      'Create and manage land use emission calculation records'),
    (gen_random_uuid(), 'land-use-emissions:soc:read',                'land-use-emissions', 'soc_read',                'View soil organic carbon assessments with IPCC Tier 1 SOC = SOCref x FLU x FMG x FI, management practice and input level factors, current/previous SOC stocks, and annual delta SOC values'),
    (gen_random_uuid(), 'land-use-emissions:soc:write',               'land-use-emissions', 'soc_write',               'Create, update, and manage soil organic carbon assessment records'),
    (gen_random_uuid(), 'land-use-emissions:compliance:read',         'land-use-emissions', 'compliance_read',         'View regulatory compliance records for GHG Protocol, IPCC, UNFCCC, EU LULUCF, and ISO 14064 with total requirements, passed/failed counts, and findings'),
    (gen_random_uuid(), 'land-use-emissions:compliance:execute',      'land-use-emissions', 'compliance_execute',      'Execute regulatory compliance checks against GHG Protocol, IPCC, UNFCCC, EU LULUCF, and ISO 14064 frameworks'),
    (gen_random_uuid(), 'land-use-emissions:admin',                   'land-use-emissions', 'admin',                   'Full administrative access to land use emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('land_use_emissions_service.lu_calculation_events', INTERVAL '365 days');
SELECT add_retention_policy('land_use_emissions_service.lu_transition_events',  INTERVAL '365 days');
SELECT add_retention_policy('land_use_emissions_service.lu_compliance_events',  INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE land_use_emissions_service.lu_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'land_category',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('land_use_emissions_service.lu_calculation_events', INTERVAL '30 days');

ALTER TABLE land_use_emissions_service.lu_transition_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'from_category',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('land_use_emissions_service.lu_transition_events', INTERVAL '30 days');

ALTER TABLE land_use_emissions_service.lu_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('land_use_emissions_service.lu_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Land Use Emissions Agent (GL-MRV-SCOPE1-006)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-006',
    'Land Use Emissions Agent',
    'Land use emission and removal calculator for GreenLang Climate OS. Manages land parcel registry with area in hectares, IPCC climate zones (tropical_wet/tropical_moist/tropical_dry/tropical_montane/warm_temperate_moist/warm_temperate_dry/cool_temperate_moist/cool_temperate_dry/boreal_moist/boreal_dry/polar_moist/polar_dry), soil types (high_activity_clay/low_activity_clay/sandy/spodic/volcanic/wetland/organic/other), land categories (forest_land/cropland/grassland/wetlands/settlements/other_land), geographic coordinates (latitude/longitude/elevation), country codes, management practices (full_tillage/reduced_tillage/no_tillage/improved/unimproved/degraded/native/managed/plantation/other), input levels (low/medium/high/high_without_manure/high_with_manure), and peatland status (not_peatland/intact/drained/rewetted/extracted). Maintains IPCC default carbon stock factor registry per land category, climate zone, soil type, and carbon pool (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon) with stock values in tC/ha, growth rates, root-shoot ratios, dead wood fractions, litter stocks, tiered sourcing from IPCC/GPG/UNFCCC/NATIONAL/CUSTOM, confidence percentages, and validity date ranges. Stores emission factor database for land-use conversion from/to category pairs by climate zone and gas (CO2/CH4/N2O) with methodology references. Records land-use transitions at parcel level with from/to categories, transition dates, areas, REMAINING/CONVERSION types, disturbance types (fire/storm/insect/drought/flood/logging/clearing/cultivation/urbanization/mining/drainage/rewetting/other), deforestation flags, and peatland change flags. Captures point-in-time carbon stock snapshots per pool with stock values in tC/ha and total tC, measurement methods, tiers 1/2/3, sources (IPCC/GPG/UNFCCC/NATIONAL/FIELD_MEASUREMENT/REMOTE_SENSING/CUSTOM), and uncertainty percentages. Executes deterministic land use emission calculations using stock_difference, gain_loss, default_factor, biomass_expansion, allometric, and direct_measurement methods with GWP sources AR4/AR5/AR6, producing total emissions, total removals, and net emissions in tCO2e with SHA-256 calculation hashes and step-by-step trace JSONB. Produces per-gas per-pool calculation detail breakdowns with emission values in tC and tCO2e, removal flags, factor values, and formulas. Performs IPCC Tier 1 soil organic carbon (SOC) assessments using SOC = SOCref x FLU x FMG x FI with management practice and input level stock change factors, reference SOC values, current/previous SOC stocks, annual delta SOC, configurable depth (default 30 cm) and transition period (default 20 years). Checks regulatory compliance against GHG Protocol, IPCC, UNFCCC, EU LULUCF, and ISO 14064 frameworks with total requirements, passed/failed counts, and findings. Generates entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/TRANSITION/SNAPSHOT/SOC_ASSESS), prev_hash/entry_hash chaining for tamper-evident provenance, and actor attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/land-use-emissions',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-006', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/land-use-emissions-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"land-use", "lulucf", "carbon-stock", "scope-1", "deforestation", "soc", "ghg-protocol", "ipcc", "unfccc", "eu-lulucf", "mrv"}',
    '{"forestry", "agriculture", "land-management", "conservation", "peatland", "wetland", "cross-sector"}',
    'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'land_parcel_registry',
    'configuration',
    'Register and manage land parcels with area (hectares), IPCC climate zone classification, soil type, land category (forest_land/cropland/grassland/wetlands/settlements/other_land), geographic coordinates, country codes, management practices, input levels, and peatland status tracking.',
    '{"name", "area_ha", "climate_zone", "soil_type", "land_category", "latitude", "longitude", "country_code", "management_practice", "input_level", "peatland_status"}',
    '{"parcel_id", "registration_result"}',
    '{"land_categories": ["forest_land", "cropland", "grassland", "wetlands", "settlements", "other_land"], "climate_zones": ["tropical_wet", "tropical_moist", "tropical_dry", "tropical_montane", "warm_temperate_moist", "warm_temperate_dry", "cool_temperate_moist", "cool_temperate_dry", "boreal_moist", "boreal_dry", "polar_moist", "polar_dry"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'carbon_stock_factor_management',
    'configuration',
    'Manage IPCC default carbon stock factor entries by land category, climate zone, soil type, and carbon pool (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon) with stock values in tC/ha, growth rates, root-shoot ratios, dead wood fractions, litter stocks, tiered sourcing, confidence percentages, and validity date ranges.',
    '{"land_category", "climate_zone", "soil_type", "carbon_pool", "stock_tc_ha", "source", "tier"}',
    '{"factor_id", "registration_result"}',
    '{"carbon_pools": ["above_ground_biomass", "below_ground_biomass", "dead_wood", "litter", "soil_organic_carbon"], "sources": ["IPCC", "GPG", "UNFCCC", "NATIONAL", "CUSTOM"], "tiers": [1, 2, 3]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'land_use_transition_tracking',
    'processing',
    'Record and manage land-use transition events at parcel level with from/to category classification, transition dates, areas, REMAINING/CONVERSION types, disturbance types (fire/storm/insect/drought/flood/logging/clearing/cultivation/urbanization/mining/drainage/rewetting/other), deforestation flags, and peatland change flags.',
    '{"parcel_id", "from_category", "to_category", "transition_date", "area_ha", "transition_type", "disturbance_type", "is_deforestation", "is_peatland_change"}',
    '{"transition_id", "tracking_result"}',
    '{"transition_types": ["REMAINING", "CONVERSION"], "tracks_deforestation": true, "tracks_peatland_change": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'carbon_stock_snapshot',
    'processing',
    'Capture point-in-time carbon stock measurements per parcel and carbon pool with stock values in tC/ha and total tC, measurement methods, tier levels 1/2/3, sources (IPCC/GPG/UNFCCC/NATIONAL/FIELD_MEASUREMENT/REMOTE_SENSING/CUSTOM), and uncertainty quantification.',
    '{"parcel_id", "carbon_pool", "stock_tc_ha", "measurement_date", "measurement_method", "tier", "source", "uncertainty_pct"}',
    '{"snapshot_id", "stock_total_tc"}',
    '{"supports_field_measurement": true, "supports_remote_sensing": true, "carbon_pools": ["above_ground_biomass", "below_ground_biomass", "dead_wood", "litter", "soil_organic_carbon"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic land use emission calculations using stock_difference (carbon stock change between two time points), gain_loss (annual gains minus annual losses), default_factor (IPCC default emission factors by conversion type), biomass_expansion (volume to biomass conversion with BEFs), allometric (species-specific allometric equations), and direct_measurement methods. Supports Tier 1/2/3, multi-gas CO2/CH4/N2O with GWP sources AR4/AR5/AR6. Produces total emissions, total removals, and net emissions in tCO2e with per-pool per-gas breakdowns.',
    '{"parcel_id", "from_category", "to_category", "area_ha", "climate_zone", "soil_type", "tier", "method", "gwp_source"}',
    '{"calculation_id", "total_emissions_tco2e", "total_removals_tco2e", "net_emissions_tco2e", "per_pool_breakdown", "calculation_hash"}',
    '{"methods": ["stock_difference", "gain_loss", "default_factor", "biomass_expansion", "allometric", "direct_measurement"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'soc_assessment',
    'processing',
    'Perform IPCC Tier 1 soil organic carbon assessments using SOC = SOCref x FLU x FMG x FI formula with management practice land-use factor (FLU), management factor (FMG), and input factor (FI). Computes current SOC stock, annual delta SOC over configurable transition period (default 20 years), and tracks previous SOC for change analysis. Supports configurable sampling depth (default 30 cm).',
    '{"parcel_id", "climate_zone", "soil_type", "land_category", "management_practice", "input_level", "soc_ref_tc_ha", "f_lu", "f_mg", "f_i", "depth_cm", "transition_years"}',
    '{"assessment_id", "soc_current_tc_ha", "delta_soc_annual_tc_ha"}',
    '{"default_depth_cm": 30, "default_transition_years": 20, "formula": "SOC = SOCref x FLU x FMG x FI", "supports_time_series": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of land use emission calculations against GHG Protocol (LULUCF guidance), IPCC (2006 Guidelines for National Greenhouse Gas Inventories), UNFCCC (National Communications reporting), EU LULUCF (Regulation 2018/841), and ISO 14064 frameworks. Produce check results with total requirements, passed/failed counts, and detailed findings.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "status", "total_requirements", "passed", "failed", "findings"}',
    '{"frameworks": ["GHG_PROTOCOL", "IPCC", "UNFCCC", "EU_LULUCF", "ISO_14064", "CUSTOM"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/TRANSITION/SNAPSHOT/SOC_ASSESS), prev_hash/entry_hash SHA-256 chaining for tamper-evident provenance, detail JSONB payloads, and actor attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT", "TRANSITION", "SNAPSHOT", "SOC_ASSESS"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage land use emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for area unit conversions (ha/acre/km2), carbon stock unit alignment (tC/ha to tCO2e), and GWP value lookups'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for IPCC carbon stock factor sources, emission factor methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection across stock-difference and gain-loss methods'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, transition events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-006', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of parcel areas, carbon stock measurements, and transition records'),
    ('GL-MRV-SCOPE1-006', 'GL-DATA-G-006',  '>=1.0.0', true,  'GIS/Mapping Connector for spatial analysis of land parcels, coordinate validation, and area verification from geospatial data'),
    ('GL-MRV-SCOPE1-006', 'GL-DATA-G-007',  '>=1.0.0', true,  'Deforestation Satellite Connector for cross-referencing satellite-detected deforestation events with land-use transition records')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-006',
    'Land Use Emissions Agent',
    'Land use emission and removal calculator. Land parcel registry (area ha, IPCC climate zones, soil types, land categories forest_land/cropland/grassland/wetlands/settlements/other_land, coordinates, management practices, input levels, peatland status). IPCC carbon stock factors (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon pools, tC/ha stocks, growth rates, root-shoot ratios, IPCC/GPG/UNFCCC/NATIONAL/CUSTOM sources, Tier 1/2/3). Emission factors (from/to category conversion pairs, CO2/CH4/N2O). Land-use transitions (REMAINING/CONVERSION, deforestation flags, peatland change). Carbon stock snapshots (per-pool tC/ha, field measurement, remote sensing). Emission calculations (stock_difference/gain_loss/default_factor/biomass_expansion/allometric/direct_measurement, emissions/removals/net tCO2e, AR4/AR5/AR6 GWP). Per-gas per-pool breakdowns. SOC assessments (SOCref x FLU x FMG x FI, 30cm depth, 20-year transition). Compliance checks (GHG Protocol/IPCC/UNFCCC/EU LULUCF/ISO 14064). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA land_use_emissions_service IS
    'Land Use Emissions Agent (AGENT-MRV-006) - land parcel registry, IPCC carbon stock factors, emission factors by conversion type, land-use transitions, carbon stock snapshots, emission calculations (emissions/removals/net), per-gas per-pool breakdowns, SOC assessments, compliance records, audit trail, provenance chains';

COMMENT ON TABLE land_use_emissions_service.lu_land_parcels IS
    'Land parcel registry: tenant_id, name, area_ha (>0), climate_zone (tropical_wet/tropical_moist/tropical_dry/tropical_montane/warm_temperate_moist/warm_temperate_dry/cool_temperate_moist/cool_temperate_dry/boreal_moist/boreal_dry/polar_moist/polar_dry), soil_type (high_activity_clay/low_activity_clay/sandy/spodic/volcanic/wetland/organic/other), land_category (forest_land/cropland/grassland/wetlands/settlements/other_land), latitude, longitude, elevation_m, country_code, management_practice, input_level, peatland_status, is_active, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_carbon_stock_factors IS
    'IPCC default carbon stock factors: land_category, climate_zone, soil_type, carbon_pool (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon), stock_tc_ha, growth_rate_tc_ha_yr, root_shoot_ratio, dead_wood_fraction, litter_stock_tc_ha, source (IPCC/GPG/UNFCCC/NATIONAL/CUSTOM), tier (1/2/3), confidence_pct, valid_from/to, UNIQUE(land_category, climate_zone, soil_type, carbon_pool, source)';

COMMENT ON TABLE land_use_emissions_service.lu_emission_factors IS
    'Emission factors by conversion type: from_category, to_category (forest_land/cropland/grassland/wetlands/settlements/other_land), climate_zone, gas (CO2/CH4/N2O), ef_value, ef_unit, source (IPCC/GPG/UNFCCC/NATIONAL/CUSTOM), methodology, UNIQUE(from_category, to_category, climate_zone, gas, source)';

COMMENT ON TABLE land_use_emissions_service.lu_land_use_transitions IS
    'Land-use change records: tenant_id, parcel_id (FK CASCADE), from_category, to_category, transition_date, area_ha, transition_type (REMAINING/CONVERSION), disturbance_type, is_deforestation (requires from_category=forest_land), is_peatland_change, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_carbon_stock_snapshots IS
    'Point-in-time carbon stock measurements: tenant_id, parcel_id (FK CASCADE), carbon_pool, stock_tc_ha, stock_total_tc (=stock_tc_ha x area), measurement_date, measurement_method, tier (1/2/3), source (IPCC/GPG/UNFCCC/NATIONAL/FIELD_MEASUREMENT/REMOTE_SENSING/CUSTOM), uncertainty_pct, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_calculations IS
    'Emission calculation results: tenant_id, parcel_id (FK), from_category, to_category, area_ha, climate_zone, soil_type, tier (1/2/3), method (stock_difference/gain_loss/default_factor/biomass_expansion/allometric/direct_measurement), gwp_source (AR4/AR5/AR6), total_emissions_tco2e, total_removals_tco2e, net_emissions_tco2e, calculation_hash (SHA-256), trace_json JSONB, reporting_period, uncertainty_pct, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_calculation_details IS
    'Per-gas per-pool calculation breakdown: calculation_id (FK CASCADE), carbon_pool (above_ground_biomass/below_ground_biomass/dead_wood/litter/soil_organic_carbon), gas (CO2/CH4/N2O), emission_tc, emission_tco2e, is_removal, factor_used, formula, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_soc_assessments IS
    'Soil organic carbon assessments: tenant_id, parcel_id (FK CASCADE), climate_zone, soil_type, land_category, management_practice, input_level, depth_cm (default 30), soc_ref_tc_ha, f_lu, f_mg, f_i (SOC = SOCref x FLU x FMG x FI), soc_current_tc_ha, soc_previous_tc_ha, delta_soc_annual_tc_ha, transition_years (default 20), source, uncertainty_pct, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_compliance_records IS
    'Regulatory compliance records: tenant_id, calculation_id (FK), framework (GHG_PROTOCOL/IPCC/UNFCCC/EU_LULUCF/ISO_14064/CUSTOM), status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), total_requirements, passed, failed, findings JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE land_use_emissions_service.lu_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/TRANSITION/SNAPSHOT/SOC_ASSESS), actor, details JSONB, prev_hash, entry_hash (SHA-256 chain)';

COMMENT ON TABLE land_use_emissions_service.lu_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, event_type, calculation_id, land_category, method, emissions_tco2e, removals_tco2e, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE land_use_emissions_service.lu_transition_events IS
    'TimescaleDB hypertable: transition events with tenant_id, event_type, transition_id, from_category, to_category, area_ha, is_deforestation, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE land_use_emissions_service.lu_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, event_type, compliance_id, framework, status, check_count, pass_count, fail_count, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW land_use_emissions_service.lu_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by land_category and method (total calculations, sum emissions tCO2e, sum removals tCO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW land_use_emissions_service.lu_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by land_category (total calculations, sum emissions tCO2e, sum removals tCO2e per day)';
