-- =============================================================================
-- V059: Agricultural Emissions Service Schema
-- =============================================================================
-- Component: AGENT-MRV-009 (GL-MRV-SCOPE1-009)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Agricultural Emissions Agent (GL-MRV-SCOPE1-009) with capabilities
-- for farm/facility registry management (dairy_farm, beef_ranch,
-- mixed_livestock, crop_farm, rice_farm, mixed_crop_livestock,
-- poultry_farm, other farm types with country, region, latitude,
-- longitude, 8 IPCC climate zones, total/arable/pasture area in
-- hectares), livestock population tracking (20 animal types with
-- head count, average body weight, milk yield, fat percentage,
-- feed digestibility, methane conversion factor Ym%, activity
-- coefficient, and reporting year), animal waste management system
-- (AWMS) allocation per animal type (15 AWMS types with allocation
-- fraction 0-1 and mean temperature), Tier 2 feed characteristic
-- data (gross energy MJ/day, digestible energy %, crude protein %,
-- Ym%, feed type), IPCC/EPA/DEFRA/UNFCCC/GHG_PROTOCOL/NATIONAL/
-- CUSTOM emission factor database per emission source (enteric
-- fermentation, manure management, rice cultivation, cropland
-- direct N2O, cropland indirect N2O, field burning) with animal
-- type, crop type, gas CO2/CH4/N2O, factor values, units, source
-- versioning, geographic scoping, climate zone, and uncertainty
-- percentages with validity date ranges), cropland input records
-- (synthetic N, organic N, crop residue N, SOM N, limestone,
-- dolomite, urea, organic soil area, PRP N, PRP animal type),
-- rice paddy field definitions (7 water regimes, 3 pre-season
-- flooding types, cultivation days, organic amendments JSONB with
-- type/rate, soil type), emission calculation results (6 methods:
-- ipcc_tier1, ipcc_tier2, ipcc_tier3, emission_factor, mass_balance,
-- direct_measurement with GWP sources AR4/AR5/AR6/AR6_20YR, total
-- tCO2e, per-gas CO2/CH4/N2O, scope classification, SHA-256
-- provenance hashes, and reporting period), per-gas per-source
-- calculation detail breakdowns (emission tonnes and tCO2e with
-- emission factor used, source reference, animal/crop/AWMS type,
-- and biogenic classification), field burning event records
-- (12 crop types with area burned, crop yield, burn fraction,
-- combustion factor, burn date), regulatory compliance records
-- (IPCC_2006/IPCC_2019/GHG_PROTOCOL/ISO_14064/CSRD_ESRS/
-- EPA_40CFR98/DEFRA/CUSTOM framework checks with total
-- requirements, passed/failed counts, findings, and
-- recommendations), and step-by-step audit trail entries
-- (entity-level action trace with parent_hash/hash_value chaining
-- for tamper-evident provenance and actor attribution).
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (12):
--   1. ag_farms                     - Farm/facility registry (type, area, climate zone, location)
--   2. ag_livestock_populations     - Animal herds (type, head count, body weight, milk yield, Ym%)
--   3. ag_manure_systems            - AWMS allocation per animal type (15 types, fraction, temperature)
--   4. ag_feed_characteristics      - Tier 2 feed data (gross energy, digestibility, crude protein, Ym%)
--   5. ag_emission_factors          - IPCC/EPA/DEFRA emission factors (per source, animal, crop, gas)
--   6. ag_cropland_inputs           - Fertilizer/lime/urea inputs (synthetic N, organic N, residue N, SOM N)
--   7. ag_rice_fields               - Rice paddy definitions (water regime, flooding, amendments, soil)
--   8. ag_calculations              - Emission calculation results (total tCO2e, per-gas breakdown)
--   9. ag_calculation_details       - Per-gas per-source breakdown (emission factor, source, biogenic)
--  10. ag_field_burning_events      - Crop residue burning events (crop type, area, yield, burn fraction)
--  11. ag_compliance_records        - Regulatory compliance checks (IPCC/GHG Protocol/ISO/CSRD/EPA/DEFRA)
--  12. ag_audit_entries             - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  13. ag_calculation_events        - Calculation event time-series (hypertable on event_time)
--  14. ag_livestock_events_ts       - Livestock event time-series (hypertable on event_time)
--  15. ag_compliance_events         - Compliance event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. ag_hourly_calculation_stats  - Hourly count/sum(emissions) by emission_source and calculation_method
--   2. ag_daily_emission_totals     - Daily count/sum(emissions) by emission_source and animal_type
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-009.
-- Previous: V058__waste_treatment_emissions_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS agricultural_emissions_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION agricultural_emissions_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: agricultural_emissions_service.ag_farms
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_farms (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_name               VARCHAR(500)    NOT NULL,
    farm_type               VARCHAR(30)     NOT NULL,
    country_code            VARCHAR(3),
    region                  VARCHAR(100),
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    climate_zone            VARCHAR(30),
    total_area_ha           NUMERIC(15,4),
    arable_area_ha          NUMERIC(15,4),
    pasture_area_ha         NUMERIC(15,4),
    is_active               BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_farm_name_not_empty CHECK (LENGTH(TRIM(farm_name)) > 0);

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_farm_type CHECK (farm_type IN (
        'dairy_farm', 'beef_ranch', 'mixed_livestock', 'crop_farm',
        'rice_farm', 'mixed_crop_livestock', 'poultry_farm', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_climate_zone CHECK (climate_zone IS NULL OR climate_zone IN (
        'tropical_wet', 'tropical_dry', 'warm_temperate_wet', 'warm_temperate_dry',
        'cool_temperate_wet', 'cool_temperate_dry', 'boreal_wet', 'boreal_dry'
    ));

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_total_area_non_negative CHECK (total_area_ha IS NULL OR total_area_ha >= 0);

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_arable_area_non_negative CHECK (arable_area_ha IS NULL OR arable_area_ha >= 0);

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_pasture_area_non_negative CHECK (pasture_area_ha IS NULL OR pasture_area_ha >= 0);

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_latitude_range CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    );

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_longitude_range CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    );

ALTER TABLE agricultural_emissions_service.ag_farms
    ADD CONSTRAINT chk_af_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_af_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_farms
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 2: agricultural_emissions_service.ag_livestock_populations
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_livestock_populations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    animal_type             VARCHAR(50)     NOT NULL,
    head_count              INTEGER         NOT NULL,
    avg_body_weight_kg      NUMERIC(10,2),
    milk_yield_kg_day       NUMERIC(10,2),
    fat_pct                 NUMERIC(5,2),
    feed_digestibility_pct  NUMERIC(5,2),
    ym_pct                  NUMERIC(5,2),
    activity_coefficient    NUMERIC(5,3),
    reporting_year          INTEGER,
    is_active               BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_animal_type CHECK (animal_type IN (
        'dairy_cattle', 'non_dairy_cattle', 'buffalo', 'sheep', 'goats',
        'camels', 'horses', 'mules_asses', 'swine_market', 'swine_breeding',
        'poultry_layers', 'poultry_broilers', 'turkeys', 'ducks',
        'deer', 'elk', 'rabbits', 'fur_animals', 'llamas_alpacas', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_head_count_positive CHECK (head_count > 0);

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_avg_body_weight_positive CHECK (avg_body_weight_kg IS NULL OR avg_body_weight_kg > 0);

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_milk_yield_non_negative CHECK (milk_yield_kg_day IS NULL OR milk_yield_kg_day >= 0);

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_fat_pct_range CHECK (
        fat_pct IS NULL OR (fat_pct >= 0 AND fat_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_feed_digestibility_range CHECK (
        feed_digestibility_pct IS NULL OR (feed_digestibility_pct >= 0 AND feed_digestibility_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_ym_pct_range CHECK (
        ym_pct IS NULL OR (ym_pct >= 0 AND ym_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_activity_coefficient_non_negative CHECK (activity_coefficient IS NULL OR activity_coefficient >= 0);

ALTER TABLE agricultural_emissions_service.ag_livestock_populations
    ADD CONSTRAINT chk_lp_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_lp_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_livestock_populations
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 3: agricultural_emissions_service.ag_manure_systems
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_manure_systems (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   VARCHAR(100)    NOT NULL,
    farm_id                     UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    livestock_population_id     UUID            REFERENCES agricultural_emissions_service.ag_livestock_populations(id),
    awms_type                   VARCHAR(50)     NOT NULL,
    allocation_fraction         NUMERIC(5,4)    NOT NULL,
    mean_temperature_c          NUMERIC(6,2),
    metadata                    JSONB           DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_manure_systems
    ADD CONSTRAINT chk_ms_awms_type CHECK (awms_type IN (
        'anaerobic_lagoon', 'liquid_slurry', 'solid_storage', 'dry_lot',
        'pasture_range_paddock', 'daily_spread', 'digester', 'burned_for_fuel',
        'deep_bedding_no_mixing', 'deep_bedding_active_mixing',
        'composting_vessel', 'composting_static_pile', 'composting_intensive_windrow',
        'composting_passive_windrow', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_manure_systems
    ADD CONSTRAINT chk_ms_allocation_fraction_range CHECK (
        allocation_fraction >= 0 AND allocation_fraction <= 1
    );

ALTER TABLE agricultural_emissions_service.ag_manure_systems
    ADD CONSTRAINT chk_ms_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ms_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_manure_systems
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 4: agricultural_emissions_service.ag_feed_characteristics
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_feed_characteristics (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    livestock_population_id UUID            REFERENCES agricultural_emissions_service.ag_livestock_populations(id),
    gross_energy_mj_day     NUMERIC(10,4),
    digestible_energy_pct   NUMERIC(5,2),
    crude_protein_pct       NUMERIC(5,2),
    ym_pct                  NUMERIC(5,2),
    feed_type               VARCHAR(100),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_feed_characteristics
    ADD CONSTRAINT chk_fc_gross_energy_positive CHECK (gross_energy_mj_day IS NULL OR gross_energy_mj_day > 0);

ALTER TABLE agricultural_emissions_service.ag_feed_characteristics
    ADD CONSTRAINT chk_fc_digestible_energy_range CHECK (
        digestible_energy_pct IS NULL OR (digestible_energy_pct >= 0 AND digestible_energy_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_feed_characteristics
    ADD CONSTRAINT chk_fc_crude_protein_range CHECK (
        crude_protein_pct IS NULL OR (crude_protein_pct >= 0 AND crude_protein_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_feed_characteristics
    ADD CONSTRAINT chk_fc_ym_pct_range CHECK (
        ym_pct IS NULL OR (ym_pct >= 0 AND ym_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_feed_characteristics
    ADD CONSTRAINT chk_fc_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_fc_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_feed_characteristics
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 5: agricultural_emissions_service.ag_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100),
    source              VARCHAR(50)     NOT NULL,
    emission_source     VARCHAR(50)     NOT NULL,
    animal_type         VARCHAR(50),
    crop_type           VARCHAR(50),
    gas                 VARCHAR(10)     NOT NULL,
    factor_value        NUMERIC(20,10)  NOT NULL,
    unit                VARCHAR(50)     NOT NULL,
    uncertainty_pct     NUMERIC(5,2),
    geographic_scope    VARCHAR(100)    DEFAULT 'global',
    climate_zone        VARCHAR(30),
    valid_from          DATE,
    valid_to            DATE,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'IPCC_2006', 'IPCC_2019', 'EPA_AP42', 'DEFRA', 'UNFCCC', 'GHG_PROTOCOL', 'NATIONAL', 'CUSTOM'
    ));

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_emission_source CHECK (emission_source IN (
        'enteric_fermentation', 'manure_management', 'rice_cultivation',
        'cropland_direct_n2o', 'cropland_indirect_n2o', 'field_burning'
    ));

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_animal_type CHECK (animal_type IS NULL OR animal_type IN (
        'dairy_cattle', 'non_dairy_cattle', 'buffalo', 'sheep', 'goats',
        'camels', 'horses', 'mules_asses', 'swine_market', 'swine_breeding',
        'poultry_layers', 'poultry_broilers', 'turkeys', 'ducks',
        'deer', 'elk', 'rabbits', 'fur_animals', 'llamas_alpacas', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O'
    ));

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_unit_not_empty CHECK (LENGTH(TRIM(unit)) > 0);

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_uncertainty_pct_range CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    );

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_climate_zone CHECK (climate_zone IS NULL OR climate_zone IN (
        'tropical_wet', 'tropical_dry', 'warm_temperate_wet', 'warm_temperate_dry',
        'cool_temperate_wet', 'cool_temperate_dry', 'boreal_wet', 'boreal_dry'
    ));

ALTER TABLE agricultural_emissions_service.ag_emission_factors
    ADD CONSTRAINT chk_ef_date_order CHECK (
        valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from
    );

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_emission_factors
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 6: agricultural_emissions_service.ag_cropland_inputs
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_cropland_inputs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    input_year              INTEGER         NOT NULL,
    synthetic_n_kg          NUMERIC(15,4),
    organic_n_kg            NUMERIC(15,4),
    crop_residue_n_kg       NUMERIC(15,4),
    som_n_kg                NUMERIC(15,4),
    limestone_tonnes        NUMERIC(15,4),
    dolomite_tonnes         NUMERIC(15,4),
    urea_tonnes             NUMERIC(15,4),
    organic_soil_area_ha    NUMERIC(15,4),
    prp_n_kg                NUMERIC(15,4),
    prp_animal_type         VARCHAR(50),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_synthetic_n_non_negative CHECK (synthetic_n_kg IS NULL OR synthetic_n_kg >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_organic_n_non_negative CHECK (organic_n_kg IS NULL OR organic_n_kg >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_crop_residue_n_non_negative CHECK (crop_residue_n_kg IS NULL OR crop_residue_n_kg >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_som_n_non_negative CHECK (som_n_kg IS NULL OR som_n_kg >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_limestone_non_negative CHECK (limestone_tonnes IS NULL OR limestone_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_dolomite_non_negative CHECK (dolomite_tonnes IS NULL OR dolomite_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_urea_non_negative CHECK (urea_tonnes IS NULL OR urea_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_organic_soil_area_non_negative CHECK (organic_soil_area_ha IS NULL OR organic_soil_area_ha >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_prp_n_non_negative CHECK (prp_n_kg IS NULL OR prp_n_kg >= 0);

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_prp_animal_type CHECK (prp_animal_type IS NULL OR prp_animal_type IN (
        'dairy_cattle', 'non_dairy_cattle', 'buffalo', 'sheep', 'goats',
        'camels', 'horses', 'mules_asses', 'swine_market', 'swine_breeding',
        'poultry_layers', 'poultry_broilers', 'turkeys', 'ducks',
        'deer', 'elk', 'rabbits', 'fur_animals', 'llamas_alpacas', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_cropland_inputs
    ADD CONSTRAINT chk_ci_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ci_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_cropland_inputs
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 7: agricultural_emissions_service.ag_rice_fields
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_rice_fields (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    field_name              VARCHAR(500),
    area_ha                 NUMERIC(15,4)   NOT NULL,
    water_regime            VARCHAR(50)     NOT NULL,
    pre_season_flooding     VARCHAR(50),
    cultivation_days        INTEGER,
    crop_year               INTEGER,
    organic_amendments      JSONB           DEFAULT '[]'::jsonb,
    soil_type               VARCHAR(50),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_rice_fields
    ADD CONSTRAINT chk_rf_area_ha_positive CHECK (area_ha > 0);

ALTER TABLE agricultural_emissions_service.ag_rice_fields
    ADD CONSTRAINT chk_rf_water_regime CHECK (water_regime IN (
        'continuously_flooded', 'intermittent_single', 'intermittent_multiple',
        'rainfed_regular', 'rainfed_drought_prone', 'deep_water', 'upland'
    ));

ALTER TABLE agricultural_emissions_service.ag_rice_fields
    ADD CONSTRAINT chk_rf_pre_season_flooding CHECK (pre_season_flooding IS NULL OR pre_season_flooding IN (
        'flooded_less_30_days', 'flooded_more_30_days', 'not_flooded'
    ));

ALTER TABLE agricultural_emissions_service.ag_rice_fields
    ADD CONSTRAINT chk_rf_cultivation_days_positive CHECK (cultivation_days IS NULL OR cultivation_days > 0);

ALTER TABLE agricultural_emissions_service.ag_rice_fields
    ADD CONSTRAINT chk_rf_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_rf_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_rice_fields
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 8: agricultural_emissions_service.ag_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    calculation_method      VARCHAR(30)     NOT NULL,
    emission_source         VARCHAR(50)     NOT NULL,
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    total_co2e_tonnes       NUMERIC(20,8)   NOT NULL,
    co2_tonnes              NUMERIC(20,8)   DEFAULT 0,
    ch4_tonnes              NUMERIC(20,8)   DEFAULT 0,
    n2o_tonnes              NUMERIC(20,8)   DEFAULT 0,
    scope                   VARCHAR(10)     DEFAULT 'SCOPE_1',
    provenance_hash         VARCHAR(64),
    reporting_period        VARCHAR(20)     DEFAULT 'annual',
    reporting_year          INTEGER,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'ipcc_tier1', 'ipcc_tier2', 'ipcc_tier3',
        'emission_factor', 'mass_balance', 'direct_measurement'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_emission_source CHECK (emission_source IN (
        'enteric_fermentation', 'manure_management', 'rice_cultivation',
        'cropland_direct_n2o', 'cropland_indirect_n2o', 'field_burning'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6', 'AR6_20YR'));

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_total_co2e_non_negative CHECK (total_co2e_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_co2_non_negative CHECK (co2_tonnes IS NULL OR co2_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_ch4_non_negative CHECK (ch4_tonnes IS NULL OR ch4_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_n2o_non_negative CHECK (n2o_tonnes IS NULL OR n2o_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_scope CHECK (scope IN (
        'SCOPE_1', 'SCOPE_2', 'SCOPE_3'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_reporting_period CHECK (reporting_period IS NULL OR reporting_period IN (
        'annual', 'quarterly', 'monthly', 'custom'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON agricultural_emissions_service.ag_calculations
    FOR EACH ROW EXECUTE FUNCTION agricultural_emissions_service.set_updated_at();

-- =============================================================================
-- Table 9: agricultural_emissions_service.ag_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id          UUID            NOT NULL REFERENCES agricultural_emissions_service.ag_calculations(id) ON DELETE CASCADE,
    emission_source         VARCHAR(50),
    gas_type                VARCHAR(10),
    tonnes                  NUMERIC(20,8),
    co2e_tonnes             NUMERIC(20,8),
    emission_factor_used    NUMERIC(20,10),
    emission_factor_source  VARCHAR(50),
    animal_type             VARCHAR(50),
    crop_type               VARCHAR(50),
    awms_type               VARCHAR(50),
    is_biogenic             BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_emission_source CHECK (emission_source IS NULL OR emission_source IN (
        'enteric_fermentation', 'manure_management', 'rice_cultivation',
        'cropland_direct_n2o', 'cropland_indirect_n2o', 'field_burning'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_gas_type CHECK (gas_type IS NULL OR gas_type IN (
        'CO2', 'CH4', 'N2O'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_tonnes_non_negative CHECK (tonnes IS NULL OR tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_co2e_tonnes_non_negative CHECK (co2e_tonnes IS NULL OR co2e_tonnes >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_used_non_negative CHECK (emission_factor_used IS NULL OR emission_factor_used >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_source CHECK (emission_factor_source IS NULL OR emission_factor_source IN (
        'IPCC_2006', 'IPCC_2019', 'EPA_AP42', 'DEFRA', 'UNFCCC', 'GHG_PROTOCOL', 'NATIONAL', 'CUSTOM'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_animal_type CHECK (animal_type IS NULL OR animal_type IN (
        'dairy_cattle', 'non_dairy_cattle', 'buffalo', 'sheep', 'goats',
        'camels', 'horses', 'mules_asses', 'swine_market', 'swine_breeding',
        'poultry_layers', 'poultry_broilers', 'turkeys', 'ducks',
        'deer', 'elk', 'rabbits', 'fur_animals', 'llamas_alpacas', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_calculation_details
    ADD CONSTRAINT chk_cd_awms_type CHECK (awms_type IS NULL OR awms_type IN (
        'anaerobic_lagoon', 'liquid_slurry', 'solid_storage', 'dry_lot',
        'pasture_range_paddock', 'daily_spread', 'digester', 'burned_for_fuel',
        'deep_bedding_no_mixing', 'deep_bedding_active_mixing',
        'composting_vessel', 'composting_static_pile', 'composting_intensive_windrow',
        'composting_passive_windrow', 'other'
    ));

-- =============================================================================
-- Table 10: agricultural_emissions_service.ag_field_burning_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_field_burning_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    crop_type               VARCHAR(50)     NOT NULL,
    area_burned_ha          NUMERIC(15,4)   NOT NULL,
    crop_yield_tonnes_ha    NUMERIC(10,4),
    burn_fraction           NUMERIC(5,4),
    combustion_factor       NUMERIC(5,4),
    burn_date               DATE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_crop_type CHECK (crop_type IN (
        'rice', 'wheat', 'maize', 'sugarcane', 'barley', 'oats',
        'sorghum', 'millet', 'cotton', 'soybean', 'rapeseed', 'other'
    ));

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_area_burned_positive CHECK (area_burned_ha > 0);

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_crop_yield_non_negative CHECK (crop_yield_tonnes_ha IS NULL OR crop_yield_tonnes_ha >= 0);

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_burn_fraction_range CHECK (
        burn_fraction IS NULL OR (burn_fraction >= 0 AND burn_fraction <= 1)
    );

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_combustion_factor_range CHECK (
        combustion_factor IS NULL OR (combustion_factor >= 0 AND combustion_factor <= 1)
    );

ALTER TABLE agricultural_emissions_service.ag_field_burning_events
    ADD CONSTRAINT chk_fb_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: agricultural_emissions_service.ag_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100)    NOT NULL,
    farm_id             UUID            REFERENCES agricultural_emissions_service.ag_farms(id),
    calculation_id      UUID            REFERENCES agricultural_emissions_service.ag_calculations(id),
    framework           VARCHAR(30)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    total_requirements  INTEGER         NOT NULL DEFAULT 0,
    passed_checks       INTEGER         NOT NULL DEFAULT 0,
    failed_checks       INTEGER         NOT NULL DEFAULT 0,
    findings            JSONB           DEFAULT '[]'::jsonb,
    recommendations     JSONB           DEFAULT '[]'::jsonb,
    check_date          DATE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'IPCC_2006', 'IPCC_2019', 'GHG_PROTOCOL', 'ISO_14064',
        'CSRD_ESRS', 'EPA_40CFR98', 'DEFRA', 'CUSTOM'
    ));

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'compliant', 'non_compliant', 'partial', 'not_assessed'
    ));

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_total_requirements_non_negative CHECK (total_requirements >= 0);

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_passed_non_negative CHECK (passed_checks >= 0);

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_failed_non_negative CHECK (failed_checks >= 0);

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_passed_plus_failed CHECK (passed_checks + failed_checks <= total_requirements);

ALTER TABLE agricultural_emissions_service.ag_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 12: agricultural_emissions_service.ag_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100)    NOT NULL,
    entity_type         VARCHAR(30)     NOT NULL,
    entity_id           VARCHAR(100)    NOT NULL,
    action              VARCHAR(30)     NOT NULL,
    parent_hash         VARCHAR(64),
    hash_value          VARCHAR(64)     NOT NULL,
    actor               VARCHAR(200)    DEFAULT 'system',
    data                JSONB           DEFAULT '{}'::jsonb,
    timestamp           TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE agricultural_emissions_service.ag_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE agricultural_emissions_service.ag_audit_entries
    ADD CONSTRAINT chk_ae_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);

ALTER TABLE agricultural_emissions_service.ag_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT',
        'COMPLIANCE_CHECK', 'FEED_UPDATE', 'MANURE_ALLOCATION'
    ));

ALTER TABLE agricultural_emissions_service.ag_audit_entries
    ADD CONSTRAINT chk_ae_hash_value_not_empty CHECK (LENGTH(TRIM(hash_value)) > 0);

ALTER TABLE agricultural_emissions_service.ag_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 13: agricultural_emissions_service.ag_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_calculation_events (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID,
    emission_source         VARCHAR(50),
    calculation_method      VARCHAR(30),
    emissions_tco2e         NUMERIC(20,8),
    gas                     VARCHAR(10),
    duration_ms             NUMERIC(12,2),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'agricultural_emissions_service.ag_calculation_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE agricultural_emissions_service.ag_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE agricultural_emissions_service.ag_calculation_events
    ADD CONSTRAINT chk_cae_emission_source CHECK (
        emission_source IS NULL OR emission_source IN (
            'enteric_fermentation', 'manure_management', 'rice_cultivation',
            'cropland_direct_n2o', 'cropland_indirect_n2o', 'field_burning'
        )
    );

ALTER TABLE agricultural_emissions_service.ag_calculation_events
    ADD CONSTRAINT chk_cae_gas CHECK (
        gas IS NULL OR gas IN ('CO2', 'CH4', 'N2O')
    );

-- =============================================================================
-- Table 14: agricultural_emissions_service.ag_livestock_events_ts (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_livestock_events_ts (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID,
    animal_type             VARCHAR(50),
    head_count              INTEGER,
    event_type              VARCHAR(50),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'agricultural_emissions_service.ag_livestock_events_ts',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE agricultural_emissions_service.ag_livestock_events_ts
    ADD CONSTRAINT chk_lets_head_count_non_negative CHECK (head_count IS NULL OR head_count >= 0);

ALTER TABLE agricultural_emissions_service.ag_livestock_events_ts
    ADD CONSTRAINT chk_lets_animal_type CHECK (
        animal_type IS NULL OR animal_type IN (
            'dairy_cattle', 'non_dairy_cattle', 'buffalo', 'sheep', 'goats',
            'camels', 'horses', 'mules_asses', 'swine_market', 'swine_breeding',
            'poultry_layers', 'poultry_broilers', 'turkeys', 'ducks',
            'deer', 'elk', 'rabbits', 'fur_animals', 'llamas_alpacas', 'other'
        )
    );

-- =============================================================================
-- Table 15: agricultural_emissions_service.ag_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS agricultural_emissions_service.ag_compliance_events (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    farm_id                 UUID,
    framework               VARCHAR(30),
    status                  VARCHAR(20),
    check_id                UUID,
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'agricultural_emissions_service.ag_compliance_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE agricultural_emissions_service.ag_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('IPCC_2006', 'IPCC_2019', 'GHG_PROTOCOL', 'ISO_14064', 'CSRD_ESRS', 'EPA_40CFR98', 'DEFRA', 'CUSTOM')
    );

ALTER TABLE agricultural_emissions_service.ag_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('compliant', 'non_compliant', 'partial', 'not_assessed')
    );

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- ag_hourly_calculation_stats: hourly count/sum(emissions) by emission_source and calculation_method
CREATE MATERIALIZED VIEW agricultural_emissions_service.ag_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time)   AS bucket,
    emission_source,
    calculation_method,
    COUNT(*)                            AS total_calculations,
    SUM(emissions_tco2e)                AS sum_emissions_tco2e,
    AVG(emissions_tco2e)                AS avg_emissions_tco2e,
    AVG(duration_ms)                    AS avg_duration_ms,
    MAX(duration_ms)                    AS max_duration_ms
FROM agricultural_emissions_service.ag_calculation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, emission_source, calculation_method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'agricultural_emissions_service.ag_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- ag_daily_emission_totals: daily count/sum(emissions) by emission_source and animal_type
CREATE MATERIALIZED VIEW agricultural_emissions_service.ag_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time)    AS bucket,
    animal_type,
    event_type,
    COUNT(*)                            AS total_events,
    SUM(head_count)                     AS sum_head_count,
    AVG(head_count)                     AS avg_head_count
FROM agricultural_emissions_service.ag_livestock_events_ts
WHERE event_time IS NOT NULL
GROUP BY bucket, animal_type, event_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'agricultural_emissions_service.ag_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- ag_farms indexes
CREATE INDEX IF NOT EXISTS idx_ag_af_tenant_id              ON agricultural_emissions_service.ag_farms(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_af_farm_name              ON agricultural_emissions_service.ag_farms(farm_name);
CREATE INDEX IF NOT EXISTS idx_ag_af_farm_type              ON agricultural_emissions_service.ag_farms(farm_type);
CREATE INDEX IF NOT EXISTS idx_ag_af_country_code           ON agricultural_emissions_service.ag_farms(country_code);
CREATE INDEX IF NOT EXISTS idx_ag_af_region                 ON agricultural_emissions_service.ag_farms(region);
CREATE INDEX IF NOT EXISTS idx_ag_af_climate_zone           ON agricultural_emissions_service.ag_farms(climate_zone);
CREATE INDEX IF NOT EXISTS idx_ag_af_is_active              ON agricultural_emissions_service.ag_farms(is_active);
CREATE INDEX IF NOT EXISTS idx_ag_af_tenant_type            ON agricultural_emissions_service.ag_farms(tenant_id, farm_type);
CREATE INDEX IF NOT EXISTS idx_ag_af_tenant_country         ON agricultural_emissions_service.ag_farms(tenant_id, country_code);
CREATE INDEX IF NOT EXISTS idx_ag_af_tenant_active          ON agricultural_emissions_service.ag_farms(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_ag_af_type_country           ON agricultural_emissions_service.ag_farms(farm_type, country_code);
CREATE INDEX IF NOT EXISTS idx_ag_af_created_at             ON agricultural_emissions_service.ag_farms(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_af_updated_at             ON agricultural_emissions_service.ag_farms(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_af_metadata               ON agricultural_emissions_service.ag_farms USING GIN (metadata);

-- Partial index: active farms only
CREATE INDEX IF NOT EXISTS idx_ag_af_active_farms           ON agricultural_emissions_service.ag_farms(tenant_id, farm_type)
    WHERE is_active = TRUE;

-- ag_livestock_populations indexes
CREATE INDEX IF NOT EXISTS idx_ag_lp_tenant_id              ON agricultural_emissions_service.ag_livestock_populations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_lp_farm_id                ON agricultural_emissions_service.ag_livestock_populations(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_lp_animal_type            ON agricultural_emissions_service.ag_livestock_populations(animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_lp_reporting_year         ON agricultural_emissions_service.ag_livestock_populations(reporting_year);
CREATE INDEX IF NOT EXISTS idx_ag_lp_is_active              ON agricultural_emissions_service.ag_livestock_populations(is_active);
CREATE INDEX IF NOT EXISTS idx_ag_lp_tenant_farm            ON agricultural_emissions_service.ag_livestock_populations(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_lp_tenant_animal          ON agricultural_emissions_service.ag_livestock_populations(tenant_id, animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_lp_farm_animal            ON agricultural_emissions_service.ag_livestock_populations(farm_id, animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_lp_farm_year              ON agricultural_emissions_service.ag_livestock_populations(farm_id, reporting_year);
CREATE INDEX IF NOT EXISTS idx_ag_lp_animal_year            ON agricultural_emissions_service.ag_livestock_populations(animal_type, reporting_year);
CREATE INDEX IF NOT EXISTS idx_ag_lp_created_at             ON agricultural_emissions_service.ag_livestock_populations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lp_updated_at             ON agricultural_emissions_service.ag_livestock_populations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lp_metadata               ON agricultural_emissions_service.ag_livestock_populations USING GIN (metadata);

-- Partial index: active livestock populations only
CREATE INDEX IF NOT EXISTS idx_ag_lp_active_populations     ON agricultural_emissions_service.ag_livestock_populations(tenant_id, animal_type)
    WHERE is_active = TRUE;

-- ag_manure_systems indexes
CREATE INDEX IF NOT EXISTS idx_ag_ms_tenant_id              ON agricultural_emissions_service.ag_manure_systems(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_ms_farm_id                ON agricultural_emissions_service.ag_manure_systems(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_ms_livestock_population_id ON agricultural_emissions_service.ag_manure_systems(livestock_population_id);
CREATE INDEX IF NOT EXISTS idx_ag_ms_awms_type              ON agricultural_emissions_service.ag_manure_systems(awms_type);
CREATE INDEX IF NOT EXISTS idx_ag_ms_tenant_farm            ON agricultural_emissions_service.ag_manure_systems(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_ms_farm_livestock          ON agricultural_emissions_service.ag_manure_systems(farm_id, livestock_population_id);
CREATE INDEX IF NOT EXISTS idx_ag_ms_farm_awms              ON agricultural_emissions_service.ag_manure_systems(farm_id, awms_type);
CREATE INDEX IF NOT EXISTS idx_ag_ms_livestock_awms         ON agricultural_emissions_service.ag_manure_systems(livestock_population_id, awms_type);
CREATE INDEX IF NOT EXISTS idx_ag_ms_created_at             ON agricultural_emissions_service.ag_manure_systems(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ms_updated_at             ON agricultural_emissions_service.ag_manure_systems(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ms_metadata               ON agricultural_emissions_service.ag_manure_systems USING GIN (metadata);

-- ag_feed_characteristics indexes
CREATE INDEX IF NOT EXISTS idx_ag_fc_tenant_id              ON agricultural_emissions_service.ag_feed_characteristics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_fc_farm_id                ON agricultural_emissions_service.ag_feed_characteristics(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_fc_livestock_population_id ON agricultural_emissions_service.ag_feed_characteristics(livestock_population_id);
CREATE INDEX IF NOT EXISTS idx_ag_fc_feed_type              ON agricultural_emissions_service.ag_feed_characteristics(feed_type);
CREATE INDEX IF NOT EXISTS idx_ag_fc_tenant_farm            ON agricultural_emissions_service.ag_feed_characteristics(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_fc_farm_livestock          ON agricultural_emissions_service.ag_feed_characteristics(farm_id, livestock_population_id);
CREATE INDEX IF NOT EXISTS idx_ag_fc_created_at             ON agricultural_emissions_service.ag_feed_characteristics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_fc_updated_at             ON agricultural_emissions_service.ag_feed_characteristics(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_fc_metadata               ON agricultural_emissions_service.ag_feed_characteristics USING GIN (metadata);

-- ag_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_ag_ef_tenant_id              ON agricultural_emissions_service.ag_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_ef_source                 ON agricultural_emissions_service.ag_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_source        ON agricultural_emissions_service.ag_emission_factors(emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_ef_animal_type            ON agricultural_emissions_service.ag_emission_factors(animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_ef_crop_type              ON agricultural_emissions_service.ag_emission_factors(crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_ef_gas                    ON agricultural_emissions_service.ag_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_ag_ef_is_active              ON agricultural_emissions_service.ag_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_ag_ef_geographic_scope       ON agricultural_emissions_service.ag_emission_factors(geographic_scope);
CREATE INDEX IF NOT EXISTS idx_ag_ef_climate_zone           ON agricultural_emissions_service.ag_emission_factors(climate_zone);
CREATE INDEX IF NOT EXISTS idx_ag_ef_source_emission        ON agricultural_emissions_service.ag_emission_factors(source, emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_animal        ON agricultural_emissions_service.ag_emission_factors(emission_source, animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_crop          ON agricultural_emissions_service.ag_emission_factors(emission_source, crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_gas           ON agricultural_emissions_service.ag_emission_factors(emission_source, gas);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_animal_gas    ON agricultural_emissions_service.ag_emission_factors(emission_source, animal_type, gas);
CREATE INDEX IF NOT EXISTS idx_ag_ef_emission_crop_gas      ON agricultural_emissions_service.ag_emission_factors(emission_source, crop_type, gas);
CREATE INDEX IF NOT EXISTS idx_ag_ef_valid_from             ON agricultural_emissions_service.ag_emission_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ef_valid_to               ON agricultural_emissions_service.ag_emission_factors(valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ef_created_at             ON agricultural_emissions_service.ag_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ef_updated_at             ON agricultural_emissions_service.ag_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ef_metadata               ON agricultural_emissions_service.ag_emission_factors USING GIN (metadata);

-- Partial index: active factors for lookups
CREATE INDEX IF NOT EXISTS idx_ag_ef_active_factors         ON agricultural_emissions_service.ag_emission_factors(emission_source, animal_type, gas, source)
    WHERE is_active = TRUE;

-- ag_cropland_inputs indexes
CREATE INDEX IF NOT EXISTS idx_ag_ci_tenant_id              ON agricultural_emissions_service.ag_cropland_inputs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_ci_farm_id                ON agricultural_emissions_service.ag_cropland_inputs(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_ci_input_year             ON agricultural_emissions_service.ag_cropland_inputs(input_year);
CREATE INDEX IF NOT EXISTS idx_ag_ci_prp_animal_type        ON agricultural_emissions_service.ag_cropland_inputs(prp_animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_ci_tenant_farm            ON agricultural_emissions_service.ag_cropland_inputs(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_ci_tenant_year            ON agricultural_emissions_service.ag_cropland_inputs(tenant_id, input_year);
CREATE INDEX IF NOT EXISTS idx_ag_ci_farm_year              ON agricultural_emissions_service.ag_cropland_inputs(farm_id, input_year);
CREATE INDEX IF NOT EXISTS idx_ag_ci_created_at             ON agricultural_emissions_service.ag_cropland_inputs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ci_updated_at             ON agricultural_emissions_service.ag_cropland_inputs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ci_metadata               ON agricultural_emissions_service.ag_cropland_inputs USING GIN (metadata);

-- ag_rice_fields indexes
CREATE INDEX IF NOT EXISTS idx_ag_rf_tenant_id              ON agricultural_emissions_service.ag_rice_fields(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_rf_farm_id                ON agricultural_emissions_service.ag_rice_fields(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_rf_water_regime           ON agricultural_emissions_service.ag_rice_fields(water_regime);
CREATE INDEX IF NOT EXISTS idx_ag_rf_crop_year              ON agricultural_emissions_service.ag_rice_fields(crop_year);
CREATE INDEX IF NOT EXISTS idx_ag_rf_soil_type              ON agricultural_emissions_service.ag_rice_fields(soil_type);
CREATE INDEX IF NOT EXISTS idx_ag_rf_tenant_farm            ON agricultural_emissions_service.ag_rice_fields(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_rf_farm_regime            ON agricultural_emissions_service.ag_rice_fields(farm_id, water_regime);
CREATE INDEX IF NOT EXISTS idx_ag_rf_farm_year              ON agricultural_emissions_service.ag_rice_fields(farm_id, crop_year);
CREATE INDEX IF NOT EXISTS idx_ag_rf_created_at             ON agricultural_emissions_service.ag_rice_fields(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_rf_updated_at             ON agricultural_emissions_service.ag_rice_fields(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_rf_metadata               ON agricultural_emissions_service.ag_rice_fields USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_ag_rf_organic_amendments     ON agricultural_emissions_service.ag_rice_fields USING GIN (organic_amendments);

-- ag_calculations indexes
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_id            ON agricultural_emissions_service.ag_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_calc_farm_id              ON agricultural_emissions_service.ag_calculations(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_calc_calculation_method   ON agricultural_emissions_service.ag_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_ag_calc_emission_source      ON agricultural_emissions_service.ag_calculations(emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_calc_gwp_source           ON agricultural_emissions_service.ag_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_ag_calc_scope                ON agricultural_emissions_service.ag_calculations(scope);
CREATE INDEX IF NOT EXISTS idx_ag_calc_provenance_hash      ON agricultural_emissions_service.ag_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_ag_calc_reporting_year       ON agricultural_emissions_service.ag_calculations(reporting_year);
CREATE INDEX IF NOT EXISTS idx_ag_calc_reporting_period     ON agricultural_emissions_service.ag_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_ag_calc_total_co2e           ON agricultural_emissions_service.ag_calculations(total_co2e_tonnes DESC);
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_farm          ON agricultural_emissions_service.ag_calculations(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_source        ON agricultural_emissions_service.ag_calculations(tenant_id, emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_method        ON agricultural_emissions_service.ag_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_year          ON agricultural_emissions_service.ag_calculations(tenant_id, reporting_year);
CREATE INDEX IF NOT EXISTS idx_ag_calc_tenant_scope         ON agricultural_emissions_service.ag_calculations(tenant_id, scope);
CREATE INDEX IF NOT EXISTS idx_ag_calc_farm_source          ON agricultural_emissions_service.ag_calculations(farm_id, emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_calc_farm_created         ON agricultural_emissions_service.ag_calculations(farm_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_calc_created_at           ON agricultural_emissions_service.ag_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_calc_updated_at           ON agricultural_emissions_service.ag_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_calc_metadata             ON agricultural_emissions_service.ag_calculations USING GIN (metadata);

-- ag_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_ag_cd_calculation_id         ON agricultural_emissions_service.ag_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_ag_cd_emission_source        ON agricultural_emissions_service.ag_calculation_details(emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_cd_gas_type               ON agricultural_emissions_service.ag_calculation_details(gas_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_animal_type            ON agricultural_emissions_service.ag_calculation_details(animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_crop_type              ON agricultural_emissions_service.ag_calculation_details(crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_awms_type              ON agricultural_emissions_service.ag_calculation_details(awms_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_is_biogenic            ON agricultural_emissions_service.ag_calculation_details(is_biogenic);
CREATE INDEX IF NOT EXISTS idx_ag_cd_calc_gas               ON agricultural_emissions_service.ag_calculation_details(calculation_id, gas_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_calc_source            ON agricultural_emissions_service.ag_calculation_details(calculation_id, emission_source);
CREATE INDEX IF NOT EXISTS idx_ag_cd_calc_animal            ON agricultural_emissions_service.ag_calculation_details(calculation_id, animal_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_calc_source_gas        ON agricultural_emissions_service.ag_calculation_details(calculation_id, emission_source, gas_type);
CREATE INDEX IF NOT EXISTS idx_ag_cd_created_at             ON agricultural_emissions_service.ag_calculation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cd_metadata               ON agricultural_emissions_service.ag_calculation_details USING GIN (metadata);

-- Partial index: biogenic emissions only
CREATE INDEX IF NOT EXISTS idx_ag_cd_biogenic_emissions     ON agricultural_emissions_service.ag_calculation_details(calculation_id, emission_source)
    WHERE is_biogenic = TRUE;

-- ag_field_burning_events indexes
CREATE INDEX IF NOT EXISTS idx_ag_fb_tenant_id              ON agricultural_emissions_service.ag_field_burning_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_fb_farm_id                ON agricultural_emissions_service.ag_field_burning_events(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_fb_crop_type              ON agricultural_emissions_service.ag_field_burning_events(crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_fb_burn_date              ON agricultural_emissions_service.ag_field_burning_events(burn_date DESC);
CREATE INDEX IF NOT EXISTS idx_ag_fb_tenant_farm            ON agricultural_emissions_service.ag_field_burning_events(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_fb_tenant_crop            ON agricultural_emissions_service.ag_field_burning_events(tenant_id, crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_fb_farm_crop              ON agricultural_emissions_service.ag_field_burning_events(farm_id, crop_type);
CREATE INDEX IF NOT EXISTS idx_ag_fb_farm_date              ON agricultural_emissions_service.ag_field_burning_events(farm_id, burn_date DESC);
CREATE INDEX IF NOT EXISTS idx_ag_fb_created_at             ON agricultural_emissions_service.ag_field_burning_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ag_fb_metadata               ON agricultural_emissions_service.ag_field_burning_events USING GIN (metadata);

-- ag_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_ag_cr_tenant_id              ON agricultural_emissions_service.ag_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_cr_farm_id                ON agricultural_emissions_service.ag_compliance_records(farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_cr_calculation_id         ON agricultural_emissions_service.ag_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_ag_cr_framework              ON agricultural_emissions_service.ag_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_ag_cr_status                 ON agricultural_emissions_service.ag_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_ag_cr_check_date             ON agricultural_emissions_service.ag_compliance_records(check_date DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cr_tenant_framework       ON agricultural_emissions_service.ag_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_ag_cr_tenant_status          ON agricultural_emissions_service.ag_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_ag_cr_framework_status       ON agricultural_emissions_service.ag_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_ag_cr_tenant_calculation     ON agricultural_emissions_service.ag_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_ag_cr_tenant_farm            ON agricultural_emissions_service.ag_compliance_records(tenant_id, farm_id);
CREATE INDEX IF NOT EXISTS idx_ag_cr_findings               ON agricultural_emissions_service.ag_compliance_records USING GIN (findings);
CREATE INDEX IF NOT EXISTS idx_ag_cr_recommendations        ON agricultural_emissions_service.ag_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_ag_cr_metadata               ON agricultural_emissions_service.ag_compliance_records USING GIN (metadata);

-- ag_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_ag_ae_tenant_id              ON agricultural_emissions_service.ag_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ag_ae_entity_type            ON agricultural_emissions_service.ag_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_ag_ae_entity_id              ON agricultural_emissions_service.ag_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_ag_ae_action                 ON agricultural_emissions_service.ag_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_ag_ae_actor                  ON agricultural_emissions_service.ag_audit_entries(actor);
CREATE INDEX IF NOT EXISTS idx_ag_ae_parent_hash            ON agricultural_emissions_service.ag_audit_entries(parent_hash);
CREATE INDEX IF NOT EXISTS idx_ag_ae_hash_value             ON agricultural_emissions_service.ag_audit_entries(hash_value);
CREATE INDEX IF NOT EXISTS idx_ag_ae_tenant_entity          ON agricultural_emissions_service.ag_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_ag_ae_tenant_action          ON agricultural_emissions_service.ag_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_ag_ae_timestamp              ON agricultural_emissions_service.ag_audit_entries(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ag_ae_data                   ON agricultural_emissions_service.ag_audit_entries USING GIN (data);

-- ag_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_ag_cae_tenant_id             ON agricultural_emissions_service.ag_calculation_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_farm_id               ON agricultural_emissions_service.ag_calculation_events(farm_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_emission_source       ON agricultural_emissions_service.ag_calculation_events(emission_source, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_calculation_method    ON agricultural_emissions_service.ag_calculation_events(calculation_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_tenant_source         ON agricultural_emissions_service.ag_calculation_events(tenant_id, emission_source, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_tenant_method         ON agricultural_emissions_service.ag_calculation_events(tenant_id, calculation_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_cae_metadata              ON agricultural_emissions_service.ag_calculation_events USING GIN (metadata);

-- ag_livestock_events_ts indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_ag_lets_tenant_id            ON agricultural_emissions_service.ag_livestock_events_ts(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lets_farm_id              ON agricultural_emissions_service.ag_livestock_events_ts(farm_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lets_animal_type          ON agricultural_emissions_service.ag_livestock_events_ts(animal_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lets_event_type           ON agricultural_emissions_service.ag_livestock_events_ts(event_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lets_tenant_animal        ON agricultural_emissions_service.ag_livestock_events_ts(tenant_id, animal_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_lets_metadata             ON agricultural_emissions_service.ag_livestock_events_ts USING GIN (metadata);

-- ag_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_ag_coe_tenant_id             ON agricultural_emissions_service.ag_compliance_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_farm_id               ON agricultural_emissions_service.ag_compliance_events(farm_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_framework             ON agricultural_emissions_service.ag_compliance_events(framework, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_status                ON agricultural_emissions_service.ag_compliance_events(status, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_tenant_framework      ON agricultural_emissions_service.ag_compliance_events(tenant_id, framework, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_tenant_status         ON agricultural_emissions_service.ag_compliance_events(tenant_id, status, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_ag_coe_metadata              ON agricultural_emissions_service.ag_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- ag_farms: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_farms ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_af_read  ON agricultural_emissions_service.ag_farms FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_af_write ON agricultural_emissions_service.ag_farms FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_livestock_populations: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_livestock_populations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_lp_read  ON agricultural_emissions_service.ag_livestock_populations FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_lp_write ON agricultural_emissions_service.ag_livestock_populations FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_manure_systems: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_manure_systems ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_ms_read  ON agricultural_emissions_service.ag_manure_systems FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_ms_write ON agricultural_emissions_service.ag_manure_systems FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_feed_characteristics: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_feed_characteristics ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_fc_read  ON agricultural_emissions_service.ag_feed_characteristics FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_fc_write ON agricultural_emissions_service.ag_feed_characteristics FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_emission_factors: shared reference data (open read, admin write)
ALTER TABLE agricultural_emissions_service.ag_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_ef_read  ON agricultural_emissions_service.ag_emission_factors FOR SELECT USING (TRUE);
CREATE POLICY ag_ef_write ON agricultural_emissions_service.ag_emission_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- ag_cropland_inputs: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_cropland_inputs ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_ci_read  ON agricultural_emissions_service.ag_cropland_inputs FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_ci_write ON agricultural_emissions_service.ag_cropland_inputs FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_rice_fields: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_rice_fields ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_rf_read  ON agricultural_emissions_service.ag_rice_fields FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_rf_write ON agricultural_emissions_service.ag_rice_fields FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_calculations: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_calc_read  ON agricultural_emissions_service.ag_calculations FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_calc_write ON agricultural_emissions_service.ag_calculations FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_calculation_details: cascade through calculation_id (no tenant_id column, secured via parent)
-- Note: calculation_details has ON DELETE CASCADE from ag_calculations; no tenant_id column needed

-- ag_field_burning_events: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_field_burning_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_fb_read  ON agricultural_emissions_service.ag_field_burning_events FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_fb_write ON agricultural_emissions_service.ag_field_burning_events FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_compliance_records: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_cr_read  ON agricultural_emissions_service.ag_compliance_records FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_cr_write ON agricultural_emissions_service.ag_compliance_records FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_audit_entries: tenant-isolated
ALTER TABLE agricultural_emissions_service.ag_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_ae_read  ON agricultural_emissions_service.ag_audit_entries FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY ag_ae_write ON agricultural_emissions_service.ag_audit_entries FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- ag_calculation_events: open read/write (time-series telemetry)
ALTER TABLE agricultural_emissions_service.ag_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_cae_read  ON agricultural_emissions_service.ag_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY ag_cae_write ON agricultural_emissions_service.ag_calculation_events FOR ALL   USING (TRUE);

-- ag_livestock_events_ts: open read/write (time-series telemetry)
ALTER TABLE agricultural_emissions_service.ag_livestock_events_ts ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_lets_read  ON agricultural_emissions_service.ag_livestock_events_ts FOR SELECT USING (TRUE);
CREATE POLICY ag_lets_write ON agricultural_emissions_service.ag_livestock_events_ts FOR ALL   USING (TRUE);

-- ag_compliance_events: open read/write (time-series telemetry)
ALTER TABLE agricultural_emissions_service.ag_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY ag_coe_read  ON agricultural_emissions_service.ag_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY ag_coe_write ON agricultural_emissions_service.ag_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA agricultural_emissions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agricultural_emissions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA agricultural_emissions_service TO greenlang_app;
GRANT SELECT ON agricultural_emissions_service.ag_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON agricultural_emissions_service.ag_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA agricultural_emissions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA agricultural_emissions_service TO greenlang_readonly;
GRANT SELECT ON agricultural_emissions_service.ag_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON agricultural_emissions_service.ag_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA agricultural_emissions_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA agricultural_emissions_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA agricultural_emissions_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'agricultural-emissions:read',                    'agricultural-emissions', 'read',                    'View all agricultural emissions service data including farms, livestock populations, manure systems, feed characteristics, emission factors, cropland inputs, rice fields, calculations, field burning events, and compliance records'),
    (gen_random_uuid(), 'agricultural-emissions:write',                   'agricultural-emissions', 'write',                   'Create, update, and manage all agricultural emissions service data'),
    (gen_random_uuid(), 'agricultural-emissions:execute',                 'agricultural-emissions', 'execute',                 'Execute agricultural emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'agricultural-emissions:farms:read',              'agricultural-emissions', 'farms_read',              'View farm registry with farm type (dairy_farm/beef_ranch/mixed_livestock/crop_farm/rice_farm/mixed_crop_livestock/poultry_farm/other), country, region, latitude/longitude, 8 IPCC climate zones, total/arable/pasture area in hectares'),
    (gen_random_uuid(), 'agricultural-emissions:farms:write',             'agricultural-emissions', 'farms_write',             'Create, update, and manage farm registry entries'),
    (gen_random_uuid(), 'agricultural-emissions:livestock:read',          'agricultural-emissions', 'livestock_read',          'View livestock populations with 20 animal types, head count, average body weight, milk yield, fat percentage, feed digestibility, Ym%, activity coefficient, and reporting year'),
    (gen_random_uuid(), 'agricultural-emissions:livestock:write',         'agricultural-emissions', 'livestock_write',         'Create, update, and manage livestock population records'),
    (gen_random_uuid(), 'agricultural-emissions:manure:read',             'agricultural-emissions', 'manure_read',             'View AWMS allocation per animal type with 15 AWMS types, allocation fraction (0-1), and mean temperature'),
    (gen_random_uuid(), 'agricultural-emissions:manure:write',            'agricultural-emissions', 'manure_write',            'Create, update, and manage manure system allocation records'),
    (gen_random_uuid(), 'agricultural-emissions:feed:read',               'agricultural-emissions', 'feed_read',               'View Tier 2 feed characteristics with gross energy MJ/day, digestible energy %, crude protein %, Ym%, and feed type'),
    (gen_random_uuid(), 'agricultural-emissions:feed:write',              'agricultural-emissions', 'feed_write',              'Create, update, and manage feed characteristic records'),
    (gen_random_uuid(), 'agricultural-emissions:factors:read',            'agricultural-emissions', 'factors_read',            'View IPCC/EPA/DEFRA/UNFCCC/GHG_PROTOCOL emission factors per emission source, animal type, crop type, and gas (CO2/CH4/N2O) with source versioning, geographic scoping, climate zone, and uncertainty percentages'),
    (gen_random_uuid(), 'agricultural-emissions:factors:write',           'agricultural-emissions', 'factors_write',           'Create, update, and manage emission factor entries'),
    (gen_random_uuid(), 'agricultural-emissions:cropland:read',           'agricultural-emissions', 'cropland_read',           'View cropland input records with synthetic N, organic N, crop residue N, SOM N, limestone, dolomite, urea, organic soil area, PRP N, and PRP animal type'),
    (gen_random_uuid(), 'agricultural-emissions:cropland:write',          'agricultural-emissions', 'cropland_write',          'Create, update, and manage cropland input records'),
    (gen_random_uuid(), 'agricultural-emissions:rice:read',               'agricultural-emissions', 'rice_read',               'View rice paddy field definitions with 7 water regimes, 3 pre-season flooding types, cultivation days, organic amendments, and soil type'),
    (gen_random_uuid(), 'agricultural-emissions:rice:write',              'agricultural-emissions', 'rice_write',              'Create, update, and manage rice field definitions'),
    (gen_random_uuid(), 'agricultural-emissions:calculations:read',       'agricultural-emissions', 'calculations_read',       'View emission calculation results with total tCO2e, per-gas breakdown (CO2, CH4, N2O), scope classification, provenance hashes, and per-source detail breakdowns'),
    (gen_random_uuid(), 'agricultural-emissions:compliance:read',         'agricultural-emissions', 'compliance_read',         'View regulatory compliance records for IPCC_2006, IPCC_2019, GHG_PROTOCOL, ISO_14064, CSRD_ESRS, EPA_40CFR98, and DEFRA with total requirements, passed/failed counts, findings, and recommendations'),
    (gen_random_uuid(), 'agricultural-emissions:admin',                   'agricultural-emissions', 'admin',                   'Full administrative access to agricultural emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('agricultural_emissions_service.ag_calculation_events',  INTERVAL '365 days');
SELECT add_retention_policy('agricultural_emissions_service.ag_livestock_events_ts', INTERVAL '365 days');
SELECT add_retention_policy('agricultural_emissions_service.ag_compliance_events',   INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE agricultural_emissions_service.ag_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'emission_source',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('agricultural_emissions_service.ag_calculation_events', INTERVAL '30 days');

ALTER TABLE agricultural_emissions_service.ag_livestock_events_ts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'animal_type',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('agricultural_emissions_service.ag_livestock_events_ts', INTERVAL '30 days');

ALTER TABLE agricultural_emissions_service.ag_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('agricultural_emissions_service.ag_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Agricultural Emissions Agent (GL-MRV-SCOPE1-009)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-009',
    'Agricultural Emissions Agent',
    'Agricultural emission calculator for GreenLang Climate OS. Manages farm/facility registry with farm types (dairy_farm/beef_ranch/mixed_livestock/crop_farm/rice_farm/mixed_crop_livestock/poultry_farm/other), country code, region, latitude/longitude coordinates, 8 IPCC climate zones (tropical_wet/tropical_dry/warm_temperate_wet/warm_temperate_dry/cool_temperate_wet/cool_temperate_dry/boreal_wet/boreal_dry), and total/arable/pasture area in hectares. Tracks livestock populations with 20 animal types (dairy_cattle/non_dairy_cattle/buffalo/sheep/goats/camels/horses/mules_asses/swine_market/swine_breeding/poultry_layers/poultry_broilers/turkeys/ducks/deer/elk/rabbits/fur_animals/llamas_alpacas/other) with head count, average body weight kg, milk yield kg/day, fat percentage, feed digestibility percentage, methane conversion factor Ym%, activity coefficient, and reporting year. Manages animal waste management system (AWMS) allocation per animal type with 15 AWMS types (anaerobic_lagoon/liquid_slurry/solid_storage/dry_lot/pasture_range_paddock/daily_spread/digester/burned_for_fuel/deep_bedding_no_mixing/deep_bedding_active_mixing/composting_vessel/composting_static_pile/composting_intensive_windrow/composting_passive_windrow/other) with allocation fraction 0-1 and mean temperature. Stores Tier 2 feed characteristic data with gross energy MJ/day, digestible energy %, crude protein %, Ym%, and feed type. Maintains IPCC/EPA/DEFRA/UNFCCC/GHG_PROTOCOL/NATIONAL/CUSTOM emission factor database per emission source (enteric_fermentation/manure_management/rice_cultivation/cropland_direct_n2o/cropland_indirect_n2o/field_burning) with animal type, crop type, gas (CO2/CH4/N2O), factor values, units, source versioning, geographic scoping, climate zone, uncertainty percentages, and validity date ranges. Records cropland inputs with synthetic N, organic N, crop residue N, SOM N, limestone, dolomite, urea tonnes, organic soil area, PRP N, and PRP animal type. Defines rice paddy fields with 7 water regimes (continuously_flooded/intermittent_single/intermittent_multiple/rainfed_regular/rainfed_drought_prone/deep_water/upland), 3 pre-season flooding types, cultivation days, organic amendments JSONB with type/rate_tonnes_ha, and soil type. Executes deterministic emission calculations using ipcc_tier1, ipcc_tier2, ipcc_tier3, emission_factor, mass_balance, and direct_measurement methods with GWP sources AR4/AR5/AR6/AR6_20YR, producing total emissions tCO2e with per-gas breakdown (CO2, CH4, N2O), scope classification (SCOPE_1/SCOPE_2/SCOPE_3), SHA-256 provenance hashes, and reporting period. Produces per-gas per-source calculation detail breakdowns with emission tonnes and tCO2e, emission factor used, source reference, animal/crop/AWMS type, and biogenic classification. Records field burning events for 12 crop types (rice/wheat/maize/sugarcane/barley/oats/sorghum/millet/cotton/soybean/rapeseed/other) with area burned, crop yield, burn fraction, combustion factor, and burn date. Checks regulatory compliance against IPCC_2006, IPCC_2019, GHG_PROTOCOL, ISO_14064, CSRD_ESRS, EPA_40CFR98, DEFRA, and CUSTOM frameworks with total requirements, passed/failed counts, findings, and recommendations. Generates entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/COMPLIANCE_CHECK/FEED_UPDATE/MANURE_ALLOCATION), parent_hash/hash_value chaining for tamper-evident provenance, and actor attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/agricultural-emissions',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-009', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/agricultural-emissions-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"agriculture", "livestock", "enteric-fermentation", "manure-management", "rice-cultivation", "cropland", "field-burning", "scope-1", "ghg-protocol", "ipcc", "mrv"}',
    '{"agriculture", "dairy", "beef", "poultry", "crop-farming", "rice-farming", "cross-sector"}',
    'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'farm_registry',
    'configuration',
    'Register and manage farms/facilities with farm type (dairy_farm/beef_ranch/mixed_livestock/crop_farm/rice_farm/mixed_crop_livestock/poultry_farm/other), country code, region, latitude/longitude, 8 IPCC climate zones (tropical_wet/tropical_dry/warm_temperate_wet/warm_temperate_dry/cool_temperate_wet/cool_temperate_dry/boreal_wet/boreal_dry), and total/arable/pasture area in hectares.',
    '{"farm_name", "farm_type", "country_code", "region", "latitude", "longitude", "climate_zone", "total_area_ha", "arable_area_ha", "pasture_area_ha"}',
    '{"farm_id", "registration_result"}',
    '{"farm_types": ["dairy_farm", "beef_ranch", "mixed_livestock", "crop_farm", "rice_farm", "mixed_crop_livestock", "poultry_farm", "other"], "climate_zones": ["tropical_wet", "tropical_dry", "warm_temperate_wet", "warm_temperate_dry", "cool_temperate_wet", "cool_temperate_dry", "boreal_wet", "boreal_dry"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'livestock_management',
    'configuration',
    'Track livestock populations with 20 animal types, head count, average body weight kg, milk yield kg/day, fat percentage, feed digestibility percentage, methane conversion factor Ym%, activity coefficient, and reporting year. Supports Tier 1 and Tier 2 enteric fermentation and manure management calculations.',
    '{"farm_id", "animal_type", "head_count", "avg_body_weight_kg", "milk_yield_kg_day", "fat_pct", "feed_digestibility_pct", "ym_pct", "activity_coefficient", "reporting_year"}',
    '{"population_id", "registration_result"}',
    '{"animal_types": ["dairy_cattle", "non_dairy_cattle", "buffalo", "sheep", "goats", "camels", "horses", "mules_asses", "swine_market", "swine_breeding", "poultry_layers", "poultry_broilers", "turkeys", "ducks", "deer", "elk", "rabbits", "fur_animals", "llamas_alpacas", "other"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'manure_systems',
    'configuration',
    'Manage animal waste management system (AWMS) allocation per animal type with 15 AWMS types, allocation fraction (0-1 summing to 1.0 per animal type), and mean annual temperature for MCF lookup. Supports IPCC Table 10.17 methane conversion factors.',
    '{"farm_id", "livestock_population_id", "awms_type", "allocation_fraction", "mean_temperature_c"}',
    '{"manure_system_id", "allocation_result"}',
    '{"awms_types": ["anaerobic_lagoon", "liquid_slurry", "solid_storage", "dry_lot", "pasture_range_paddock", "daily_spread", "digester", "burned_for_fuel", "deep_bedding_no_mixing", "deep_bedding_active_mixing", "composting_vessel", "composting_static_pile", "composting_intensive_windrow", "composting_passive_windrow", "other"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'feed_characteristics',
    'configuration',
    'Store Tier 2 feed characteristic data per livestock population with gross energy intake MJ/day, digestible energy percentage, crude protein percentage, methane conversion factor Ym%, and feed type description. Enables Tier 2 enteric fermentation calculation using IPCC Eq. 10.21.',
    '{"farm_id", "livestock_population_id", "gross_energy_mj_day", "digestible_energy_pct", "crude_protein_pct", "ym_pct", "feed_type"}',
    '{"feed_id", "registration_result"}',
    '{"supports_tier2": true, "ipcc_equation": "10.21"}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'cropland_inputs',
    'configuration',
    'Record cropland nitrogen and carbon inputs per farm per year. Synthetic nitrogen fertilizer kg, organic nitrogen amendments kg, crop residue nitrogen kg, soil organic matter mineralization nitrogen kg, limestone and dolomite tonnes for liming CO2, urea tonnes for urea CO2, organic soil area in hectares for drainage N2O, and pasture range paddock (PRP) nitrogen deposits with animal type.',
    '{"farm_id", "input_year", "synthetic_n_kg", "organic_n_kg", "crop_residue_n_kg", "som_n_kg", "limestone_tonnes", "dolomite_tonnes", "urea_tonnes", "organic_soil_area_ha", "prp_n_kg", "prp_animal_type"}',
    '{"input_id", "registration_result"}',
    '{"supports_direct_n2o": true, "supports_indirect_n2o": true, "supports_liming_co2": true, "supports_urea_co2": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'rice_fields',
    'configuration',
    'Define rice paddy fields with area in hectares, 7 water regimes (continuously_flooded/intermittent_single/intermittent_multiple/rainfed_regular/rainfed_drought_prone/deep_water/upland), 3 pre-season flooding types (flooded_less_30_days/flooded_more_30_days/not_flooded), cultivation days, crop year, organic amendments JSONB array with type and rate_tonnes_ha, and soil type. Enables IPCC Chapter 5 rice CH4 calculations.',
    '{"farm_id", "field_name", "area_ha", "water_regime", "pre_season_flooding", "cultivation_days", "crop_year", "organic_amendments", "soil_type"}',
    '{"field_id", "registration_result"}',
    '{"water_regimes": ["continuously_flooded", "intermittent_single", "intermittent_multiple", "rainfed_regular", "rainfed_drought_prone", "deep_water", "upland"], "pre_season_flooding": ["flooded_less_30_days", "flooded_more_30_days", "not_flooded"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic agricultural emission calculations using ipcc_tier1, ipcc_tier2, ipcc_tier3, emission_factor, mass_balance, and direct_measurement methods. Supports 6 emission sources (enteric_fermentation/manure_management/rice_cultivation/cropland_direct_n2o/cropland_indirect_n2o/field_burning) with multi-gas CO2/CH4/N2O and GWP sources AR4/AR5/AR6/AR6_20YR. Produces total emissions tCO2e with per-gas breakdown, scope classification, and per-source detail breakdowns with biogenic classification.',
    '{"farm_id", "emission_source", "calculation_method", "gwp_source"}',
    '{"calculation_id", "total_co2e_tonnes", "co2_tonnes", "ch4_tonnes", "n2o_tonnes", "per_source_breakdown", "provenance_hash"}',
    '{"calculation_methods": ["ipcc_tier1", "ipcc_tier2", "ipcc_tier3", "emission_factor", "mass_balance", "direct_measurement"], "emission_sources": ["enteric_fermentation", "manure_management", "rice_cultivation", "cropland_direct_n2o", "cropland_indirect_n2o", "field_burning"], "gwp_sources": ["AR4", "AR5", "AR6", "AR6_20YR"], "gases": ["CO2", "CH4", "N2O"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of agricultural emission calculations against IPCC_2006 (2006 Guidelines Vol 4 Agriculture), IPCC_2019 (2019 Refinement Vol 4), GHG_PROTOCOL (Corporate Standard and Agricultural Guidance), ISO_14064, CSRD_ESRS (European Sustainability Reporting Standards), EPA_40CFR98 (Subpart JJ), DEFRA (Environmental Reporting Guidelines), and CUSTOM frameworks. Produce check results with total requirements, passed/failed counts, findings, and recommendations.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "status", "total_requirements", "passed_checks", "failed_checks", "findings", "recommendations"}',
    '{"frameworks": ["IPCC_2006", "IPCC_2019", "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS", "EPA_40CFR98", "DEFRA", "CUSTOM"], "statuses": ["compliant", "non_compliant", "partial", "not_assessed"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-009', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/COMPLIANCE_CHECK/FEED_UPDATE/MANURE_ALLOCATION), parent_hash/hash_value SHA-256 chaining for tamper-evident provenance, data JSONB payloads, and actor attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT", "COMPLIANCE_CHECK", "FEED_UPDATE", "MANURE_ALLOCATION"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-009', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage agricultural emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-009', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for mass unit conversions (tonnes/kg/lbs), energy unit alignment (MJ/GJ/kWh), and GWP value lookups'),
    ('GL-MRV-SCOPE1-009', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for IPCC/EPA/DEFRA emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-009', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection across calculation methods'),
    ('GL-MRV-SCOPE1-009', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, livestock events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-009', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of livestock populations, feed characteristics, cropland inputs, and emission factors'),
    ('GL-MRV-SCOPE1-009', 'GL-MRV-SCOPE1-006', '>=1.0.0', true, 'Land Use Emissions Agent for cross-referencing land use change carbon stock impacts on agricultural land (AFOLU integration)')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-009',
    'Agricultural Emissions Agent',
    'Agricultural emission calculator. Farm registry (dairy_farm/beef_ranch/mixed_livestock/crop_farm/rice_farm/mixed_crop_livestock/poultry_farm/other, country, region, lat/lon, 8 IPCC climate zones, total/arable/pasture area ha). Livestock populations (20 animal types, head count, body weight, milk yield, fat %, feed digestibility, Ym%, activity coefficient). AWMS allocation (15 types, allocation fraction 0-1, mean temperature). Tier 2 feed characteristics (gross energy MJ/day, digestible energy %, crude protein %, Ym%). IPCC/EPA/DEFRA/UNFCCC/GHG_PROTOCOL emission factors (6 emission sources, CO2/CH4/N2O, geographic scope, climate zone). Cropland inputs (synthetic/organic/residue/SOM N, limestone/dolomite/urea, organic soil, PRP). Rice fields (7 water regimes, 3 pre-season flooding, organic amendments, soil type). Emission calculations (ipcc_tier1/tier2/tier3/emission_factor/mass_balance/direct_measurement, AR4/AR5/AR6/AR6_20YR GWP). Per-gas per-source breakdowns (biogenic flag). Field burning events (12 crop types). Compliance checks (IPCC_2006/IPCC_2019/GHG_PROTOCOL/ISO_14064/CSRD_ESRS/EPA_40CFR98/DEFRA). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA agricultural_emissions_service IS
    'Agricultural Emissions Agent (AGENT-MRV-009) - farm/facility registry, livestock populations, manure systems, feed characteristics, IPCC/EPA/DEFRA emission factors, cropland inputs, rice fields, emission calculations (CO2, CH4, N2O), per-gas per-source breakdowns, field burning events, compliance records, audit trail, provenance chains';

COMMENT ON TABLE agricultural_emissions_service.ag_farms IS
    'Farm/facility registry: tenant_id, farm_name, farm_type (dairy_farm/beef_ranch/mixed_livestock/crop_farm/rice_farm/mixed_crop_livestock/poultry_farm/other), country_code, region, latitude, longitude, climate_zone (tropical_wet/tropical_dry/warm_temperate_wet/warm_temperate_dry/cool_temperate_wet/cool_temperate_dry/boreal_wet/boreal_dry), total_area_ha, arable_area_ha, pasture_area_ha, is_active, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_livestock_populations IS
    'Livestock populations: tenant_id, farm_id (FK), animal_type (20 types: dairy_cattle/non_dairy_cattle/buffalo/sheep/goats/camels/horses/mules_asses/swine_market/swine_breeding/poultry_layers/poultry_broilers/turkeys/ducks/deer/elk/rabbits/fur_animals/llamas_alpacas/other), head_count (>0), avg_body_weight_kg, milk_yield_kg_day, fat_pct, feed_digestibility_pct, ym_pct, activity_coefficient, reporting_year, is_active, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_manure_systems IS
    'AWMS allocation: tenant_id, farm_id (FK), livestock_population_id (FK), awms_type (15 types: anaerobic_lagoon/liquid_slurry/solid_storage/dry_lot/pasture_range_paddock/daily_spread/digester/burned_for_fuel/deep_bedding_no_mixing/deep_bedding_active_mixing/composting_vessel/composting_static_pile/composting_intensive_windrow/composting_passive_windrow/other), allocation_fraction (0-1), mean_temperature_c, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_feed_characteristics IS
    'Tier 2 feed characteristics: tenant_id, farm_id (FK), livestock_population_id (FK), gross_energy_mj_day, digestible_energy_pct, crude_protein_pct, ym_pct, feed_type, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_emission_factors IS
    'IPCC/EPA/DEFRA emission factors: tenant_id (nullable for shared), source (IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/UNFCCC/GHG_PROTOCOL/NATIONAL/CUSTOM), emission_source (enteric_fermentation/manure_management/rice_cultivation/cropland_direct_n2o/cropland_indirect_n2o/field_burning), animal_type, crop_type, gas (CO2/CH4/N2O), factor_value, unit, uncertainty_pct, geographic_scope, climate_zone, valid_from/to, is_active, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_cropland_inputs IS
    'Cropland inputs: tenant_id, farm_id (FK), input_year, synthetic_n_kg, organic_n_kg, crop_residue_n_kg, som_n_kg, limestone_tonnes, dolomite_tonnes, urea_tonnes, organic_soil_area_ha, prp_n_kg, prp_animal_type, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_rice_fields IS
    'Rice paddy fields: tenant_id, farm_id (FK), field_name, area_ha (>0), water_regime (continuously_flooded/intermittent_single/intermittent_multiple/rainfed_regular/rainfed_drought_prone/deep_water/upland), pre_season_flooding (flooded_less_30_days/flooded_more_30_days/not_flooded), cultivation_days, crop_year, organic_amendments JSONB [{type, rate_tonnes_ha}], soil_type, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_calculations IS
    'Emission calculation results: tenant_id, farm_id (FK), calculation_method (ipcc_tier1/ipcc_tier2/ipcc_tier3/emission_factor/mass_balance/direct_measurement), emission_source, gwp_source (AR4/AR5/AR6/AR6_20YR), total_co2e_tonnes, co2_tonnes, ch4_tonnes, n2o_tonnes, scope (SCOPE_1/SCOPE_2/SCOPE_3), provenance_hash (SHA-256), reporting_period, reporting_year, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_calculation_details IS
    'Per-gas per-source calculation breakdown: calculation_id (FK CASCADE), emission_source, gas_type (CO2/CH4/N2O), tonnes, co2e_tonnes, emission_factor_used, emission_factor_source, animal_type, crop_type, awms_type, is_biogenic, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_field_burning_events IS
    'Field burning events: tenant_id, farm_id (FK), crop_type (12 types: rice/wheat/maize/sugarcane/barley/oats/sorghum/millet/cotton/soybean/rapeseed/other), area_burned_ha (>0), crop_yield_tonnes_ha, burn_fraction (0-1), combustion_factor (0-1), burn_date, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_compliance_records IS
    'Regulatory compliance records: tenant_id, farm_id (FK), calculation_id (FK), framework (IPCC_2006/IPCC_2019/GHG_PROTOCOL/ISO_14064/CSRD_ESRS/EPA_40CFR98/DEFRA/CUSTOM), status (compliant/non_compliant/partial/not_assessed), total_requirements, passed_checks, failed_checks, findings JSONB, recommendations JSONB, check_date, metadata JSONB';

COMMENT ON TABLE agricultural_emissions_service.ag_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/COMPLIANCE_CHECK/FEED_UPDATE/MANURE_ALLOCATION), parent_hash, hash_value (SHA-256 chain), actor, data JSONB, timestamp';

COMMENT ON TABLE agricultural_emissions_service.ag_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, farm_id, emission_source, calculation_method, emissions_tco2e, gas, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE agricultural_emissions_service.ag_livestock_events_ts IS
    'TimescaleDB hypertable: livestock events with tenant_id, farm_id, animal_type, head_count, event_type, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE agricultural_emissions_service.ag_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, farm_id, framework, status, check_id, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW agricultural_emissions_service.ag_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by emission_source and calculation_method (total calculations, sum emissions tCO2e, avg emissions tCO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW agricultural_emissions_service.ag_daily_emission_totals IS
    'Continuous aggregate: daily livestock event totals by animal_type and event_type (total events, sum head count, avg head count per day)';
