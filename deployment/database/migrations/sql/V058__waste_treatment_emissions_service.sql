-- =============================================================================
-- V058: Waste Treatment Emissions Service Schema
-- =============================================================================
-- Component: AGENT-MRV-008 (GL-MRV-SCOPE1-008)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- On-site Waste Treatment Emissions Agent (GL-MRV-SCOPE1-008) with
-- capabilities for treatment facility registry management (industrial
-- on-site, municipal treatment, waste-to-energy, composting, anaerobic
-- digestion, MBT, wastewater, chemical treatment, and multi-stream
-- facilities with capacity tracking, methane/energy recovery flags,
-- collection efficiency, flare DRE, and climate zone classification),
-- waste stream definitions (19 waste categories and 15 treatment methods
-- with composition breakdowns, moisture/carbon/fossil carbon/DOC/volatile
-- solids fractions per stream), IPCC/EPA/DEFRA emission factor database
-- (per treatment method, waste category, and gas CO2/CH4/N2O/CO with
-- factor values, units, source versioning from IPCC_2006/IPCC_2019/
-- EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM, geographic scoping, and
-- uncertainty percentages with validity date ranges), treatment event
-- records (per-facility per-stream event-level waste tonnage with
-- treatment temperature and retention time tracking), emission
-- calculation results (per-facility calculation with calculation method,
-- treatment method, GWP source AR4/AR5/AR6, total emissions tCO2e,
-- per-gas breakdown CO2 fossil/biogenic, CH4, N2O, CO, scope
-- classification, SHA-256 provenance hashes, and reporting period),
-- per-gas per-stream calculation detail breakdowns (emission tonnes and
-- tCO2e with emission factor used, source reference, and fossil/biogenic
-- classification), methane recovery tracking (CH4 generated/captured/
-- flared/utilized/vented with collection efficiency, flare DRE, and
-- energy generated in GJ), energy recovery and offset tracking (waste
-- NCV, electricity/heat generated, electric/thermal efficiency, grid
-- emission factors, and displaced emissions tCO2e), regulatory
-- compliance records (IPCC/GHG Protocol/ISO 14064/EU ETS/EPA/DEFRA/
-- CSRD framework checks with total requirements, passed/failed counts,
-- findings, and recommendations), and step-by-step audit trail entries
-- (entity-level action trace with parent_hash/hash_value chaining for
-- tamper-evident provenance and actor attribution).
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. wt_treatment_facilities      - Treatment facility registry (type, capacity, recovery flags)
--   2. wt_waste_streams             - Waste stream definitions (category, method, composition, fractions)
--   3. wt_emission_factors          - IPCC/EPA/DEFRA emission factors (per method, category, gas)
--   4. wt_treatment_events          - Individual treatment events (per-facility per-stream tonnage)
--   5. wt_calculations              - Emission calculation results (total tCO2e, per-gas breakdown)
--   6. wt_calculation_details       - Per-gas per-stream breakdown (emission factor, source)
--   7. wt_methane_recovery          - CH4 recovery tracking (captured, flared, utilized, vented)
--   8. wt_energy_recovery           - Energy recovery and offsets (electricity, heat, displaced emissions)
--   9. wt_compliance_records        - Regulatory compliance checks (IPCC/GHG Protocol/ISO/EU ETS/EPA/DEFRA/CSRD)
--  10. wt_audit_entries             - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  11. wt_calculation_events        - Calculation event time-series (hypertable on event_time)
--  12. wt_treatment_events_ts       - Treatment event time-series (hypertable on event_time)
--  13. wt_compliance_events         - Compliance event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. wt_hourly_calculation_stats  - Hourly count/sum(emissions) by treatment_method and calculation_method
--   2. wt_daily_emission_totals     - Daily count/sum(emissions) by treatment_method and waste_category
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-008.
-- Previous: V057__land_use_emissions_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS waste_treatment_emissions_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION waste_treatment_emissions_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: waste_treatment_emissions_service.wt_treatment_facilities
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_treatment_facilities (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_name           VARCHAR(500)    NOT NULL,
    facility_type           VARCHAR(50)     NOT NULL,
    treatment_methods       JSONB           NOT NULL DEFAULT '[]'::jsonb,
    capacity_tonnes_per_year NUMERIC(15,2),
    location_country        VARCHAR(3),
    location_region         VARCHAR(100),
    climate_zone            VARCHAR(30),
    has_methane_recovery    BOOLEAN         DEFAULT FALSE,
    has_energy_recovery     BOOLEAN         DEFAULT FALSE,
    collection_efficiency   NUMERIC(5,4),
    flare_efficiency        NUMERIC(5,4),
    is_active               BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_facility_name_not_empty CHECK (LENGTH(TRIM(facility_name)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_facility_type CHECK (facility_type IN (
        'industrial_onsite', 'municipal_treatment', 'waste_to_energy',
        'composting_facility', 'ad_plant', 'mbt_plant',
        'wastewater_plant', 'chemical_treatment', 'multi_stream'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_capacity_non_negative CHECK (capacity_tonnes_per_year IS NULL OR capacity_tonnes_per_year >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_climate_zone CHECK (climate_zone IS NULL OR climate_zone IN (
        'tropical', 'subtropical', 'temperate', 'boreal', 'polar'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_collection_efficiency_range CHECK (
        collection_efficiency IS NULL OR (collection_efficiency >= 0 AND collection_efficiency <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_flare_efficiency_range CHECK (
        flare_efficiency IS NULL OR (flare_efficiency >= 0 AND flare_efficiency <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities
    ADD CONSTRAINT chk_tf_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_tf_updated_at
    BEFORE UPDATE ON waste_treatment_emissions_service.wt_treatment_facilities
    FOR EACH ROW EXECUTE FUNCTION waste_treatment_emissions_service.set_updated_at();

-- =============================================================================
-- Table 2: waste_treatment_emissions_service.wt_waste_streams
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_waste_streams (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID            REFERENCES waste_treatment_emissions_service.wt_treatment_facilities(id),
    stream_name             VARCHAR(500)    NOT NULL,
    waste_category          VARCHAR(50)     NOT NULL,
    treatment_method        VARCHAR(50)     NOT NULL,
    composition             JSONB           DEFAULT '{}'::jsonb,
    annual_tonnes           NUMERIC(15,2),
    moisture_content        NUMERIC(5,4),
    carbon_content          NUMERIC(5,4),
    fossil_carbon_fraction  NUMERIC(5,4),
    doc_value               NUMERIC(5,4),
    volatile_solids_fraction NUMERIC(5,4),
    is_active               BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_stream_name_not_empty CHECK (LENGTH(TRIM(stream_name)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_waste_category CHECK (waste_category IN (
        'food_waste', 'garden_waste', 'paper_cardboard', 'wood_waste',
        'textiles', 'plastics', 'rubber', 'industrial_sludge',
        'municipal_solid_waste', 'construction_demolition', 'healthcare_waste',
        'chemical_waste', 'electronic_waste', 'agricultural_waste',
        'sewage_sludge', 'animal_waste', 'mixed_organic', 'hazardous_waste',
        'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_treatment_method CHECK (treatment_method IN (
        'incineration', 'open_burning', 'pyrolysis', 'gasification',
        'anaerobic_digestion', 'composting', 'mechanical_biological',
        'autoclaving', 'chemical_treatment', 'thermal_desorption',
        'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
        'deep_well_injection', 'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_annual_tonnes_non_negative CHECK (annual_tonnes IS NULL OR annual_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_moisture_content_range CHECK (
        moisture_content IS NULL OR (moisture_content >= 0 AND moisture_content <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_carbon_content_range CHECK (
        carbon_content IS NULL OR (carbon_content >= 0 AND carbon_content <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_fossil_carbon_fraction_range CHECK (
        fossil_carbon_fraction IS NULL OR (fossil_carbon_fraction >= 0 AND fossil_carbon_fraction <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_doc_value_range CHECK (
        doc_value IS NULL OR (doc_value >= 0 AND doc_value <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_volatile_solids_fraction_range CHECK (
        volatile_solids_fraction IS NULL OR (volatile_solids_fraction >= 0 AND volatile_solids_fraction <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_waste_streams
    ADD CONSTRAINT chk_ws_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ws_updated_at
    BEFORE UPDATE ON waste_treatment_emissions_service.wt_waste_streams
    FOR EACH ROW EXECUTE FUNCTION waste_treatment_emissions_service.set_updated_at();

-- =============================================================================
-- Table 3: waste_treatment_emissions_service.wt_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100),
    factor_name         VARCHAR(200)    NOT NULL,
    treatment_method    VARCHAR(50)     NOT NULL,
    waste_category      VARCHAR(50),
    gas                 VARCHAR(10)     NOT NULL,
    factor_value        NUMERIC(20,10)  NOT NULL,
    factor_unit         VARCHAR(50)     NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    source_version      VARCHAR(20),
    geographic_scope    VARCHAR(100)    DEFAULT 'global',
    uncertainty_pct     NUMERIC(5,2),
    valid_from          DATE,
    valid_to            DATE,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_factor_name_not_empty CHECK (LENGTH(TRIM(factor_name)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_treatment_method CHECK (treatment_method IN (
        'incineration', 'open_burning', 'pyrolysis', 'gasification',
        'anaerobic_digestion', 'composting', 'mechanical_biological',
        'autoclaving', 'chemical_treatment', 'thermal_desorption',
        'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
        'deep_well_injection', 'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_waste_category CHECK (waste_category IS NULL OR waste_category IN (
        'food_waste', 'garden_waste', 'paper_cardboard', 'wood_waste',
        'textiles', 'plastics', 'rubber', 'industrial_sludge',
        'municipal_solid_waste', 'construction_demolition', 'healthcare_waste',
        'chemical_waste', 'electronic_waste', 'agricultural_waste',
        'sewage_sludge', 'animal_waste', 'mixed_organic', 'hazardous_waste',
        'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'CO'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty CHECK (LENGTH(TRIM(factor_unit)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'IPCC_2006', 'IPCC_2019', 'EPA_AP42', 'DEFRA', 'ECOINVENT', 'NATIONAL', 'CUSTOM'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_uncertainty_pct_range CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    );

ALTER TABLE waste_treatment_emissions_service.wt_emission_factors
    ADD CONSTRAINT chk_ef_date_order CHECK (
        valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from
    );

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON waste_treatment_emissions_service.wt_emission_factors
    FOR EACH ROW EXECUTE FUNCTION waste_treatment_emissions_service.set_updated_at();

-- =============================================================================
-- Table 4: waste_treatment_emissions_service.wt_treatment_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_treatment_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID            REFERENCES waste_treatment_emissions_service.wt_treatment_facilities(id),
    stream_id               UUID            REFERENCES waste_treatment_emissions_service.wt_waste_streams(id),
    event_date              DATE            NOT NULL,
    treatment_method        VARCHAR(50)     NOT NULL,
    waste_category          VARCHAR(50)     NOT NULL,
    waste_tonnes            NUMERIC(15,4)   NOT NULL,
    treatment_temperature   NUMERIC(8,2),
    retention_time_days     NUMERIC(8,2),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_treatment_method CHECK (treatment_method IN (
        'incineration', 'open_burning', 'pyrolysis', 'gasification',
        'anaerobic_digestion', 'composting', 'mechanical_biological',
        'autoclaving', 'chemical_treatment', 'thermal_desorption',
        'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
        'deep_well_injection', 'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_waste_category CHECK (waste_category IN (
        'food_waste', 'garden_waste', 'paper_cardboard', 'wood_waste',
        'textiles', 'plastics', 'rubber', 'industrial_sludge',
        'municipal_solid_waste', 'construction_demolition', 'healthcare_waste',
        'chemical_waste', 'electronic_waste', 'agricultural_waste',
        'sewage_sludge', 'animal_waste', 'mixed_organic', 'hazardous_waste',
        'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_waste_tonnes_positive CHECK (waste_tonnes > 0);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_treatment_temperature_range CHECK (
        treatment_temperature IS NULL OR treatment_temperature >= 0
    );

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_retention_time_days_positive CHECK (
        retention_time_days IS NULL OR retention_time_days > 0
    );

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events
    ADD CONSTRAINT chk_te_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_te_updated_at
    BEFORE UPDATE ON waste_treatment_emissions_service.wt_treatment_events
    FOR EACH ROW EXECUTE FUNCTION waste_treatment_emissions_service.set_updated_at();

-- =============================================================================
-- Table 5: waste_treatment_emissions_service.wt_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID            REFERENCES waste_treatment_emissions_service.wt_treatment_facilities(id),
    calculation_method      VARCHAR(30)     NOT NULL,
    treatment_method        VARCHAR(50)     NOT NULL,
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    waste_category          VARCHAR(50),
    waste_tonnes            NUMERIC(15,4),
    total_emissions_tco2e   NUMERIC(20,8)   NOT NULL,
    fossil_co2_tonnes       NUMERIC(20,8)   DEFAULT 0,
    biogenic_co2_tonnes     NUMERIC(20,8)   DEFAULT 0,
    ch4_tonnes              NUMERIC(20,8)   DEFAULT 0,
    n2o_tonnes              NUMERIC(20,8)   DEFAULT 0,
    co_tonnes               NUMERIC(20,8)   DEFAULT 0,
    scope                   VARCHAR(10)     DEFAULT 'scope_1',
    provenance_hash         VARCHAR(64),
    reporting_year          INTEGER,
    reporting_period        VARCHAR(20)     DEFAULT 'annual',
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'ipcc_default', 'ipcc_first_order', 'mass_balance',
        'stoichiometric', 'direct_measurement', 'emission_factor',
        'continuous_monitoring'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_treatment_method CHECK (treatment_method IN (
        'incineration', 'open_burning', 'pyrolysis', 'gasification',
        'anaerobic_digestion', 'composting', 'mechanical_biological',
        'autoclaving', 'chemical_treatment', 'thermal_desorption',
        'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
        'deep_well_injection', 'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_waste_category CHECK (waste_category IS NULL OR waste_category IN (
        'food_waste', 'garden_waste', 'paper_cardboard', 'wood_waste',
        'textiles', 'plastics', 'rubber', 'industrial_sludge',
        'municipal_solid_waste', 'construction_demolition', 'healthcare_waste',
        'chemical_waste', 'electronic_waste', 'agricultural_waste',
        'sewage_sludge', 'animal_waste', 'mixed_organic', 'hazardous_waste',
        'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_waste_tonnes_non_negative CHECK (waste_tonnes IS NULL OR waste_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_total_emissions_non_negative CHECK (total_emissions_tco2e >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_fossil_co2_non_negative CHECK (fossil_co2_tonnes IS NULL OR fossil_co2_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_biogenic_co2_non_negative CHECK (biogenic_co2_tonnes IS NULL OR biogenic_co2_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_ch4_non_negative CHECK (ch4_tonnes IS NULL OR ch4_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_n2o_non_negative CHECK (n2o_tonnes IS NULL OR n2o_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_co_non_negative CHECK (co_tonnes IS NULL OR co_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_scope CHECK (scope IN (
        'scope_1', 'scope_2', 'scope_3'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_reporting_period CHECK (reporting_period IS NULL OR reporting_period IN (
        'annual', 'quarterly', 'monthly', 'custom'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON waste_treatment_emissions_service.wt_calculations
    FOR EACH ROW EXECUTE FUNCTION waste_treatment_emissions_service.set_updated_at();

-- =============================================================================
-- Table 6: waste_treatment_emissions_service.wt_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id          UUID            NOT NULL REFERENCES waste_treatment_emissions_service.wt_calculations(id) ON DELETE CASCADE,
    tenant_id               VARCHAR(100)    NOT NULL,
    stream_name             VARCHAR(500),
    waste_category          VARCHAR(50),
    treatment_method        VARCHAR(50),
    gas                     VARCHAR(10),
    emission_tonnes         NUMERIC(20,8),
    emission_tco2e          NUMERIC(20,8),
    emission_factor_used    NUMERIC(20,10),
    emission_factor_source  VARCHAR(50),
    is_fossil               BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_waste_category CHECK (waste_category IS NULL OR waste_category IN (
        'food_waste', 'garden_waste', 'paper_cardboard', 'wood_waste',
        'textiles', 'plastics', 'rubber', 'industrial_sludge',
        'municipal_solid_waste', 'construction_demolition', 'healthcare_waste',
        'chemical_waste', 'electronic_waste', 'agricultural_waste',
        'sewage_sludge', 'animal_waste', 'mixed_organic', 'hazardous_waste',
        'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_treatment_method CHECK (treatment_method IS NULL OR treatment_method IN (
        'incineration', 'open_burning', 'pyrolysis', 'gasification',
        'anaerobic_digestion', 'composting', 'mechanical_biological',
        'autoclaving', 'chemical_treatment', 'thermal_desorption',
        'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
        'deep_well_injection', 'other'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IS NULL OR gas IN (
        'CO2', 'CH4', 'N2O', 'CO'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_emission_tonnes_non_negative CHECK (emission_tonnes IS NULL OR emission_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_emission_tco2e_non_negative CHECK (emission_tco2e IS NULL OR emission_tco2e >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_used_non_negative CHECK (emission_factor_used IS NULL OR emission_factor_used >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_source CHECK (emission_factor_source IS NULL OR emission_factor_source IN (
        'IPCC_2006', 'IPCC_2019', 'EPA_AP42', 'DEFRA', 'ECOINVENT', 'NATIONAL', 'CUSTOM'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 7: waste_treatment_emissions_service.wt_methane_recovery
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_methane_recovery (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID            REFERENCES waste_treatment_emissions_service.wt_treatment_facilities(id),
    calculation_id          UUID            REFERENCES waste_treatment_emissions_service.wt_calculations(id),
    event_date              DATE            NOT NULL,
    ch4_generated_tonnes    NUMERIC(20,8),
    ch4_captured_tonnes     NUMERIC(20,8),
    ch4_flared_tonnes       NUMERIC(20,8),
    ch4_utilized_tonnes     NUMERIC(20,8),
    ch4_vented_tonnes       NUMERIC(20,8),
    collection_efficiency   NUMERIC(5,4),
    flare_dre               NUMERIC(5,4),
    energy_generated_gj     NUMERIC(15,4),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_ch4_generated_non_negative CHECK (ch4_generated_tonnes IS NULL OR ch4_generated_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_ch4_captured_non_negative CHECK (ch4_captured_tonnes IS NULL OR ch4_captured_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_ch4_flared_non_negative CHECK (ch4_flared_tonnes IS NULL OR ch4_flared_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_ch4_utilized_non_negative CHECK (ch4_utilized_tonnes IS NULL OR ch4_utilized_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_ch4_vented_non_negative CHECK (ch4_vented_tonnes IS NULL OR ch4_vented_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_collection_efficiency_range CHECK (
        collection_efficiency IS NULL OR (collection_efficiency >= 0 AND collection_efficiency <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_flare_dre_range CHECK (
        flare_dre IS NULL OR (flare_dre >= 0 AND flare_dre <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_energy_generated_non_negative CHECK (energy_generated_gj IS NULL OR energy_generated_gj >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery
    ADD CONSTRAINT chk_mr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: waste_treatment_emissions_service.wt_energy_recovery
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_energy_recovery (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   VARCHAR(100)    NOT NULL,
    facility_id                 UUID            REFERENCES waste_treatment_emissions_service.wt_treatment_facilities(id),
    calculation_id              UUID            REFERENCES waste_treatment_emissions_service.wt_calculations(id),
    event_date                  DATE            NOT NULL,
    waste_tonnes                NUMERIC(15,4),
    ncv_gj_per_tonne            NUMERIC(10,4),
    electricity_generated_gj    NUMERIC(15,4),
    heat_generated_gj           NUMERIC(15,4),
    electric_efficiency         NUMERIC(5,4),
    thermal_efficiency          NUMERIC(5,4),
    grid_ef_electric            NUMERIC(10,6),
    grid_ef_heat                NUMERIC(10,6),
    displaced_emissions_tco2e   NUMERIC(20,8),
    metadata                    JSONB           DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_waste_tonnes_non_negative CHECK (waste_tonnes IS NULL OR waste_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_ncv_non_negative CHECK (ncv_gj_per_tonne IS NULL OR ncv_gj_per_tonne >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_electricity_non_negative CHECK (electricity_generated_gj IS NULL OR electricity_generated_gj >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_heat_non_negative CHECK (heat_generated_gj IS NULL OR heat_generated_gj >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_electric_efficiency_range CHECK (
        electric_efficiency IS NULL OR (electric_efficiency >= 0 AND electric_efficiency <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_thermal_efficiency_range CHECK (
        thermal_efficiency IS NULL OR (thermal_efficiency >= 0 AND thermal_efficiency <= 1)
    );

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_grid_ef_electric_non_negative CHECK (grid_ef_electric IS NULL OR grid_ef_electric >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_grid_ef_heat_non_negative CHECK (grid_ef_heat IS NULL OR grid_ef_heat >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_displaced_emissions_non_negative CHECK (displaced_emissions_tco2e IS NULL OR displaced_emissions_tco2e >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery
    ADD CONSTRAINT chk_er_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 9: waste_treatment_emissions_service.wt_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100)    NOT NULL,
    calculation_id      UUID            REFERENCES waste_treatment_emissions_service.wt_calculations(id),
    framework           VARCHAR(30)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    total_requirements  INTEGER         NOT NULL DEFAULT 0,
    passed              INTEGER         NOT NULL DEFAULT 0,
    failed              INTEGER         NOT NULL DEFAULT 0,
    findings            JSONB           DEFAULT '[]'::jsonb,
    recommendations     JSONB           DEFAULT '[]'::jsonb,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'IPCC', 'GHG_PROTOCOL', 'ISO_14064', 'EU_ETS', 'EPA', 'DEFRA', 'CSRD', 'CUSTOM'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'compliant', 'non_compliant', 'partial', 'not_assessed'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_total_requirements_non_negative CHECK (total_requirements >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_passed_non_negative CHECK (passed >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_failed_non_negative CHECK (failed >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_passed_plus_failed CHECK (passed + failed <= total_requirements);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: waste_treatment_emissions_service.wt_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           VARCHAR(100)    NOT NULL,
    entity_type         VARCHAR(30)     NOT NULL,
    entity_id           VARCHAR(100)    NOT NULL,
    action              VARCHAR(30)     NOT NULL,
    parent_hash         VARCHAR(64),
    hash_value          VARCHAR(64)     NOT NULL,
    actor               VARCHAR(200),
    details             JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE waste_treatment_emissions_service.wt_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_audit_entries
    ADD CONSTRAINT chk_ae_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT',
        'RECOVER_CH4', 'RECOVER_ENERGY', 'COMPLIANCE_CHECK'
    ));

ALTER TABLE waste_treatment_emissions_service.wt_audit_entries
    ADD CONSTRAINT chk_ae_hash_value_not_empty CHECK (LENGTH(TRIM(hash_value)) > 0);

ALTER TABLE waste_treatment_emissions_service.wt_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 11: waste_treatment_emissions_service.wt_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_calculation_events (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID,
    treatment_method        VARCHAR(50),
    waste_category          VARCHAR(50),
    calculation_method      VARCHAR(30),
    emissions_tco2e         NUMERIC(20,8),
    gas                     VARCHAR(10),
    duration_ms             NUMERIC(12,2),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'waste_treatment_emissions_service.wt_calculation_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_calculation_events
    ADD CONSTRAINT chk_cae_treatment_method CHECK (
        treatment_method IS NULL OR treatment_method IN (
            'incineration', 'open_burning', 'pyrolysis', 'gasification',
            'anaerobic_digestion', 'composting', 'mechanical_biological',
            'autoclaving', 'chemical_treatment', 'thermal_desorption',
            'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
            'deep_well_injection', 'other'
        )
    );

ALTER TABLE waste_treatment_emissions_service.wt_calculation_events
    ADD CONSTRAINT chk_cae_gas CHECK (
        gas IS NULL OR gas IN ('CO2', 'CH4', 'N2O', 'CO')
    );

-- =============================================================================
-- Table 12: waste_treatment_emissions_service.wt_treatment_events_ts (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_treatment_events_ts (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    facility_id             UUID,
    treatment_method        VARCHAR(50),
    waste_category          VARCHAR(50),
    waste_tonnes            NUMERIC(15,4),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'waste_treatment_emissions_service.wt_treatment_events_ts',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events_ts
    ADD CONSTRAINT chk_tets_waste_tonnes_non_negative CHECK (waste_tonnes IS NULL OR waste_tonnes >= 0);

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events_ts
    ADD CONSTRAINT chk_tets_treatment_method CHECK (
        treatment_method IS NULL OR treatment_method IN (
            'incineration', 'open_burning', 'pyrolysis', 'gasification',
            'anaerobic_digestion', 'composting', 'mechanical_biological',
            'autoclaving', 'chemical_treatment', 'thermal_desorption',
            'landfill_onsite', 'wastewater_aerobic', 'wastewater_anaerobic',
            'deep_well_injection', 'other'
        )
    );

-- =============================================================================
-- Table 13: waste_treatment_emissions_service.wt_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS waste_treatment_emissions_service.wt_compliance_events (
    event_time              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               VARCHAR(100)    NOT NULL,
    framework               VARCHAR(30),
    status                  VARCHAR(20),
    findings_count          INTEGER,
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'waste_treatment_emissions_service.wt_compliance_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE waste_treatment_emissions_service.wt_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('IPCC', 'GHG_PROTOCOL', 'ISO_14064', 'EU_ETS', 'EPA', 'DEFRA', 'CSRD', 'CUSTOM')
    );

ALTER TABLE waste_treatment_emissions_service.wt_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('compliant', 'non_compliant', 'partial', 'not_assessed')
    );

ALTER TABLE waste_treatment_emissions_service.wt_compliance_events
    ADD CONSTRAINT chk_coe_findings_count_non_negative CHECK (findings_count IS NULL OR findings_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- wt_hourly_calculation_stats: hourly count/sum(emissions) by treatment_method and calculation_method
CREATE MATERIALIZED VIEW waste_treatment_emissions_service.wt_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time)   AS bucket,
    treatment_method,
    calculation_method,
    COUNT(*)                            AS total_calculations,
    SUM(emissions_tco2e)                AS sum_emissions_tco2e,
    AVG(emissions_tco2e)                AS avg_emissions_tco2e,
    AVG(duration_ms)                    AS avg_duration_ms,
    MAX(duration_ms)                    AS max_duration_ms
FROM waste_treatment_emissions_service.wt_calculation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, treatment_method, calculation_method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'waste_treatment_emissions_service.wt_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- wt_daily_emission_totals: daily count/sum(emissions) by treatment_method and waste_category
CREATE MATERIALIZED VIEW waste_treatment_emissions_service.wt_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time)    AS bucket,
    treatment_method,
    waste_category,
    COUNT(*)                            AS total_calculations,
    SUM(emissions_tco2e)                AS sum_emissions_tco2e,
    AVG(emissions_tco2e)                AS avg_emissions_tco2e
FROM waste_treatment_emissions_service.wt_calculation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, treatment_method, waste_category
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'waste_treatment_emissions_service.wt_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- wt_treatment_facilities indexes
CREATE INDEX IF NOT EXISTS idx_wt_tf_tenant_id              ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_tf_facility_name          ON waste_treatment_emissions_service.wt_treatment_facilities(facility_name);
CREATE INDEX IF NOT EXISTS idx_wt_tf_facility_type          ON waste_treatment_emissions_service.wt_treatment_facilities(facility_type);
CREATE INDEX IF NOT EXISTS idx_wt_tf_location_country       ON waste_treatment_emissions_service.wt_treatment_facilities(location_country);
CREATE INDEX IF NOT EXISTS idx_wt_tf_location_region        ON waste_treatment_emissions_service.wt_treatment_facilities(location_region);
CREATE INDEX IF NOT EXISTS idx_wt_tf_climate_zone           ON waste_treatment_emissions_service.wt_treatment_facilities(climate_zone);
CREATE INDEX IF NOT EXISTS idx_wt_tf_has_methane_recovery   ON waste_treatment_emissions_service.wt_treatment_facilities(has_methane_recovery);
CREATE INDEX IF NOT EXISTS idx_wt_tf_has_energy_recovery    ON waste_treatment_emissions_service.wt_treatment_facilities(has_energy_recovery);
CREATE INDEX IF NOT EXISTS idx_wt_tf_is_active              ON waste_treatment_emissions_service.wt_treatment_facilities(is_active);
CREATE INDEX IF NOT EXISTS idx_wt_tf_tenant_type            ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, facility_type);
CREATE INDEX IF NOT EXISTS idx_wt_tf_tenant_country         ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, location_country);
CREATE INDEX IF NOT EXISTS idx_wt_tf_tenant_active          ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_wt_tf_type_country           ON waste_treatment_emissions_service.wt_treatment_facilities(facility_type, location_country);
CREATE INDEX IF NOT EXISTS idx_wt_tf_created_at             ON waste_treatment_emissions_service.wt_treatment_facilities(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tf_updated_at             ON waste_treatment_emissions_service.wt_treatment_facilities(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tf_metadata               ON waste_treatment_emissions_service.wt_treatment_facilities USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_wt_tf_treatment_methods      ON waste_treatment_emissions_service.wt_treatment_facilities USING GIN (treatment_methods);

-- Partial index: active facilities only
CREATE INDEX IF NOT EXISTS idx_wt_tf_active_facilities      ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, facility_type)
    WHERE is_active = TRUE;

-- Partial index: facilities with methane recovery
CREATE INDEX IF NOT EXISTS idx_wt_tf_methane_recovery       ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, facility_type)
    WHERE has_methane_recovery = TRUE;

-- Partial index: facilities with energy recovery
CREATE INDEX IF NOT EXISTS idx_wt_tf_energy_recovery        ON waste_treatment_emissions_service.wt_treatment_facilities(tenant_id, facility_type)
    WHERE has_energy_recovery = TRUE;

-- wt_waste_streams indexes
CREATE INDEX IF NOT EXISTS idx_wt_ws_tenant_id              ON waste_treatment_emissions_service.wt_waste_streams(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_ws_facility_id            ON waste_treatment_emissions_service.wt_waste_streams(facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_ws_waste_category         ON waste_treatment_emissions_service.wt_waste_streams(waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_ws_treatment_method       ON waste_treatment_emissions_service.wt_waste_streams(treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_ws_is_active              ON waste_treatment_emissions_service.wt_waste_streams(is_active);
CREATE INDEX IF NOT EXISTS idx_wt_ws_tenant_facility        ON waste_treatment_emissions_service.wt_waste_streams(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_ws_tenant_category        ON waste_treatment_emissions_service.wt_waste_streams(tenant_id, waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_ws_tenant_method          ON waste_treatment_emissions_service.wt_waste_streams(tenant_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_ws_facility_category      ON waste_treatment_emissions_service.wt_waste_streams(facility_id, waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_ws_facility_method        ON waste_treatment_emissions_service.wt_waste_streams(facility_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_ws_category_method        ON waste_treatment_emissions_service.wt_waste_streams(waste_category, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_ws_created_at             ON waste_treatment_emissions_service.wt_waste_streams(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ws_updated_at             ON waste_treatment_emissions_service.wt_waste_streams(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ws_metadata               ON waste_treatment_emissions_service.wt_waste_streams USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_wt_ws_composition            ON waste_treatment_emissions_service.wt_waste_streams USING GIN (composition);

-- Partial index: active streams only
CREATE INDEX IF NOT EXISTS idx_wt_ws_active_streams         ON waste_treatment_emissions_service.wt_waste_streams(tenant_id, waste_category)
    WHERE is_active = TRUE;

-- wt_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_wt_ef_tenant_id              ON waste_treatment_emissions_service.wt_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_ef_treatment_method       ON waste_treatment_emissions_service.wt_emission_factors(treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_ef_waste_category         ON waste_treatment_emissions_service.wt_emission_factors(waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_ef_gas                    ON waste_treatment_emissions_service.wt_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_wt_ef_source                 ON waste_treatment_emissions_service.wt_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_wt_ef_is_active              ON waste_treatment_emissions_service.wt_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_wt_ef_geographic_scope       ON waste_treatment_emissions_service.wt_emission_factors(geographic_scope);
CREATE INDEX IF NOT EXISTS idx_wt_ef_method_category        ON waste_treatment_emissions_service.wt_emission_factors(treatment_method, waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_ef_method_gas             ON waste_treatment_emissions_service.wt_emission_factors(treatment_method, gas);
CREATE INDEX IF NOT EXISTS idx_wt_ef_method_category_gas    ON waste_treatment_emissions_service.wt_emission_factors(treatment_method, waste_category, gas);
CREATE INDEX IF NOT EXISTS idx_wt_ef_method_source          ON waste_treatment_emissions_service.wt_emission_factors(treatment_method, source);
CREATE INDEX IF NOT EXISTS idx_wt_ef_valid_from             ON waste_treatment_emissions_service.wt_emission_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ef_valid_to               ON waste_treatment_emissions_service.wt_emission_factors(valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ef_created_at             ON waste_treatment_emissions_service.wt_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ef_updated_at             ON waste_treatment_emissions_service.wt_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ef_metadata               ON waste_treatment_emissions_service.wt_emission_factors USING GIN (metadata);

-- Partial index: active factors for lookups
CREATE INDEX IF NOT EXISTS idx_wt_ef_active_factors         ON waste_treatment_emissions_service.wt_emission_factors(treatment_method, waste_category, gas, source)
    WHERE is_active = TRUE;

-- wt_treatment_events indexes
CREATE INDEX IF NOT EXISTS idx_wt_te_tenant_id              ON waste_treatment_emissions_service.wt_treatment_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_te_facility_id            ON waste_treatment_emissions_service.wt_treatment_events(facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_te_stream_id              ON waste_treatment_emissions_service.wt_treatment_events(stream_id);
CREATE INDEX IF NOT EXISTS idx_wt_te_event_date             ON waste_treatment_emissions_service.wt_treatment_events(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_te_treatment_method       ON waste_treatment_emissions_service.wt_treatment_events(treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_te_waste_category         ON waste_treatment_emissions_service.wt_treatment_events(waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_te_tenant_facility        ON waste_treatment_emissions_service.wt_treatment_events(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_te_tenant_date            ON waste_treatment_emissions_service.wt_treatment_events(tenant_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_te_tenant_method          ON waste_treatment_emissions_service.wt_treatment_events(tenant_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_te_facility_date          ON waste_treatment_emissions_service.wt_treatment_events(facility_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_te_facility_method        ON waste_treatment_emissions_service.wt_treatment_events(facility_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_te_method_category        ON waste_treatment_emissions_service.wt_treatment_events(treatment_method, waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_te_created_at             ON waste_treatment_emissions_service.wt_treatment_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_te_updated_at             ON waste_treatment_emissions_service.wt_treatment_events(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_te_metadata               ON waste_treatment_emissions_service.wt_treatment_events USING GIN (metadata);

-- wt_calculations indexes
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_id            ON waste_treatment_emissions_service.wt_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_calc_facility_id          ON waste_treatment_emissions_service.wt_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_calc_calculation_method   ON waste_treatment_emissions_service.wt_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_wt_calc_treatment_method     ON waste_treatment_emissions_service.wt_calculations(treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_calc_gwp_source           ON waste_treatment_emissions_service.wt_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_wt_calc_waste_category       ON waste_treatment_emissions_service.wt_calculations(waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_calc_scope                ON waste_treatment_emissions_service.wt_calculations(scope);
CREATE INDEX IF NOT EXISTS idx_wt_calc_provenance_hash      ON waste_treatment_emissions_service.wt_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_wt_calc_reporting_year       ON waste_treatment_emissions_service.wt_calculations(reporting_year);
CREATE INDEX IF NOT EXISTS idx_wt_calc_reporting_period     ON waste_treatment_emissions_service.wt_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_wt_calc_total_emissions      ON waste_treatment_emissions_service.wt_calculations(total_emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_facility      ON waste_treatment_emissions_service.wt_calculations(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_method        ON waste_treatment_emissions_service.wt_calculations(tenant_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_calc_method   ON waste_treatment_emissions_service.wt_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_year          ON waste_treatment_emissions_service.wt_calculations(tenant_id, reporting_year);
CREATE INDEX IF NOT EXISTS idx_wt_calc_tenant_scope         ON waste_treatment_emissions_service.wt_calculations(tenant_id, scope);
CREATE INDEX IF NOT EXISTS idx_wt_calc_facility_method      ON waste_treatment_emissions_service.wt_calculations(facility_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_calc_created_at           ON waste_treatment_emissions_service.wt_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_calc_updated_at           ON waste_treatment_emissions_service.wt_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_calc_metadata             ON waste_treatment_emissions_service.wt_calculations USING GIN (metadata);

-- wt_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_wt_cd_calculation_id         ON waste_treatment_emissions_service.wt_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_wt_cd_tenant_id              ON waste_treatment_emissions_service.wt_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_cd_waste_category         ON waste_treatment_emissions_service.wt_calculation_details(waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_cd_treatment_method       ON waste_treatment_emissions_service.wt_calculation_details(treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_cd_gas                    ON waste_treatment_emissions_service.wt_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_wt_cd_is_fossil              ON waste_treatment_emissions_service.wt_calculation_details(is_fossil);
CREATE INDEX IF NOT EXISTS idx_wt_cd_calc_gas               ON waste_treatment_emissions_service.wt_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_wt_cd_calc_category          ON waste_treatment_emissions_service.wt_calculation_details(calculation_id, waste_category);
CREATE INDEX IF NOT EXISTS idx_wt_cd_calc_method            ON waste_treatment_emissions_service.wt_calculation_details(calculation_id, treatment_method);
CREATE INDEX IF NOT EXISTS idx_wt_cd_calc_category_gas      ON waste_treatment_emissions_service.wt_calculation_details(calculation_id, waste_category, gas);
CREATE INDEX IF NOT EXISTS idx_wt_cd_created_at             ON waste_treatment_emissions_service.wt_calculation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cd_metadata               ON waste_treatment_emissions_service.wt_calculation_details USING GIN (metadata);

-- Partial index: fossil emissions only
CREATE INDEX IF NOT EXISTS idx_wt_cd_fossil_emissions       ON waste_treatment_emissions_service.wt_calculation_details(calculation_id, waste_category)
    WHERE is_fossil = TRUE;

-- wt_methane_recovery indexes
CREATE INDEX IF NOT EXISTS idx_wt_mr_tenant_id              ON waste_treatment_emissions_service.wt_methane_recovery(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_mr_facility_id            ON waste_treatment_emissions_service.wt_methane_recovery(facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_mr_calculation_id         ON waste_treatment_emissions_service.wt_methane_recovery(calculation_id);
CREATE INDEX IF NOT EXISTS idx_wt_mr_event_date             ON waste_treatment_emissions_service.wt_methane_recovery(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_mr_tenant_facility        ON waste_treatment_emissions_service.wt_methane_recovery(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_mr_tenant_date            ON waste_treatment_emissions_service.wt_methane_recovery(tenant_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_mr_facility_date          ON waste_treatment_emissions_service.wt_methane_recovery(facility_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_mr_created_at             ON waste_treatment_emissions_service.wt_methane_recovery(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_mr_metadata               ON waste_treatment_emissions_service.wt_methane_recovery USING GIN (metadata);

-- wt_energy_recovery indexes
CREATE INDEX IF NOT EXISTS idx_wt_er_tenant_id              ON waste_treatment_emissions_service.wt_energy_recovery(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_er_facility_id            ON waste_treatment_emissions_service.wt_energy_recovery(facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_er_calculation_id         ON waste_treatment_emissions_service.wt_energy_recovery(calculation_id);
CREATE INDEX IF NOT EXISTS idx_wt_er_event_date             ON waste_treatment_emissions_service.wt_energy_recovery(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_er_tenant_facility        ON waste_treatment_emissions_service.wt_energy_recovery(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_wt_er_tenant_date            ON waste_treatment_emissions_service.wt_energy_recovery(tenant_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_er_facility_date          ON waste_treatment_emissions_service.wt_energy_recovery(facility_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_wt_er_created_at             ON waste_treatment_emissions_service.wt_energy_recovery(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_er_metadata               ON waste_treatment_emissions_service.wt_energy_recovery USING GIN (metadata);

-- wt_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_wt_cr_tenant_id              ON waste_treatment_emissions_service.wt_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_cr_calculation_id         ON waste_treatment_emissions_service.wt_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_wt_cr_framework              ON waste_treatment_emissions_service.wt_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_wt_cr_status                 ON waste_treatment_emissions_service.wt_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_wt_cr_checked_at             ON waste_treatment_emissions_service.wt_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cr_tenant_framework       ON waste_treatment_emissions_service.wt_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_wt_cr_tenant_status          ON waste_treatment_emissions_service.wt_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_wt_cr_framework_status       ON waste_treatment_emissions_service.wt_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_wt_cr_tenant_calculation     ON waste_treatment_emissions_service.wt_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_wt_cr_findings               ON waste_treatment_emissions_service.wt_compliance_records USING GIN (findings);
CREATE INDEX IF NOT EXISTS idx_wt_cr_recommendations        ON waste_treatment_emissions_service.wt_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_wt_cr_metadata               ON waste_treatment_emissions_service.wt_compliance_records USING GIN (metadata);

-- wt_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_wt_ae_tenant_id              ON waste_treatment_emissions_service.wt_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_wt_ae_entity_type            ON waste_treatment_emissions_service.wt_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_wt_ae_entity_id              ON waste_treatment_emissions_service.wt_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_wt_ae_action                 ON waste_treatment_emissions_service.wt_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_wt_ae_actor                  ON waste_treatment_emissions_service.wt_audit_entries(actor);
CREATE INDEX IF NOT EXISTS idx_wt_ae_parent_hash            ON waste_treatment_emissions_service.wt_audit_entries(parent_hash);
CREATE INDEX IF NOT EXISTS idx_wt_ae_hash_value             ON waste_treatment_emissions_service.wt_audit_entries(hash_value);
CREATE INDEX IF NOT EXISTS idx_wt_ae_tenant_entity          ON waste_treatment_emissions_service.wt_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_wt_ae_tenant_action          ON waste_treatment_emissions_service.wt_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_wt_ae_created_at             ON waste_treatment_emissions_service.wt_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_wt_ae_details                ON waste_treatment_emissions_service.wt_audit_entries USING GIN (details);

-- wt_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_wt_cae_tenant_id             ON waste_treatment_emissions_service.wt_calculation_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_facility_id           ON waste_treatment_emissions_service.wt_calculation_events(facility_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_treatment_method      ON waste_treatment_emissions_service.wt_calculation_events(treatment_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_waste_category        ON waste_treatment_emissions_service.wt_calculation_events(waste_category, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_calculation_method    ON waste_treatment_emissions_service.wt_calculation_events(calculation_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_tenant_method         ON waste_treatment_emissions_service.wt_calculation_events(tenant_id, treatment_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_tenant_category       ON waste_treatment_emissions_service.wt_calculation_events(tenant_id, waste_category, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_cae_metadata              ON waste_treatment_emissions_service.wt_calculation_events USING GIN (metadata);

-- wt_treatment_events_ts indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_wt_tets_tenant_id            ON waste_treatment_emissions_service.wt_treatment_events_ts(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tets_facility_id          ON waste_treatment_emissions_service.wt_treatment_events_ts(facility_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tets_treatment_method     ON waste_treatment_emissions_service.wt_treatment_events_ts(treatment_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tets_waste_category       ON waste_treatment_emissions_service.wt_treatment_events_ts(waste_category, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tets_tenant_method        ON waste_treatment_emissions_service.wt_treatment_events_ts(tenant_id, treatment_method, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_tets_metadata             ON waste_treatment_emissions_service.wt_treatment_events_ts USING GIN (metadata);

-- wt_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_wt_coe_tenant_id             ON waste_treatment_emissions_service.wt_compliance_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_coe_framework             ON waste_treatment_emissions_service.wt_compliance_events(framework, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_coe_status                ON waste_treatment_emissions_service.wt_compliance_events(status, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_coe_tenant_framework      ON waste_treatment_emissions_service.wt_compliance_events(tenant_id, framework, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_coe_tenant_status         ON waste_treatment_emissions_service.wt_compliance_events(tenant_id, status, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_wt_coe_metadata              ON waste_treatment_emissions_service.wt_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- wt_treatment_facilities: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_treatment_facilities ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_tf_read  ON waste_treatment_emissions_service.wt_treatment_facilities FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_tf_write ON waste_treatment_emissions_service.wt_treatment_facilities FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_waste_streams: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_waste_streams ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_ws_read  ON waste_treatment_emissions_service.wt_waste_streams FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_ws_write ON waste_treatment_emissions_service.wt_waste_streams FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_emission_factors: shared reference data (open read, admin write)
ALTER TABLE waste_treatment_emissions_service.wt_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_ef_read  ON waste_treatment_emissions_service.wt_emission_factors FOR SELECT USING (TRUE);
CREATE POLICY wt_ef_write ON waste_treatment_emissions_service.wt_emission_factors FOR ALL USING (
    current_setting('app.is_admin', true) = 'true'
);

-- wt_treatment_events: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_treatment_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_te_read  ON waste_treatment_emissions_service.wt_treatment_events FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_te_write ON waste_treatment_emissions_service.wt_treatment_events FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_calculations: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_calc_read  ON waste_treatment_emissions_service.wt_calculations FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_calc_write ON waste_treatment_emissions_service.wt_calculations FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_calculation_details: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_cd_read  ON waste_treatment_emissions_service.wt_calculation_details FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_cd_write ON waste_treatment_emissions_service.wt_calculation_details FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_methane_recovery: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_methane_recovery ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_mr_read  ON waste_treatment_emissions_service.wt_methane_recovery FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_mr_write ON waste_treatment_emissions_service.wt_methane_recovery FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_energy_recovery: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_energy_recovery ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_er_read  ON waste_treatment_emissions_service.wt_energy_recovery FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_er_write ON waste_treatment_emissions_service.wt_energy_recovery FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_compliance_records: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_cr_read  ON waste_treatment_emissions_service.wt_compliance_records FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_cr_write ON waste_treatment_emissions_service.wt_compliance_records FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_audit_entries: tenant-isolated
ALTER TABLE waste_treatment_emissions_service.wt_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_ae_read  ON waste_treatment_emissions_service.wt_audit_entries FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY wt_ae_write ON waste_treatment_emissions_service.wt_audit_entries FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- wt_calculation_events: open read/write (time-series telemetry)
ALTER TABLE waste_treatment_emissions_service.wt_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_cae_read  ON waste_treatment_emissions_service.wt_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY wt_cae_write ON waste_treatment_emissions_service.wt_calculation_events FOR ALL   USING (TRUE);

-- wt_treatment_events_ts: open read/write (time-series telemetry)
ALTER TABLE waste_treatment_emissions_service.wt_treatment_events_ts ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_tets_read  ON waste_treatment_emissions_service.wt_treatment_events_ts FOR SELECT USING (TRUE);
CREATE POLICY wt_tets_write ON waste_treatment_emissions_service.wt_treatment_events_ts FOR ALL   USING (TRUE);

-- wt_compliance_events: open read/write (time-series telemetry)
ALTER TABLE waste_treatment_emissions_service.wt_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY wt_coe_read  ON waste_treatment_emissions_service.wt_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY wt_coe_write ON waste_treatment_emissions_service.wt_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA waste_treatment_emissions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA waste_treatment_emissions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA waste_treatment_emissions_service TO greenlang_app;
GRANT SELECT ON waste_treatment_emissions_service.wt_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON waste_treatment_emissions_service.wt_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA waste_treatment_emissions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA waste_treatment_emissions_service TO greenlang_readonly;
GRANT SELECT ON waste_treatment_emissions_service.wt_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON waste_treatment_emissions_service.wt_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA waste_treatment_emissions_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA waste_treatment_emissions_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA waste_treatment_emissions_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'waste-treatment-emissions:read',                    'waste-treatment-emissions', 'read',                    'View all waste treatment emissions service data including facilities, waste streams, emission factors, treatment events, calculations, methane/energy recovery, and compliance records'),
    (gen_random_uuid(), 'waste-treatment-emissions:write',                   'waste-treatment-emissions', 'write',                   'Create, update, and manage all waste treatment emissions service data'),
    (gen_random_uuid(), 'waste-treatment-emissions:execute',                 'waste-treatment-emissions', 'execute',                 'Execute waste treatment emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'waste-treatment-emissions:facilities:read',         'waste-treatment-emissions', 'facilities_read',         'View treatment facility registry with facility type (industrial_onsite/municipal_treatment/waste_to_energy/composting_facility/ad_plant/mbt_plant/wastewater_plant/chemical_treatment/multi_stream), capacity, location, climate zone, methane/energy recovery flags, collection efficiency, and flare DRE'),
    (gen_random_uuid(), 'waste-treatment-emissions:facilities:write',        'waste-treatment-emissions', 'facilities_write',        'Create, update, and manage treatment facility registry entries'),
    (gen_random_uuid(), 'waste-treatment-emissions:streams:read',            'waste-treatment-emissions', 'streams_read',            'View waste stream definitions with 19 waste categories, 15 treatment methods, composition breakdowns, moisture/carbon/fossil carbon/DOC/volatile solids fractions'),
    (gen_random_uuid(), 'waste-treatment-emissions:streams:write',           'waste-treatment-emissions', 'streams_write',           'Create, update, and manage waste stream definitions'),
    (gen_random_uuid(), 'waste-treatment-emissions:factors:read',            'waste-treatment-emissions', 'factors_read',            'View IPCC/EPA/DEFRA emission factors per treatment method, waste category, and gas (CO2/CH4/N2O/CO) with source versioning from IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM, geographic scoping, and uncertainty percentages'),
    (gen_random_uuid(), 'waste-treatment-emissions:factors:write',           'waste-treatment-emissions', 'factors_write',           'Create, update, and manage emission factor entries'),
    (gen_random_uuid(), 'waste-treatment-emissions:events:read',             'waste-treatment-emissions', 'events_read',             'View individual treatment events with per-facility per-stream waste tonnage, treatment temperature, and retention time tracking'),
    (gen_random_uuid(), 'waste-treatment-emissions:events:write',            'waste-treatment-emissions', 'events_write',            'Create, update, and manage treatment event records'),
    (gen_random_uuid(), 'waste-treatment-emissions:calculations:read',       'waste-treatment-emissions', 'calculations_read',       'View emission calculation results with total tCO2e, per-gas breakdown (fossil CO2, biogenic CO2, CH4, N2O, CO), scope classification, provenance hashes, and per-stream detail breakdowns'),
    (gen_random_uuid(), 'waste-treatment-emissions:calculations:write',      'waste-treatment-emissions', 'calculations_write',      'Create and manage emission calculation records'),
    (gen_random_uuid(), 'waste-treatment-emissions:methane-recovery:read',   'waste-treatment-emissions', 'methane_recovery_read',   'View methane recovery tracking with CH4 generated/captured/flared/utilized/vented, collection efficiency, flare DRE, and energy generated in GJ'),
    (gen_random_uuid(), 'waste-treatment-emissions:methane-recovery:write',  'waste-treatment-emissions', 'methane_recovery_write',  'Create and manage methane recovery tracking records'),
    (gen_random_uuid(), 'waste-treatment-emissions:energy-recovery:read',    'waste-treatment-emissions', 'energy_recovery_read',    'View energy recovery and offset tracking with waste NCV, electricity/heat generated, electric/thermal efficiency, grid emission factors, and displaced emissions tCO2e'),
    (gen_random_uuid(), 'waste-treatment-emissions:energy-recovery:write',   'waste-treatment-emissions', 'energy_recovery_write',   'Create and manage energy recovery tracking records'),
    (gen_random_uuid(), 'waste-treatment-emissions:compliance:read',         'waste-treatment-emissions', 'compliance_read',         'View regulatory compliance records for IPCC, GHG Protocol, ISO 14064, EU ETS, EPA, DEFRA, and CSRD with total requirements, passed/failed counts, findings, and recommendations'),
    (gen_random_uuid(), 'waste-treatment-emissions:compliance:execute',      'waste-treatment-emissions', 'compliance_execute',      'Execute regulatory compliance checks against IPCC, GHG Protocol, ISO 14064, EU ETS, EPA, DEFRA, and CSRD frameworks'),
    (gen_random_uuid(), 'waste-treatment-emissions:admin',                   'waste-treatment-emissions', 'admin',                   'Full administrative access to waste treatment emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('waste_treatment_emissions_service.wt_calculation_events',  INTERVAL '365 days');
SELECT add_retention_policy('waste_treatment_emissions_service.wt_treatment_events_ts', INTERVAL '365 days');
SELECT add_retention_policy('waste_treatment_emissions_service.wt_compliance_events',   INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE waste_treatment_emissions_service.wt_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'treatment_method',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('waste_treatment_emissions_service.wt_calculation_events', INTERVAL '30 days');

ALTER TABLE waste_treatment_emissions_service.wt_treatment_events_ts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'treatment_method',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('waste_treatment_emissions_service.wt_treatment_events_ts', INTERVAL '30 days');

ALTER TABLE waste_treatment_emissions_service.wt_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('waste_treatment_emissions_service.wt_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Waste Treatment Emissions Agent (GL-MRV-SCOPE1-008)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-008',
    'On-site Waste Treatment Emissions Agent',
    'On-site waste treatment emission calculator for GreenLang Climate OS. Manages treatment facility registry with facility types (industrial_onsite/municipal_treatment/waste_to_energy/composting_facility/ad_plant/mbt_plant/wastewater_plant/chemical_treatment/multi_stream), capacity in tonnes per year, location (country/region), climate zones (tropical/subtropical/temperate/boreal/polar), methane recovery flags, energy recovery flags, gas collection efficiency (0-1), and flare destruction removal efficiency (DRE 0-1). Maintains waste stream definitions with 19 waste categories (food_waste/garden_waste/paper_cardboard/wood_waste/textiles/plastics/rubber/industrial_sludge/municipal_solid_waste/construction_demolition/healthcare_waste/chemical_waste/electronic_waste/agricultural_waste/sewage_sludge/animal_waste/mixed_organic/hazardous_waste/other) and 15 treatment methods (incineration/open_burning/pyrolysis/gasification/anaerobic_digestion/composting/mechanical_biological/autoclaving/chemical_treatment/thermal_desorption/landfill_onsite/wastewater_aerobic/wastewater_anaerobic/deep_well_injection/other) with composition JSONB breakdowns, moisture content, carbon content, fossil carbon fraction, degradable organic carbon (DOC), and volatile solids fraction. Stores IPCC/EPA/DEFRA emission factor database per treatment method, waste category, and gas (CO2/CH4/N2O/CO) with factor values, units, source versioning from IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM, geographic scoping, uncertainty percentages, and validity date ranges. Records individual treatment events with per-facility per-stream waste tonnage, treatment temperature, and retention time in days. Executes deterministic emission calculations using ipcc_default, ipcc_first_order, mass_balance, stoichiometric, direct_measurement, emission_factor, and continuous_monitoring methods with GWP sources AR4/AR5/AR6, producing total emissions tCO2e with per-gas breakdown (fossil CO2, biogenic CO2, CH4, N2O, CO), scope classification (scope_1/scope_2/scope_3), SHA-256 provenance hashes, and reporting period. Produces per-gas per-stream calculation detail breakdowns with emission tonnes and tCO2e, emission factor used, source reference, and fossil/biogenic classification. Tracks methane recovery with CH4 generated/captured/flared/utilized/vented, collection efficiency, flare DRE, and energy generated in GJ. Tracks energy recovery and offsets with waste NCV (GJ/tonne), electricity/heat generated (GJ), electric/thermal efficiency, grid emission factors, and displaced emissions tCO2e. Checks regulatory compliance against IPCC, GHG Protocol, ISO 14064, EU ETS, EPA, DEFRA, and CSRD frameworks with total requirements, passed/failed counts, findings, and recommendations. Generates entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/RECOVER_CH4/RECOVER_ENERGY/COMPLIANCE_CHECK), parent_hash/hash_value chaining for tamper-evident provenance, and actor attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/waste-treatment-emissions',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-008', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/waste-treatment-emissions-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"waste-treatment", "incineration", "composting", "anaerobic-digestion", "scope-1", "methane-recovery", "energy-recovery", "ghg-protocol", "ipcc", "mrv"}',
    '{"waste-management", "industrial", "municipal", "wastewater", "energy-from-waste", "cross-sector"}',
    'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'treatment_facility_registry',
    'configuration',
    'Register and manage waste treatment facilities with facility type (industrial_onsite/municipal_treatment/waste_to_energy/composting_facility/ad_plant/mbt_plant/wastewater_plant/chemical_treatment/multi_stream), capacity (tonnes/year), location (country/region), climate zone, methane/energy recovery flags, collection efficiency, and flare DRE.',
    '{"facility_name", "facility_type", "treatment_methods", "capacity_tonnes_per_year", "location_country", "location_region", "climate_zone", "has_methane_recovery", "has_energy_recovery", "collection_efficiency", "flare_efficiency"}',
    '{"facility_id", "registration_result"}',
    '{"facility_types": ["industrial_onsite", "municipal_treatment", "waste_to_energy", "composting_facility", "ad_plant", "mbt_plant", "wastewater_plant", "chemical_treatment", "multi_stream"], "climate_zones": ["tropical", "subtropical", "temperate", "boreal", "polar"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'waste_stream_management',
    'configuration',
    'Define and manage waste streams with 19 waste categories, 15 treatment methods, composition JSONB breakdowns by material percentage, annual tonnage, moisture content, carbon content, fossil carbon fraction, degradable organic carbon (DOC), and volatile solids fraction.',
    '{"facility_id", "stream_name", "waste_category", "treatment_method", "composition", "annual_tonnes", "moisture_content", "carbon_content", "fossil_carbon_fraction", "doc_value", "volatile_solids_fraction"}',
    '{"stream_id", "registration_result"}',
    '{"waste_categories": ["food_waste", "garden_waste", "paper_cardboard", "wood_waste", "textiles", "plastics", "rubber", "industrial_sludge", "municipal_solid_waste", "construction_demolition", "healthcare_waste", "chemical_waste", "electronic_waste", "agricultural_waste", "sewage_sludge", "animal_waste", "mixed_organic", "hazardous_waste", "other"], "treatment_methods": ["incineration", "open_burning", "pyrolysis", "gasification", "anaerobic_digestion", "composting", "mechanical_biological", "autoclaving", "chemical_treatment", "thermal_desorption", "landfill_onsite", "wastewater_aerobic", "wastewater_anaerobic", "deep_well_injection", "other"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'emission_factor_management',
    'configuration',
    'Manage IPCC/EPA/DEFRA emission factors per treatment method, waste category, and gas (CO2/CH4/N2O/CO) with factor values, units, source versioning from IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM, geographic scoping, uncertainty percentages, and validity date ranges.',
    '{"factor_name", "treatment_method", "waste_category", "gas", "factor_value", "factor_unit", "source", "source_version", "geographic_scope", "uncertainty_pct", "valid_from", "valid_to"}',
    '{"factor_id", "registration_result"}',
    '{"sources": ["IPCC_2006", "IPCC_2019", "EPA_AP42", "DEFRA", "ECOINVENT", "NATIONAL", "CUSTOM"], "gases": ["CO2", "CH4", "N2O", "CO"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'treatment_event_tracking',
    'processing',
    'Record and manage individual treatment events with per-facility per-stream waste tonnage, treatment method, waste category, treatment temperature, and retention time in days.',
    '{"facility_id", "stream_id", "event_date", "treatment_method", "waste_category", "waste_tonnes", "treatment_temperature", "retention_time_days"}',
    '{"event_id", "tracking_result"}',
    '{"tracks_temperature": true, "tracks_retention_time": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic waste treatment emission calculations using ipcc_default, ipcc_first_order, mass_balance, stoichiometric, direct_measurement, emission_factor, and continuous_monitoring methods. Supports multi-gas CO2/CH4/N2O/CO with GWP sources AR4/AR5/AR6. Produces total emissions tCO2e with per-gas breakdown (fossil CO2, biogenic CO2, CH4, N2O, CO), scope classification, and per-stream detail breakdowns.',
    '{"facility_id", "treatment_method", "waste_category", "waste_tonnes", "calculation_method", "gwp_source"}',
    '{"calculation_id", "total_emissions_tco2e", "fossil_co2_tonnes", "biogenic_co2_tonnes", "ch4_tonnes", "n2o_tonnes", "co_tonnes", "per_stream_breakdown", "provenance_hash"}',
    '{"calculation_methods": ["ipcc_default", "ipcc_first_order", "mass_balance", "stoichiometric", "direct_measurement", "emission_factor", "continuous_monitoring"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O", "CO"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'methane_recovery_tracking',
    'processing',
    'Track methane recovery at treatment facilities with CH4 generated, captured, flared, utilized, and vented quantities, collection efficiency (0-1), flare destruction removal efficiency (DRE 0-1), and energy generated in GJ. Supports net emissions calculation by accounting for recovered methane.',
    '{"facility_id", "calculation_id", "event_date", "ch4_generated_tonnes", "ch4_captured_tonnes", "ch4_flared_tonnes", "ch4_utilized_tonnes", "ch4_vented_tonnes", "collection_efficiency", "flare_dre", "energy_generated_gj"}',
    '{"recovery_id", "net_ch4_vented_tonnes", "energy_generated_gj"}',
    '{"tracks_flaring": true, "tracks_utilization": true, "tracks_energy_generation": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'energy_recovery_tracking',
    'processing',
    'Track energy recovery and offsets from waste treatment with waste NCV (GJ/tonne), electricity and heat generated (GJ), electric and thermal efficiency (0-1), grid emission factors for electricity and heat, and displaced emissions in tCO2e.',
    '{"facility_id", "calculation_id", "event_date", "waste_tonnes", "ncv_gj_per_tonne", "electricity_generated_gj", "heat_generated_gj", "electric_efficiency", "thermal_efficiency", "grid_ef_electric", "grid_ef_heat"}',
    '{"recovery_id", "displaced_emissions_tco2e"}',
    '{"tracks_electricity": true, "tracks_heat": true, "calculates_displaced_emissions": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of waste treatment emission calculations against IPCC (2006/2019 Guidelines), GHG Protocol (Corporate Standard), ISO 14064, EU ETS (Monitoring and Reporting Regulation), EPA (40 CFR Part 98), DEFRA (Environmental Reporting Guidelines), and CSRD frameworks. Produce check results with total requirements, passed/failed counts, findings, and recommendations.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "status", "total_requirements", "passed", "failed", "findings", "recommendations"}',
    '{"frameworks": ["IPCC", "GHG_PROTOCOL", "ISO_14064", "EU_ETS", "EPA", "DEFRA", "CSRD", "CUSTOM"], "statuses": ["compliant", "non_compliant", "partial", "not_assessed"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-008', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with action tracking (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/RECOVER_CH4/RECOVER_ENERGY/COMPLIANCE_CHECK), parent_hash/hash_value SHA-256 chaining for tamper-evident provenance, detail JSONB payloads, and actor attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT", "RECOVER_CH4", "RECOVER_ENERGY", "COMPLIANCE_CHECK"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-008', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage waste treatment emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-008', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for mass unit conversions (tonnes/kg/lbs), energy unit alignment (GJ/MWh/kWh), and GWP value lookups'),
    ('GL-MRV-SCOPE1-008', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for IPCC/EPA/DEFRA emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-008', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection across calculation methods'),
    ('GL-MRV-SCOPE1-008', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, treatment events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-008', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of waste tonnage, composition fractions, and emission factors'),
    ('GL-MRV-SCOPE1-008', 'GL-MRV-SCOPE1-005', '>=1.0.0', true, 'Fugitive Emissions Agent for cross-referencing fugitive methane emissions from waste treatment with LDAR tracking')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-008',
    'On-site Waste Treatment Emissions Agent',
    'On-site waste treatment emission calculator. Treatment facility registry (industrial_onsite/municipal_treatment/waste_to_energy/composting_facility/ad_plant/mbt_plant/wastewater_plant/chemical_treatment/multi_stream, capacity tonnes/year, methane/energy recovery, collection efficiency, flare DRE). Waste streams (19 categories, 15 treatment methods, composition, moisture/carbon/fossil carbon/DOC/volatile solids fractions). IPCC/EPA/DEFRA emission factors (CO2/CH4/N2O/CO, IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM sources). Treatment events (per-facility per-stream tonnage, temperature, retention time). Emission calculations (ipcc_default/ipcc_first_order/mass_balance/stoichiometric/direct_measurement/emission_factor/continuous_monitoring, fossil/biogenic CO2, CH4, N2O, CO, AR4/AR5/AR6 GWP). Per-gas per-stream breakdowns. Methane recovery (CH4 generated/captured/flared/utilized/vented, collection efficiency, flare DRE, energy GJ). Energy recovery (NCV, electricity/heat GJ, efficiency, grid EF, displaced emissions). Compliance checks (IPCC/GHG Protocol/ISO 14064/EU ETS/EPA/DEFRA/CSRD). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA waste_treatment_emissions_service IS
    'On-site Waste Treatment Emissions Agent (AGENT-MRV-008) - treatment facility registry, waste stream definitions, IPCC/EPA/DEFRA emission factors, treatment events, emission calculations (fossil/biogenic CO2, CH4, N2O, CO), per-gas per-stream breakdowns, methane recovery tracking, energy recovery and offsets, compliance records, audit trail, provenance chains';

COMMENT ON TABLE waste_treatment_emissions_service.wt_treatment_facilities IS
    'Treatment facility registry: tenant_id, facility_name, facility_type (industrial_onsite/municipal_treatment/waste_to_energy/composting_facility/ad_plant/mbt_plant/wastewater_plant/chemical_treatment/multi_stream), treatment_methods JSONB, capacity_tonnes_per_year, location_country, location_region, climate_zone (tropical/subtropical/temperate/boreal/polar), has_methane_recovery, has_energy_recovery, collection_efficiency (0-1), flare_efficiency (0-1), is_active, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_waste_streams IS
    'Waste stream definitions: tenant_id, facility_id (FK), stream_name, waste_category (19 types: food_waste/garden_waste/paper_cardboard/wood_waste/textiles/plastics/rubber/industrial_sludge/municipal_solid_waste/construction_demolition/healthcare_waste/chemical_waste/electronic_waste/agricultural_waste/sewage_sludge/animal_waste/mixed_organic/hazardous_waste/other), treatment_method (15 methods), composition JSONB, annual_tonnes, moisture_content, carbon_content, fossil_carbon_fraction, doc_value, volatile_solids_fraction, is_active, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_emission_factors IS
    'IPCC/EPA/DEFRA emission factors: tenant_id (nullable for shared), factor_name, treatment_method, waste_category, gas (CO2/CH4/N2O/CO), factor_value, factor_unit, source (IPCC_2006/IPCC_2019/EPA_AP42/DEFRA/ECOINVENT/NATIONAL/CUSTOM), source_version, geographic_scope, uncertainty_pct, valid_from/to, is_active, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_treatment_events IS
    'Individual treatment events: tenant_id, facility_id (FK), stream_id (FK), event_date, treatment_method, waste_category, waste_tonnes (>0), treatment_temperature, retention_time_days, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_calculations IS
    'Emission calculation results: tenant_id, facility_id (FK), calculation_method (ipcc_default/ipcc_first_order/mass_balance/stoichiometric/direct_measurement/emission_factor/continuous_monitoring), treatment_method, gwp_source (AR4/AR5/AR6), waste_category, waste_tonnes, total_emissions_tco2e, fossil_co2_tonnes, biogenic_co2_tonnes, ch4_tonnes, n2o_tonnes, co_tonnes, scope (scope_1/scope_2/scope_3), provenance_hash (SHA-256), reporting_year, reporting_period, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_calculation_details IS
    'Per-gas per-stream calculation breakdown: calculation_id (FK CASCADE), tenant_id, stream_name, waste_category, treatment_method, gas (CO2/CH4/N2O/CO), emission_tonnes, emission_tco2e, emission_factor_used, emission_factor_source, is_fossil, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_methane_recovery IS
    'Methane recovery tracking: tenant_id, facility_id (FK), calculation_id (FK), event_date, ch4_generated_tonnes, ch4_captured_tonnes, ch4_flared_tonnes, ch4_utilized_tonnes, ch4_vented_tonnes, collection_efficiency (0-1), flare_dre (0-1), energy_generated_gj, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_energy_recovery IS
    'Energy recovery and offsets: tenant_id, facility_id (FK), calculation_id (FK), event_date, waste_tonnes, ncv_gj_per_tonne, electricity_generated_gj, heat_generated_gj, electric_efficiency (0-1), thermal_efficiency (0-1), grid_ef_electric, grid_ef_heat, displaced_emissions_tco2e, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_compliance_records IS
    'Regulatory compliance records: tenant_id, calculation_id (FK), framework (IPCC/GHG_PROTOCOL/ISO_14064/EU_ETS/EPA/DEFRA/CSRD/CUSTOM), status (compliant/non_compliant/partial/not_assessed), total_requirements, passed, failed, findings JSONB, recommendations JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT/RECOVER_CH4/RECOVER_ENERGY/COMPLIANCE_CHECK), parent_hash, hash_value (SHA-256 chain), actor, details JSONB';

COMMENT ON TABLE waste_treatment_emissions_service.wt_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, facility_id, treatment_method, waste_category, calculation_method, emissions_tco2e, gas, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE waste_treatment_emissions_service.wt_treatment_events_ts IS
    'TimescaleDB hypertable: treatment events with tenant_id, facility_id, treatment_method, waste_category, waste_tonnes, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE waste_treatment_emissions_service.wt_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, framework, status, findings_count, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW waste_treatment_emissions_service.wt_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by treatment_method and calculation_method (total calculations, sum emissions tCO2e, avg emissions tCO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW waste_treatment_emissions_service.wt_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by treatment_method and waste_category (total calculations, sum emissions tCO2e, avg emissions tCO2e per day)';
