-- =============================================================================
-- V053: Mobile Combustion Service Schema
-- =============================================================================
-- Component: AGENT-MRV-003 (GL-MRV-SCOPE1-003)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Mobile Combustion Agent (GL-MRV-SCOPE1-003) with capabilities for
-- vehicle type registry management (ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL
-- categories with fuel economy defaults, weight class, and emission standard
-- tracking), fuel type registry (gasoline/diesel/CNG/LNG/LPG/biofuel/jet
-- fuel/marine fuel with density, heating values MJ/L and MJ/kg, per-gas
-- emission factors CO2/CH4/N2O in kg/L, biofuel fraction, renewable
-- fraction, and source year tracking from EPA/IPCC/DEFRA/EU_ETS/CUSTOM),
-- tiered emission factor database (vehicle type x fuel type x gas with
-- Tier 1/2/3 factors, model year ranges, emission control technology, and
-- source references), vehicle registry (VIN, make, model, year, fuel type,
-- emission control, department, location, odometer, annual mileage estimate,
-- acquisition/disposal dates), trip logging (distance, fuel consumed, start/
-- end time and location, purpose, load factor, passengers, cargo tonnes),
-- mobile combustion emission calculations (fuel-based and distance-based
-- methods converting fuel quantity or distance to CO2e with multi-gas GWP
-- weighting CO2/CH4/N2O, biogenic CO2 tracking, and provenance hashing),
-- per-gas calculation detail breakdowns (individual emission factors, GWP
-- values, and biogenic flags), fleet-level aggregations (total CO2e with
-- breakdowns by vehicle type, fuel type, and department, plus intensity
-- metrics g CO2e/km), regulatory compliance records (GHG Protocol/EPA/
-- DEFRA/EU ETS/ISO 14064 framework checks with findings and
-- recommendations), and step-by-step audit trail entries (entity type,
-- action, input/output hashes, previous hash chaining).
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. mc_vehicle_types          - Vehicle type registry (ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL)
--   2. mc_fuel_types             - Fuel type registry (density, heating values, per-gas EFs, biofuel fraction)
--   3. mc_emission_factors       - Emission factor database (vehicle x fuel x gas, Tier 1/2/3, model year range)
--   4. mc_vehicle_registry       - Vehicle fleet registry (VIN, make, model, year, department, odometer)
--   5. mc_trips                  - Trip records (distance, fuel consumed, start/end, purpose, load factor)
--   6. mc_calculations           - Calculation results (fuel/distance to CO2e with provenance)
--   7. mc_calculation_details    - Per-gas breakdown (CO2/CH4/N2O with individual EFs and GWP)
--   8. mc_fleet_aggregations     - Fleet roll-ups (by vehicle type/fuel type/department, intensity g/km)
--   9. mc_compliance_records     - Regulatory compliance (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064)
--  10. mc_audit_entries          - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  11. mc_calculation_events     - Calculation event time-series (hypertable on time)
--  12. mc_trip_events            - Trip event time-series (hypertable on time)
--  13. mc_compliance_events      - Compliance event time-series (hypertable on time)
--
-- Continuous Aggregates (2):
--   1. mc_hourly_calculation_stats  - Hourly count/sum(co2e)/avg(co2e) by vehicle_type and method
--   2. mc_daily_emission_totals     - Daily count/sum(co2e) by vehicle_type and fuel_type
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-003.
-- Previous: V052__refrigerants_fgas_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS mobile_combustion_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION mobile_combustion_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: mobile_combustion_service.mc_vehicle_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_vehicle_types (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID          NOT NULL,
    vehicle_type            VARCHAR(100)  NOT NULL,
    category                VARCHAR(50)   NOT NULL,
    display_name            VARCHAR(200)  NOT NULL,
    description             TEXT,
    default_fuel_type       VARCHAR(100),
    default_fuel_economy    DECIMAL(12,4),
    default_fuel_economy_unit VARCHAR(20),
    weight_class            VARCHAR(50),
    emission_standard       VARCHAR(50),
    is_active               BOOLEAN       DEFAULT TRUE,
    metadata                JSONB         DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_vt_tenant_vehicle_type UNIQUE (tenant_id, vehicle_type)
);

ALTER TABLE mobile_combustion_service.mc_vehicle_types
    ADD CONSTRAINT chk_vt_vehicle_type_not_empty CHECK (LENGTH(TRIM(vehicle_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_types
    ADD CONSTRAINT chk_vt_display_name_not_empty CHECK (LENGTH(TRIM(display_name)) > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_types
    ADD CONSTRAINT chk_vt_category CHECK (category IN (
        'ON_ROAD', 'OFF_ROAD', 'MARINE', 'AVIATION', 'RAIL'
    ));

ALTER TABLE mobile_combustion_service.mc_vehicle_types
    ADD CONSTRAINT chk_vt_default_fuel_economy_positive CHECK (default_fuel_economy IS NULL OR default_fuel_economy > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_types
    ADD CONSTRAINT chk_vt_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_vt_updated_at
    BEFORE UPDATE ON mobile_combustion_service.mc_vehicle_types
    FOR EACH ROW EXECUTE FUNCTION mobile_combustion_service.set_updated_at();

-- =============================================================================
-- Table 2: mobile_combustion_service.mc_fuel_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_fuel_types (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    fuel_type           VARCHAR(100)    NOT NULL,
    display_name        VARCHAR(200),
    density_kg_per_l    DECIMAL(10,6),
    heating_value_mj_per_l  DECIMAL(10,4),
    heating_value_mj_per_kg DECIMAL(10,4),
    co2_ef_kg_per_l     DECIMAL(12,8),
    ch4_ef_kg_per_l     DECIMAL(12,8),
    n2o_ef_kg_per_l     DECIMAL(12,8),
    biofuel_fraction    DECIMAL(6,4)    DEFAULT 0,
    renewable_fraction  DECIMAL(6,4)    DEFAULT 0,
    is_biofuel          BOOLEAN         DEFAULT FALSE,
    source              VARCHAR(50),
    source_year         INTEGER,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_ft_tenant_fuel_type UNIQUE (tenant_id, fuel_type)
);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_density_positive CHECK (density_kg_per_l IS NULL OR density_kg_per_l > 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_heating_value_l_positive CHECK (heating_value_mj_per_l IS NULL OR heating_value_mj_per_l > 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_heating_value_kg_positive CHECK (heating_value_mj_per_kg IS NULL OR heating_value_mj_per_kg > 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_co2_ef_non_negative CHECK (co2_ef_kg_per_l IS NULL OR co2_ef_kg_per_l >= 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_ch4_ef_non_negative CHECK (ch4_ef_kg_per_l IS NULL OR ch4_ef_kg_per_l >= 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_n2o_ef_non_negative CHECK (n2o_ef_kg_per_l IS NULL OR n2o_ef_kg_per_l >= 0);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_biofuel_fraction_range CHECK (biofuel_fraction >= 0 AND biofuel_fraction <= 1);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_renewable_fraction_range CHECK (renewable_fraction >= 0 AND renewable_fraction <= 1);

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_source CHECK (source IS NULL OR source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_source_year_reasonable CHECK (source_year IS NULL OR (source_year >= 1990 AND source_year <= 2100));

ALTER TABLE mobile_combustion_service.mc_fuel_types
    ADD CONSTRAINT chk_ft_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ft_updated_at
    BEFORE UPDATE ON mobile_combustion_service.mc_fuel_types
    FOR EACH ROW EXECUTE FUNCTION mobile_combustion_service.set_updated_at();

-- =============================================================================
-- Table 3: mobile_combustion_service.mc_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    vehicle_type        VARCHAR(100)    NOT NULL,
    fuel_type           VARCHAR(100)    NOT NULL,
    gas                 VARCHAR(10)     NOT NULL,
    factor_value        DECIMAL(16,10)  NOT NULL,
    factor_unit         VARCHAR(50)     NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    source_year         INTEGER,
    model_year_start    INTEGER,
    model_year_end      INTEGER,
    emission_control    VARCHAR(100),
    tier                VARCHAR(20),
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_vehicle_type_not_empty CHECK (LENGTH(TRIM(vehicle_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN ('CO2', 'CH4', 'N2O'));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty CHECK (LENGTH(TRIM(factor_unit)) > 0);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_source_year_reasonable CHECK (source_year IS NULL OR (source_year >= 1990 AND source_year <= 2100));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_model_year_start_reasonable CHECK (model_year_start IS NULL OR (model_year_start >= 1900 AND model_year_start <= 2100));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_model_year_end_reasonable CHECK (model_year_end IS NULL OR (model_year_end >= 1900 AND model_year_end <= 2100));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_model_year_order CHECK (model_year_end IS NULL OR model_year_start IS NULL OR model_year_end >= model_year_start);

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_tier CHECK (tier IS NULL OR tier IN ('TIER_1', 'TIER_2', 'TIER_3'));

ALTER TABLE mobile_combustion_service.mc_emission_factors
    ADD CONSTRAINT chk_ef_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 4: mobile_combustion_service.mc_vehicle_registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_vehicle_registry (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID          NOT NULL,
    vehicle_id              VARCHAR(100)  NOT NULL,
    vin                     VARCHAR(50),
    make                    VARCHAR(100),
    model                   VARCHAR(100),
    year                    INTEGER,
    vehicle_type            VARCHAR(100)  NOT NULL,
    fuel_type               VARCHAR(100)  NOT NULL,
    emission_control        VARCHAR(100),
    department              VARCHAR(200),
    location                VARCHAR(200),
    status                  VARCHAR(20)   DEFAULT 'ACTIVE',
    acquisition_date        DATE,
    disposal_date           DATE,
    odometer_reading        DECIMAL(12,2),
    annual_mileage_estimate DECIMAL(10,2),
    metadata                JSONB         DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_vr_tenant_vehicle_id UNIQUE (tenant_id, vehicle_id)
);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_vehicle_id_not_empty CHECK (LENGTH(TRIM(vehicle_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_vehicle_type_not_empty CHECK (LENGTH(TRIM(vehicle_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_year_reasonable CHECK (year IS NULL OR (year >= 1900 AND year <= 2100));

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_status CHECK (status IN (
        'ACTIVE', 'INACTIVE', 'DECOMMISSIONED', 'MAINTENANCE', 'DISPOSED'
    ));

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_disposal_after_acquisition CHECK (
        disposal_date IS NULL OR acquisition_date IS NULL OR disposal_date >= acquisition_date
    );

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_odometer_non_negative CHECK (odometer_reading IS NULL OR odometer_reading >= 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_annual_mileage_non_negative CHECK (annual_mileage_estimate IS NULL OR annual_mileage_estimate >= 0);

ALTER TABLE mobile_combustion_service.mc_vehicle_registry
    ADD CONSTRAINT chk_vr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_vr_updated_at
    BEFORE UPDATE ON mobile_combustion_service.mc_vehicle_registry
    FOR EACH ROW EXECUTE FUNCTION mobile_combustion_service.set_updated_at();

-- =============================================================================
-- Table 5: mobile_combustion_service.mc_trips
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_trips (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    trip_id             VARCHAR(100)    NOT NULL,
    vehicle_id          VARCHAR(100)    NOT NULL,
    distance_km         DECIMAL(12,4),
    distance_unit       VARCHAR(20)     DEFAULT 'KM',
    fuel_consumed_liters DECIMAL(12,4),
    fuel_type           VARCHAR(100),
    start_time          TIMESTAMPTZ,
    end_time            TIMESTAMPTZ,
    start_location      VARCHAR(500),
    end_location        VARCHAR(500),
    purpose             VARCHAR(200),
    load_factor         DECIMAL(6,4),
    passengers          INTEGER,
    cargo_tonnes        DECIMAL(10,4),
    status              VARCHAR(20)     DEFAULT 'COMPLETED',
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_tr_tenant_trip_id UNIQUE (tenant_id, trip_id)
);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_trip_id_not_empty CHECK (LENGTH(TRIM(trip_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_vehicle_id_not_empty CHECK (LENGTH(TRIM(vehicle_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_distance_non_negative CHECK (distance_km IS NULL OR distance_km >= 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_distance_unit CHECK (distance_unit IS NULL OR distance_unit IN (
        'KM', 'MI', 'NM'
    ));

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_fuel_consumed_non_negative CHECK (fuel_consumed_liters IS NULL OR fuel_consumed_liters >= 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_time_order CHECK (
        end_time IS NULL OR start_time IS NULL OR end_time >= start_time
    );

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_load_factor_range CHECK (load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1));

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_passengers_non_negative CHECK (passengers IS NULL OR passengers >= 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_cargo_non_negative CHECK (cargo_tonnes IS NULL OR cargo_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_status CHECK (status IN (
        'PLANNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED'
    ));

ALTER TABLE mobile_combustion_service.mc_trips
    ADD CONSTRAINT chk_tr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 6: mobile_combustion_service.mc_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          VARCHAR(100)    NOT NULL,
    vehicle_type            VARCHAR(100)    NOT NULL,
    fuel_type               VARCHAR(100)    NOT NULL,
    calculation_method      VARCHAR(50)     NOT NULL,
    calculation_tier        VARCHAR(20),
    fuel_quantity           DECIMAL(16,8),
    fuel_unit               VARCHAR(20),
    distance_km             DECIMAL(12,4),
    total_co2e_kg           DECIMAL(16,8)   NOT NULL,
    total_co2e_tonnes       DECIMAL(16,10),
    biogenic_co2_kg         DECIMAL(16,8)   DEFAULT 0,
    gwp_source              VARCHAR(20)     NOT NULL,
    provenance_hash         VARCHAR(128)    NOT NULL,
    vehicle_id              VARCHAR(100),
    trip_id                 VARCHAR(100),
    period_start            TIMESTAMPTZ,
    period_end              TIMESTAMPTZ,
    status                  VARCHAR(20)     DEFAULT 'COMPLETED',
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_calc_tenant_calculation_id UNIQUE (tenant_id, calculation_id)
);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_calculation_id_not_empty CHECK (LENGTH(TRIM(calculation_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_vehicle_type_not_empty CHECK (LENGTH(TRIM(vehicle_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'FUEL_BASED', 'DISTANCE_BASED', 'SPEND_BASED', 'HYBRID', 'DIRECT_MEASUREMENT'
    ));

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_calculation_tier CHECK (calculation_tier IS NULL OR calculation_tier IN ('TIER_1', 'TIER_2', 'TIER_3'));

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_fuel_quantity_non_negative CHECK (fuel_quantity IS NULL OR fuel_quantity >= 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_distance_non_negative CHECK (distance_km IS NULL OR distance_km >= 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_total_co2e_kg_non_negative CHECK (total_co2e_kg >= 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_total_co2e_tonnes_non_negative CHECK (total_co2e_tonnes IS NULL OR total_co2e_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_biogenic_non_negative CHECK (biogenic_co2_kg IS NULL OR biogenic_co2_kg >= 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_period_order CHECK (period_end IS NULL OR period_start IS NULL OR period_end >= period_start);

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_status CHECK (status IN (
        'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    ));

ALTER TABLE mobile_combustion_service.mc_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 7: mobile_combustion_service.mc_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          VARCHAR(100)    NOT NULL,
    gas                     VARCHAR(10)     NOT NULL,
    emissions_kg            DECIMAL(16,8)   NOT NULL,
    emissions_tco2e         DECIMAL(16,10)  NOT NULL,
    emission_factor         DECIMAL(16,10)  NOT NULL,
    emission_factor_unit    VARCHAR(50)     NOT NULL,
    gwp_applied             DECIMAL(8,2)    NOT NULL,
    is_biogenic             BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_calculation_id_not_empty CHECK (LENGTH(TRIM(calculation_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN ('CO2', 'CH4', 'N2O'));

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_emissions_kg_non_negative CHECK (emissions_kg >= 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_emissions_tco2e_non_negative CHECK (emissions_tco2e >= 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_non_negative CHECK (emission_factor >= 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_unit_not_empty CHECK (LENGTH(TRIM(emission_factor_unit)) > 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_gwp_positive CHECK (gwp_applied > 0);

ALTER TABLE mobile_combustion_service.mc_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 8: mobile_combustion_service.mc_fleet_aggregations
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_fleet_aggregations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    aggregation_id              VARCHAR(100)    NOT NULL,
    period_start                TIMESTAMPTZ     NOT NULL,
    period_end                  TIMESTAMPTZ     NOT NULL,
    total_co2e_tonnes           DECIMAL(16,10)  NOT NULL,
    total_co2_tonnes            DECIMAL(16,10),
    total_ch4_tonnes            DECIMAL(16,10),
    total_n2o_tonnes            DECIMAL(16,10),
    total_biogenic_tonnes       DECIMAL(16,10),
    vehicle_count               INTEGER,
    trip_count                  INTEGER,
    total_distance_km           DECIMAL(16,4),
    total_fuel_liters           DECIMAL(16,4),
    intensity_g_co2e_per_km     DECIMAL(12,6),
    breakdown_by_vehicle_type   JSONB           DEFAULT '{}'::jsonb,
    breakdown_by_fuel_type      JSONB           DEFAULT '{}'::jsonb,
    breakdown_by_department     JSONB           DEFAULT '{}'::jsonb,
    provenance_hash             VARCHAR(128)    NOT NULL,
    metadata                    JSONB           DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_mc_fa_tenant_aggregation_id UNIQUE (tenant_id, aggregation_id)
);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_aggregation_id_not_empty CHECK (LENGTH(TRIM(aggregation_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_period_order CHECK (period_end >= period_start);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_co2e_non_negative CHECK (total_co2e_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_co2_non_negative CHECK (total_co2_tonnes IS NULL OR total_co2_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_ch4_non_negative CHECK (total_ch4_tonnes IS NULL OR total_ch4_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_n2o_non_negative CHECK (total_n2o_tonnes IS NULL OR total_n2o_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_biogenic_non_negative CHECK (total_biogenic_tonnes IS NULL OR total_biogenic_tonnes >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_vehicle_count_non_negative CHECK (vehicle_count IS NULL OR vehicle_count >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_trip_count_non_negative CHECK (trip_count IS NULL OR trip_count >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_distance_non_negative CHECK (total_distance_km IS NULL OR total_distance_km >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_total_fuel_non_negative CHECK (total_fuel_liters IS NULL OR total_fuel_liters >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_intensity_non_negative CHECK (intensity_g_co2e_per_km IS NULL OR intensity_g_co2e_per_km >= 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE mobile_combustion_service.mc_fleet_aggregations
    ADD CONSTRAINT chk_fa_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 9: mobile_combustion_service.mc_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    framework           VARCHAR(50)     NOT NULL,
    status              VARCHAR(20)     NOT NULL,
    findings            JSONB           DEFAULT '[]'::jsonb,
    recommendations     JSONB           DEFAULT '[]'::jsonb,
    calculation_ids     JSONB           DEFAULT '[]'::jsonb,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

ALTER TABLE mobile_combustion_service.mc_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM'
    ));

ALTER TABLE mobile_combustion_service.mc_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE mobile_combustion_service.mc_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 10: mobile_combustion_service.mc_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           VARCHAR(200)    NOT NULL,
    action              VARCHAR(50)     NOT NULL,
    input_hash          VARCHAR(128),
    output_hash         VARCHAR(128),
    previous_hash       VARCHAR(128),
    step_name           VARCHAR(200),
    details             JSONB           DEFAULT '{}'::jsonb,
    user_id             VARCHAR(100),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE mobile_combustion_service.mc_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE mobile_combustion_service.mc_audit_entries
    ADD CONSTRAINT chk_ae_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);

ALTER TABLE mobile_combustion_service.mc_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT'
    ));

ALTER TABLE mobile_combustion_service.mc_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: mobile_combustion_service.mc_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_calculation_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    vehicle_type        VARCHAR(100),
    fuel_type           VARCHAR(100),
    method              VARCHAR(50),
    emissions_kg_co2e   DECIMAL(16,8),
    status              VARCHAR(20),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'mobile_combustion_service.mc_calculation_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE mobile_combustion_service.mc_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_kg_co2e IS NULL OR emissions_kg_co2e >= 0);

ALTER TABLE mobile_combustion_service.mc_calculation_events
    ADD CONSTRAINT chk_cae_method CHECK (
        method IS NULL OR method IN ('FUEL_BASED', 'DISTANCE_BASED', 'SPEND_BASED', 'HYBRID', 'DIRECT_MEASUREMENT')
    );

ALTER TABLE mobile_combustion_service.mc_calculation_events
    ADD CONSTRAINT chk_cae_status CHECK (
        status IS NULL OR status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')
    );

-- =============================================================================
-- Table 12: mobile_combustion_service.mc_trip_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_trip_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    vehicle_id          VARCHAR(100),
    vehicle_type        VARCHAR(100),
    distance_km         DECIMAL(12,4),
    fuel_liters         DECIMAL(12,4),
    purpose             VARCHAR(200),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'mobile_combustion_service.mc_trip_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE mobile_combustion_service.mc_trip_events
    ADD CONSTRAINT chk_tre_distance_non_negative CHECK (distance_km IS NULL OR distance_km >= 0);

ALTER TABLE mobile_combustion_service.mc_trip_events
    ADD CONSTRAINT chk_tre_fuel_non_negative CHECK (fuel_liters IS NULL OR fuel_liters >= 0);

-- =============================================================================
-- Table 13: mobile_combustion_service.mc_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS mobile_combustion_service.mc_compliance_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    framework           VARCHAR(50),
    status              VARCHAR(20),
    findings_count      INTEGER,
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'mobile_combustion_service.mc_compliance_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE mobile_combustion_service.mc_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM')
    );

ALTER TABLE mobile_combustion_service.mc_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE mobile_combustion_service.mc_compliance_events
    ADD CONSTRAINT chk_coe_findings_count_non_negative CHECK (findings_count IS NULL OR findings_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- mc_hourly_calculation_stats: hourly count/sum(co2e)/avg(co2e) by vehicle_type and method
CREATE MATERIALIZED VIEW mobile_combustion_service.mc_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    vehicle_type,
    method,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e,
    AVG(emissions_kg_co2e)          AS avg_emissions_kg_co2e
FROM mobile_combustion_service.mc_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, vehicle_type, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'mobile_combustion_service.mc_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- mc_daily_emission_totals: daily count/sum(co2e) by vehicle_type and fuel_type
CREATE MATERIALIZED VIEW mobile_combustion_service.mc_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    vehicle_type,
    fuel_type,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e
FROM mobile_combustion_service.mc_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, vehicle_type, fuel_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'mobile_combustion_service.mc_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- mc_vehicle_types indexes (10)
CREATE INDEX IF NOT EXISTS idx_mc_vt_tenant_id              ON mobile_combustion_service.mc_vehicle_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_vt_vehicle_type           ON mobile_combustion_service.mc_vehicle_types(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_vt_category               ON mobile_combustion_service.mc_vehicle_types(category);
CREATE INDEX IF NOT EXISTS idx_mc_vt_display_name           ON mobile_combustion_service.mc_vehicle_types(display_name);
CREATE INDEX IF NOT EXISTS idx_mc_vt_default_fuel_type      ON mobile_combustion_service.mc_vehicle_types(default_fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_vt_weight_class           ON mobile_combustion_service.mc_vehicle_types(weight_class);
CREATE INDEX IF NOT EXISTS idx_mc_vt_emission_standard      ON mobile_combustion_service.mc_vehicle_types(emission_standard);
CREATE INDEX IF NOT EXISTS idx_mc_vt_is_active              ON mobile_combustion_service.mc_vehicle_types(is_active);
CREATE INDEX IF NOT EXISTS idx_mc_vt_tenant_category        ON mobile_combustion_service.mc_vehicle_types(tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_mc_vt_created_at             ON mobile_combustion_service.mc_vehicle_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_vt_updated_at             ON mobile_combustion_service.mc_vehicle_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_vt_metadata               ON mobile_combustion_service.mc_vehicle_types USING GIN (metadata);

-- mc_fuel_types indexes (14)
CREATE INDEX IF NOT EXISTS idx_mc_ft_tenant_id              ON mobile_combustion_service.mc_fuel_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_ft_fuel_type              ON mobile_combustion_service.mc_fuel_types(fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_ft_display_name           ON mobile_combustion_service.mc_fuel_types(display_name);
CREATE INDEX IF NOT EXISTS idx_mc_ft_source                 ON mobile_combustion_service.mc_fuel_types(source);
CREATE INDEX IF NOT EXISTS idx_mc_ft_source_year            ON mobile_combustion_service.mc_fuel_types(source_year DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ft_is_biofuel             ON mobile_combustion_service.mc_fuel_types(is_biofuel);
CREATE INDEX IF NOT EXISTS idx_mc_ft_is_active              ON mobile_combustion_service.mc_fuel_types(is_active);
CREATE INDEX IF NOT EXISTS idx_mc_ft_tenant_fuel_type       ON mobile_combustion_service.mc_fuel_types(tenant_id, fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_ft_tenant_source          ON mobile_combustion_service.mc_fuel_types(tenant_id, source);
CREATE INDEX IF NOT EXISTS idx_mc_ft_tenant_biofuel         ON mobile_combustion_service.mc_fuel_types(tenant_id, is_biofuel);
CREATE INDEX IF NOT EXISTS idx_mc_ft_fuel_source            ON mobile_combustion_service.mc_fuel_types(fuel_type, source);
CREATE INDEX IF NOT EXISTS idx_mc_ft_created_at             ON mobile_combustion_service.mc_fuel_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ft_updated_at             ON mobile_combustion_service.mc_fuel_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ft_metadata               ON mobile_combustion_service.mc_fuel_types USING GIN (metadata);

-- mc_emission_factors indexes (14)
CREATE INDEX IF NOT EXISTS idx_mc_ef_tenant_id              ON mobile_combustion_service.mc_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_ef_vehicle_type           ON mobile_combustion_service.mc_emission_factors(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_ef_fuel_type              ON mobile_combustion_service.mc_emission_factors(fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_ef_gas                    ON mobile_combustion_service.mc_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_mc_ef_source                 ON mobile_combustion_service.mc_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_mc_ef_source_year            ON mobile_combustion_service.mc_emission_factors(source_year DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ef_tier                   ON mobile_combustion_service.mc_emission_factors(tier);
CREATE INDEX IF NOT EXISTS idx_mc_ef_emission_control       ON mobile_combustion_service.mc_emission_factors(emission_control);
CREATE INDEX IF NOT EXISTS idx_mc_ef_is_active              ON mobile_combustion_service.mc_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_mc_ef_vehicle_fuel_gas       ON mobile_combustion_service.mc_emission_factors(vehicle_type, fuel_type, gas);
CREATE INDEX IF NOT EXISTS idx_mc_ef_vehicle_fuel_source    ON mobile_combustion_service.mc_emission_factors(vehicle_type, fuel_type, source);
CREATE INDEX IF NOT EXISTS idx_mc_ef_tenant_vehicle_fuel    ON mobile_combustion_service.mc_emission_factors(tenant_id, vehicle_type, fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_ef_created_at             ON mobile_combustion_service.mc_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ef_metadata               ON mobile_combustion_service.mc_emission_factors USING GIN (metadata);

-- mc_vehicle_registry indexes (16)
CREATE INDEX IF NOT EXISTS idx_mc_vr_tenant_id              ON mobile_combustion_service.mc_vehicle_registry(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_vr_vehicle_id             ON mobile_combustion_service.mc_vehicle_registry(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_mc_vr_vin                    ON mobile_combustion_service.mc_vehicle_registry(vin);
CREATE INDEX IF NOT EXISTS idx_mc_vr_make                   ON mobile_combustion_service.mc_vehicle_registry(make);
CREATE INDEX IF NOT EXISTS idx_mc_vr_model                  ON mobile_combustion_service.mc_vehicle_registry(model);
CREATE INDEX IF NOT EXISTS idx_mc_vr_year                   ON mobile_combustion_service.mc_vehicle_registry(year DESC);
CREATE INDEX IF NOT EXISTS idx_mc_vr_vehicle_type           ON mobile_combustion_service.mc_vehicle_registry(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_vr_fuel_type              ON mobile_combustion_service.mc_vehicle_registry(fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_vr_department             ON mobile_combustion_service.mc_vehicle_registry(department);
CREATE INDEX IF NOT EXISTS idx_mc_vr_location               ON mobile_combustion_service.mc_vehicle_registry(location);
CREATE INDEX IF NOT EXISTS idx_mc_vr_status                 ON mobile_combustion_service.mc_vehicle_registry(status);
CREATE INDEX IF NOT EXISTS idx_mc_vr_tenant_status          ON mobile_combustion_service.mc_vehicle_registry(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_mc_vr_tenant_vehicle_type    ON mobile_combustion_service.mc_vehicle_registry(tenant_id, vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_vr_tenant_fuel_type       ON mobile_combustion_service.mc_vehicle_registry(tenant_id, fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_vr_tenant_department      ON mobile_combustion_service.mc_vehicle_registry(tenant_id, department);
CREATE INDEX IF NOT EXISTS idx_mc_vr_created_at             ON mobile_combustion_service.mc_vehicle_registry(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_vr_updated_at             ON mobile_combustion_service.mc_vehicle_registry(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_vr_metadata               ON mobile_combustion_service.mc_vehicle_registry USING GIN (metadata);

-- mc_trips indexes (14)
CREATE INDEX IF NOT EXISTS idx_mc_tr_tenant_id              ON mobile_combustion_service.mc_trips(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_tr_trip_id                ON mobile_combustion_service.mc_trips(trip_id);
CREATE INDEX IF NOT EXISTS idx_mc_tr_vehicle_id             ON mobile_combustion_service.mc_trips(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_mc_tr_fuel_type              ON mobile_combustion_service.mc_trips(fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_tr_start_time             ON mobile_combustion_service.mc_trips(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tr_end_time               ON mobile_combustion_service.mc_trips(end_time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tr_purpose                ON mobile_combustion_service.mc_trips(purpose);
CREATE INDEX IF NOT EXISTS idx_mc_tr_status                 ON mobile_combustion_service.mc_trips(status);
CREATE INDEX IF NOT EXISTS idx_mc_tr_tenant_vehicle         ON mobile_combustion_service.mc_trips(tenant_id, vehicle_id);
CREATE INDEX IF NOT EXISTS idx_mc_tr_tenant_status          ON mobile_combustion_service.mc_trips(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_mc_tr_vehicle_time           ON mobile_combustion_service.mc_trips(vehicle_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tr_tenant_time_range      ON mobile_combustion_service.mc_trips(tenant_id, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_mc_tr_created_at             ON mobile_combustion_service.mc_trips(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tr_metadata               ON mobile_combustion_service.mc_trips USING GIN (metadata);

-- mc_calculations indexes (16)
CREATE INDEX IF NOT EXISTS idx_mc_calc_tenant_id            ON mobile_combustion_service.mc_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_calc_calculation_id       ON mobile_combustion_service.mc_calculations(calculation_id);
CREATE INDEX IF NOT EXISTS idx_mc_calc_vehicle_type         ON mobile_combustion_service.mc_calculations(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_calc_fuel_type            ON mobile_combustion_service.mc_calculations(fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_calc_method               ON mobile_combustion_service.mc_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_mc_calc_tier                 ON mobile_combustion_service.mc_calculations(calculation_tier);
CREATE INDEX IF NOT EXISTS idx_mc_calc_total_co2e_kg        ON mobile_combustion_service.mc_calculations(total_co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_mc_calc_total_co2e_tonnes    ON mobile_combustion_service.mc_calculations(total_co2e_tonnes DESC);
CREATE INDEX IF NOT EXISTS idx_mc_calc_gwp_source           ON mobile_combustion_service.mc_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_mc_calc_provenance_hash      ON mobile_combustion_service.mc_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_mc_calc_vehicle_id           ON mobile_combustion_service.mc_calculations(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_mc_calc_trip_id              ON mobile_combustion_service.mc_calculations(trip_id);
CREATE INDEX IF NOT EXISTS idx_mc_calc_status               ON mobile_combustion_service.mc_calculations(status);
CREATE INDEX IF NOT EXISTS idx_mc_calc_tenant_status        ON mobile_combustion_service.mc_calculations(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_mc_calc_tenant_vehicle_type  ON mobile_combustion_service.mc_calculations(tenant_id, vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_calc_tenant_fuel_type     ON mobile_combustion_service.mc_calculations(tenant_id, fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_calc_period_range         ON mobile_combustion_service.mc_calculations(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_mc_calc_created_at           ON mobile_combustion_service.mc_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_calc_metadata             ON mobile_combustion_service.mc_calculations USING GIN (metadata);

-- mc_calculation_details indexes (10)
CREATE INDEX IF NOT EXISTS idx_mc_cd_tenant_id              ON mobile_combustion_service.mc_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_cd_calculation_id         ON mobile_combustion_service.mc_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_mc_cd_gas                    ON mobile_combustion_service.mc_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_mc_cd_emissions_kg           ON mobile_combustion_service.mc_calculation_details(emissions_kg DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cd_emissions_tco2e        ON mobile_combustion_service.mc_calculation_details(emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cd_is_biogenic            ON mobile_combustion_service.mc_calculation_details(is_biogenic);
CREATE INDEX IF NOT EXISTS idx_mc_cd_tenant_calc            ON mobile_combustion_service.mc_calculation_details(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_mc_cd_calc_gas               ON mobile_combustion_service.mc_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_mc_cd_tenant_gas             ON mobile_combustion_service.mc_calculation_details(tenant_id, gas);
CREATE INDEX IF NOT EXISTS idx_mc_cd_created_at             ON mobile_combustion_service.mc_calculation_details(created_at DESC);

-- mc_fleet_aggregations indexes (14)
CREATE INDEX IF NOT EXISTS idx_mc_fa_tenant_id              ON mobile_combustion_service.mc_fleet_aggregations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_fa_aggregation_id         ON mobile_combustion_service.mc_fleet_aggregations(aggregation_id);
CREATE INDEX IF NOT EXISTS idx_mc_fa_period_start           ON mobile_combustion_service.mc_fleet_aggregations(period_start DESC);
CREATE INDEX IF NOT EXISTS idx_mc_fa_period_end             ON mobile_combustion_service.mc_fleet_aggregations(period_end DESC);
CREATE INDEX IF NOT EXISTS idx_mc_fa_total_co2e             ON mobile_combustion_service.mc_fleet_aggregations(total_co2e_tonnes DESC);
CREATE INDEX IF NOT EXISTS idx_mc_fa_intensity              ON mobile_combustion_service.mc_fleet_aggregations(intensity_g_co2e_per_km DESC);
CREATE INDEX IF NOT EXISTS idx_mc_fa_provenance_hash        ON mobile_combustion_service.mc_fleet_aggregations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_mc_fa_tenant_period          ON mobile_combustion_service.mc_fleet_aggregations(tenant_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_mc_fa_created_at             ON mobile_combustion_service.mc_fleet_aggregations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_fa_breakdown_vehicle      ON mobile_combustion_service.mc_fleet_aggregations USING GIN (breakdown_by_vehicle_type);
CREATE INDEX IF NOT EXISTS idx_mc_fa_breakdown_fuel         ON mobile_combustion_service.mc_fleet_aggregations USING GIN (breakdown_by_fuel_type);
CREATE INDEX IF NOT EXISTS idx_mc_fa_breakdown_dept         ON mobile_combustion_service.mc_fleet_aggregations USING GIN (breakdown_by_department);
CREATE INDEX IF NOT EXISTS idx_mc_fa_metadata               ON mobile_combustion_service.mc_fleet_aggregations USING GIN (metadata);

-- mc_compliance_records indexes (10)
CREATE INDEX IF NOT EXISTS idx_mc_cr_tenant_id              ON mobile_combustion_service.mc_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_cr_framework              ON mobile_combustion_service.mc_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_mc_cr_status                 ON mobile_combustion_service.mc_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_mc_cr_checked_at             ON mobile_combustion_service.mc_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cr_tenant_framework       ON mobile_combustion_service.mc_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_mc_cr_tenant_status          ON mobile_combustion_service.mc_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_mc_cr_framework_status       ON mobile_combustion_service.mc_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_mc_cr_findings               ON mobile_combustion_service.mc_compliance_records USING GIN (findings);
CREATE INDEX IF NOT EXISTS idx_mc_cr_recommendations        ON mobile_combustion_service.mc_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_mc_cr_calculation_ids        ON mobile_combustion_service.mc_compliance_records USING GIN (calculation_ids);
CREATE INDEX IF NOT EXISTS idx_mc_cr_metadata               ON mobile_combustion_service.mc_compliance_records USING GIN (metadata);

-- mc_audit_entries indexes (12)
CREATE INDEX IF NOT EXISTS idx_mc_ae_tenant_id              ON mobile_combustion_service.mc_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mc_ae_entity_type            ON mobile_combustion_service.mc_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_mc_ae_entity_id              ON mobile_combustion_service.mc_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_mc_ae_action                 ON mobile_combustion_service.mc_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_mc_ae_input_hash             ON mobile_combustion_service.mc_audit_entries(input_hash);
CREATE INDEX IF NOT EXISTS idx_mc_ae_output_hash            ON mobile_combustion_service.mc_audit_entries(output_hash);
CREATE INDEX IF NOT EXISTS idx_mc_ae_previous_hash          ON mobile_combustion_service.mc_audit_entries(previous_hash);
CREATE INDEX IF NOT EXISTS idx_mc_ae_step_name              ON mobile_combustion_service.mc_audit_entries(step_name);
CREATE INDEX IF NOT EXISTS idx_mc_ae_user_id                ON mobile_combustion_service.mc_audit_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_mc_ae_tenant_entity          ON mobile_combustion_service.mc_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_mc_ae_created_at             ON mobile_combustion_service.mc_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mc_ae_details                ON mobile_combustion_service.mc_audit_entries USING GIN (details);

-- mc_calculation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_mc_cae_tenant_id             ON mobile_combustion_service.mc_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_vehicle_type          ON mobile_combustion_service.mc_calculation_events(vehicle_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_fuel_type             ON mobile_combustion_service.mc_calculation_events(fuel_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_method                ON mobile_combustion_service.mc_calculation_events(method, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_status                ON mobile_combustion_service.mc_calculation_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_tenant_vehicle_fuel   ON mobile_combustion_service.mc_calculation_events(tenant_id, vehicle_type, fuel_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_cae_metadata              ON mobile_combustion_service.mc_calculation_events USING GIN (metadata);

-- mc_trip_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_mc_tre_tenant_id             ON mobile_combustion_service.mc_trip_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_vehicle_id            ON mobile_combustion_service.mc_trip_events(vehicle_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_vehicle_type          ON mobile_combustion_service.mc_trip_events(vehicle_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_purpose               ON mobile_combustion_service.mc_trip_events(purpose, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_tenant_vehicle        ON mobile_combustion_service.mc_trip_events(tenant_id, vehicle_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_tenant_vehicle_type   ON mobile_combustion_service.mc_trip_events(tenant_id, vehicle_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_tre_metadata              ON mobile_combustion_service.mc_trip_events USING GIN (metadata);

-- mc_compliance_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_mc_coe_tenant_id             ON mobile_combustion_service.mc_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_coe_framework             ON mobile_combustion_service.mc_compliance_events(framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_coe_status                ON mobile_combustion_service.mc_compliance_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_coe_tenant_framework      ON mobile_combustion_service.mc_compliance_events(tenant_id, framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_coe_tenant_status         ON mobile_combustion_service.mc_compliance_events(tenant_id, status, time DESC);
CREATE INDEX IF NOT EXISTS idx_mc_coe_metadata              ON mobile_combustion_service.mc_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- mc_vehicle_types: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_vehicle_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_vt_read  ON mobile_combustion_service.mc_vehicle_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_vt_write ON mobile_combustion_service.mc_vehicle_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_fuel_types: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_fuel_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_ft_read  ON mobile_combustion_service.mc_fuel_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_ft_write ON mobile_combustion_service.mc_fuel_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_emission_factors: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_ef_read  ON mobile_combustion_service.mc_emission_factors FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_ef_write ON mobile_combustion_service.mc_emission_factors FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_vehicle_registry: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_vehicle_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_vr_read  ON mobile_combustion_service.mc_vehicle_registry FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_vr_write ON mobile_combustion_service.mc_vehicle_registry FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_trips: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_trips ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_tr_read  ON mobile_combustion_service.mc_trips FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_tr_write ON mobile_combustion_service.mc_trips FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_calculations: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_calc_read  ON mobile_combustion_service.mc_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_calc_write ON mobile_combustion_service.mc_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_calculation_details: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_cd_read  ON mobile_combustion_service.mc_calculation_details FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_cd_write ON mobile_combustion_service.mc_calculation_details FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_fleet_aggregations: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_fleet_aggregations ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_fa_read  ON mobile_combustion_service.mc_fleet_aggregations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_fa_write ON mobile_combustion_service.mc_fleet_aggregations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_compliance_records: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_cr_read  ON mobile_combustion_service.mc_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_cr_write ON mobile_combustion_service.mc_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_audit_entries: tenant-isolated
ALTER TABLE mobile_combustion_service.mc_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_ae_read  ON mobile_combustion_service.mc_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY mc_ae_write ON mobile_combustion_service.mc_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- mc_calculation_events: open read/write (time-series telemetry)
ALTER TABLE mobile_combustion_service.mc_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_cae_read  ON mobile_combustion_service.mc_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY mc_cae_write ON mobile_combustion_service.mc_calculation_events FOR ALL   USING (TRUE);

-- mc_trip_events: open read/write (time-series telemetry)
ALTER TABLE mobile_combustion_service.mc_trip_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_tre_read  ON mobile_combustion_service.mc_trip_events FOR SELECT USING (TRUE);
CREATE POLICY mc_tre_write ON mobile_combustion_service.mc_trip_events FOR ALL   USING (TRUE);

-- mc_compliance_events: open read/write (time-series telemetry)
ALTER TABLE mobile_combustion_service.mc_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY mc_coe_read  ON mobile_combustion_service.mc_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY mc_coe_write ON mobile_combustion_service.mc_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA mobile_combustion_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mobile_combustion_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA mobile_combustion_service TO greenlang_app;
GRANT SELECT ON mobile_combustion_service.mc_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON mobile_combustion_service.mc_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA mobile_combustion_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA mobile_combustion_service TO greenlang_readonly;
GRANT SELECT ON mobile_combustion_service.mc_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON mobile_combustion_service.mc_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA mobile_combustion_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA mobile_combustion_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA mobile_combustion_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'mobile-combustion:read',                    'mobile-combustion', 'read',                    'View all mobile combustion service data including vehicle types, fuel types, calculations, and fleet aggregations'),
    (gen_random_uuid(), 'mobile-combustion:write',                   'mobile-combustion', 'write',                   'Create, update, and manage all mobile combustion service data'),
    (gen_random_uuid(), 'mobile-combustion:execute',                 'mobile-combustion', 'execute',                 'Execute mobile combustion emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'mobile-combustion:vehicles:read',           'mobile-combustion', 'vehicles_read',           'View vehicle type registry and vehicle fleet registry with VIN, make, model, department, and odometer data'),
    (gen_random_uuid(), 'mobile-combustion:vehicles:write',          'mobile-combustion', 'vehicles_write',          'Create, update, and manage vehicle type and vehicle fleet registry entries'),
    (gen_random_uuid(), 'mobile-combustion:trips:read',              'mobile-combustion', 'trips_read',              'View trip records with distance, fuel consumed, start/end locations, purpose, and load factor data'),
    (gen_random_uuid(), 'mobile-combustion:trips:write',             'mobile-combustion', 'trips_write',             'Create, update, and manage trip records with fuel consumption and distance data'),
    (gen_random_uuid(), 'mobile-combustion:factors:read',            'mobile-combustion', 'factors_read',            'View emission factors by vehicle type, fuel type, gas, tier, and model year range from EPA/IPCC/DEFRA/EU_ETS sources'),
    (gen_random_uuid(), 'mobile-combustion:factors:write',           'mobile-combustion', 'factors_write',           'Create, update, and manage emission factor entries with source, tier, and model year range data'),
    (gen_random_uuid(), 'mobile-combustion:fleet:read',              'mobile-combustion', 'fleet_read',              'View fleet-level emission aggregations with breakdowns by vehicle type, fuel type, and department'),
    (gen_random_uuid(), 'mobile-combustion:fleet:execute',           'mobile-combustion', 'fleet_execute',           'Execute fleet-level emission aggregation rollups with intensity metrics and provenance tracking'),
    (gen_random_uuid(), 'mobile-combustion:compliance:read',         'mobile-combustion', 'compliance_read',         'View regulatory compliance records for GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 with findings'),
    (gen_random_uuid(), 'mobile-combustion:compliance:execute',      'mobile-combustion', 'compliance_execute',      'Execute regulatory compliance checks against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks'),
    (gen_random_uuid(), 'mobile-combustion:uncertainty:execute',     'mobile-combustion', 'uncertainty_execute',     'Execute uncertainty quantification analysis on mobile combustion calculation results'),
    (gen_random_uuid(), 'mobile-combustion:audit:read',              'mobile-combustion', 'audit_read',              'View audit trail entries with entity type, action, input/output hashes, and hash chain provenance'),
    (gen_random_uuid(), 'mobile-combustion:admin:read',              'mobile-combustion', 'admin_read',              'View mobile combustion service administrative configuration, statistics, and diagnostics'),
    (gen_random_uuid(), 'mobile-combustion:admin:write',             'mobile-combustion', 'admin_write',             'Update mobile combustion service administrative configuration and manage service settings'),
    (gen_random_uuid(), 'mobile-combustion:admin:execute',           'mobile-combustion', 'admin_execute',           'Execute mobile combustion service administrative operations including bulk imports and data migrations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('mobile_combustion_service.mc_calculation_events', INTERVAL '90 days');
SELECT add_retention_policy('mobile_combustion_service.mc_trip_events',        INTERVAL '90 days');
SELECT add_retention_policy('mobile_combustion_service.mc_compliance_events',  INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE mobile_combustion_service.mc_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'vehicle_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('mobile_combustion_service.mc_calculation_events', INTERVAL '7 days');

ALTER TABLE mobile_combustion_service.mc_trip_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'vehicle_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('mobile_combustion_service.mc_trip_events', INTERVAL '7 days');

ALTER TABLE mobile_combustion_service.mc_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('mobile_combustion_service.mc_compliance_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Mobile Combustion Agent (GL-MRV-SCOPE1-003)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-003',
    'Mobile Combustion Agent',
    'Mobile combustion emission calculator for GreenLang Climate OS. Manages vehicle type registry with ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL categories including fuel economy defaults, weight class, and emission standard tracking. Maintains fuel type registry with density (kg/L), heating values (MJ/L and MJ/kg), per-gas emission factors (CO2/CH4/N2O in kg/L), biofuel and renewable fractions, and source year tracking from EPA/IPCC/DEFRA/EU_ETS/CUSTOM. Stores tiered emission factor database with vehicle type x fuel type x gas factors at Tier 1/2/3, model year ranges, and emission control technology references. Registers vehicle fleet with VIN, make, model, year, fuel type, emission control, department, location, odometer, and annual mileage estimates. Logs trip records with distance, fuel consumed, start/end time and location, purpose, load factor, passengers, and cargo tonnes. Executes deterministic mobile combustion emission calculations using fuel-based and distance-based methods converting fuel quantity or distance to CO2e with multi-gas GWP weighting (CO2/CH4/N2O) using AR4/AR5/AR6 values. Tracks biogenic CO2 separately. Produces per-gas calculation detail breakdowns with individual emission factors, GWP values, and biogenic flags. Aggregates fleet-level emissions with breakdowns by vehicle type, fuel type, and department, plus intensity metrics (g CO2e/km). Checks regulatory compliance against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks with findings and recommendations. Generates entity-level audit trail entries with action tracking, input/output hashes, and previous hash chaining for provenance. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/mobile-combustion',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-003', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/mobile-combustion-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"mobile-combustion", "scope-1", "ghg-protocol", "epa", "ipcc", "defra", "fleet", "vehicles", "mrv"}',
    '{"cross-sector", "transportation", "logistics", "construction", "mining", "agriculture", "aviation", "marine"}',
    'd4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'vehicle_type_registry',
    'configuration',
    'Register and manage vehicle type entries with category classification (ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL), default fuel type and economy, weight class, and emission standard tracking.',
    '{"vehicle_type", "category", "display_name", "default_fuel_type", "default_fuel_economy", "weight_class", "emission_standard"}',
    '{"vehicle_type_id", "registration_result"}',
    '{"categories": ["ON_ROAD", "OFF_ROAD", "MARINE", "AVIATION", "RAIL"], "supports_weight_class": true, "supports_emission_standards": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'fuel_type_registry',
    'configuration',
    'Register and manage fuel type entries with density, heating values (MJ/L, MJ/kg), per-gas emission factors (CO2/CH4/N2O in kg/L), biofuel and renewable fractions, and source tracking from EPA/IPCC/DEFRA/EU_ETS/CUSTOM.',
    '{"fuel_type", "display_name", "density_kg_per_l", "heating_value_mj_per_l", "heating_value_mj_per_kg", "co2_ef_kg_per_l", "ch4_ef_kg_per_l", "n2o_ef_kg_per_l", "biofuel_fraction", "source", "source_year"}',
    '{"fuel_type_id", "registration_result"}',
    '{"sources": ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"], "supports_biofuel": true, "supports_renewable": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic mobile combustion emission calculations using fuel-based method (fuel consumed x emission factor x GWP) or distance-based method (distance x distance-based EF x GWP). Supports Tier 1/2/3 factors, model year ranges, and emission control technology matching.',
    '{"calculation_method", "vehicle_type", "fuel_type", "fuel_quantity", "fuel_unit", "distance_km", "gwp_source", "vehicle_id", "trip_id", "period_start", "period_end"}',
    '{"calculation_id", "total_co2e_kg", "total_co2e_tonnes", "biogenic_co2_kg", "per_gas_breakdown", "provenance_hash"}',
    '{"methods": ["FUEL_BASED", "DISTANCE_BASED", "SPEND_BASED", "HYBRID", "DIRECT_MEASUREMENT"], "tiers": ["TIER_1", "TIER_2", "TIER_3"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'vehicle_fleet_management',
    'configuration',
    'Register and manage vehicle fleet entries with VIN, make, model, year, vehicle type, fuel type, emission control technology, department assignment, location, odometer readings, and annual mileage estimates.',
    '{"vehicle_id", "vin", "make", "model", "year", "vehicle_type", "fuel_type", "emission_control", "department", "location", "odometer_reading", "annual_mileage_estimate"}',
    '{"vehicle_id", "registration_result"}',
    '{"statuses": ["ACTIVE", "INACTIVE", "DECOMMISSIONED", "MAINTENANCE", "DISPOSED"], "supports_vin_lookup": true, "supports_odometer": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'trip_logging',
    'processing',
    'Log and manage trip records with distance, fuel consumed, start/end time and location, purpose classification, load factor, passenger count, and cargo tonnes for detailed activity-based accounting.',
    '{"trip_id", "vehicle_id", "distance_km", "fuel_consumed_liters", "fuel_type", "start_time", "end_time", "start_location", "end_location", "purpose", "load_factor", "passengers", "cargo_tonnes"}',
    '{"trip_id", "logging_result"}',
    '{"distance_units": ["KM", "MI", "NM"], "statuses": ["PLANNED", "IN_PROGRESS", "COMPLETED", "CANCELLED"], "supports_load_factor": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'fleet_aggregation',
    'reporting',
    'Aggregate fleet-level emissions across vehicles, trips, and time periods with breakdowns by vehicle type, fuel type, and department. Calculate intensity metrics (g CO2e/km) and per-gas totals (CO2/CH4/N2O) with biogenic tracking.',
    '{"period_start", "period_end", "group_by_vehicle_type", "group_by_fuel_type", "group_by_department"}',
    '{"aggregation_id", "total_co2e_tonnes", "per_gas_totals", "vehicle_count", "trip_count", "total_distance_km", "total_fuel_liters", "intensity_g_co2e_per_km", "breakdowns", "provenance_hash"}',
    '{"breakdown_dimensions": ["vehicle_type", "fuel_type", "department"], "intensity_metrics": ["g_co2e_per_km"], "supports_biogenic_tracking": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of mobile combustion calculations against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks. Produce findings with severity levels and actionable recommendations.',
    '{"calculation_ids", "framework"}',
    '{"compliance_id", "status", "findings", "recommendations"}',
    '{"frameworks": ["GHG_PROTOCOL", "EPA", "DEFRA", "EU_ETS", "ISO_14064", "CUSTOM"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-003', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with action tracking, input/output SHA-256 hashes, previous hash chaining for tamper-evident provenance, step names, and user attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-003', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage mobile combustion calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-003', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for fuel quantities, distances, and emission factor unit conversions'),
    ('GL-MRV-SCOPE1-003', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-003', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-SCOPE1-003', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, trip events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-003', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of fuel consumption and trip records'),
    ('GL-MRV-SCOPE1-003', 'GL-MRV-X-001',   '>=1.0.0', true,  'Stationary Combustion Calculator for cross-referencing fuel types and shared emission factor database'),
    ('GL-MRV-SCOPE1-003', 'GL-MRV-SCOPE1-002', '>=1.0.0', true, 'Refrigerants & F-Gas Agent for vehicle AC refrigerant leakage cross-referencing')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-003',
    'Mobile Combustion Agent',
    'Mobile combustion emission calculator. Vehicle type registry (ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL, fuel economy, weight class, emission standards). Fuel type registry (density, heating values MJ/L and MJ/kg, per-gas EFs CO2/CH4/N2O in kg/L, biofuel/renewable fractions, EPA/IPCC/DEFRA/EU_ETS/CUSTOM sources). Tiered emission factor database (vehicle x fuel x gas, Tier 1/2/3, model year ranges, emission control). Vehicle fleet registry (VIN, make, model, year, department, odometer). Trip logging (distance, fuel, start/end, purpose, load factor, passengers, cargo). Emission calculations (fuel-based/distance-based methods, multi-gas CO2/CH4/N2O, GWP AR4/AR5/AR6, biogenic tracking). Per-gas breakdowns. Fleet aggregations (by vehicle type/fuel type/department, intensity g CO2e/km). Compliance checks (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA mobile_combustion_service IS
    'Mobile Combustion Agent (AGENT-MRV-003) - vehicle type registry, fuel type registry, emission factor database, vehicle fleet registry, trip logging, emission calculations, per-gas breakdowns, fleet aggregations, compliance records, audit trail, provenance chains';

COMMENT ON TABLE mobile_combustion_service.mc_vehicle_types IS
    'Vehicle type registry: tenant_id, vehicle_type (unique per tenant), category (ON_ROAD/OFF_ROAD/MARINE/AVIATION/RAIL), display_name, description, default_fuel_type, default_fuel_economy, default_fuel_economy_unit, weight_class, emission_standard, is_active, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_fuel_types IS
    'Fuel type registry: tenant_id, fuel_type (unique per tenant), display_name, density_kg_per_l, heating_value_mj_per_l, heating_value_mj_per_kg, co2_ef_kg_per_l, ch4_ef_kg_per_l, n2o_ef_kg_per_l, biofuel_fraction, renewable_fraction, is_biofuel, source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), source_year, is_active, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_emission_factors IS
    'Emission factor database: tenant_id, vehicle_type, fuel_type, gas (CO2/CH4/N2O), factor_value, factor_unit, source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), source_year, model_year_start/end, emission_control, tier (TIER_1/TIER_2/TIER_3), is_active, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_vehicle_registry IS
    'Vehicle fleet registry: tenant_id, vehicle_id (unique per tenant), vin, make, model, year, vehicle_type, fuel_type, emission_control, department, location, status (ACTIVE/INACTIVE/DECOMMISSIONED/MAINTENANCE/DISPOSED), acquisition_date, disposal_date, odometer_reading, annual_mileage_estimate, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_trips IS
    'Trip records: tenant_id, trip_id (unique per tenant), vehicle_id, distance_km, distance_unit (KM/MI/NM), fuel_consumed_liters, fuel_type, start_time, end_time, start_location, end_location, purpose, load_factor, passengers, cargo_tonnes, status (PLANNED/IN_PROGRESS/COMPLETED/CANCELLED), metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_calculations IS
    'Calculation results: tenant_id, calculation_id (unique per tenant), vehicle_type, fuel_type, calculation_method (FUEL_BASED/DISTANCE_BASED/SPEND_BASED/HYBRID/DIRECT_MEASUREMENT), calculation_tier (TIER_1/TIER_2/TIER_3), fuel_quantity/unit, distance_km, total_co2e_kg/tonnes, biogenic_co2_kg, gwp_source (AR4/AR5/AR6), provenance_hash, vehicle_id, trip_id, period_start/end, status, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_calculation_details IS
    'Per-gas calculation breakdown: tenant_id, calculation_id, gas (CO2/CH4/N2O), emissions_kg, emissions_tco2e, emission_factor, emission_factor_unit, gwp_applied, is_biogenic';

COMMENT ON TABLE mobile_combustion_service.mc_fleet_aggregations IS
    'Fleet-level emission aggregations: tenant_id, aggregation_id (unique per tenant), period_start/end, total_co2e/co2/ch4/n2o/biogenic_tonnes, vehicle_count, trip_count, total_distance_km, total_fuel_liters, intensity_g_co2e_per_km, breakdown_by_vehicle_type/fuel_type/department JSONB, provenance_hash, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_compliance_records IS
    'Regulatory compliance records: tenant_id, framework (GHG_PROTOCOL/EPA/DEFRA/EU_ETS/ISO_14064/CUSTOM), status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), findings JSONB, recommendations JSONB, calculation_ids JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE mobile_combustion_service.mc_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT), input_hash, output_hash, previous_hash (chain), step_name, details JSONB, user_id';

COMMENT ON TABLE mobile_combustion_service.mc_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, vehicle_type, fuel_type, method, emissions_kg_co2e, status, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE mobile_combustion_service.mc_trip_events IS
    'TimescaleDB hypertable: trip events with tenant_id, vehicle_id, vehicle_type, distance_km, fuel_liters, purpose, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE mobile_combustion_service.mc_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, framework, status, findings_count, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW mobile_combustion_service.mc_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by vehicle_type and method (total calculations, sum emissions kg CO2e, avg emissions kg CO2e per hour)';

COMMENT ON MATERIALIZED VIEW mobile_combustion_service.mc_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by vehicle_type and fuel_type (total calculations, sum emissions kg CO2e per day)';
