-- =============================================================================
-- V051: Stationary Combustion Service Schema
-- =============================================================================
-- Component: AGENT-MRV-001 (GL-MRV-X-001)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Stationary Combustion Calculator (GL-MRV-X-001) with capabilities for
-- fuel type registry management (gaseous/liquid/solid/biomass with HHV/NCV
-- density and carbon content), emission factor database (EPA/IPCC/DEFRA/
-- EU_ETS/CUSTOM sources across Tier 1-3 with geography and effective dates),
-- heating value storage (HHV/NCV basis per fuel type), oxidation factor
-- registry (0-1 range per fuel type), equipment profile management
-- (capacity/efficiency curves/load factors/maintenance status), combustion
-- emission calculations (fuel quantity to CO2e with energy conversion,
-- heating value application, oxidation factor correction, and multi-gas
-- GWP weighting), per-gas calculation detail breakdowns (CO2/CH4/N2O with
-- individual emission factors and GWP values), facility-level aggregations
-- (operational/financial/equity share control approaches with monthly/
-- quarterly/annual reporting periods), step-by-step audit trail entries
-- (input/output data per calculation step with methodology references),
-- and user-defined custom emission factors with approval workflows.
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. sc_fuel_types              - Fuel type registry (gaseous/liquid/solid/biomass, HHV/NCV/density)
--   2. sc_emission_factors        - Emission factor database (EPA/IPCC/DEFRA/EU_ETS/CUSTOM, Tier 1-3)
--   3. sc_heating_values          - Heating value storage (HHV/NCV basis per fuel type)
--   4. sc_oxidation_factors       - Oxidation factor registry (0-1 range per fuel type)
--   5. sc_equipment_profiles      - Equipment registry (capacity/efficiency/load factors)
--   6. sc_calculations            - Calculation results (fuel to CO2e with provenance)
--   7. sc_calculation_details     - Per-gas breakdown (CO2/CH4/N2O with EF and GWP)
--   8. sc_facility_aggregations   - Facility roll-ups (operational/financial/equity share)
--   9. sc_audit_entries           - Audit trail (step-by-step calculation trace)
--  10. sc_custom_factors          - User-defined emission factors with approval workflow
--
-- Hypertables (3):
--  11. sc_calculation_events      - Calculation event time-series (hypertable on event_time)
--  12. sc_factor_updates          - Factor update time-series (hypertable on event_time)
--  13. sc_audit_events            - Audit event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. sc_hourly_calculation_stats  - Hourly count/sum(co2e)/avg(duration)/max(duration) by fuel_type
--   2. sc_daily_emission_totals     - Daily count/sum(co2e)/sum(co2)/sum(ch4)/sum(n2o)/sum(biogenic) by fuel_type
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-X-001.
-- Previous: V050__climate_hazard_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS stationary_combustion_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION stationary_combustion_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: stationary_combustion_service.sc_fuel_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_fuel_types (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    fuel_type_id        VARCHAR(255)  UNIQUE NOT NULL,
    name                VARCHAR(255)  NOT NULL,
    category            VARCHAR(50)   NOT NULL,
    hhv                 DECIMAL(12,6) NOT NULL,
    hhv_unit            VARCHAR(50)   NOT NULL,
    ncv                 DECIMAL(12,6),
    ncv_unit            VARCHAR(50),
    density             DECIMAL(10,4),
    density_unit        VARCHAR(50),
    carbon_content_pct  DECIMAL(6,4),
    is_biogenic         BOOLEAN       DEFAULT FALSE,
    ipcc_code           VARCHAR(50),
    properties          JSONB         DEFAULT '{}'::jsonb,
    tenant_id           VARCHAR(255)  NOT NULL,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_fuel_type_id_not_empty CHECK (LENGTH(TRIM(fuel_type_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_category CHECK (category IN (
        'gaseous', 'liquid', 'solid', 'biomass'
    ));

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_hhv_positive CHECK (hhv > 0);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_ncv_positive CHECK (ncv IS NULL OR ncv > 0);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_density_positive CHECK (density IS NULL OR density > 0);

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_carbon_content_range CHECK (carbon_content_pct IS NULL OR (carbon_content_pct >= 0 AND carbon_content_pct <= 1));

ALTER TABLE stationary_combustion_service.sc_fuel_types
    ADD CONSTRAINT chk_ft_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ft_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_fuel_types
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 2: stationary_combustion_service.sc_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_emission_factors (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    factor_id       VARCHAR(255)  UNIQUE NOT NULL,
    fuel_type       VARCHAR(100)  NOT NULL,
    gas             VARCHAR(10)   NOT NULL,
    value           DECIMAL(20,10) NOT NULL,
    unit            VARCHAR(50)   NOT NULL,
    source          VARCHAR(50)   NOT NULL,
    tier            INTEGER       NOT NULL,
    geography       VARCHAR(100)  DEFAULT 'GLOBAL',
    effective_date  DATE,
    expiry_date     DATE,
    reference       TEXT,
    notes           TEXT,
    metadata        JSONB         DEFAULT '{}'::jsonb,
    tenant_id       VARCHAR(255)  NOT NULL,
    created_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_factor_id_not_empty CHECK (LENGTH(TRIM(factor_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN ('CO2', 'CH4', 'N2O'));

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_value_non_negative CHECK (value >= 0);

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_tier CHECK (tier IN (1, 2, 3));

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_date_order CHECK (expiry_date IS NULL OR effective_date IS NULL OR expiry_date >= effective_date);

ALTER TABLE stationary_combustion_service.sc_emission_factors
    ADD CONSTRAINT chk_ef_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_emission_factors
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 3: stationary_combustion_service.sc_heating_values
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_heating_values (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    heating_value_id    VARCHAR(255)  UNIQUE NOT NULL,
    fuel_type           VARCHAR(100)  NOT NULL,
    basis               VARCHAR(10)   NOT NULL,
    value               DECIMAL(15,8) NOT NULL,
    unit                VARCHAR(50)   NOT NULL,
    source              VARCHAR(50)   NOT NULL,
    reference           TEXT,
    tenant_id           VARCHAR(255)  NOT NULL,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_heating_values
    ADD CONSTRAINT chk_hv_heating_value_id_not_empty CHECK (LENGTH(TRIM(heating_value_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_heating_values
    ADD CONSTRAINT chk_hv_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_heating_values
    ADD CONSTRAINT chk_hv_basis CHECK (basis IN ('HHV', 'NCV'));

ALTER TABLE stationary_combustion_service.sc_heating_values
    ADD CONSTRAINT chk_hv_value_positive CHECK (value > 0);

ALTER TABLE stationary_combustion_service.sc_heating_values
    ADD CONSTRAINT chk_hv_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_hv_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_heating_values
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 4: stationary_combustion_service.sc_oxidation_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_oxidation_factors (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    oxidation_factor_id     VARCHAR(255)  UNIQUE NOT NULL,
    fuel_type               VARCHAR(100)  NOT NULL,
    value                   DECIMAL(6,4)  NOT NULL,
    source                  VARCHAR(50)   NOT NULL,
    reference               TEXT,
    tenant_id               VARCHAR(255)  NOT NULL,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_oxidation_factors
    ADD CONSTRAINT chk_of_oxidation_factor_id_not_empty CHECK (LENGTH(TRIM(oxidation_factor_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_oxidation_factors
    ADD CONSTRAINT chk_of_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_oxidation_factors
    ADD CONSTRAINT chk_of_value_range CHECK (value >= 0 AND value <= 1);

ALTER TABLE stationary_combustion_service.sc_oxidation_factors
    ADD CONSTRAINT chk_of_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_of_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_oxidation_factors
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 5: stationary_combustion_service.sc_equipment_profiles
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_equipment_profiles (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            VARCHAR(255)  UNIQUE NOT NULL,
    equipment_type          VARCHAR(100)  NOT NULL,
    name                    VARCHAR(255)  NOT NULL,
    facility_id             VARCHAR(255),
    rated_capacity_mmbtu_hr DECIMAL(10,2),
    efficiency_curve        JSONB         DEFAULT '[]'::jsonb,
    age_years               INTEGER       DEFAULT 0,
    maintenance_status      VARCHAR(50)   DEFAULT 'good',
    load_factor_min         DECIMAL(4,2)  DEFAULT 0.0,
    load_factor_max         DECIMAL(4,2)  DEFAULT 1.0,
    metadata                JSONB         DEFAULT '{}'::jsonb,
    tenant_id               VARCHAR(255)  NOT NULL,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_equipment_id_not_empty CHECK (LENGTH(TRIM(equipment_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_equipment_type_not_empty CHECK (LENGTH(TRIM(equipment_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_rated_capacity_positive CHECK (rated_capacity_mmbtu_hr IS NULL OR rated_capacity_mmbtu_hr > 0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_age_non_negative CHECK (age_years >= 0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_maintenance_status CHECK (maintenance_status IN (
        'good', 'fair', 'poor', 'maintenance_required', 'decommissioned'
    ));

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_load_factor_min_range CHECK (load_factor_min >= 0.0 AND load_factor_min <= 1.0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_load_factor_max_range CHECK (load_factor_max >= 0.0 AND load_factor_max <= 1.0);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_load_factor_order CHECK (load_factor_max >= load_factor_min);

ALTER TABLE stationary_combustion_service.sc_equipment_profiles
    ADD CONSTRAINT chk_ep_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_ep_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_equipment_profiles
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 6: stationary_combustion_service.sc_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    calculation_id          VARCHAR(255)    UNIQUE NOT NULL,
    fuel_type               VARCHAR(100)    NOT NULL,
    equipment_type          VARCHAR(100),
    fuel_quantity           DECIMAL(20,8)   NOT NULL,
    fuel_unit               VARCHAR(50)     NOT NULL,
    energy_gj              DECIMAL(20,8)   NOT NULL,
    heating_value_used      DECIMAL(15,8)   NOT NULL,
    heating_value_basis     VARCHAR(10),
    oxidation_factor_used   DECIMAL(6,4)    NOT NULL,
    tier_used               INTEGER         NOT NULL,
    total_co2e_kg           DECIMAL(20,8)   NOT NULL,
    total_co2e_tonnes       DECIMAL(20,10)  NOT NULL,
    biogenic_co2_kg         DECIMAL(20,8)   DEFAULT 0,
    gwp_source              VARCHAR(10)     NOT NULL,
    regulatory_framework    VARCHAR(50),
    provenance_hash         VARCHAR(64)     NOT NULL,
    calculation_trace       JSONB           DEFAULT '[]'::jsonb,
    facility_id             VARCHAR(255),
    source_id               VARCHAR(255),
    period_start            TIMESTAMPTZ,
    period_end              TIMESTAMPTZ,
    organization_id         VARCHAR(255),
    status                  VARCHAR(20)     DEFAULT 'completed',
    metadata                JSONB           DEFAULT '{}'::jsonb,
    tenant_id               VARCHAR(255)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_calculation_id_not_empty CHECK (LENGTH(TRIM(calculation_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_fuel_quantity_positive CHECK (fuel_quantity > 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_energy_gj_non_negative CHECK (energy_gj >= 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_heating_value_positive CHECK (heating_value_used > 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_heating_value_basis CHECK (heating_value_basis IS NULL OR heating_value_basis IN ('HHV', 'NCV'));

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_oxidation_factor_range CHECK (oxidation_factor_used >= 0 AND oxidation_factor_used <= 1);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_tier CHECK (tier_used IN (1, 2, 3));

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_total_co2e_kg_non_negative CHECK (total_co2e_kg >= 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_total_co2e_tonnes_non_negative CHECK (total_co2e_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_biogenic_non_negative CHECK (biogenic_co2_kg IS NULL OR biogenic_co2_kg >= 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_status CHECK (status IN (
        'pending', 'running', 'completed', 'failed', 'cancelled'
    ));

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_period_order CHECK (period_end IS NULL OR period_start IS NULL OR period_end >= period_start);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE stationary_combustion_service.sc_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_calculations
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 7: stationary_combustion_service.sc_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    detail_id               VARCHAR(255)    UNIQUE NOT NULL,
    calculation_id          VARCHAR(255)    NOT NULL,
    gas                     VARCHAR(10)     NOT NULL,
    emissions_kg            DECIMAL(20,8)   NOT NULL,
    emissions_tco2e         DECIMAL(20,10)  NOT NULL,
    emission_factor_value   DECIMAL(20,10)  NOT NULL,
    emission_factor_unit    VARCHAR(50),
    emission_factor_source  VARCHAR(50),
    gwp_applied             DECIMAL(10,4)   NOT NULL,
    is_biogenic             BOOLEAN         DEFAULT FALSE,
    tenant_id               VARCHAR(255)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_detail_id_not_empty CHECK (LENGTH(TRIM(detail_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT fk_cd_calculation_id
        FOREIGN KEY (calculation_id)
        REFERENCES stationary_combustion_service.sc_calculations(calculation_id)
        ON DELETE CASCADE;

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN ('CO2', 'CH4', 'N2O'));

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_emissions_kg_non_negative CHECK (emissions_kg >= 0);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_emissions_tco2e_non_negative CHECK (emissions_tco2e >= 0);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_non_negative CHECK (emission_factor_value >= 0);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_source CHECK (emission_factor_source IS NULL OR emission_factor_source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_gwp_positive CHECK (gwp_applied > 0);

ALTER TABLE stationary_combustion_service.sc_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 8: stationary_combustion_service.sc_facility_aggregations
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_facility_aggregations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregation_id          VARCHAR(255)    UNIQUE NOT NULL,
    facility_id             VARCHAR(255)    NOT NULL,
    organization_id         VARCHAR(255),
    control_approach        VARCHAR(50)     NOT NULL,
    reporting_period_type   VARCHAR(20)     NOT NULL,
    period_start            TIMESTAMPTZ     NOT NULL,
    period_end              TIMESTAMPTZ     NOT NULL,
    total_co2e_tonnes       DECIMAL(20,10),
    total_co2_tonnes        DECIMAL(20,10),
    total_ch4_tonnes        DECIMAL(20,10),
    total_n2o_tonnes        DECIMAL(20,10),
    biogenic_co2_tonnes     DECIMAL(20,10),
    calculation_count       INTEGER         DEFAULT 0,
    equipment_count         INTEGER         DEFAULT 0,
    fuel_types_used         JSONB           DEFAULT '[]'::jsonb,
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    tenant_id               VARCHAR(255)    NOT NULL,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_aggregation_id_not_empty CHECK (LENGTH(TRIM(aggregation_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_facility_id_not_empty CHECK (LENGTH(TRIM(facility_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_control_approach CHECK (control_approach IN (
        'OPERATIONAL', 'FINANCIAL', 'EQUITY_SHARE'
    ));

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_reporting_period_type CHECK (reporting_period_type IN (
        'MONTHLY', 'QUARTERLY', 'ANNUAL'
    ));

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_period_order CHECK (period_end >= period_start);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_total_co2e_non_negative CHECK (total_co2e_tonnes IS NULL OR total_co2e_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_total_co2_non_negative CHECK (total_co2_tonnes IS NULL OR total_co2_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_total_ch4_non_negative CHECK (total_ch4_tonnes IS NULL OR total_ch4_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_total_n2o_non_negative CHECK (total_n2o_tonnes IS NULL OR total_n2o_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_biogenic_non_negative CHECK (biogenic_co2_tonnes IS NULL OR biogenic_co2_tonnes >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_calculation_count_non_negative CHECK (calculation_count >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_equipment_count_non_negative CHECK (equipment_count >= 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE stationary_combustion_service.sc_facility_aggregations
    ADD CONSTRAINT chk_fa_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_fa_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_facility_aggregations
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 9: stationary_combustion_service.sc_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_audit_entries (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id                VARCHAR(255)  UNIQUE NOT NULL,
    calculation_id          VARCHAR(255)  NOT NULL,
    step_number             INTEGER       NOT NULL,
    step_name               VARCHAR(100)  NOT NULL,
    input_data              JSONB         DEFAULT '{}'::jsonb,
    output_data             JSONB         DEFAULT '{}'::jsonb,
    emission_factor_used    VARCHAR(255),
    methodology_reference   TEXT,
    provenance_hash         VARCHAR(64)   NOT NULL,
    tenant_id               VARCHAR(255)  NOT NULL,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_entry_id_not_empty CHECK (LENGTH(TRIM(entry_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_calculation_id_not_empty CHECK (LENGTH(TRIM(calculation_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_step_number_positive CHECK (step_number > 0);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_step_name_not_empty CHECK (LENGTH(TRIM(step_name)) > 0);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE stationary_combustion_service.sc_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

-- =============================================================================
-- Table 10: stationary_combustion_service.sc_custom_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_custom_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    custom_factor_id    VARCHAR(255)    UNIQUE NOT NULL,
    fuel_type           VARCHAR(100)    NOT NULL,
    gas                 VARCHAR(10)     NOT NULL,
    value               DECIMAL(20,10)  NOT NULL,
    unit                VARCHAR(50)     NOT NULL,
    geography           VARCHAR(100),
    reference           TEXT,
    approved_by         VARCHAR(255),
    approval_status     VARCHAR(20)     DEFAULT 'pending',
    approval_date       TIMESTAMPTZ,
    valid_from          DATE,
    valid_until         DATE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    tenant_id           VARCHAR(255)    NOT NULL,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_custom_factor_id_not_empty CHECK (LENGTH(TRIM(custom_factor_id)) > 0);

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_fuel_type_not_empty CHECK (LENGTH(TRIM(fuel_type)) > 0);

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_gas CHECK (gas IN ('CO2', 'CH4', 'N2O'));

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_value_non_negative CHECK (value >= 0);

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_approval_status CHECK (approval_status IN (
        'pending', 'approved', 'rejected', 'expired', 'revoked'
    ));

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_valid_date_order CHECK (valid_until IS NULL OR valid_from IS NULL OR valid_until >= valid_from);

ALTER TABLE stationary_combustion_service.sc_custom_factors
    ADD CONSTRAINT chk_cf_tenant_id_not_empty CHECK (LENGTH(TRIM(tenant_id)) > 0);

CREATE TRIGGER trg_cf_updated_at
    BEFORE UPDATE ON stationary_combustion_service.sc_custom_factors
    FOR EACH ROW EXECUTE FUNCTION stationary_combustion_service.set_updated_at();

-- =============================================================================
-- Table 11: stationary_combustion_service.sc_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_calculation_events (
    event_time      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculation_id  VARCHAR(255),
    fuel_type       VARCHAR(100),
    tier            INTEGER,
    total_co2e_kg   DECIMAL(20,8),
    duration_ms     NUMERIC,
    status          VARCHAR(20),
    tenant_id       VARCHAR(255),
    metadata        JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'stationary_combustion_service.sc_calculation_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE stationary_combustion_service.sc_calculation_events
    ADD CONSTRAINT chk_ce_total_co2e_non_negative CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0);

ALTER TABLE stationary_combustion_service.sc_calculation_events
    ADD CONSTRAINT chk_ce_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE stationary_combustion_service.sc_calculation_events
    ADD CONSTRAINT chk_ce_tier CHECK (tier IS NULL OR tier IN (1, 2, 3));

ALTER TABLE stationary_combustion_service.sc_calculation_events
    ADD CONSTRAINT chk_ce_status CHECK (
        status IS NULL OR status IN ('pending', 'running', 'completed', 'failed', 'cancelled')
    );

-- =============================================================================
-- Table 12: stationary_combustion_service.sc_factor_updates (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_factor_updates (
    event_time      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    factor_id       VARCHAR(255),
    fuel_type       VARCHAR(100),
    gas             VARCHAR(10),
    old_value       DECIMAL(20,10),
    new_value       DECIMAL(20,10),
    source          VARCHAR(50),
    updated_by      VARCHAR(255),
    tenant_id       VARCHAR(255),
    metadata        JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'stationary_combustion_service.sc_factor_updates',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE stationary_combustion_service.sc_factor_updates
    ADD CONSTRAINT chk_fu_gas CHECK (
        gas IS NULL OR gas IN ('CO2', 'CH4', 'N2O')
    );

ALTER TABLE stationary_combustion_service.sc_factor_updates
    ADD CONSTRAINT chk_fu_old_value_non_negative CHECK (old_value IS NULL OR old_value >= 0);

ALTER TABLE stationary_combustion_service.sc_factor_updates
    ADD CONSTRAINT chk_fu_new_value_non_negative CHECK (new_value IS NULL OR new_value >= 0);

-- =============================================================================
-- Table 13: stationary_combustion_service.sc_audit_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS stationary_combustion_service.sc_audit_events (
    event_time          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    calculation_id      VARCHAR(255),
    step_name           VARCHAR(100),
    provenance_hash     VARCHAR(64),
    tenant_id           VARCHAR(255),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'stationary_combustion_service.sc_audit_events',
    'event_time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- sc_hourly_calculation_stats: hourly count/sum(co2e)/avg(duration)/max(duration) by fuel_type
CREATE MATERIALIZED VIEW stationary_combustion_service.sc_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', event_time) AS bucket,
    fuel_type,
    COUNT(*)                           AS total_calculations,
    SUM(total_co2e_kg)                 AS sum_co2e_kg,
    AVG(duration_ms)                   AS avg_duration_ms,
    MAX(duration_ms)                   AS max_duration_ms
FROM stationary_combustion_service.sc_calculation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, fuel_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'stationary_combustion_service.sc_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- sc_daily_emission_totals: daily count/sum(co2e)/sum(co2)/sum(ch4)/sum(n2o)/sum(biogenic) by fuel_type
-- NOTE: Per-gas daily totals are derived from sc_calculation_events combined with
-- sc_calculation_details. This aggregate provides the total CO2e from calculation events.
-- For per-gas breakdowns, join with sc_calculation_details at query time.
CREATE MATERIALIZED VIEW stationary_combustion_service.sc_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', event_time)  AS bucket,
    fuel_type,
    COUNT(*)                           AS total_calculations,
    SUM(total_co2e_kg)                 AS sum_co2e_kg
FROM stationary_combustion_service.sc_calculation_events
WHERE event_time IS NOT NULL
GROUP BY bucket, fuel_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'stationary_combustion_service.sc_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- sc_fuel_types indexes (10)
CREATE INDEX IF NOT EXISTS idx_sc_ft_fuel_type_id        ON stationary_combustion_service.sc_fuel_types(fuel_type_id);
CREATE INDEX IF NOT EXISTS idx_sc_ft_name                ON stationary_combustion_service.sc_fuel_types(name);
CREATE INDEX IF NOT EXISTS idx_sc_ft_category            ON stationary_combustion_service.sc_fuel_types(category);
CREATE INDEX IF NOT EXISTS idx_sc_ft_is_biogenic         ON stationary_combustion_service.sc_fuel_types(is_biogenic);
CREATE INDEX IF NOT EXISTS idx_sc_ft_ipcc_code           ON stationary_combustion_service.sc_fuel_types(ipcc_code);
CREATE INDEX IF NOT EXISTS idx_sc_ft_created_at          ON stationary_combustion_service.sc_fuel_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ft_updated_at          ON stationary_combustion_service.sc_fuel_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ft_tenant_id           ON stationary_combustion_service.sc_fuel_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_ft_category_tenant     ON stationary_combustion_service.sc_fuel_types(category, tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_ft_properties          ON stationary_combustion_service.sc_fuel_types USING GIN (properties);

-- sc_emission_factors indexes (12)
CREATE INDEX IF NOT EXISTS idx_sc_ef_factor_id           ON stationary_combustion_service.sc_emission_factors(factor_id);
CREATE INDEX IF NOT EXISTS idx_sc_ef_fuel_type           ON stationary_combustion_service.sc_emission_factors(fuel_type);
CREATE INDEX IF NOT EXISTS idx_sc_ef_gas                 ON stationary_combustion_service.sc_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_sc_ef_source              ON stationary_combustion_service.sc_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_sc_ef_tier                ON stationary_combustion_service.sc_emission_factors(tier);
CREATE INDEX IF NOT EXISTS idx_sc_ef_geography           ON stationary_combustion_service.sc_emission_factors(geography);
CREATE INDEX IF NOT EXISTS idx_sc_ef_effective_date      ON stationary_combustion_service.sc_emission_factors(effective_date DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ef_expiry_date         ON stationary_combustion_service.sc_emission_factors(expiry_date DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ef_tenant_id           ON stationary_combustion_service.sc_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_ef_fuel_gas            ON stationary_combustion_service.sc_emission_factors(fuel_type, gas);
CREATE INDEX IF NOT EXISTS idx_sc_ef_fuel_source_tier    ON stationary_combustion_service.sc_emission_factors(fuel_type, source, tier);
CREATE INDEX IF NOT EXISTS idx_sc_ef_metadata            ON stationary_combustion_service.sc_emission_factors USING GIN (metadata);

-- sc_heating_values indexes (8)
CREATE INDEX IF NOT EXISTS idx_sc_hv_heating_value_id    ON stationary_combustion_service.sc_heating_values(heating_value_id);
CREATE INDEX IF NOT EXISTS idx_sc_hv_fuel_type           ON stationary_combustion_service.sc_heating_values(fuel_type);
CREATE INDEX IF NOT EXISTS idx_sc_hv_basis               ON stationary_combustion_service.sc_heating_values(basis);
CREATE INDEX IF NOT EXISTS idx_sc_hv_source              ON stationary_combustion_service.sc_heating_values(source);
CREATE INDEX IF NOT EXISTS idx_sc_hv_tenant_id           ON stationary_combustion_service.sc_heating_values(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_hv_fuel_basis          ON stationary_combustion_service.sc_heating_values(fuel_type, basis);
CREATE INDEX IF NOT EXISTS idx_sc_hv_created_at          ON stationary_combustion_service.sc_heating_values(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_hv_updated_at          ON stationary_combustion_service.sc_heating_values(updated_at DESC);

-- sc_oxidation_factors indexes (6)
CREATE INDEX IF NOT EXISTS idx_sc_of_oxidation_factor_id ON stationary_combustion_service.sc_oxidation_factors(oxidation_factor_id);
CREATE INDEX IF NOT EXISTS idx_sc_of_fuel_type           ON stationary_combustion_service.sc_oxidation_factors(fuel_type);
CREATE INDEX IF NOT EXISTS idx_sc_of_source              ON stationary_combustion_service.sc_oxidation_factors(source);
CREATE INDEX IF NOT EXISTS idx_sc_of_tenant_id           ON stationary_combustion_service.sc_oxidation_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_of_created_at          ON stationary_combustion_service.sc_oxidation_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_of_updated_at          ON stationary_combustion_service.sc_oxidation_factors(updated_at DESC);

-- sc_equipment_profiles indexes (12)
CREATE INDEX IF NOT EXISTS idx_sc_ep_equipment_id        ON stationary_combustion_service.sc_equipment_profiles(equipment_id);
CREATE INDEX IF NOT EXISTS idx_sc_ep_equipment_type      ON stationary_combustion_service.sc_equipment_profiles(equipment_type);
CREATE INDEX IF NOT EXISTS idx_sc_ep_name                ON stationary_combustion_service.sc_equipment_profiles(name);
CREATE INDEX IF NOT EXISTS idx_sc_ep_facility_id         ON stationary_combustion_service.sc_equipment_profiles(facility_id);
CREATE INDEX IF NOT EXISTS idx_sc_ep_maintenance_status  ON stationary_combustion_service.sc_equipment_profiles(maintenance_status);
CREATE INDEX IF NOT EXISTS idx_sc_ep_created_at          ON stationary_combustion_service.sc_equipment_profiles(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ep_updated_at          ON stationary_combustion_service.sc_equipment_profiles(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ep_tenant_id           ON stationary_combustion_service.sc_equipment_profiles(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_ep_type_facility       ON stationary_combustion_service.sc_equipment_profiles(equipment_type, facility_id);
CREATE INDEX IF NOT EXISTS idx_sc_ep_tenant_type         ON stationary_combustion_service.sc_equipment_profiles(tenant_id, equipment_type);
CREATE INDEX IF NOT EXISTS idx_sc_ep_efficiency_curve    ON stationary_combustion_service.sc_equipment_profiles USING GIN (efficiency_curve);
CREATE INDEX IF NOT EXISTS idx_sc_ep_metadata            ON stationary_combustion_service.sc_equipment_profiles USING GIN (metadata);

-- sc_calculations indexes (14)
CREATE INDEX IF NOT EXISTS idx_sc_calc_calculation_id    ON stationary_combustion_service.sc_calculations(calculation_id);
CREATE INDEX IF NOT EXISTS idx_sc_calc_fuel_type         ON stationary_combustion_service.sc_calculations(fuel_type);
CREATE INDEX IF NOT EXISTS idx_sc_calc_equipment_type    ON stationary_combustion_service.sc_calculations(equipment_type);
CREATE INDEX IF NOT EXISTS idx_sc_calc_tier_used         ON stationary_combustion_service.sc_calculations(tier_used);
CREATE INDEX IF NOT EXISTS idx_sc_calc_total_co2e_kg     ON stationary_combustion_service.sc_calculations(total_co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_sc_calc_total_co2e_tonnes ON stationary_combustion_service.sc_calculations(total_co2e_tonnes DESC);
CREATE INDEX IF NOT EXISTS idx_sc_calc_gwp_source        ON stationary_combustion_service.sc_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_sc_calc_facility_id       ON stationary_combustion_service.sc_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_sc_calc_organization_id   ON stationary_combustion_service.sc_calculations(organization_id);
CREATE INDEX IF NOT EXISTS idx_sc_calc_status            ON stationary_combustion_service.sc_calculations(status);
CREATE INDEX IF NOT EXISTS idx_sc_calc_provenance_hash   ON stationary_combustion_service.sc_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_sc_calc_created_at        ON stationary_combustion_service.sc_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_calc_tenant_id         ON stationary_combustion_service.sc_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_calc_fuel_tenant       ON stationary_combustion_service.sc_calculations(fuel_type, tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_calc_facility_period   ON stationary_combustion_service.sc_calculations(facility_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_sc_calc_trace             ON stationary_combustion_service.sc_calculations USING GIN (calculation_trace);
CREATE INDEX IF NOT EXISTS idx_sc_calc_metadata          ON stationary_combustion_service.sc_calculations USING GIN (metadata);

-- sc_calculation_details indexes (10)
CREATE INDEX IF NOT EXISTS idx_sc_cd_detail_id           ON stationary_combustion_service.sc_calculation_details(detail_id);
CREATE INDEX IF NOT EXISTS idx_sc_cd_calculation_id      ON stationary_combustion_service.sc_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_sc_cd_gas                 ON stationary_combustion_service.sc_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_sc_cd_emissions_kg        ON stationary_combustion_service.sc_calculation_details(emissions_kg DESC);
CREATE INDEX IF NOT EXISTS idx_sc_cd_emissions_tco2e     ON stationary_combustion_service.sc_calculation_details(emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_sc_cd_ef_source           ON stationary_combustion_service.sc_calculation_details(emission_factor_source);
CREATE INDEX IF NOT EXISTS idx_sc_cd_is_biogenic         ON stationary_combustion_service.sc_calculation_details(is_biogenic);
CREATE INDEX IF NOT EXISTS idx_sc_cd_tenant_id           ON stationary_combustion_service.sc_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_cd_calc_gas            ON stationary_combustion_service.sc_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_sc_cd_tenant_calc         ON stationary_combustion_service.sc_calculation_details(tenant_id, calculation_id);

-- sc_facility_aggregations indexes (12)
CREATE INDEX IF NOT EXISTS idx_sc_fa_aggregation_id      ON stationary_combustion_service.sc_facility_aggregations(aggregation_id);
CREATE INDEX IF NOT EXISTS idx_sc_fa_facility_id         ON stationary_combustion_service.sc_facility_aggregations(facility_id);
CREATE INDEX IF NOT EXISTS idx_sc_fa_organization_id     ON stationary_combustion_service.sc_facility_aggregations(organization_id);
CREATE INDEX IF NOT EXISTS idx_sc_fa_control_approach    ON stationary_combustion_service.sc_facility_aggregations(control_approach);
CREATE INDEX IF NOT EXISTS idx_sc_fa_period_type         ON stationary_combustion_service.sc_facility_aggregations(reporting_period_type);
CREATE INDEX IF NOT EXISTS idx_sc_fa_period_start        ON stationary_combustion_service.sc_facility_aggregations(period_start DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fa_period_end          ON stationary_combustion_service.sc_facility_aggregations(period_end DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fa_total_co2e          ON stationary_combustion_service.sc_facility_aggregations(total_co2e_tonnes DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fa_provenance_hash     ON stationary_combustion_service.sc_facility_aggregations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_sc_fa_tenant_id           ON stationary_combustion_service.sc_facility_aggregations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_fa_facility_period     ON stationary_combustion_service.sc_facility_aggregations(facility_id, period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_sc_fa_tenant_facility     ON stationary_combustion_service.sc_facility_aggregations(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_sc_fa_fuel_types_used     ON stationary_combustion_service.sc_facility_aggregations USING GIN (fuel_types_used);
CREATE INDEX IF NOT EXISTS idx_sc_fa_metadata            ON stationary_combustion_service.sc_facility_aggregations USING GIN (metadata);

-- sc_audit_entries indexes (10)
CREATE INDEX IF NOT EXISTS idx_sc_ae_entry_id            ON stationary_combustion_service.sc_audit_entries(entry_id);
CREATE INDEX IF NOT EXISTS idx_sc_ae_calculation_id      ON stationary_combustion_service.sc_audit_entries(calculation_id);
CREATE INDEX IF NOT EXISTS idx_sc_ae_step_number         ON stationary_combustion_service.sc_audit_entries(step_number);
CREATE INDEX IF NOT EXISTS idx_sc_ae_step_name           ON stationary_combustion_service.sc_audit_entries(step_name);
CREATE INDEX IF NOT EXISTS idx_sc_ae_ef_used             ON stationary_combustion_service.sc_audit_entries(emission_factor_used);
CREATE INDEX IF NOT EXISTS idx_sc_ae_provenance_hash     ON stationary_combustion_service.sc_audit_entries(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_sc_ae_created_at          ON stationary_combustion_service.sc_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ae_tenant_id           ON stationary_combustion_service.sc_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_ae_calc_step           ON stationary_combustion_service.sc_audit_entries(calculation_id, step_number);
CREATE INDEX IF NOT EXISTS idx_sc_ae_input_data          ON stationary_combustion_service.sc_audit_entries USING GIN (input_data);
CREATE INDEX IF NOT EXISTS idx_sc_ae_output_data         ON stationary_combustion_service.sc_audit_entries USING GIN (output_data);

-- sc_custom_factors indexes (12)
CREATE INDEX IF NOT EXISTS idx_sc_cf_custom_factor_id    ON stationary_combustion_service.sc_custom_factors(custom_factor_id);
CREATE INDEX IF NOT EXISTS idx_sc_cf_fuel_type           ON stationary_combustion_service.sc_custom_factors(fuel_type);
CREATE INDEX IF NOT EXISTS idx_sc_cf_gas                 ON stationary_combustion_service.sc_custom_factors(gas);
CREATE INDEX IF NOT EXISTS idx_sc_cf_geography           ON stationary_combustion_service.sc_custom_factors(geography);
CREATE INDEX IF NOT EXISTS idx_sc_cf_approved_by         ON stationary_combustion_service.sc_custom_factors(approved_by);
CREATE INDEX IF NOT EXISTS idx_sc_cf_approval_status     ON stationary_combustion_service.sc_custom_factors(approval_status);
CREATE INDEX IF NOT EXISTS idx_sc_cf_valid_from          ON stationary_combustion_service.sc_custom_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_sc_cf_valid_until         ON stationary_combustion_service.sc_custom_factors(valid_until DESC);
CREATE INDEX IF NOT EXISTS idx_sc_cf_tenant_id           ON stationary_combustion_service.sc_custom_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sc_cf_fuel_gas            ON stationary_combustion_service.sc_custom_factors(fuel_type, gas);
CREATE INDEX IF NOT EXISTS idx_sc_cf_tenant_status       ON stationary_combustion_service.sc_custom_factors(tenant_id, approval_status);
CREATE INDEX IF NOT EXISTS idx_sc_cf_metadata            ON stationary_combustion_service.sc_custom_factors USING GIN (metadata);

-- sc_calculation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_sc_ce_calculation_id      ON stationary_combustion_service.sc_calculation_events(calculation_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ce_fuel_type           ON stationary_combustion_service.sc_calculation_events(fuel_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ce_tier                ON stationary_combustion_service.sc_calculation_events(tier, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ce_status              ON stationary_combustion_service.sc_calculation_events(status, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ce_tenant_id           ON stationary_combustion_service.sc_calculation_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_ce_fuel_tenant         ON stationary_combustion_service.sc_calculation_events(fuel_type, tenant_id, event_time DESC);

-- sc_factor_updates indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_sc_fu_factor_id           ON stationary_combustion_service.sc_factor_updates(factor_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fu_fuel_type           ON stationary_combustion_service.sc_factor_updates(fuel_type, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fu_gas                 ON stationary_combustion_service.sc_factor_updates(gas, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fu_source              ON stationary_combustion_service.sc_factor_updates(source, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fu_tenant_id           ON stationary_combustion_service.sc_factor_updates(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_fu_fuel_gas            ON stationary_combustion_service.sc_factor_updates(fuel_type, gas, event_time DESC);

-- sc_audit_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_sc_aue_calculation_id     ON stationary_combustion_service.sc_audit_events(calculation_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_aue_step_name          ON stationary_combustion_service.sc_audit_events(step_name, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_aue_provenance_hash    ON stationary_combustion_service.sc_audit_events(provenance_hash, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_aue_tenant_id          ON stationary_combustion_service.sc_audit_events(tenant_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_aue_calc_step          ON stationary_combustion_service.sc_audit_events(calculation_id, step_name, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_sc_aue_tenant_calc        ON stationary_combustion_service.sc_audit_events(tenant_id, calculation_id, event_time DESC);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- sc_fuel_types: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_fuel_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_ft_read  ON stationary_combustion_service.sc_fuel_types FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_ft_write ON stationary_combustion_service.sc_fuel_types FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_emission_factors: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_ef_read  ON stationary_combustion_service.sc_emission_factors FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_ef_write ON stationary_combustion_service.sc_emission_factors FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_heating_values: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_heating_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_hv_read  ON stationary_combustion_service.sc_heating_values FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_hv_write ON stationary_combustion_service.sc_heating_values FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_oxidation_factors: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_oxidation_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_of_read  ON stationary_combustion_service.sc_oxidation_factors FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_of_write ON stationary_combustion_service.sc_oxidation_factors FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_equipment_profiles: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_equipment_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_ep_read  ON stationary_combustion_service.sc_equipment_profiles FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_ep_write ON stationary_combustion_service.sc_equipment_profiles FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_calculations: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_calc_read  ON stationary_combustion_service.sc_calculations FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_calc_write ON stationary_combustion_service.sc_calculations FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_calculation_details: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_cd_read  ON stationary_combustion_service.sc_calculation_details FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_cd_write ON stationary_combustion_service.sc_calculation_details FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_facility_aggregations: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_facility_aggregations ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_fa_read  ON stationary_combustion_service.sc_facility_aggregations FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_fa_write ON stationary_combustion_service.sc_facility_aggregations FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_audit_entries: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_ae_read  ON stationary_combustion_service.sc_audit_entries FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_ae_write ON stationary_combustion_service.sc_audit_entries FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_custom_factors: tenant-isolated
ALTER TABLE stationary_combustion_service.sc_custom_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_cf_read  ON stationary_combustion_service.sc_custom_factors FOR SELECT USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY sc_cf_write ON stationary_combustion_service.sc_custom_factors FOR ALL USING (
    tenant_id = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- sc_calculation_events: open read/write (time-series telemetry)
ALTER TABLE stationary_combustion_service.sc_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_ce_read  ON stationary_combustion_service.sc_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY sc_ce_write ON stationary_combustion_service.sc_calculation_events FOR ALL   USING (TRUE);

-- sc_factor_updates: open read/write (time-series telemetry)
ALTER TABLE stationary_combustion_service.sc_factor_updates ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_fu_read  ON stationary_combustion_service.sc_factor_updates FOR SELECT USING (TRUE);
CREATE POLICY sc_fu_write ON stationary_combustion_service.sc_factor_updates FOR ALL   USING (TRUE);

-- sc_audit_events: open read/write (time-series telemetry)
ALTER TABLE stationary_combustion_service.sc_audit_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY sc_aue_read  ON stationary_combustion_service.sc_audit_events FOR SELECT USING (TRUE);
CREATE POLICY sc_aue_write ON stationary_combustion_service.sc_audit_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA stationary_combustion_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA stationary_combustion_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA stationary_combustion_service TO greenlang_app;
GRANT SELECT ON stationary_combustion_service.sc_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON stationary_combustion_service.sc_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA stationary_combustion_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA stationary_combustion_service TO greenlang_readonly;
GRANT SELECT ON stationary_combustion_service.sc_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON stationary_combustion_service.sc_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA stationary_combustion_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA stationary_combustion_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA stationary_combustion_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'stationary-combustion:fuels:read',             'stationary-combustion', 'fuels_read',             'View fuel type registry with HHV/NCV, density, carbon content, and IPCC codes'),
    (gen_random_uuid(), 'stationary-combustion:fuels:write',            'stationary-combustion', 'fuels_write',            'Create, update, and manage fuel type entries in the registry'),
    (gen_random_uuid(), 'stationary-combustion:factors:read',           'stationary-combustion', 'factors_read',           'View emission factors from EPA/IPCC/DEFRA/EU_ETS sources across Tier 1-3'),
    (gen_random_uuid(), 'stationary-combustion:factors:write',          'stationary-combustion', 'factors_write',          'Create, update, and manage emission factor entries with source and tier'),
    (gen_random_uuid(), 'stationary-combustion:equipment:read',         'stationary-combustion', 'equipment_read',         'View equipment profiles with capacity, efficiency curves, and load factors'),
    (gen_random_uuid(), 'stationary-combustion:equipment:write',        'stationary-combustion', 'equipment_write',        'Create, update, and manage equipment profile entries'),
    (gen_random_uuid(), 'stationary-combustion:calculations:read',      'stationary-combustion', 'calculations_read',      'View calculation results with CO2e totals, per-gas breakdowns, and provenance'),
    (gen_random_uuid(), 'stationary-combustion:calculations:write',     'stationary-combustion', 'calculations_write',     'Create and update calculation result records'),
    (gen_random_uuid(), 'stationary-combustion:calculations:execute',   'stationary-combustion', 'calculations_execute',   'Execute stationary combustion emission calculations with full audit trail'),
    (gen_random_uuid(), 'stationary-combustion:aggregations:read',      'stationary-combustion', 'aggregations_read',      'View facility-level emission aggregations by control approach and period'),
    (gen_random_uuid(), 'stationary-combustion:aggregations:execute',   'stationary-combustion', 'aggregations_execute',   'Execute facility emission aggregation rollups with provenance tracking'),
    (gen_random_uuid(), 'stationary-combustion:uncertainty:execute',    'stationary-combustion', 'uncertainty_execute',    'Execute uncertainty quantification analysis on calculation results'),
    (gen_random_uuid(), 'stationary-combustion:audit:read',             'stationary-combustion', 'audit_read',             'View step-by-step audit trail entries for calculations with methodology references'),
    (gen_random_uuid(), 'stationary-combustion:validate:execute',       'stationary-combustion', 'validate_execute',       'Execute validation checks on input data and calculation results'),
    (gen_random_uuid(), 'stationary-combustion:health:read',            'stationary-combustion', 'health_read',            'View stationary combustion service health status and diagnostics'),
    (gen_random_uuid(), 'stationary-combustion:stats:read',             'stationary-combustion', 'stats_read',             'View stationary combustion service statistics and continuous aggregates'),
    (gen_random_uuid(), 'stationary-combustion:custom-factors:read',    'stationary-combustion', 'custom_factors_read',    'View user-defined custom emission factors with approval status'),
    (gen_random_uuid(), 'stationary-combustion:custom-factors:write',   'stationary-combustion', 'custom_factors_write',   'Create, update, and manage custom emission factors with approval workflow')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('stationary_combustion_service.sc_calculation_events', INTERVAL '90 days');
SELECT add_retention_policy('stationary_combustion_service.sc_factor_updates',     INTERVAL '90 days');
SELECT add_retention_policy('stationary_combustion_service.sc_audit_events',       INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE stationary_combustion_service.sc_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'fuel_type',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('stationary_combustion_service.sc_calculation_events', INTERVAL '7 days');

ALTER TABLE stationary_combustion_service.sc_factor_updates
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'fuel_type',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('stationary_combustion_service.sc_factor_updates', INTERVAL '7 days');

ALTER TABLE stationary_combustion_service.sc_audit_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'calculation_id',
         timescaledb.compress_orderby   = 'event_time DESC');
SELECT add_compression_policy('stationary_combustion_service.sc_audit_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Stationary Combustion Calculator (GL-MRV-X-001)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-X-001',
    'Stationary Combustion Calculator',
    'Stationary combustion emission calculator for GreenLang Climate OS. Manages fuel type registry with gaseous/liquid/solid/biomass categories including HHV/NCV heating values, density, carbon content, and IPCC codes. Maintains emission factor database from EPA/IPCC/DEFRA/EU_ETS/CUSTOM sources across Tier 1-3 with geography-specific factors and effective date ranges. Stores heating values (HHV/NCV basis) and oxidation factors (0-1 range) per fuel type. Registers equipment profiles with rated capacity (MMBtu/hr), efficiency curves, age, maintenance status, and load factor ranges. Executes deterministic combustion emission calculations converting fuel quantity through energy (GJ) via heating values, applying oxidation factors, and computing per-gas emissions (CO2/CH4/N2O) with GWP weighting (AR4/AR5/AR6) to total CO2e. Tracks biogenic CO2 separately. Produces per-gas calculation detail breakdowns with individual emission factors, GWP values, and biogenic flags. Aggregates facility-level emissions by operational/financial/equity share control approaches across monthly/quarterly/annual reporting periods. Generates step-by-step audit trail entries with input/output data, emission factors used, and methodology references. Supports user-defined custom emission factors with approval workflows. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/stationary-combustion-calculator',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-X-001', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/stationary-combustion-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"stationary-combustion", "scope-1", "ghg-protocol", "epa", "ipcc", "emission-factors", "mrv"}',
    '{"cross-sector", "energy", "manufacturing", "utilities", "mining", "real-estate"}',
    'b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-X-001', '1.0.0',
    'fuel_database',
    'configuration',
    'Register and manage fuel type entries with category classification (gaseous/liquid/solid/biomass), HHV/NCV heating values, density, carbon content percentage, biogenic flag, and IPCC codes.',
    '{"fuel_type_id", "name", "category", "hhv", "hhv_unit", "ncv", "density", "carbon_content_pct", "is_biogenic", "ipcc_code"}',
    '{"fuel_type_id", "registration_result"}',
    '{"categories": ["gaseous", "liquid", "solid", "biomass"], "supports_ipcc_codes": true, "supports_biogenic": true}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'combustion_calculation',
    'processing',
    'Execute deterministic stationary combustion emission calculations converting fuel quantity to CO2e through energy conversion (GJ), heating value application (HHV/NCV), oxidation factor correction, and multi-gas GWP weighting (CO2/CH4/N2O).',
    '{"fuel_type", "fuel_quantity", "fuel_unit", "tier", "gwp_source", "equipment_type", "facility_id", "period_start", "period_end"}',
    '{"calculation_id", "total_co2e_kg", "total_co2e_tonnes", "energy_gj", "per_gas_breakdown", "provenance_hash"}',
    '{"tiers": [1, 2, 3], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'equipment_profiling',
    'configuration',
    'Register and manage equipment profiles with type classification, rated capacity (MMBtu/hr), efficiency curves, age tracking, maintenance status, and load factor ranges for combustion equipment.',
    '{"equipment_id", "equipment_type", "name", "facility_id", "rated_capacity_mmbtu_hr", "efficiency_curve", "age_years", "maintenance_status"}',
    '{"equipment_id", "registration_result"}',
    '{"maintenance_statuses": ["good", "fair", "poor", "maintenance_required", "decommissioned"], "supports_efficiency_curves": true}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'emission_factor_selection',
    'processing',
    'Select appropriate emission factors based on fuel type, gas, tier level, source preference, geography, and effective date. Supports EPA/IPCC/DEFRA/EU_ETS/CUSTOM sources with tier-based selection.',
    '{"fuel_type", "gas", "tier", "source_preference", "geography", "effective_date"}',
    '{"factor_id", "value", "unit", "source", "tier", "reference"}',
    '{"sources": ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"], "tiers": [1, 2, 3], "default_geography": "GLOBAL"}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'uncertainty_quantification',
    'processing',
    'Quantify uncertainty in emission calculations based on IPCC Tier methodology, emission factor confidence intervals, measurement uncertainty, and equipment calibration status.',
    '{"calculation_id", "tier", "emission_factor_source", "measurement_method"}',
    '{"uncertainty_pct", "confidence_interval", "methodology_reference"}',
    '{"tier_uncertainty_ranges": {"1": "+-30%", "2": "+-15%", "3": "+-5%"}, "ipcc_2006_guidelines": true}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive step-by-step audit trail for each calculation with input/output data per step, emission factors used, methodology references, and SHA-256 provenance hashes.',
    '{"calculation_id"}',
    '{"audit_entries", "provenance_chain", "methodology_references"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_step": true, "includes_methodology_refs": true}'::jsonb
),
(
    'GL-MRV-X-001', '1.0.0',
    'pipeline_orchestration',
    'processing',
    'Orchestrate end-to-end stationary combustion calculation pipelines including data intake, validation, factor selection, calculation, per-gas breakdown, facility aggregation, and audit trail generation.',
    '{"fuel_records", "facility_id", "organization_id", "control_approach", "reporting_period_type", "period_start", "period_end"}',
    '{"aggregation_id", "total_co2e_tonnes", "per_gas_totals", "calculation_count", "provenance_hash"}',
    '{"control_approaches": ["OPERATIONAL", "FINANCIAL", "EQUITY_SHARE"], "reporting_periods": ["MONTHLY", "QUARTERLY", "ANNUAL"], "batch_processing": true}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-X-001', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage stationary combustion calculation pipeline execution ordering'),
    ('GL-MRV-X-001', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for fuel quantities, heating values, and emission factor unit conversions'),
    ('GL-MRV-X-001', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-X-001', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-X-001', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, factor updates, and audit event telemetry'),
    ('GL-MRV-X-001', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of fuel consumption records')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-X-001',
    'Stationary Combustion Calculator',
    'Stationary combustion emission calculator. Fuel type registry (gaseous/liquid/solid/biomass, HHV/NCV, density, carbon content, IPCC codes). Emission factor database (EPA/IPCC/DEFRA/EU_ETS/CUSTOM, Tier 1-3, geography-specific, effective dates). Heating values (HHV/NCV basis). Oxidation factors (0-1 range). Equipment profiles (capacity, efficiency curves, load factors, maintenance status). Deterministic combustion calculations (fuel to energy GJ to CO2e, multi-gas CO2/CH4/N2O, GWP AR4/AR5/AR6). Per-gas breakdowns with individual EFs and GWP. Biogenic CO2 tracking. Facility aggregations (OPERATIONAL/FINANCIAL/EQUITY_SHARE, MONTHLY/QUARTERLY/ANNUAL). Custom emission factors with approval workflow. Step-by-step audit trail. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA stationary_combustion_service IS
    'Stationary Combustion Calculator (AGENT-MRV-001) - fuel type registry, emission factor database, heating values, oxidation factors, equipment profiles, combustion calculations, per-gas breakdowns, facility aggregations, audit trail, custom factors, provenance chains';

COMMENT ON TABLE stationary_combustion_service.sc_fuel_types IS
    'Fuel type registry: fuel_type_id (unique), name, category (gaseous/liquid/solid/biomass), hhv, hhv_unit, ncv, ncv_unit, density, density_unit, carbon_content_pct, is_biogenic, ipcc_code, properties JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_emission_factors IS
    'Emission factor database: factor_id (unique), fuel_type, gas (CO2/CH4/N2O), value, unit, source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), tier (1/2/3), geography, effective_date, expiry_date, reference, notes, metadata JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_heating_values IS
    'Heating value storage: heating_value_id (unique), fuel_type, basis (HHV/NCV), value, unit, source, reference, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_oxidation_factors IS
    'Oxidation factor registry: oxidation_factor_id (unique), fuel_type, value (0-1), source, reference, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_equipment_profiles IS
    'Equipment profile registry: equipment_id (unique), equipment_type, name, facility_id, rated_capacity_mmbtu_hr, efficiency_curve JSONB, age_years, maintenance_status, load_factor_min/max, metadata JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_calculations IS
    'Calculation results: calculation_id (unique), fuel_type, equipment_type, fuel_quantity/unit, energy_gj, heating_value_used/basis, oxidation_factor_used, tier_used, total_co2e_kg/tonnes, biogenic_co2_kg, gwp_source (AR4/AR5/AR6), regulatory_framework, provenance_hash, calculation_trace JSONB, facility_id, source_id, period_start/end, organization_id, status, metadata JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_calculation_details IS
    'Per-gas calculation breakdown: detail_id (unique), calculation_id FK, gas (CO2/CH4/N2O), emissions_kg, emissions_tco2e, emission_factor_value/unit/source, gwp_applied, is_biogenic, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_facility_aggregations IS
    'Facility-level emission aggregations: aggregation_id (unique), facility_id, organization_id, control_approach (OPERATIONAL/FINANCIAL/EQUITY_SHARE), reporting_period_type (MONTHLY/QUARTERLY/ANNUAL), period_start/end, total_co2e/co2/ch4/n2o/biogenic_tonnes, calculation_count, equipment_count, fuel_types_used JSONB, provenance_hash, metadata JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_audit_entries IS
    'Audit trail entries: entry_id (unique), calculation_id, step_number, step_name, input_data JSONB, output_data JSONB, emission_factor_used, methodology_reference, provenance_hash, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_custom_factors IS
    'User-defined custom emission factors: custom_factor_id (unique), fuel_type, gas (CO2/CH4/N2O), value, unit, geography, reference, approved_by, approval_status (pending/approved/rejected/expired/revoked), approval_date, valid_from/until, metadata JSONB, tenant_id';

COMMENT ON TABLE stationary_combustion_service.sc_calculation_events IS
    'TimescaleDB hypertable: calculation events with calculation_id, fuel_type, tier, total_co2e_kg, duration_ms, status, tenant_id, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE stationary_combustion_service.sc_factor_updates IS
    'TimescaleDB hypertable: factor update events with factor_id, fuel_type, gas, old_value, new_value, source, updated_by, tenant_id, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE stationary_combustion_service.sc_audit_events IS
    'TimescaleDB hypertable: audit events with calculation_id, step_name, provenance_hash, tenant_id, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW stationary_combustion_service.sc_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by fuel_type (total calculations, sum CO2e kg, avg/max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW stationary_combustion_service.sc_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by fuel_type (total calculations, sum CO2e kg per day)';
