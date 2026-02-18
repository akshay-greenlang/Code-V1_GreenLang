-- =============================================================================
-- V054: Process Emissions Service Schema
-- =============================================================================
-- Component: AGENT-MRV-004 (GL-MRV-SCOPE1-004)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Process Emissions Agent (GL-MRV-SCOPE1-004) with capabilities for
-- industrial process type registry management (mineral, chemical, metal,
-- electronics, pulp_paper, other categories with primary greenhouse gases,
-- applicable tiers, EPA subpart references, default emission factors, and
-- production route tracking), raw material registry (material types with
-- carbon content, carbonate content and type, molecular weight, moisture
-- content, heating value, and density properties), emission factor database
-- (process type x gas factors with EPA/IPCC/DEFRA/EU_ETS/CUSTOM sources
-- across Tier 1/2/3, production routes, validity date ranges, and
-- references), process unit management (unit name, type, process type,
-- facility, capacity, process mode CONTINUOUS/BATCH/SEMI_BATCH,
-- commissioning and decommission dates), material input tracking (per-
-- calculation material inputs with quantities, carbon/carbonate/moisture
-- content, product and by-product classification), process emission
-- calculations (mass balance, emission factor, and direct measurement
-- methods with multi-gas GWP weighting CO2/CH4/N2O/PFC/SF6/NF3 using
-- AR4/AR5/AR6 values, abatement efficiency, uncertainty quantification,
-- and provenance hashing), per-gas calculation detail breakdowns
-- (individual emission factors, GWP values, raw and CO2e emissions with
-- calculation trace), abatement records (destruction efficiency,
-- verification status, monitoring frequency, cost per tonne CO2e),
-- regulatory compliance records (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064
-- framework checks with findings and recommendations), and step-by-step
-- audit trail entries (entity-level action trace with hash chaining).
-- SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. pe_process_types          - Industrial process type registry (mineral/chemical/metal/electronics/pulp_paper/other)
--   2. pe_raw_materials          - Raw material registry (carbon content, carbonate, molecular weight, density)
--   3. pe_emission_factors       - Emission factor database (process x gas, Tier 1/2/3, production routes)
--   4. pe_process_units          - Process unit registry (capacity, mode CONTINUOUS/BATCH/SEMI_BATCH)
--   5. pe_material_inputs        - Material input records per calculation (quantities, product/by-product flags)
--   6. pe_calculations           - Calculation results (multi-gas CO2e with abatement and provenance)
--   7. pe_calculation_details    - Per-gas breakdown (EF, GWP, raw and CO2e emissions, trace)
--   8. pe_abatement_records      - Abatement equipment records (efficiency, verification, cost)
--   9. pe_compliance_records     - Regulatory compliance (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064)
--  10. pe_audit_entries          - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  11. pe_calculation_events     - Calculation event time-series (hypertable on event_time)
--  12. pe_material_events        - Material event time-series (hypertable on event_time)
--  13. pe_compliance_events      - Compliance event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. pe_hourly_calculation_stats  - Hourly count/sum(co2e)/avg(co2e) by process_type and method
--   2. pe_daily_emission_totals     - Daily count/sum(co2e) by process_type
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-004.
-- Previous: V053__mobile_combustion_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS process_emissions_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION process_emissions_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: process_emissions_service.pe_process_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_process_types (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID          NOT NULL,
    process_type            VARCHAR(100)  NOT NULL,
    category                VARCHAR(50)   NOT NULL,
    name                    VARCHAR(200)  NOT NULL,
    description             TEXT,
    primary_gases           JSONB         NOT NULL DEFAULT '[]'::jsonb,
    applicable_tiers        JSONB         NOT NULL DEFAULT '["TIER_1"]'::jsonb,
    epa_subpart             VARCHAR(20),
    default_emission_factor DECIMAL(20,10),
    default_ef_unit         VARCHAR(50),
    default_ef_source       VARCHAR(50)   DEFAULT 'IPCC',
    production_routes       JSONB         DEFAULT '[]'::jsonb,
    is_active               BOOLEAN       DEFAULT TRUE,
    metadata                JSONB         DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_pe_pt_tenant_process_type UNIQUE (tenant_id, process_type)
);

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_process_type_not_empty CHECK (LENGTH(TRIM(process_type)) > 0);

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_category CHECK (category IN (
        'mineral', 'chemical', 'metal', 'electronics', 'pulp_paper', 'other'
    ));

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_default_ef_positive CHECK (default_emission_factor IS NULL OR default_emission_factor > 0);

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_default_ef_source CHECK (default_ef_source IS NULL OR default_ef_source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE process_emissions_service.pe_process_types
    ADD CONSTRAINT chk_pt_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_pt_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_process_types
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 2: process_emissions_service.pe_raw_materials
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_raw_materials (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    material_type       VARCHAR(100)    NOT NULL,
    name                VARCHAR(200)    NOT NULL,
    description         TEXT,
    carbon_content      DECIMAL(10,6),
    carbonate_content   DECIMAL(10,6),
    carbonate_type      VARCHAR(50),
    molecular_weight    DECIMAL(10,4),
    moisture_content    DECIMAL(10,6)   DEFAULT 0,
    heating_value       DECIMAL(15,6),
    heating_value_unit  VARCHAR(30),
    density             DECIMAL(10,4),
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_pe_rm_tenant_material_type UNIQUE (tenant_id, material_type)
);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_material_type_not_empty CHECK (LENGTH(TRIM(material_type)) > 0);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_carbon_content_range CHECK (carbon_content IS NULL OR (carbon_content >= 0 AND carbon_content <= 1));

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_carbonate_content_range CHECK (carbonate_content IS NULL OR (carbonate_content >= 0 AND carbonate_content <= 1));

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_molecular_weight_positive CHECK (molecular_weight IS NULL OR molecular_weight > 0);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_moisture_content_range CHECK (moisture_content IS NULL OR (moisture_content >= 0 AND moisture_content <= 1));

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_heating_value_positive CHECK (heating_value IS NULL OR heating_value > 0);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_density_positive CHECK (density IS NULL OR density > 0);

ALTER TABLE process_emissions_service.pe_raw_materials
    ADD CONSTRAINT chk_rm_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_rm_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_raw_materials
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 3: process_emissions_service.pe_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    process_type        VARCHAR(100)    NOT NULL,
    gas                 VARCHAR(20)     NOT NULL,
    factor_value        DECIMAL(20,10)  NOT NULL,
    factor_unit         VARCHAR(100)    NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    tier                VARCHAR(20)     NOT NULL DEFAULT 'TIER_1',
    production_route    VARCHAR(50),
    valid_from          DATE,
    valid_to            DATE,
    reference           TEXT,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_process_type_not_empty CHECK (LENGTH(TRIM(process_type)) > 0);

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'CF4', 'C2F6', 'SF6', 'NF3', 'HFC'
    ));

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty CHECK (LENGTH(TRIM(factor_unit)) > 0);

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_tier CHECK (tier IN ('TIER_1', 'TIER_2', 'TIER_3'));

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_date_order CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from);

ALTER TABLE process_emissions_service.pe_emission_factors
    ADD CONSTRAINT chk_ef_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_emission_factors
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 4: process_emissions_service.pe_process_units
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_process_units (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    unit_name           VARCHAR(200)    NOT NULL,
    unit_type           VARCHAR(50)     NOT NULL,
    process_type        VARCHAR(100)    NOT NULL,
    facility_id         VARCHAR(100),
    capacity            DECIMAL(15,4),
    capacity_unit       VARCHAR(50),
    process_mode        VARCHAR(30)     DEFAULT 'CONTINUOUS',
    commissioning_date  DATE,
    decommission_date   DATE,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_unit_name_not_empty CHECK (LENGTH(TRIM(unit_name)) > 0);

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_unit_type_not_empty CHECK (LENGTH(TRIM(unit_type)) > 0);

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_process_type_not_empty CHECK (LENGTH(TRIM(process_type)) > 0);

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_capacity_positive CHECK (capacity IS NULL OR capacity > 0);

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_process_mode CHECK (process_mode IN (
        'CONTINUOUS', 'BATCH', 'SEMI_BATCH'
    ));

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_decommission_after_commissioning CHECK (
        decommission_date IS NULL OR commissioning_date IS NULL OR decommission_date >= commissioning_date
    );

ALTER TABLE process_emissions_service.pe_process_units
    ADD CONSTRAINT chk_pu_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_pu_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_process_units
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 5: process_emissions_service.pe_material_inputs
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_material_inputs (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            NOT NULL,
    material_type       VARCHAR(100)    NOT NULL,
    quantity            DECIMAL(20,6)   NOT NULL,
    quantity_unit       VARCHAR(50)     NOT NULL,
    carbon_content      DECIMAL(10,6),
    carbonate_content   DECIMAL(10,6),
    moisture_content    DECIMAL(10,6),
    is_product          BOOLEAN         DEFAULT FALSE,
    is_by_product       BOOLEAN         DEFAULT FALSE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_material_type_not_empty CHECK (LENGTH(TRIM(material_type)) > 0);

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_quantity_non_negative CHECK (quantity >= 0);

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_quantity_unit_not_empty CHECK (LENGTH(TRIM(quantity_unit)) > 0);

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_carbon_content_range CHECK (carbon_content IS NULL OR (carbon_content >= 0 AND carbon_content <= 1));

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_carbonate_content_range CHECK (carbonate_content IS NULL OR (carbonate_content >= 0 AND carbonate_content <= 1));

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_moisture_content_range CHECK (moisture_content IS NULL OR (moisture_content >= 0 AND moisture_content <= 1));

ALTER TABLE process_emissions_service.pe_material_inputs
    ADD CONSTRAINT chk_mi_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 6: process_emissions_service.pe_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    process_type            VARCHAR(100)    NOT NULL,
    calculation_method      VARCHAR(30)     NOT NULL,
    calculation_tier        VARCHAR(20)     NOT NULL,
    activity_data           DECIMAL(20,6)   NOT NULL,
    activity_unit           VARCHAR(50)     NOT NULL,
    production_route        VARCHAR(50),
    process_unit_id         UUID            REFERENCES process_emissions_service.pe_process_units(id),
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    ef_source               VARCHAR(50)     NOT NULL DEFAULT 'IPCC',
    total_co2e_kg           DECIMAL(20,8)   NOT NULL,
    co2_kg                  DECIMAL(20,8)   DEFAULT 0,
    ch4_kg                  DECIMAL(20,8)   DEFAULT 0,
    n2o_kg                  DECIMAL(20,8)   DEFAULT 0,
    pfc_co2e_kg             DECIMAL(20,8)   DEFAULT 0,
    sf6_co2e_kg             DECIMAL(20,8)   DEFAULT 0,
    nf3_co2e_kg             DECIMAL(20,8)   DEFAULT 0,
    abatement_applied       BOOLEAN         DEFAULT FALSE,
    abatement_efficiency    DECIMAL(10,6)   DEFAULT 0,
    uncertainty_pct         DECIMAL(10,4),
    confidence_level        DECIMAL(5,2),
    facility_id             VARCHAR(100),
    reporting_period        VARCHAR(30),
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_process_type_not_empty CHECK (LENGTH(TRIM(process_type)) > 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'MASS_BALANCE', 'EMISSION_FACTOR', 'DIRECT_MEASUREMENT', 'HYBRID', 'CEMS'
    ));

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_calculation_tier CHECK (calculation_tier IN ('TIER_1', 'TIER_2', 'TIER_3'));

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_activity_data_non_negative CHECK (activity_data >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_activity_unit_not_empty CHECK (LENGTH(TRIM(activity_unit)) > 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_ef_source CHECK (ef_source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_total_co2e_kg_non_negative CHECK (total_co2e_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_co2_kg_non_negative CHECK (co2_kg IS NULL OR co2_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_ch4_kg_non_negative CHECK (ch4_kg IS NULL OR ch4_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_n2o_kg_non_negative CHECK (n2o_kg IS NULL OR n2o_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_pfc_co2e_kg_non_negative CHECK (pfc_co2e_kg IS NULL OR pfc_co2e_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_sf6_co2e_kg_non_negative CHECK (sf6_co2e_kg IS NULL OR sf6_co2e_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_nf3_co2e_kg_non_negative CHECK (nf3_co2e_kg IS NULL OR nf3_co2e_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_abatement_efficiency_range CHECK (abatement_efficiency >= 0 AND abatement_efficiency <= 1);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_uncertainty_pct_non_negative CHECK (uncertainty_pct IS NULL OR uncertainty_pct >= 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_confidence_level_range CHECK (confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 100));

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE process_emissions_service.pe_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_calculations
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 7: process_emissions_service.pe_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL REFERENCES process_emissions_service.pe_calculations(id) ON DELETE CASCADE,
    gas                     VARCHAR(20)     NOT NULL,
    emission_factor         DECIMAL(20,10)  NOT NULL,
    emission_factor_unit    VARCHAR(100),
    emission_factor_source  VARCHAR(50),
    raw_emissions_kg        DECIMAL(20,8)   NOT NULL,
    gwp_value               DECIMAL(15,4)   NOT NULL,
    co2e_kg                 DECIMAL(20,8)   NOT NULL,
    calculation_trace       JSONB           DEFAULT '[]'::jsonb,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'CF4', 'C2F6', 'SF6', 'NF3', 'HFC'
    ));

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_non_negative CHECK (emission_factor >= 0);

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_source CHECK (emission_factor_source IS NULL OR emission_factor_source IN (
        'EPA', 'IPCC', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_raw_emissions_kg_non_negative CHECK (raw_emissions_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_gwp_value_positive CHECK (gwp_value > 0);

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_co2e_kg_non_negative CHECK (co2e_kg >= 0);

ALTER TABLE process_emissions_service.pe_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 8: process_emissions_service.pe_abatement_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_abatement_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    process_unit_id         UUID            REFERENCES process_emissions_service.pe_process_units(id),
    abatement_type          VARCHAR(50)     NOT NULL,
    target_gas              VARCHAR(20)     NOT NULL,
    destruction_efficiency  DECIMAL(10,6)   NOT NULL,
    verification_status     VARCHAR(30)     DEFAULT 'UNVERIFIED',
    installation_date       DATE,
    last_verification_date  DATE,
    monitoring_frequency    VARCHAR(30),
    cost_per_tonne_co2e     DECIMAL(15,4),
    is_active               BOOLEAN         DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_abatement_type_not_empty CHECK (LENGTH(TRIM(abatement_type)) > 0);

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_target_gas CHECK (target_gas IN (
        'CO2', 'CH4', 'N2O', 'CF4', 'C2F6', 'SF6', 'NF3', 'HFC'
    ));

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_destruction_efficiency_range CHECK (destruction_efficiency >= 0 AND destruction_efficiency <= 1);

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_verification_status CHECK (verification_status IN (
        'UNVERIFIED', 'VERIFIED', 'EXPIRED', 'PENDING', 'REJECTED'
    ));

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_last_verification_after_installation CHECK (
        last_verification_date IS NULL OR installation_date IS NULL OR last_verification_date >= installation_date
    );

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_monitoring_frequency CHECK (monitoring_frequency IS NULL OR monitoring_frequency IN (
        'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUALLY', 'CONTINUOUS'
    ));

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_cost_non_negative CHECK (cost_per_tonne_co2e IS NULL OR cost_per_tonne_co2e >= 0);

ALTER TABLE process_emissions_service.pe_abatement_records
    ADD CONSTRAINT chk_ar_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ar_updated_at
    BEFORE UPDATE ON process_emissions_service.pe_abatement_records
    FOR EACH ROW EXECUTE FUNCTION process_emissions_service.set_updated_at();

-- =============================================================================
-- Table 9: process_emissions_service.pe_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            REFERENCES process_emissions_service.pe_calculations(id),
    framework           VARCHAR(50)     NOT NULL,
    check_name          VARCHAR(200)    NOT NULL,
    status              VARCHAR(30)     NOT NULL,
    details             TEXT,
    recommendations     JSONB           DEFAULT '[]'::jsonb,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM'
    ));

ALTER TABLE process_emissions_service.pe_compliance_records
    ADD CONSTRAINT chk_cr_check_name_not_empty CHECK (LENGTH(TRIM(check_name)) > 0);

ALTER TABLE process_emissions_service.pe_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE process_emissions_service.pe_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 10: process_emissions_service.pe_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           VARCHAR(100)    NOT NULL,
    action              VARCHAR(30)     NOT NULL,
    details             JSONB           DEFAULT '{}'::jsonb,
    parent_hash         VARCHAR(64),
    entry_hash          VARCHAR(64)     NOT NULL,
    user_id             VARCHAR(100),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE process_emissions_service.pe_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE process_emissions_service.pe_audit_entries
    ADD CONSTRAINT chk_ae_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);

ALTER TABLE process_emissions_service.pe_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT'
    ));

ALTER TABLE process_emissions_service.pe_audit_entries
    ADD CONSTRAINT chk_ae_entry_hash_not_empty CHECK (LENGTH(TRIM(entry_hash)) > 0);

ALTER TABLE process_emissions_service.pe_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: process_emissions_service.pe_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_calculation_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    process_type        VARCHAR(100),
    method              VARCHAR(30),
    emissions_kg_co2e   DECIMAL(20,8),
    duration_ms         DECIMAL(12,2),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'process_emissions_service.pe_calculation_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE process_emissions_service.pe_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_kg_co2e IS NULL OR emissions_kg_co2e >= 0);

ALTER TABLE process_emissions_service.pe_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE process_emissions_service.pe_calculation_events
    ADD CONSTRAINT chk_cae_method CHECK (
        method IS NULL OR method IN ('MASS_BALANCE', 'EMISSION_FACTOR', 'DIRECT_MEASUREMENT', 'HYBRID', 'CEMS')
    );

-- =============================================================================
-- Table 12: process_emissions_service.pe_material_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_material_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    material_type       VARCHAR(100),
    quantity            DECIMAL(20,6),
    unit                VARCHAR(50),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'process_emissions_service.pe_material_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE process_emissions_service.pe_material_events
    ADD CONSTRAINT chk_mae_quantity_non_negative CHECK (quantity IS NULL OR quantity >= 0);

-- =============================================================================
-- Table 13: process_emissions_service.pe_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS process_emissions_service.pe_compliance_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    framework           VARCHAR(50),
    status              VARCHAR(30),
    check_count         INTEGER,
    pass_count          INTEGER,
    fail_count          INTEGER,
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'process_emissions_service.pe_compliance_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE process_emissions_service.pe_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM')
    );

ALTER TABLE process_emissions_service.pe_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE process_emissions_service.pe_compliance_events
    ADD CONSTRAINT chk_coe_check_count_non_negative CHECK (check_count IS NULL OR check_count >= 0);

ALTER TABLE process_emissions_service.pe_compliance_events
    ADD CONSTRAINT chk_coe_pass_count_non_negative CHECK (pass_count IS NULL OR pass_count >= 0);

ALTER TABLE process_emissions_service.pe_compliance_events
    ADD CONSTRAINT chk_coe_fail_count_non_negative CHECK (fail_count IS NULL OR fail_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- pe_hourly_calculation_stats: hourly count/sum(co2e)/avg(co2e) by process_type and method
CREATE MATERIALIZED VIEW process_emissions_service.pe_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    process_type,
    method,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e,
    AVG(emissions_kg_co2e)          AS avg_emissions_kg_co2e,
    AVG(duration_ms)                AS avg_duration_ms,
    MAX(duration_ms)                AS max_duration_ms
FROM process_emissions_service.pe_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, process_type, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'process_emissions_service.pe_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- pe_daily_emission_totals: daily count/sum(co2e) by process_type
CREATE MATERIALIZED VIEW process_emissions_service.pe_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    process_type,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e
FROM process_emissions_service.pe_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, process_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'process_emissions_service.pe_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- pe_process_types indexes
CREATE INDEX IF NOT EXISTS idx_pe_pt_tenant_id              ON process_emissions_service.pe_process_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_pt_process_type           ON process_emissions_service.pe_process_types(process_type);
CREATE INDEX IF NOT EXISTS idx_pe_pt_category               ON process_emissions_service.pe_process_types(category);
CREATE INDEX IF NOT EXISTS idx_pe_pt_name                   ON process_emissions_service.pe_process_types(name);
CREATE INDEX IF NOT EXISTS idx_pe_pt_epa_subpart            ON process_emissions_service.pe_process_types(epa_subpart);
CREATE INDEX IF NOT EXISTS idx_pe_pt_default_ef_source      ON process_emissions_service.pe_process_types(default_ef_source);
CREATE INDEX IF NOT EXISTS idx_pe_pt_is_active              ON process_emissions_service.pe_process_types(is_active);
CREATE INDEX IF NOT EXISTS idx_pe_pt_tenant_category        ON process_emissions_service.pe_process_types(tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_pe_pt_tenant_active          ON process_emissions_service.pe_process_types(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_pe_pt_created_at             ON process_emissions_service.pe_process_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_pt_updated_at             ON process_emissions_service.pe_process_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_pt_primary_gases          ON process_emissions_service.pe_process_types USING GIN (primary_gases);
CREATE INDEX IF NOT EXISTS idx_pe_pt_applicable_tiers       ON process_emissions_service.pe_process_types USING GIN (applicable_tiers);
CREATE INDEX IF NOT EXISTS idx_pe_pt_production_routes      ON process_emissions_service.pe_process_types USING GIN (production_routes);
CREATE INDEX IF NOT EXISTS idx_pe_pt_metadata               ON process_emissions_service.pe_process_types USING GIN (metadata);

-- pe_raw_materials indexes
CREATE INDEX IF NOT EXISTS idx_pe_rm_tenant_id              ON process_emissions_service.pe_raw_materials(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_rm_material_type          ON process_emissions_service.pe_raw_materials(material_type);
CREATE INDEX IF NOT EXISTS idx_pe_rm_name                   ON process_emissions_service.pe_raw_materials(name);
CREATE INDEX IF NOT EXISTS idx_pe_rm_carbonate_type         ON process_emissions_service.pe_raw_materials(carbonate_type);
CREATE INDEX IF NOT EXISTS idx_pe_rm_is_active              ON process_emissions_service.pe_raw_materials(is_active);
CREATE INDEX IF NOT EXISTS idx_pe_rm_tenant_material_type   ON process_emissions_service.pe_raw_materials(tenant_id, material_type);
CREATE INDEX IF NOT EXISTS idx_pe_rm_tenant_active          ON process_emissions_service.pe_raw_materials(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_pe_rm_created_at             ON process_emissions_service.pe_raw_materials(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_rm_updated_at             ON process_emissions_service.pe_raw_materials(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_rm_metadata               ON process_emissions_service.pe_raw_materials USING GIN (metadata);

-- pe_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_pe_ef_tenant_id              ON process_emissions_service.pe_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_ef_process_type           ON process_emissions_service.pe_emission_factors(process_type);
CREATE INDEX IF NOT EXISTS idx_pe_ef_gas                    ON process_emissions_service.pe_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_pe_ef_source                 ON process_emissions_service.pe_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_pe_ef_tier                   ON process_emissions_service.pe_emission_factors(tier);
CREATE INDEX IF NOT EXISTS idx_pe_ef_production_route       ON process_emissions_service.pe_emission_factors(production_route);
CREATE INDEX IF NOT EXISTS idx_pe_ef_is_active              ON process_emissions_service.pe_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_pe_ef_valid_from             ON process_emissions_service.pe_emission_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ef_valid_to               ON process_emissions_service.pe_emission_factors(valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ef_process_gas            ON process_emissions_service.pe_emission_factors(process_type, gas);
CREATE INDEX IF NOT EXISTS idx_pe_ef_process_gas_source     ON process_emissions_service.pe_emission_factors(process_type, gas, source);
CREATE INDEX IF NOT EXISTS idx_pe_ef_tenant_process_gas     ON process_emissions_service.pe_emission_factors(tenant_id, process_type, gas);
CREATE INDEX IF NOT EXISTS idx_pe_ef_tenant_process_tier    ON process_emissions_service.pe_emission_factors(tenant_id, process_type, tier);
CREATE INDEX IF NOT EXISTS idx_pe_ef_created_at             ON process_emissions_service.pe_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ef_updated_at             ON process_emissions_service.pe_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ef_metadata               ON process_emissions_service.pe_emission_factors USING GIN (metadata);

-- pe_process_units indexes
CREATE INDEX IF NOT EXISTS idx_pe_pu_tenant_id              ON process_emissions_service.pe_process_units(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_pu_unit_name              ON process_emissions_service.pe_process_units(unit_name);
CREATE INDEX IF NOT EXISTS idx_pe_pu_unit_type              ON process_emissions_service.pe_process_units(unit_type);
CREATE INDEX IF NOT EXISTS idx_pe_pu_process_type           ON process_emissions_service.pe_process_units(process_type);
CREATE INDEX IF NOT EXISTS idx_pe_pu_facility_id            ON process_emissions_service.pe_process_units(facility_id);
CREATE INDEX IF NOT EXISTS idx_pe_pu_process_mode           ON process_emissions_service.pe_process_units(process_mode);
CREATE INDEX IF NOT EXISTS idx_pe_pu_is_active              ON process_emissions_service.pe_process_units(is_active);
CREATE INDEX IF NOT EXISTS idx_pe_pu_tenant_process_type    ON process_emissions_service.pe_process_units(tenant_id, process_type);
CREATE INDEX IF NOT EXISTS idx_pe_pu_tenant_facility        ON process_emissions_service.pe_process_units(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_pe_pu_tenant_active          ON process_emissions_service.pe_process_units(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_pe_pu_created_at             ON process_emissions_service.pe_process_units(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_pu_updated_at             ON process_emissions_service.pe_process_units(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_pu_metadata               ON process_emissions_service.pe_process_units USING GIN (metadata);

-- pe_material_inputs indexes
CREATE INDEX IF NOT EXISTS idx_pe_mi_tenant_id              ON process_emissions_service.pe_material_inputs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_mi_calculation_id         ON process_emissions_service.pe_material_inputs(calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_mi_material_type          ON process_emissions_service.pe_material_inputs(material_type);
CREATE INDEX IF NOT EXISTS idx_pe_mi_is_product             ON process_emissions_service.pe_material_inputs(is_product);
CREATE INDEX IF NOT EXISTS idx_pe_mi_is_by_product          ON process_emissions_service.pe_material_inputs(is_by_product);
CREATE INDEX IF NOT EXISTS idx_pe_mi_tenant_calculation     ON process_emissions_service.pe_material_inputs(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_mi_tenant_material        ON process_emissions_service.pe_material_inputs(tenant_id, material_type);
CREATE INDEX IF NOT EXISTS idx_pe_mi_created_at             ON process_emissions_service.pe_material_inputs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_mi_metadata               ON process_emissions_service.pe_material_inputs USING GIN (metadata);

-- pe_calculations indexes
CREATE INDEX IF NOT EXISTS idx_pe_calc_tenant_id            ON process_emissions_service.pe_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_calc_process_type         ON process_emissions_service.pe_calculations(process_type);
CREATE INDEX IF NOT EXISTS idx_pe_calc_method               ON process_emissions_service.pe_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_pe_calc_tier                 ON process_emissions_service.pe_calculations(calculation_tier);
CREATE INDEX IF NOT EXISTS idx_pe_calc_production_route     ON process_emissions_service.pe_calculations(production_route);
CREATE INDEX IF NOT EXISTS idx_pe_calc_process_unit_id      ON process_emissions_service.pe_calculations(process_unit_id);
CREATE INDEX IF NOT EXISTS idx_pe_calc_gwp_source           ON process_emissions_service.pe_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_pe_calc_ef_source            ON process_emissions_service.pe_calculations(ef_source);
CREATE INDEX IF NOT EXISTS idx_pe_calc_total_co2e_kg        ON process_emissions_service.pe_calculations(total_co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_pe_calc_abatement_applied    ON process_emissions_service.pe_calculations(abatement_applied);
CREATE INDEX IF NOT EXISTS idx_pe_calc_facility_id          ON process_emissions_service.pe_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_pe_calc_reporting_period     ON process_emissions_service.pe_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_pe_calc_provenance_hash      ON process_emissions_service.pe_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_pe_calc_tenant_process_type  ON process_emissions_service.pe_calculations(tenant_id, process_type);
CREATE INDEX IF NOT EXISTS idx_pe_calc_tenant_method        ON process_emissions_service.pe_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_pe_calc_tenant_facility      ON process_emissions_service.pe_calculations(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_pe_calc_tenant_period        ON process_emissions_service.pe_calculations(tenant_id, reporting_period);
CREATE INDEX IF NOT EXISTS idx_pe_calc_created_at           ON process_emissions_service.pe_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_calc_updated_at           ON process_emissions_service.pe_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_calc_metadata             ON process_emissions_service.pe_calculations USING GIN (metadata);

-- pe_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_pe_cd_tenant_id              ON process_emissions_service.pe_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_cd_calculation_id         ON process_emissions_service.pe_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_cd_gas                    ON process_emissions_service.pe_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_pe_cd_raw_emissions_kg       ON process_emissions_service.pe_calculation_details(raw_emissions_kg DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cd_co2e_kg                ON process_emissions_service.pe_calculation_details(co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cd_emission_factor_source ON process_emissions_service.pe_calculation_details(emission_factor_source);
CREATE INDEX IF NOT EXISTS idx_pe_cd_tenant_calc            ON process_emissions_service.pe_calculation_details(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_cd_calc_gas               ON process_emissions_service.pe_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_pe_cd_tenant_gas             ON process_emissions_service.pe_calculation_details(tenant_id, gas);
CREATE INDEX IF NOT EXISTS idx_pe_cd_created_at             ON process_emissions_service.pe_calculation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cd_calculation_trace      ON process_emissions_service.pe_calculation_details USING GIN (calculation_trace);
CREATE INDEX IF NOT EXISTS idx_pe_cd_metadata               ON process_emissions_service.pe_calculation_details USING GIN (metadata);

-- pe_abatement_records indexes
CREATE INDEX IF NOT EXISTS idx_pe_ar_tenant_id              ON process_emissions_service.pe_abatement_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_ar_process_unit_id        ON process_emissions_service.pe_abatement_records(process_unit_id);
CREATE INDEX IF NOT EXISTS idx_pe_ar_abatement_type         ON process_emissions_service.pe_abatement_records(abatement_type);
CREATE INDEX IF NOT EXISTS idx_pe_ar_target_gas             ON process_emissions_service.pe_abatement_records(target_gas);
CREATE INDEX IF NOT EXISTS idx_pe_ar_verification_status    ON process_emissions_service.pe_abatement_records(verification_status);
CREATE INDEX IF NOT EXISTS idx_pe_ar_is_active              ON process_emissions_service.pe_abatement_records(is_active);
CREATE INDEX IF NOT EXISTS idx_pe_ar_installation_date      ON process_emissions_service.pe_abatement_records(installation_date DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ar_last_verification      ON process_emissions_service.pe_abatement_records(last_verification_date DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ar_tenant_unit            ON process_emissions_service.pe_abatement_records(tenant_id, process_unit_id);
CREATE INDEX IF NOT EXISTS idx_pe_ar_tenant_gas             ON process_emissions_service.pe_abatement_records(tenant_id, target_gas);
CREATE INDEX IF NOT EXISTS idx_pe_ar_tenant_active          ON process_emissions_service.pe_abatement_records(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_pe_ar_created_at             ON process_emissions_service.pe_abatement_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ar_updated_at             ON process_emissions_service.pe_abatement_records(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ar_metadata               ON process_emissions_service.pe_abatement_records USING GIN (metadata);

-- pe_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_pe_cr_tenant_id              ON process_emissions_service.pe_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_cr_calculation_id         ON process_emissions_service.pe_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_cr_framework              ON process_emissions_service.pe_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_pe_cr_check_name             ON process_emissions_service.pe_compliance_records(check_name);
CREATE INDEX IF NOT EXISTS idx_pe_cr_status                 ON process_emissions_service.pe_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_pe_cr_checked_at             ON process_emissions_service.pe_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cr_tenant_framework       ON process_emissions_service.pe_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_pe_cr_tenant_status          ON process_emissions_service.pe_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_pe_cr_framework_status       ON process_emissions_service.pe_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_pe_cr_tenant_calculation     ON process_emissions_service.pe_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_pe_cr_created_at             ON process_emissions_service.pe_compliance_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cr_recommendations        ON process_emissions_service.pe_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_pe_cr_metadata               ON process_emissions_service.pe_compliance_records USING GIN (metadata);

-- pe_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_pe_ae_tenant_id              ON process_emissions_service.pe_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_pe_ae_entity_type            ON process_emissions_service.pe_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_pe_ae_entity_id              ON process_emissions_service.pe_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_pe_ae_action                 ON process_emissions_service.pe_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_pe_ae_parent_hash            ON process_emissions_service.pe_audit_entries(parent_hash);
CREATE INDEX IF NOT EXISTS idx_pe_ae_entry_hash             ON process_emissions_service.pe_audit_entries(entry_hash);
CREATE INDEX IF NOT EXISTS idx_pe_ae_user_id                ON process_emissions_service.pe_audit_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_pe_ae_tenant_entity          ON process_emissions_service.pe_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_pe_ae_tenant_action          ON process_emissions_service.pe_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_pe_ae_created_at             ON process_emissions_service.pe_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pe_ae_details                ON process_emissions_service.pe_audit_entries USING GIN (details);

-- pe_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_pe_cae_tenant_id             ON process_emissions_service.pe_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_event_type            ON process_emissions_service.pe_calculation_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_process_type          ON process_emissions_service.pe_calculation_events(process_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_method                ON process_emissions_service.pe_calculation_events(method, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_tenant_process        ON process_emissions_service.pe_calculation_events(tenant_id, process_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_tenant_method         ON process_emissions_service.pe_calculation_events(tenant_id, method, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_cae_metadata              ON process_emissions_service.pe_calculation_events USING GIN (metadata);

-- pe_material_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_pe_mae_tenant_id             ON process_emissions_service.pe_material_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_mae_event_type            ON process_emissions_service.pe_material_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_mae_material_type         ON process_emissions_service.pe_material_events(material_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_mae_tenant_material       ON process_emissions_service.pe_material_events(tenant_id, material_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_mae_metadata              ON process_emissions_service.pe_material_events USING GIN (metadata);

-- pe_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_pe_coe_tenant_id             ON process_emissions_service.pe_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_coe_framework             ON process_emissions_service.pe_compliance_events(framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_coe_status                ON process_emissions_service.pe_compliance_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_coe_tenant_framework      ON process_emissions_service.pe_compliance_events(tenant_id, framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_coe_tenant_status         ON process_emissions_service.pe_compliance_events(tenant_id, status, time DESC);
CREATE INDEX IF NOT EXISTS idx_pe_coe_metadata              ON process_emissions_service.pe_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- pe_process_types: tenant-isolated
ALTER TABLE process_emissions_service.pe_process_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_pt_read  ON process_emissions_service.pe_process_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_pt_write ON process_emissions_service.pe_process_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_raw_materials: tenant-isolated
ALTER TABLE process_emissions_service.pe_raw_materials ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_rm_read  ON process_emissions_service.pe_raw_materials FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_rm_write ON process_emissions_service.pe_raw_materials FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_emission_factors: tenant-isolated
ALTER TABLE process_emissions_service.pe_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_ef_read  ON process_emissions_service.pe_emission_factors FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_ef_write ON process_emissions_service.pe_emission_factors FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_process_units: tenant-isolated
ALTER TABLE process_emissions_service.pe_process_units ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_pu_read  ON process_emissions_service.pe_process_units FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_pu_write ON process_emissions_service.pe_process_units FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_material_inputs: tenant-isolated
ALTER TABLE process_emissions_service.pe_material_inputs ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_mi_read  ON process_emissions_service.pe_material_inputs FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_mi_write ON process_emissions_service.pe_material_inputs FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_calculations: tenant-isolated
ALTER TABLE process_emissions_service.pe_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_calc_read  ON process_emissions_service.pe_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_calc_write ON process_emissions_service.pe_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_calculation_details: tenant-isolated
ALTER TABLE process_emissions_service.pe_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_cd_read  ON process_emissions_service.pe_calculation_details FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_cd_write ON process_emissions_service.pe_calculation_details FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_abatement_records: tenant-isolated
ALTER TABLE process_emissions_service.pe_abatement_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_ar_read  ON process_emissions_service.pe_abatement_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_ar_write ON process_emissions_service.pe_abatement_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_compliance_records: tenant-isolated
ALTER TABLE process_emissions_service.pe_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_cr_read  ON process_emissions_service.pe_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_cr_write ON process_emissions_service.pe_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_audit_entries: tenant-isolated
ALTER TABLE process_emissions_service.pe_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_ae_read  ON process_emissions_service.pe_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY pe_ae_write ON process_emissions_service.pe_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- pe_calculation_events: open read/write (time-series telemetry)
ALTER TABLE process_emissions_service.pe_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_cae_read  ON process_emissions_service.pe_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY pe_cae_write ON process_emissions_service.pe_calculation_events FOR ALL   USING (TRUE);

-- pe_material_events: open read/write (time-series telemetry)
ALTER TABLE process_emissions_service.pe_material_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_mae_read  ON process_emissions_service.pe_material_events FOR SELECT USING (TRUE);
CREATE POLICY pe_mae_write ON process_emissions_service.pe_material_events FOR ALL   USING (TRUE);

-- pe_compliance_events: open read/write (time-series telemetry)
ALTER TABLE process_emissions_service.pe_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY pe_coe_read  ON process_emissions_service.pe_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY pe_coe_write ON process_emissions_service.pe_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA process_emissions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA process_emissions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA process_emissions_service TO greenlang_app;
GRANT SELECT ON process_emissions_service.pe_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON process_emissions_service.pe_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA process_emissions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA process_emissions_service TO greenlang_readonly;
GRANT SELECT ON process_emissions_service.pe_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON process_emissions_service.pe_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA process_emissions_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA process_emissions_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA process_emissions_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'process-emissions:read',                    'process-emissions', 'read',                    'View all process emissions service data including process types, raw materials, calculations, and abatement records'),
    (gen_random_uuid(), 'process-emissions:write',                   'process-emissions', 'write',                   'Create, update, and manage all process emissions service data'),
    (gen_random_uuid(), 'process-emissions:execute',                 'process-emissions', 'execute',                 'Execute process emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'process-emissions:processes:read',          'process-emissions', 'processes_read',          'View industrial process type registry with categories, primary gases, applicable tiers, EPA subparts, and production routes'),
    (gen_random_uuid(), 'process-emissions:processes:write',         'process-emissions', 'processes_write',         'Create, update, and manage industrial process type registry entries'),
    (gen_random_uuid(), 'process-emissions:materials:read',          'process-emissions', 'materials_read',          'View raw material registry with carbon content, carbonate content, molecular weight, moisture, heating value, and density'),
    (gen_random_uuid(), 'process-emissions:materials:write',         'process-emissions', 'materials_write',         'Create, update, and manage raw material registry entries'),
    (gen_random_uuid(), 'process-emissions:units:read',              'process-emissions', 'units_read',              'View process unit registry with capacity, process mode, facility assignment, and commissioning dates'),
    (gen_random_uuid(), 'process-emissions:units:write',             'process-emissions', 'units_write',             'Create, update, and manage process unit registry entries'),
    (gen_random_uuid(), 'process-emissions:factors:read',            'process-emissions', 'factors_read',            'View emission factors by process type, gas, tier, production route, and source from EPA/IPCC/DEFRA/EU_ETS/CUSTOM'),
    (gen_random_uuid(), 'process-emissions:factors:write',           'process-emissions', 'factors_write',           'Create, update, and manage emission factor entries with source, tier, and validity date range data'),
    (gen_random_uuid(), 'process-emissions:abatement:read',          'process-emissions', 'abatement_read',          'View abatement records with destruction efficiency, verification status, monitoring frequency, and cost per tonne CO2e'),
    (gen_random_uuid(), 'process-emissions:abatement:write',         'process-emissions', 'abatement_write',         'Create, update, and manage abatement records with efficiency and verification data'),
    (gen_random_uuid(), 'process-emissions:calculations:read',       'process-emissions', 'calculations_read',       'View process emission calculation results with multi-gas breakdown, abatement, uncertainty, and provenance data'),
    (gen_random_uuid(), 'process-emissions:calculations:write',      'process-emissions', 'calculations_write',      'Create and manage process emission calculation records'),
    (gen_random_uuid(), 'process-emissions:compliance:read',         'process-emissions', 'compliance_read',         'View regulatory compliance records for GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 with findings and recommendations'),
    (gen_random_uuid(), 'process-emissions:compliance:execute',      'process-emissions', 'compliance_execute',      'Execute regulatory compliance checks against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks'),
    (gen_random_uuid(), 'process-emissions:admin',                   'process-emissions', 'admin',                   'Full administrative access to process emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('process_emissions_service.pe_calculation_events', INTERVAL '365 days');
SELECT add_retention_policy('process_emissions_service.pe_material_events',    INTERVAL '365 days');
SELECT add_retention_policy('process_emissions_service.pe_compliance_events',  INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE process_emissions_service.pe_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'process_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('process_emissions_service.pe_calculation_events', INTERVAL '30 days');

ALTER TABLE process_emissions_service.pe_material_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'material_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('process_emissions_service.pe_material_events', INTERVAL '30 days');

ALTER TABLE process_emissions_service.pe_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('process_emissions_service.pe_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Process Emissions Agent (GL-MRV-SCOPE1-004)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-004',
    'Process Emissions Agent',
    'Industrial process emission calculator for GreenLang Climate OS. Manages process type registry with mineral, chemical, metal, electronics, pulp_paper, and other categories including primary greenhouse gases (CO2/CH4/N2O/CF4/C2F6/SF6/NF3/HFC), applicable calculation tiers (Tier 1/2/3), EPA subpart references, default emission factors, and production route tracking. Maintains raw material registry with carbon content (fraction 0-1), carbonate content and type, molecular weight, moisture content, heating value, and density properties. Stores tiered emission factor database with process type x gas factors at Tier 1/2/3 from EPA/IPCC/DEFRA/EU_ETS/CUSTOM sources with production routes and validity date ranges. Registers process units with capacity, process mode (CONTINUOUS/BATCH/SEMI_BATCH), facility assignment, and commissioning/decommission dates. Tracks material inputs per calculation with quantities, carbon/carbonate/moisture content, and product/by-product classification for mass balance accounting. Executes deterministic process emission calculations using mass balance, emission factor, direct measurement, hybrid, and CEMS methods with multi-gas GWP weighting (CO2/CH4/N2O/PFC/SF6/NF3) using AR4/AR5/AR6 values. Applies abatement efficiency corrections. Quantifies uncertainty and confidence levels. Produces per-gas calculation detail breakdowns with individual emission factors, GWP values, raw and CO2e emissions, and step-by-step calculation traces. Manages abatement equipment records with destruction efficiency, verification status (UNVERIFIED/VERIFIED/EXPIRED/PENDING/REJECTED), monitoring frequency, and cost per tonne CO2e. Checks regulatory compliance against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks with check names, findings, and recommendations. Generates entity-level audit trail entries with action tracking, parent/entry hash chaining for tamper-evident provenance, and user attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/process-emissions',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-004', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/process-emissions-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"process-emissions", "scope-1", "industrial-processes", "ghg-protocol", "epa", "ipcc", "defra", "mrv"}',
    '{"cross-sector", "mineral-products", "chemical-industry", "metal-industry", "electronics", "pulp-paper", "manufacturing"}',
    'e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'process_type_registry',
    'configuration',
    'Register and manage industrial process type entries with category classification (mineral/chemical/metal/electronics/pulp_paper/other), primary greenhouse gases, applicable tiers, EPA subpart references, default emission factors, and production route tracking.',
    '{"process_type", "category", "name", "primary_gases", "applicable_tiers", "epa_subpart", "default_emission_factor", "default_ef_unit", "default_ef_source", "production_routes"}',
    '{"process_type_id", "registration_result"}',
    '{"categories": ["mineral", "chemical", "metal", "electronics", "pulp_paper", "other"], "gases": ["CO2", "CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC"], "tiers": ["TIER_1", "TIER_2", "TIER_3"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'raw_material_registry',
    'configuration',
    'Register and manage raw material entries with carbon content (fraction 0-1), carbonate content and type, molecular weight, moisture content, heating value, and density properties for mass balance calculations.',
    '{"material_type", "name", "carbon_content", "carbonate_content", "carbonate_type", "molecular_weight", "moisture_content", "heating_value", "heating_value_unit", "density"}',
    '{"material_id", "registration_result"}',
    '{"content_fractions": true, "supports_carbonate_types": true, "supports_heating_value": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic industrial process emission calculations using mass balance (material inputs/outputs x carbon content x 44/12), emission factor (activity data x EF x GWP), direct measurement (CEMS data), or hybrid methods. Supports multi-gas GWP weighting for CO2/CH4/N2O/PFC/SF6/NF3 with abatement efficiency corrections and uncertainty quantification.',
    '{"process_type", "calculation_method", "calculation_tier", "activity_data", "activity_unit", "production_route", "process_unit_id", "gwp_source", "ef_source", "material_inputs"}',
    '{"calculation_id", "total_co2e_kg", "per_gas_breakdown", "abatement_applied", "uncertainty_pct", "confidence_level", "provenance_hash"}',
    '{"methods": ["MASS_BALANCE", "EMISSION_FACTOR", "DIRECT_MEASUREMENT", "HYBRID", "CEMS"], "tiers": ["TIER_1", "TIER_2", "TIER_3"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'process_unit_management',
    'configuration',
    'Register and manage process unit entries with unit name, type, process type assignment, facility association, capacity with units, process mode (CONTINUOUS/BATCH/SEMI_BATCH), and commissioning/decommission date tracking.',
    '{"unit_name", "unit_type", "process_type", "facility_id", "capacity", "capacity_unit", "process_mode", "commissioning_date", "decommission_date"}',
    '{"process_unit_id", "registration_result"}',
    '{"modes": ["CONTINUOUS", "BATCH", "SEMI_BATCH"], "supports_facility_assignment": true, "supports_capacity_tracking": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'abatement_management',
    'configuration',
    'Register and manage abatement equipment records with abatement type, target gas, destruction efficiency (0-1), verification status tracking (UNVERIFIED/VERIFIED/EXPIRED/PENDING/REJECTED), monitoring frequency, and cost per tonne CO2e for marginal abatement cost curves.',
    '{"process_unit_id", "abatement_type", "target_gas", "destruction_efficiency", "verification_status", "installation_date", "monitoring_frequency", "cost_per_tonne_co2e"}',
    '{"abatement_id", "registration_result"}',
    '{"verification_statuses": ["UNVERIFIED", "VERIFIED", "EXPIRED", "PENDING", "REJECTED"], "monitoring_frequencies": ["DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "ANNUALLY", "CONTINUOUS"], "supports_cost_tracking": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'material_input_tracking',
    'processing',
    'Track material inputs and outputs per calculation with quantities, carbon/carbonate/moisture content overrides, and product/by-product classification for detailed mass balance accounting.',
    '{"calculation_id", "material_type", "quantity", "quantity_unit", "carbon_content", "carbonate_content", "moisture_content", "is_product", "is_by_product"}',
    '{"material_input_id", "tracking_result"}',
    '{"supports_product_classification": true, "supports_content_overrides": true, "supports_mass_balance": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of process emission calculations against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks. Produce named check results with findings detail text and actionable recommendations.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "check_name", "status", "details", "recommendations"}',
    '{"frameworks": ["GHG_PROTOCOL", "EPA", "DEFRA", "EU_ETS", "ISO_14064", "CUSTOM"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-004', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with action tracking, parent/entry SHA-256 hash chaining for tamper-evident provenance, detail JSONB payloads, and user attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-004', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage process emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-004', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for activity data quantities, emission factor unit conversions, and mass balance unit alignment'),
    ('GL-MRV-SCOPE1-004', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-004', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-SCOPE1-004', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, material events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-004', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of material inputs and activity data'),
    ('GL-MRV-SCOPE1-004', 'GL-MRV-X-001',   '>=1.0.0', true,  'Stationary Combustion Calculator for cross-referencing fuel types and shared emission factor database'),
    ('GL-MRV-SCOPE1-004', 'GL-MRV-SCOPE1-002', '>=1.0.0', true, 'Refrigerants & F-Gas Agent for cross-referencing industrial process fluorinated gas leakage data')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-004',
    'Process Emissions Agent',
    'Industrial process emission calculator. Process type registry (mineral/chemical/metal/electronics/pulp_paper/other, primary gases CO2/CH4/N2O/CF4/C2F6/SF6/NF3/HFC, Tier 1/2/3, EPA subparts, production routes). Raw material registry (carbon content, carbonate content/type, molecular weight, moisture, heating value, density). Emission factor database (process x gas, Tier 1/2/3, EPA/IPCC/DEFRA/EU_ETS/CUSTOM, production routes, validity dates). Process unit management (capacity, mode CONTINUOUS/BATCH/SEMI_BATCH, facility assignment). Material input tracking (quantities, content overrides, product/by-product flags). Emission calculations (mass balance/emission factor/direct measurement/hybrid/CEMS, multi-gas CO2/CH4/N2O/PFC/SF6/NF3 GWP AR4/AR5/AR6, abatement efficiency). Per-gas breakdowns with calculation traces. Abatement records (destruction efficiency, verification, monitoring, cost). Compliance checks (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA process_emissions_service IS
    'Process Emissions Agent (AGENT-MRV-004) - industrial process type registry, raw material registry, emission factor database, process unit management, material input tracking, emission calculations, per-gas breakdowns, abatement records, compliance records, audit trail, provenance chains';

COMMENT ON TABLE process_emissions_service.pe_process_types IS
    'Industrial process type registry: tenant_id, process_type (unique per tenant), category (mineral/chemical/metal/electronics/pulp_paper/other), name, description, primary_gases JSONB, applicable_tiers JSONB, epa_subpart, default_emission_factor, default_ef_unit, default_ef_source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), production_routes JSONB, is_active, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_raw_materials IS
    'Raw material registry: tenant_id, material_type (unique per tenant), name, description, carbon_content (fraction 0-1), carbonate_content (fraction 0-1), carbonate_type, molecular_weight, moisture_content, heating_value, heating_value_unit, density, is_active, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_emission_factors IS
    'Emission factor database: tenant_id, process_type, gas (CO2/CH4/N2O/CF4/C2F6/SF6/NF3/HFC), factor_value, factor_unit, source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), tier (TIER_1/TIER_2/TIER_3), production_route, valid_from/to, reference, is_active, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_process_units IS
    'Process unit registry: tenant_id, unit_name, unit_type, process_type, facility_id, capacity, capacity_unit, process_mode (CONTINUOUS/BATCH/SEMI_BATCH), commissioning_date, decommission_date, is_active, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_material_inputs IS
    'Material input records per calculation: tenant_id, calculation_id, material_type, quantity, quantity_unit, carbon_content, carbonate_content, moisture_content, is_product, is_by_product, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_calculations IS
    'Calculation results: tenant_id, process_type, calculation_method (MASS_BALANCE/EMISSION_FACTOR/DIRECT_MEASUREMENT/HYBRID/CEMS), calculation_tier (TIER_1/TIER_2/TIER_3), activity_data/unit, production_route, process_unit_id, gwp_source (AR4/AR5/AR6), ef_source (EPA/IPCC/DEFRA/EU_ETS/CUSTOM), total_co2e_kg, co2_kg, ch4_kg, n2o_kg, pfc_co2e_kg, sf6_co2e_kg, nf3_co2e_kg, abatement_applied/efficiency, uncertainty_pct, confidence_level, facility_id, reporting_period, provenance_hash, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_calculation_details IS
    'Per-gas calculation breakdown: tenant_id, calculation_id (FK CASCADE), gas (CO2/CH4/N2O/CF4/C2F6/SF6/NF3/HFC), emission_factor, emission_factor_unit, emission_factor_source, raw_emissions_kg, gwp_value, co2e_kg, calculation_trace JSONB, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_abatement_records IS
    'Abatement equipment records: tenant_id, process_unit_id (FK), abatement_type, target_gas (CO2/CH4/N2O/CF4/C2F6/SF6/NF3/HFC), destruction_efficiency (0-1), verification_status (UNVERIFIED/VERIFIED/EXPIRED/PENDING/REJECTED), installation_date, last_verification_date, monitoring_frequency, cost_per_tonne_co2e, is_active, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_compliance_records IS
    'Regulatory compliance records: tenant_id, calculation_id (FK), framework (GHG_PROTOCOL/EPA/DEFRA/EU_ETS/ISO_14064/CUSTOM), check_name, status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), details, recommendations JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE process_emissions_service.pe_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT), details JSONB, parent_hash, entry_hash (SHA-256 chain), user_id';

COMMENT ON TABLE process_emissions_service.pe_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, event_type, process_type, method, emissions_kg_co2e, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE process_emissions_service.pe_material_events IS
    'TimescaleDB hypertable: material events with tenant_id, event_type, material_type, quantity, unit, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE process_emissions_service.pe_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, event_type, framework, status, check_count, pass_count, fail_count, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW process_emissions_service.pe_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by process_type and method (total calculations, sum emissions kg CO2e, avg emissions kg CO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW process_emissions_service.pe_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by process_type (total calculations, sum emissions kg CO2e per day)';
