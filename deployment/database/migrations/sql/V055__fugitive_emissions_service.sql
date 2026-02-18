-- =============================================================================
-- V055: Fugitive Emissions Service Schema
-- =============================================================================
-- Component: AGENT-MRV-005 (GL-MRV-SCOPE1-005)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Fugitive Emissions Agent (GL-MRV-SCOPE1-005) with capabilities for
-- fugitive emission source type registry management (equipment_leaks,
-- coal_mining, oil_gas_systems, wastewater, landfill, refrigeration,
-- industrial_processes, other categories with primary greenhouse gases,
-- applicable quantification methods, and EPA/IPCC methodology references),
-- component type emission factor registry (valve, connector, pump_seal,
-- compressor_seal, pressure_relief, open_ended_line, sampling_connection,
-- flange, other types with gas/light_liquid/heavy_liquid service types,
-- leak/no-leak emission factors), emission factor database (source type x
-- gas factors with EPA/IPCC/DEFRA/API/CUSTOM sources across average_factor,
-- screening_ranges, correlation_equation, unit_specific, hi_flow_sampler
-- methods, validity date ranges, and references), equipment registry
-- (facility component tracking with tag numbers, component types, service
-- types, process units, locations, installation dates, leak status
-- NO_LEAK/LEAK/REPAIRED/EXCLUDED, and last survey dates), LDAR survey
-- records (OGI/METHOD21/AVO/CONTINUOUS/SCREENING survey types with
-- facility/inspector IDs, component counts, leak detection counts,
-- coverage percentages, and methodology tracking), fugitive emission
-- calculations (average_factor, screening_ranges, correlation_equation,
-- unit_specific, mass_balance, engineering_estimate, direct_measurement
-- methods with multi-gas GWP weighting CH4/CO2/N2O/VOC using AR4/AR5/AR6
-- values, recovery rate application, uncertainty quantification, and
-- provenance hashing), per-gas calculation detail breakdowns (individual
-- emission factors, GWP values, raw and CO2e emissions with calculation
-- trace), leak repair tracking (detection date, screening value PPM,
-- repair date and method, post-repair verification, delay-of-repair
-- justification), regulatory compliance records (GHG Protocol/EPA/DEFRA/
-- EU ETS/ISO 14064 framework checks with findings and recommendations),
-- and step-by-step audit trail entries (entity-level action trace with
-- hash chaining). SHA-256 provenance chains for zero-hallucination
-- audit trail.
-- =============================================================================
-- Tables (10):
--   1. fe_source_types              - Fugitive emission source category definitions
--   2. fe_component_types           - Component type emission factors (leak/no-leak EFs)
--   3. fe_emission_factors          - Emission factors by source type, gas, and method
--   4. fe_equipment_registry        - Facility component registry (tag numbers, leak status)
--   5. fe_ldar_surveys              - LDAR survey records (OGI/METHOD21/AVO/CONTINUOUS)
--   6. fe_calculations              - Emission calculation results (multi-gas with recovery)
--   7. fe_calculation_details       - Per-gas breakdown (EF, GWP, raw and CO2e, trace)
--   8. fe_leak_repairs              - Leak repair tracking (detection, repair, verification)
--   9. fe_compliance_records        - Regulatory compliance checks (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064)
--  10. fe_audit_entries             - Audit trail (entity-level action trace with hash chaining)
--
-- Hypertables (3):
--  11. fe_calculation_events        - Calculation event time-series (hypertable on event_time)
--  12. fe_survey_events             - Survey event time-series (hypertable on event_time)
--  13. fe_compliance_events         - Compliance event time-series (hypertable on event_time)
--
-- Continuous Aggregates (2):
--   1. fe_hourly_calculation_stats  - Hourly count/sum(co2e)/avg(co2e) by source_type and method
--   2. fe_daily_emission_totals     - Daily count/sum(co2e) by source_type
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-005.
-- Previous: V054__process_emissions_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS fugitive_emissions_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION fugitive_emissions_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: fugitive_emissions_service.fe_source_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_source_types (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    source_type         VARCHAR(100)    NOT NULL,
    category            VARCHAR(50)     NOT NULL,
    name                VARCHAR(200)    NOT NULL,
    description         TEXT,
    primary_gases       JSONB           NOT NULL DEFAULT '[]'::jsonb,
    applicable_methods  JSONB           NOT NULL DEFAULT '[]'::jsonb,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_fe_st_tenant_source_type UNIQUE (tenant_id, source_type)
);

ALTER TABLE fugitive_emissions_service.fe_source_types
    ADD CONSTRAINT chk_st_source_type_not_empty CHECK (LENGTH(TRIM(source_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_source_types
    ADD CONSTRAINT chk_st_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE fugitive_emissions_service.fe_source_types
    ADD CONSTRAINT chk_st_category CHECK (category IN (
        'equipment_leaks', 'coal_mining', 'oil_gas_systems', 'wastewater',
        'landfill', 'refrigeration', 'industrial_processes', 'other'
    ));

ALTER TABLE fugitive_emissions_service.fe_source_types
    ADD CONSTRAINT chk_st_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_st_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_source_types
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 2: fugitive_emissions_service.fe_component_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_component_types (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    component_type      VARCHAR(50)     NOT NULL,
    service_type        VARCHAR(30)     NOT NULL,
    emission_factor     DECIMAL(20,10)  NOT NULL,
    ef_unit             VARCHAR(50)     NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    leak_ef             DECIMAL(20,10),
    no_leak_ef          DECIMAL(20,10),
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_component_type_not_empty CHECK (LENGTH(TRIM(component_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_component_type CHECK (component_type IN (
        'valve', 'connector', 'pump_seal', 'compressor_seal',
        'pressure_relief', 'open_ended_line', 'sampling_connection',
        'flange', 'other'
    ));

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_service_type CHECK (service_type IN (
        'gas', 'light_liquid', 'heavy_liquid'
    ));

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_emission_factor_non_negative CHECK (emission_factor >= 0);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_ef_unit_not_empty CHECK (LENGTH(TRIM(ef_unit)) > 0);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_source CHECK (source IN (
        'EPA', 'IPCC', 'DEFRA', 'API', 'CUSTOM'
    ));

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_leak_ef_non_negative CHECK (leak_ef IS NULL OR leak_ef >= 0);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_no_leak_ef_non_negative CHECK (no_leak_ef IS NULL OR no_leak_ef >= 0);

ALTER TABLE fugitive_emissions_service.fe_component_types
    ADD CONSTRAINT chk_ct_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ct_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_component_types
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 3: fugitive_emissions_service.fe_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    source_type         VARCHAR(100)    NOT NULL,
    gas                 VARCHAR(20)     NOT NULL,
    factor_value        DECIMAL(20,10)  NOT NULL,
    factor_unit         VARCHAR(100)    NOT NULL,
    source              VARCHAR(50)     NOT NULL,
    method              VARCHAR(50)     NOT NULL,
    valid_from          DATE,
    valid_to            DATE,
    reference           TEXT,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_source_type_not_empty CHECK (LENGTH(TRIM(source_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CH4', 'CO2', 'N2O', 'VOC', 'SF6', 'HFC', 'PFC'
    ));

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty CHECK (LENGTH(TRIM(factor_unit)) > 0);

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'EPA', 'IPCC', 'DEFRA', 'API', 'CUSTOM'
    ));

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_method CHECK (method IN (
        'average_factor', 'screening_ranges', 'correlation_equation',
        'unit_specific', 'hi_flow_sampler', 'mass_balance',
        'engineering_estimate', 'direct_measurement'
    ));

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_date_order CHECK (valid_to IS NULL OR valid_from IS NULL OR valid_to >= valid_from);

ALTER TABLE fugitive_emissions_service.fe_emission_factors
    ADD CONSTRAINT chk_ef_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_emission_factors
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 4: fugitive_emissions_service.fe_equipment_registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_equipment_registry (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    tag_number          VARCHAR(100)    NOT NULL,
    component_type      VARCHAR(50)     NOT NULL,
    service_type        VARCHAR(30)     NOT NULL,
    facility_id         VARCHAR(100)    NOT NULL,
    process_unit        VARCHAR(100),
    location            VARCHAR(200),
    installation_date   DATE,
    is_active           BOOLEAN         DEFAULT TRUE,
    leak_status         VARCHAR(30)     DEFAULT 'NO_LEAK',
    last_survey_date    DATE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_tag_number_not_empty CHECK (LENGTH(TRIM(tag_number)) > 0);

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_component_type_not_empty CHECK (LENGTH(TRIM(component_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_component_type CHECK (component_type IN (
        'valve', 'connector', 'pump_seal', 'compressor_seal',
        'pressure_relief', 'open_ended_line', 'sampling_connection',
        'flange', 'other'
    ));

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_service_type CHECK (service_type IN (
        'gas', 'light_liquid', 'heavy_liquid'
    ));

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_facility_id_not_empty CHECK (LENGTH(TRIM(facility_id)) > 0);

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_leak_status CHECK (leak_status IN (
        'NO_LEAK', 'LEAK', 'REPAIRED', 'EXCLUDED'
    ));

ALTER TABLE fugitive_emissions_service.fe_equipment_registry
    ADD CONSTRAINT chk_er_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_er_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_equipment_registry
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 5: fugitive_emissions_service.fe_ldar_surveys
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_ldar_surveys (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    survey_type             VARCHAR(30)     NOT NULL,
    survey_date             DATE            NOT NULL,
    facility_id             VARCHAR(100)    NOT NULL,
    inspector_id            VARCHAR(100),
    components_surveyed     INTEGER         NOT NULL,
    leaks_detected          INTEGER         NOT NULL DEFAULT 0,
    coverage_pct            DECIMAL(5,2),
    methodology             VARCHAR(50),
    is_complete             BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_survey_type CHECK (survey_type IN (
        'OGI', 'METHOD21', 'AVO', 'CONTINUOUS', 'SCREENING'
    ));

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_facility_id_not_empty CHECK (LENGTH(TRIM(facility_id)) > 0);

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_components_surveyed_non_negative CHECK (components_surveyed >= 0);

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_leaks_detected_non_negative CHECK (leaks_detected >= 0);

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_leaks_not_exceed_components CHECK (leaks_detected <= components_surveyed);

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_coverage_pct_range CHECK (coverage_pct IS NULL OR (coverage_pct >= 0 AND coverage_pct <= 100));

ALTER TABLE fugitive_emissions_service.fe_ldar_surveys
    ADD CONSTRAINT chk_ls_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 6: fugitive_emissions_service.fe_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    source_type             VARCHAR(100)    NOT NULL,
    calculation_method      VARCHAR(50)     NOT NULL,
    activity_data           DECIMAL(20,6)   NOT NULL,
    activity_unit           VARCHAR(50)     NOT NULL,
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    ef_source               VARCHAR(50)     NOT NULL DEFAULT 'EPA',
    total_co2e_kg           DECIMAL(20,8)   NOT NULL,
    ch4_kg                  DECIMAL(20,8)   DEFAULT 0,
    co2_kg                  DECIMAL(20,8)   DEFAULT 0,
    n2o_kg                  DECIMAL(20,8)   DEFAULT 0,
    voc_kg                  DECIMAL(20,8)   DEFAULT 0,
    recovery_applied        BOOLEAN         DEFAULT FALSE,
    recovery_rate           DECIMAL(10,6)   DEFAULT 0,
    uncertainty_pct         DECIMAL(10,4),
    facility_id             VARCHAR(100),
    reporting_period        VARCHAR(30),
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_source_type_not_empty CHECK (LENGTH(TRIM(source_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'average_factor', 'screening_ranges', 'correlation_equation',
        'unit_specific', 'mass_balance', 'engineering_estimate',
        'direct_measurement'
    ));

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_activity_data_non_negative CHECK (activity_data >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_activity_unit_not_empty CHECK (LENGTH(TRIM(activity_unit)) > 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_ef_source CHECK (ef_source IN (
        'EPA', 'IPCC', 'DEFRA', 'API', 'CUSTOM'
    ));

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_total_co2e_kg_non_negative CHECK (total_co2e_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_ch4_kg_non_negative CHECK (ch4_kg IS NULL OR ch4_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_co2_kg_non_negative CHECK (co2_kg IS NULL OR co2_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_n2o_kg_non_negative CHECK (n2o_kg IS NULL OR n2o_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_voc_kg_non_negative CHECK (voc_kg IS NULL OR voc_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_recovery_rate_range CHECK (recovery_rate >= 0 AND recovery_rate <= 1);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_uncertainty_pct_non_negative CHECK (uncertainty_pct IS NULL OR uncertainty_pct >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE fugitive_emissions_service.fe_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_calculations
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 7: fugitive_emissions_service.fe_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL REFERENCES fugitive_emissions_service.fe_calculations(id) ON DELETE CASCADE,
    gas                     VARCHAR(20)     NOT NULL,
    emission_factor         DECIMAL(20,10)  NOT NULL,
    raw_emissions_kg        DECIMAL(20,8)   NOT NULL,
    gwp_value               DECIMAL(15,4)   NOT NULL,
    co2e_kg                 DECIMAL(20,8)   NOT NULL,
    calculation_trace       JSONB           DEFAULT '[]'::jsonb,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN (
        'CH4', 'CO2', 'N2O', 'VOC', 'SF6', 'HFC', 'PFC'
    ));

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_non_negative CHECK (emission_factor >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_raw_emissions_kg_non_negative CHECK (raw_emissions_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_gwp_value_positive CHECK (gwp_value > 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_co2e_kg_non_negative CHECK (co2e_kg >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 8: fugitive_emissions_service.fe_leak_repairs
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_leak_repairs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    equipment_id            UUID            REFERENCES fugitive_emissions_service.fe_equipment_registry(id),
    detection_date          DATE            NOT NULL,
    screening_value_ppm     DECIMAL(15,4),
    repair_date             DATE,
    repair_method           VARCHAR(100),
    post_repair_ppm         DECIMAL(15,4),
    is_verified             BOOLEAN         DEFAULT FALSE,
    delay_of_repair         BOOLEAN         DEFAULT FALSE,
    dor_justification       TEXT,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_leak_repairs
    ADD CONSTRAINT chk_lr_screening_value_non_negative CHECK (screening_value_ppm IS NULL OR screening_value_ppm >= 0);

ALTER TABLE fugitive_emissions_service.fe_leak_repairs
    ADD CONSTRAINT chk_lr_post_repair_non_negative CHECK (post_repair_ppm IS NULL OR post_repair_ppm >= 0);

ALTER TABLE fugitive_emissions_service.fe_leak_repairs
    ADD CONSTRAINT chk_lr_repair_after_detection CHECK (repair_date IS NULL OR repair_date >= detection_date);

ALTER TABLE fugitive_emissions_service.fe_leak_repairs
    ADD CONSTRAINT chk_lr_dor_justification_required CHECK (
        delay_of_repair = FALSE OR (delay_of_repair = TRUE AND dor_justification IS NOT NULL AND LENGTH(TRIM(dor_justification)) > 0)
    );

ALTER TABLE fugitive_emissions_service.fe_leak_repairs
    ADD CONSTRAINT chk_lr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_lr_updated_at
    BEFORE UPDATE ON fugitive_emissions_service.fe_leak_repairs
    FOR EACH ROW EXECUTE FUNCTION fugitive_emissions_service.set_updated_at();

-- =============================================================================
-- Table 9: fugitive_emissions_service.fe_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_compliance_records (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    calculation_id      UUID            REFERENCES fugitive_emissions_service.fe_calculations(id),
    framework           VARCHAR(50)     NOT NULL,
    check_name          VARCHAR(200)    NOT NULL,
    status              VARCHAR(30)     NOT NULL,
    details             TEXT,
    recommendations     JSONB           DEFAULT '[]'::jsonb,
    checked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE fugitive_emissions_service.fe_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM'
    ));

ALTER TABLE fugitive_emissions_service.fe_compliance_records
    ADD CONSTRAINT chk_cr_check_name_not_empty CHECK (LENGTH(TRIM(check_name)) > 0);

ALTER TABLE fugitive_emissions_service.fe_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE fugitive_emissions_service.fe_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 10: fugitive_emissions_service.fe_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_audit_entries (
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

ALTER TABLE fugitive_emissions_service.fe_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE fugitive_emissions_service.fe_audit_entries
    ADD CONSTRAINT chk_ae_entity_id_not_empty CHECK (LENGTH(TRIM(entity_id)) > 0);

ALTER TABLE fugitive_emissions_service.fe_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT'
    ));

ALTER TABLE fugitive_emissions_service.fe_audit_entries
    ADD CONSTRAINT chk_ae_entry_hash_not_empty CHECK (LENGTH(TRIM(entry_hash)) > 0);

ALTER TABLE fugitive_emissions_service.fe_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: fugitive_emissions_service.fe_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_calculation_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    source_type         VARCHAR(100),
    method              VARCHAR(50),
    emissions_kg_co2e   DECIMAL(20,8),
    duration_ms         DECIMAL(12,2),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'fugitive_emissions_service.fe_calculation_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE fugitive_emissions_service.fe_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_kg_co2e IS NULL OR emissions_kg_co2e >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE fugitive_emissions_service.fe_calculation_events
    ADD CONSTRAINT chk_cae_method CHECK (
        method IS NULL OR method IN (
            'average_factor', 'screening_ranges', 'correlation_equation',
            'unit_specific', 'mass_balance', 'engineering_estimate',
            'direct_measurement'
        )
    );

-- =============================================================================
-- Table 12: fugitive_emissions_service.fe_survey_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_survey_events (
    time                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               UUID            NOT NULL,
    survey_type             VARCHAR(30),
    facility_id             VARCHAR(100),
    components_surveyed     INTEGER,
    leaks_detected          INTEGER,
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'fugitive_emissions_service.fe_survey_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE fugitive_emissions_service.fe_survey_events
    ADD CONSTRAINT chk_sve_survey_type CHECK (
        survey_type IS NULL OR survey_type IN ('OGI', 'METHOD21', 'AVO', 'CONTINUOUS', 'SCREENING')
    );

ALTER TABLE fugitive_emissions_service.fe_survey_events
    ADD CONSTRAINT chk_sve_components_non_negative CHECK (components_surveyed IS NULL OR components_surveyed >= 0);

ALTER TABLE fugitive_emissions_service.fe_survey_events
    ADD CONSTRAINT chk_sve_leaks_non_negative CHECK (leaks_detected IS NULL OR leaks_detected >= 0);

-- =============================================================================
-- Table 13: fugitive_emissions_service.fe_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS fugitive_emissions_service.fe_compliance_events (
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
    'fugitive_emissions_service.fe_compliance_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('GHG_PROTOCOL', 'EPA', 'DEFRA', 'EU_ETS', 'ISO_14064', 'CUSTOM')
    );

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    ADD CONSTRAINT chk_coe_check_count_non_negative CHECK (check_count IS NULL OR check_count >= 0);

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    ADD CONSTRAINT chk_coe_pass_count_non_negative CHECK (pass_count IS NULL OR pass_count >= 0);

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    ADD CONSTRAINT chk_coe_fail_count_non_negative CHECK (fail_count IS NULL OR fail_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- fe_hourly_calculation_stats: hourly count/sum(co2e)/avg(co2e) by source_type and method
CREATE MATERIALIZED VIEW fugitive_emissions_service.fe_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    source_type,
    method,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e,
    AVG(emissions_kg_co2e)          AS avg_emissions_kg_co2e,
    AVG(duration_ms)                AS avg_duration_ms,
    MAX(duration_ms)                AS max_duration_ms
FROM fugitive_emissions_service.fe_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, source_type, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'fugitive_emissions_service.fe_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- fe_daily_emission_totals: daily count/sum(co2e) by source_type
CREATE MATERIALIZED VIEW fugitive_emissions_service.fe_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    source_type,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e
FROM fugitive_emissions_service.fe_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, source_type
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'fugitive_emissions_service.fe_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- fe_source_types indexes
CREATE INDEX IF NOT EXISTS idx_fe_st_tenant_id              ON fugitive_emissions_service.fe_source_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_st_source_type            ON fugitive_emissions_service.fe_source_types(source_type);
CREATE INDEX IF NOT EXISTS idx_fe_st_category               ON fugitive_emissions_service.fe_source_types(category);
CREATE INDEX IF NOT EXISTS idx_fe_st_name                   ON fugitive_emissions_service.fe_source_types(name);
CREATE INDEX IF NOT EXISTS idx_fe_st_is_active              ON fugitive_emissions_service.fe_source_types(is_active);
CREATE INDEX IF NOT EXISTS idx_fe_st_tenant_category        ON fugitive_emissions_service.fe_source_types(tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_fe_st_tenant_active          ON fugitive_emissions_service.fe_source_types(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_fe_st_created_at             ON fugitive_emissions_service.fe_source_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_st_updated_at             ON fugitive_emissions_service.fe_source_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_st_primary_gases          ON fugitive_emissions_service.fe_source_types USING GIN (primary_gases);
CREATE INDEX IF NOT EXISTS idx_fe_st_applicable_methods     ON fugitive_emissions_service.fe_source_types USING GIN (applicable_methods);
CREATE INDEX IF NOT EXISTS idx_fe_st_metadata               ON fugitive_emissions_service.fe_source_types USING GIN (metadata);

-- fe_component_types indexes
CREATE INDEX IF NOT EXISTS idx_fe_ct_tenant_id              ON fugitive_emissions_service.fe_component_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_ct_component_type         ON fugitive_emissions_service.fe_component_types(component_type);
CREATE INDEX IF NOT EXISTS idx_fe_ct_service_type           ON fugitive_emissions_service.fe_component_types(service_type);
CREATE INDEX IF NOT EXISTS idx_fe_ct_source                 ON fugitive_emissions_service.fe_component_types(source);
CREATE INDEX IF NOT EXISTS idx_fe_ct_is_active              ON fugitive_emissions_service.fe_component_types(is_active);
CREATE INDEX IF NOT EXISTS idx_fe_ct_tenant_component       ON fugitive_emissions_service.fe_component_types(tenant_id, component_type);
CREATE INDEX IF NOT EXISTS idx_fe_ct_tenant_service         ON fugitive_emissions_service.fe_component_types(tenant_id, service_type);
CREATE INDEX IF NOT EXISTS idx_fe_ct_component_service      ON fugitive_emissions_service.fe_component_types(component_type, service_type);
CREATE INDEX IF NOT EXISTS idx_fe_ct_tenant_active          ON fugitive_emissions_service.fe_component_types(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_fe_ct_created_at             ON fugitive_emissions_service.fe_component_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ct_updated_at             ON fugitive_emissions_service.fe_component_types(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ct_metadata               ON fugitive_emissions_service.fe_component_types USING GIN (metadata);

-- fe_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_fe_ef_tenant_id              ON fugitive_emissions_service.fe_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_ef_source_type            ON fugitive_emissions_service.fe_emission_factors(source_type);
CREATE INDEX IF NOT EXISTS idx_fe_ef_gas                    ON fugitive_emissions_service.fe_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_fe_ef_source                 ON fugitive_emissions_service.fe_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_fe_ef_method                 ON fugitive_emissions_service.fe_emission_factors(method);
CREATE INDEX IF NOT EXISTS idx_fe_ef_is_active              ON fugitive_emissions_service.fe_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_fe_ef_valid_from             ON fugitive_emissions_service.fe_emission_factors(valid_from DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ef_valid_to               ON fugitive_emissions_service.fe_emission_factors(valid_to DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ef_source_gas             ON fugitive_emissions_service.fe_emission_factors(source_type, gas);
CREATE INDEX IF NOT EXISTS idx_fe_ef_source_gas_method      ON fugitive_emissions_service.fe_emission_factors(source_type, gas, method);
CREATE INDEX IF NOT EXISTS idx_fe_ef_tenant_source_gas      ON fugitive_emissions_service.fe_emission_factors(tenant_id, source_type, gas);
CREATE INDEX IF NOT EXISTS idx_fe_ef_tenant_source_method   ON fugitive_emissions_service.fe_emission_factors(tenant_id, source_type, method);
CREATE INDEX IF NOT EXISTS idx_fe_ef_created_at             ON fugitive_emissions_service.fe_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ef_updated_at             ON fugitive_emissions_service.fe_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ef_metadata               ON fugitive_emissions_service.fe_emission_factors USING GIN (metadata);

-- fe_equipment_registry indexes
CREATE INDEX IF NOT EXISTS idx_fe_er_tenant_id              ON fugitive_emissions_service.fe_equipment_registry(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_er_tag_number             ON fugitive_emissions_service.fe_equipment_registry(tag_number);
CREATE INDEX IF NOT EXISTS idx_fe_er_component_type         ON fugitive_emissions_service.fe_equipment_registry(component_type);
CREATE INDEX IF NOT EXISTS idx_fe_er_service_type           ON fugitive_emissions_service.fe_equipment_registry(service_type);
CREATE INDEX IF NOT EXISTS idx_fe_er_facility_id            ON fugitive_emissions_service.fe_equipment_registry(facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_er_process_unit           ON fugitive_emissions_service.fe_equipment_registry(process_unit);
CREATE INDEX IF NOT EXISTS idx_fe_er_is_active              ON fugitive_emissions_service.fe_equipment_registry(is_active);
CREATE INDEX IF NOT EXISTS idx_fe_er_leak_status            ON fugitive_emissions_service.fe_equipment_registry(leak_status);
CREATE INDEX IF NOT EXISTS idx_fe_er_last_survey_date       ON fugitive_emissions_service.fe_equipment_registry(last_survey_date DESC);
CREATE INDEX IF NOT EXISTS idx_fe_er_tenant_facility        ON fugitive_emissions_service.fe_equipment_registry(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_er_tenant_component       ON fugitive_emissions_service.fe_equipment_registry(tenant_id, component_type);
CREATE INDEX IF NOT EXISTS idx_fe_er_tenant_leak_status     ON fugitive_emissions_service.fe_equipment_registry(tenant_id, leak_status);
CREATE INDEX IF NOT EXISTS idx_fe_er_facility_component     ON fugitive_emissions_service.fe_equipment_registry(facility_id, component_type);
CREATE INDEX IF NOT EXISTS idx_fe_er_tenant_active          ON fugitive_emissions_service.fe_equipment_registry(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_fe_er_created_at             ON fugitive_emissions_service.fe_equipment_registry(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_er_updated_at             ON fugitive_emissions_service.fe_equipment_registry(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_er_metadata               ON fugitive_emissions_service.fe_equipment_registry USING GIN (metadata);

-- fe_ldar_surveys indexes
CREATE INDEX IF NOT EXISTS idx_fe_ls_tenant_id              ON fugitive_emissions_service.fe_ldar_surveys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_ls_survey_type            ON fugitive_emissions_service.fe_ldar_surveys(survey_type);
CREATE INDEX IF NOT EXISTS idx_fe_ls_survey_date            ON fugitive_emissions_service.fe_ldar_surveys(survey_date DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ls_facility_id            ON fugitive_emissions_service.fe_ldar_surveys(facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_ls_inspector_id           ON fugitive_emissions_service.fe_ldar_surveys(inspector_id);
CREATE INDEX IF NOT EXISTS idx_fe_ls_methodology            ON fugitive_emissions_service.fe_ldar_surveys(methodology);
CREATE INDEX IF NOT EXISTS idx_fe_ls_is_complete            ON fugitive_emissions_service.fe_ldar_surveys(is_complete);
CREATE INDEX IF NOT EXISTS idx_fe_ls_tenant_facility        ON fugitive_emissions_service.fe_ldar_surveys(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_ls_tenant_survey_type     ON fugitive_emissions_service.fe_ldar_surveys(tenant_id, survey_type);
CREATE INDEX IF NOT EXISTS idx_fe_ls_facility_date          ON fugitive_emissions_service.fe_ldar_surveys(facility_id, survey_date DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ls_created_at             ON fugitive_emissions_service.fe_ldar_surveys(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ls_metadata               ON fugitive_emissions_service.fe_ldar_surveys USING GIN (metadata);

-- fe_calculations indexes
CREATE INDEX IF NOT EXISTS idx_fe_calc_tenant_id            ON fugitive_emissions_service.fe_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_calc_source_type          ON fugitive_emissions_service.fe_calculations(source_type);
CREATE INDEX IF NOT EXISTS idx_fe_calc_method               ON fugitive_emissions_service.fe_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_fe_calc_gwp_source           ON fugitive_emissions_service.fe_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_fe_calc_ef_source            ON fugitive_emissions_service.fe_calculations(ef_source);
CREATE INDEX IF NOT EXISTS idx_fe_calc_total_co2e_kg        ON fugitive_emissions_service.fe_calculations(total_co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fe_calc_recovery_applied     ON fugitive_emissions_service.fe_calculations(recovery_applied);
CREATE INDEX IF NOT EXISTS idx_fe_calc_facility_id          ON fugitive_emissions_service.fe_calculations(facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_calc_reporting_period     ON fugitive_emissions_service.fe_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_fe_calc_provenance_hash      ON fugitive_emissions_service.fe_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_fe_calc_tenant_source_type   ON fugitive_emissions_service.fe_calculations(tenant_id, source_type);
CREATE INDEX IF NOT EXISTS idx_fe_calc_tenant_method        ON fugitive_emissions_service.fe_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_fe_calc_tenant_facility      ON fugitive_emissions_service.fe_calculations(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_fe_calc_tenant_period        ON fugitive_emissions_service.fe_calculations(tenant_id, reporting_period);
CREATE INDEX IF NOT EXISTS idx_fe_calc_created_at           ON fugitive_emissions_service.fe_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_calc_updated_at           ON fugitive_emissions_service.fe_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_calc_metadata             ON fugitive_emissions_service.fe_calculations USING GIN (metadata);

-- fe_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_fe_cd_tenant_id              ON fugitive_emissions_service.fe_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_cd_calculation_id         ON fugitive_emissions_service.fe_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_fe_cd_gas                    ON fugitive_emissions_service.fe_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_fe_cd_raw_emissions_kg       ON fugitive_emissions_service.fe_calculation_details(raw_emissions_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cd_co2e_kg                ON fugitive_emissions_service.fe_calculation_details(co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cd_tenant_calc            ON fugitive_emissions_service.fe_calculation_details(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_fe_cd_calc_gas               ON fugitive_emissions_service.fe_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_fe_cd_tenant_gas             ON fugitive_emissions_service.fe_calculation_details(tenant_id, gas);
CREATE INDEX IF NOT EXISTS idx_fe_cd_created_at             ON fugitive_emissions_service.fe_calculation_details(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cd_calculation_trace      ON fugitive_emissions_service.fe_calculation_details USING GIN (calculation_trace);
CREATE INDEX IF NOT EXISTS idx_fe_cd_metadata               ON fugitive_emissions_service.fe_calculation_details USING GIN (metadata);

-- fe_leak_repairs indexes
CREATE INDEX IF NOT EXISTS idx_fe_lr_tenant_id              ON fugitive_emissions_service.fe_leak_repairs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_lr_equipment_id           ON fugitive_emissions_service.fe_leak_repairs(equipment_id);
CREATE INDEX IF NOT EXISTS idx_fe_lr_detection_date         ON fugitive_emissions_service.fe_leak_repairs(detection_date DESC);
CREATE INDEX IF NOT EXISTS idx_fe_lr_repair_date            ON fugitive_emissions_service.fe_leak_repairs(repair_date DESC);
CREATE INDEX IF NOT EXISTS idx_fe_lr_repair_method          ON fugitive_emissions_service.fe_leak_repairs(repair_method);
CREATE INDEX IF NOT EXISTS idx_fe_lr_is_verified            ON fugitive_emissions_service.fe_leak_repairs(is_verified);
CREATE INDEX IF NOT EXISTS idx_fe_lr_delay_of_repair        ON fugitive_emissions_service.fe_leak_repairs(delay_of_repair);
CREATE INDEX IF NOT EXISTS idx_fe_lr_tenant_equipment       ON fugitive_emissions_service.fe_leak_repairs(tenant_id, equipment_id);
CREATE INDEX IF NOT EXISTS idx_fe_lr_tenant_verified        ON fugitive_emissions_service.fe_leak_repairs(tenant_id, is_verified);
CREATE INDEX IF NOT EXISTS idx_fe_lr_tenant_dor             ON fugitive_emissions_service.fe_leak_repairs(tenant_id, delay_of_repair);
CREATE INDEX IF NOT EXISTS idx_fe_lr_created_at             ON fugitive_emissions_service.fe_leak_repairs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_lr_updated_at             ON fugitive_emissions_service.fe_leak_repairs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_lr_metadata               ON fugitive_emissions_service.fe_leak_repairs USING GIN (metadata);

-- fe_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_fe_cr_tenant_id              ON fugitive_emissions_service.fe_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_cr_calculation_id         ON fugitive_emissions_service.fe_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_fe_cr_framework              ON fugitive_emissions_service.fe_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_fe_cr_check_name             ON fugitive_emissions_service.fe_compliance_records(check_name);
CREATE INDEX IF NOT EXISTS idx_fe_cr_status                 ON fugitive_emissions_service.fe_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_fe_cr_checked_at             ON fugitive_emissions_service.fe_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cr_tenant_framework       ON fugitive_emissions_service.fe_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_fe_cr_tenant_status          ON fugitive_emissions_service.fe_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_fe_cr_framework_status       ON fugitive_emissions_service.fe_compliance_records(framework, status);
CREATE INDEX IF NOT EXISTS idx_fe_cr_tenant_calculation     ON fugitive_emissions_service.fe_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_fe_cr_created_at             ON fugitive_emissions_service.fe_compliance_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cr_recommendations        ON fugitive_emissions_service.fe_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_fe_cr_metadata               ON fugitive_emissions_service.fe_compliance_records USING GIN (metadata);

-- fe_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_fe_ae_tenant_id              ON fugitive_emissions_service.fe_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fe_ae_entity_type            ON fugitive_emissions_service.fe_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_fe_ae_entity_id              ON fugitive_emissions_service.fe_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_fe_ae_action                 ON fugitive_emissions_service.fe_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_fe_ae_parent_hash            ON fugitive_emissions_service.fe_audit_entries(parent_hash);
CREATE INDEX IF NOT EXISTS idx_fe_ae_entry_hash             ON fugitive_emissions_service.fe_audit_entries(entry_hash);
CREATE INDEX IF NOT EXISTS idx_fe_ae_user_id                ON fugitive_emissions_service.fe_audit_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_fe_ae_tenant_entity          ON fugitive_emissions_service.fe_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_fe_ae_tenant_action          ON fugitive_emissions_service.fe_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_fe_ae_created_at             ON fugitive_emissions_service.fe_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fe_ae_details                ON fugitive_emissions_service.fe_audit_entries USING GIN (details);

-- fe_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fe_cae_tenant_id             ON fugitive_emissions_service.fe_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_event_type            ON fugitive_emissions_service.fe_calculation_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_source_type           ON fugitive_emissions_service.fe_calculation_events(source_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_method                ON fugitive_emissions_service.fe_calculation_events(method, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_tenant_source         ON fugitive_emissions_service.fe_calculation_events(tenant_id, source_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_tenant_method         ON fugitive_emissions_service.fe_calculation_events(tenant_id, method, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_cae_metadata              ON fugitive_emissions_service.fe_calculation_events USING GIN (metadata);

-- fe_survey_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fe_sve_tenant_id             ON fugitive_emissions_service.fe_survey_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_sve_survey_type           ON fugitive_emissions_service.fe_survey_events(survey_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_sve_facility_id           ON fugitive_emissions_service.fe_survey_events(facility_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_sve_tenant_survey         ON fugitive_emissions_service.fe_survey_events(tenant_id, survey_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_sve_tenant_facility       ON fugitive_emissions_service.fe_survey_events(tenant_id, facility_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_sve_metadata              ON fugitive_emissions_service.fe_survey_events USING GIN (metadata);

-- fe_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fe_coe_tenant_id             ON fugitive_emissions_service.fe_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_coe_framework             ON fugitive_emissions_service.fe_compliance_events(framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_coe_status                ON fugitive_emissions_service.fe_compliance_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_coe_tenant_framework      ON fugitive_emissions_service.fe_compliance_events(tenant_id, framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_coe_tenant_status         ON fugitive_emissions_service.fe_compliance_events(tenant_id, status, time DESC);
CREATE INDEX IF NOT EXISTS idx_fe_coe_metadata              ON fugitive_emissions_service.fe_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- fe_source_types: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_source_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_st_read  ON fugitive_emissions_service.fe_source_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_st_write ON fugitive_emissions_service.fe_source_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_component_types: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_component_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_ct_read  ON fugitive_emissions_service.fe_component_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_ct_write ON fugitive_emissions_service.fe_component_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_emission_factors: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_ef_read  ON fugitive_emissions_service.fe_emission_factors FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_ef_write ON fugitive_emissions_service.fe_emission_factors FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_equipment_registry: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_equipment_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_er_read  ON fugitive_emissions_service.fe_equipment_registry FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_er_write ON fugitive_emissions_service.fe_equipment_registry FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_ldar_surveys: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_ldar_surveys ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_ls_read  ON fugitive_emissions_service.fe_ldar_surveys FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_ls_write ON fugitive_emissions_service.fe_ldar_surveys FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_calculations: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_calc_read  ON fugitive_emissions_service.fe_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_calc_write ON fugitive_emissions_service.fe_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_calculation_details: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_cd_read  ON fugitive_emissions_service.fe_calculation_details FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_cd_write ON fugitive_emissions_service.fe_calculation_details FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_leak_repairs: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_leak_repairs ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_lr_read  ON fugitive_emissions_service.fe_leak_repairs FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_lr_write ON fugitive_emissions_service.fe_leak_repairs FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_compliance_records: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_cr_read  ON fugitive_emissions_service.fe_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_cr_write ON fugitive_emissions_service.fe_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_audit_entries: tenant-isolated
ALTER TABLE fugitive_emissions_service.fe_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_ae_read  ON fugitive_emissions_service.fe_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fe_ae_write ON fugitive_emissions_service.fe_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fe_calculation_events: open read/write (time-series telemetry)
ALTER TABLE fugitive_emissions_service.fe_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_cae_read  ON fugitive_emissions_service.fe_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY fe_cae_write ON fugitive_emissions_service.fe_calculation_events FOR ALL   USING (TRUE);

-- fe_survey_events: open read/write (time-series telemetry)
ALTER TABLE fugitive_emissions_service.fe_survey_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_sve_read  ON fugitive_emissions_service.fe_survey_events FOR SELECT USING (TRUE);
CREATE POLICY fe_sve_write ON fugitive_emissions_service.fe_survey_events FOR ALL   USING (TRUE);

-- fe_compliance_events: open read/write (time-series telemetry)
ALTER TABLE fugitive_emissions_service.fe_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fe_coe_read  ON fugitive_emissions_service.fe_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY fe_coe_write ON fugitive_emissions_service.fe_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA fugitive_emissions_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA fugitive_emissions_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA fugitive_emissions_service TO greenlang_app;
GRANT SELECT ON fugitive_emissions_service.fe_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON fugitive_emissions_service.fe_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA fugitive_emissions_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA fugitive_emissions_service TO greenlang_readonly;
GRANT SELECT ON fugitive_emissions_service.fe_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON fugitive_emissions_service.fe_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA fugitive_emissions_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA fugitive_emissions_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA fugitive_emissions_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'fugitive-emissions:read',                    'fugitive-emissions', 'read',                    'View all fugitive emissions service data including source types, component types, equipment registry, LDAR surveys, calculations, and leak repairs'),
    (gen_random_uuid(), 'fugitive-emissions:write',                   'fugitive-emissions', 'write',                   'Create, update, and manage all fugitive emissions service data'),
    (gen_random_uuid(), 'fugitive-emissions:execute',                 'fugitive-emissions', 'execute',                 'Execute fugitive emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'fugitive-emissions:sources:read',            'fugitive-emissions', 'sources_read',            'View fugitive emission source type registry with categories (equipment_leaks/coal_mining/oil_gas_systems/wastewater/landfill/refrigeration/industrial_processes/other), primary gases, and applicable methods'),
    (gen_random_uuid(), 'fugitive-emissions:sources:write',           'fugitive-emissions', 'sources_write',           'Create, update, and manage fugitive emission source type registry entries'),
    (gen_random_uuid(), 'fugitive-emissions:components:read',         'fugitive-emissions', 'components_read',         'View component type emission factors with leak/no-leak EFs for valve, connector, pump_seal, compressor_seal, pressure_relief, open_ended_line, sampling_connection, and flange types'),
    (gen_random_uuid(), 'fugitive-emissions:components:write',        'fugitive-emissions', 'components_write',        'Create, update, and manage component type emission factor entries'),
    (gen_random_uuid(), 'fugitive-emissions:equipment:read',          'fugitive-emissions', 'equipment_read',          'View facility equipment registry with tag numbers, component types, service types, leak status (NO_LEAK/LEAK/REPAIRED/EXCLUDED), and survey dates'),
    (gen_random_uuid(), 'fugitive-emissions:equipment:write',         'fugitive-emissions', 'equipment_write',         'Create, update, and manage facility equipment registry entries'),
    (gen_random_uuid(), 'fugitive-emissions:factors:read',            'fugitive-emissions', 'factors_read',            'View emission factors by source type, gas, and method (average_factor/screening_ranges/correlation_equation/unit_specific/hi_flow_sampler) from EPA/IPCC/DEFRA/API/CUSTOM sources'),
    (gen_random_uuid(), 'fugitive-emissions:factors:write',           'fugitive-emissions', 'factors_write',           'Create, update, and manage emission factor entries with source, method, and validity date range data'),
    (gen_random_uuid(), 'fugitive-emissions:surveys:read',            'fugitive-emissions', 'surveys_read',            'View LDAR survey records with OGI/METHOD21/AVO/CONTINUOUS/SCREENING types, component counts, leak detection results, and coverage percentages'),
    (gen_random_uuid(), 'fugitive-emissions:surveys:write',           'fugitive-emissions', 'surveys_write',           'Create, update, and manage LDAR survey records'),
    (gen_random_uuid(), 'fugitive-emissions:calculations:read',       'fugitive-emissions', 'calculations_read',       'View fugitive emission calculation results with multi-gas breakdown (CH4/CO2/N2O/VOC), recovery rate, uncertainty, and provenance data'),
    (gen_random_uuid(), 'fugitive-emissions:calculations:write',      'fugitive-emissions', 'calculations_write',      'Create and manage fugitive emission calculation records'),
    (gen_random_uuid(), 'fugitive-emissions:repairs:read',            'fugitive-emissions', 'repairs_read',            'View leak repair records with detection dates, screening values, repair methods, post-repair verification, and delay-of-repair justifications'),
    (gen_random_uuid(), 'fugitive-emissions:repairs:write',           'fugitive-emissions', 'repairs_write',           'Create, update, and manage leak repair tracking records'),
    (gen_random_uuid(), 'fugitive-emissions:compliance:read',         'fugitive-emissions', 'compliance_read',         'View regulatory compliance records for GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 with findings and recommendations'),
    (gen_random_uuid(), 'fugitive-emissions:compliance:execute',      'fugitive-emissions', 'compliance_execute',      'Execute regulatory compliance checks against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks'),
    (gen_random_uuid(), 'fugitive-emissions:admin',                   'fugitive-emissions', 'admin',                   'Full administrative access to fugitive emissions service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('fugitive_emissions_service.fe_calculation_events', INTERVAL '365 days');
SELECT add_retention_policy('fugitive_emissions_service.fe_survey_events',      INTERVAL '365 days');
SELECT add_retention_policy('fugitive_emissions_service.fe_compliance_events',  INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE fugitive_emissions_service.fe_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'source_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('fugitive_emissions_service.fe_calculation_events', INTERVAL '30 days');

ALTER TABLE fugitive_emissions_service.fe_survey_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'survey_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('fugitive_emissions_service.fe_survey_events', INTERVAL '30 days');

ALTER TABLE fugitive_emissions_service.fe_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('fugitive_emissions_service.fe_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Fugitive Emissions Agent (GL-MRV-SCOPE1-005)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-005',
    'Fugitive Emissions Agent',
    'Fugitive emission calculator for GreenLang Climate OS. Manages fugitive emission source type registry with equipment_leaks, coal_mining, oil_gas_systems, wastewater, landfill, refrigeration, industrial_processes, and other categories including primary greenhouse gases (CH4/CO2/N2O/VOC/SF6/HFC/PFC) and applicable quantification methods (average_factor, screening_ranges, correlation_equation, unit_specific, hi_flow_sampler, mass_balance, engineering_estimate, direct_measurement). Maintains component type emission factor registry for valve, connector, pump_seal, compressor_seal, pressure_relief, open_ended_line, sampling_connection, and flange types with gas/light_liquid/heavy_liquid service types, leak/no-leak emission factors from EPA/IPCC/DEFRA/API/CUSTOM sources. Stores emission factor database with source type x gas factors across multiple quantification methods with validity date ranges and references. Manages facility equipment registry with tag numbers, component types, service types, process units, locations, installation dates, and leak status tracking (NO_LEAK/LEAK/REPAIRED/EXCLUDED). Records LDAR surveys (OGI/METHOD21/AVO/CONTINUOUS/SCREENING) with inspector IDs, component counts, leak detection counts, coverage percentages, and methodology tracking. Executes deterministic fugitive emission calculations using average factor, screening ranges, correlation equation, unit-specific, mass balance, engineering estimate, and direct measurement methods with multi-gas GWP weighting (CH4/CO2/N2O/VOC) using AR4/AR5/AR6 values. Applies gas recovery rate corrections. Quantifies uncertainty. Produces per-gas calculation detail breakdowns with individual emission factors, GWP values, raw and CO2e emissions, and step-by-step calculation traces. Tracks leak repairs with detection dates, screening values (PPM), repair dates and methods, post-repair verification, and delay-of-repair justifications. Checks regulatory compliance against GHG Protocol, EPA, DEFRA, EU ETS, and ISO 14064 frameworks with check names, findings, and recommendations. Generates entity-level audit trail entries with action tracking, parent/entry hash chaining for tamper-evident provenance, and user attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/fugitive-emissions',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-005', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/fugitive-emissions-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"fugitive-emissions", "scope-1", "equipment-leaks", "ldar", "ghg-protocol", "epa", "ipcc", "defra", "api", "mrv"}',
    '{"oil-and-gas", "petrochemical", "refining", "chemical", "mining", "wastewater", "landfill", "cross-sector"}',
    'f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'source_type_registry',
    'configuration',
    'Register and manage fugitive emission source type entries with category classification (equipment_leaks/coal_mining/oil_gas_systems/wastewater/landfill/refrigeration/industrial_processes/other), primary greenhouse gases, and applicable quantification methods.',
    '{"source_type", "category", "name", "primary_gases", "applicable_methods"}',
    '{"source_type_id", "registration_result"}',
    '{"categories": ["equipment_leaks", "coal_mining", "oil_gas_systems", "wastewater", "landfill", "refrigeration", "industrial_processes", "other"], "gases": ["CH4", "CO2", "N2O", "VOC", "SF6", "HFC", "PFC"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'component_type_registry',
    'configuration',
    'Register and manage component type emission factor entries with component type (valve/connector/pump_seal/compressor_seal/pressure_relief/open_ended_line/sampling_connection/flange/other), service type (gas/light_liquid/heavy_liquid), overall emission factor, leak/no-leak emission factors, and source attribution.',
    '{"component_type", "service_type", "emission_factor", "ef_unit", "source", "leak_ef", "no_leak_ef"}',
    '{"component_type_id", "registration_result"}',
    '{"component_types": ["valve", "connector", "pump_seal", "compressor_seal", "pressure_relief", "open_ended_line", "sampling_connection", "flange", "other"], "service_types": ["gas", "light_liquid", "heavy_liquid"], "sources": ["EPA", "IPCC", "DEFRA", "API", "CUSTOM"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'equipment_registry',
    'configuration',
    'Register and manage facility equipment components with tag numbers, component types, service types, facility/process unit assignment, location, installation dates, leak status tracking (NO_LEAK/LEAK/REPAIRED/EXCLUDED), and last LDAR survey date tracking.',
    '{"tag_number", "component_type", "service_type", "facility_id", "process_unit", "location", "installation_date", "leak_status"}',
    '{"equipment_id", "registration_result"}',
    '{"leak_statuses": ["NO_LEAK", "LEAK", "REPAIRED", "EXCLUDED"], "supports_tag_tracking": true, "supports_survey_date_tracking": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'ldar_survey_management',
    'processing',
    'Record and manage LDAR (Leak Detection and Repair) survey records with survey type (OGI/METHOD21/AVO/CONTINUOUS/SCREENING), facility and inspector assignment, component counts, leak detection results, coverage percentages, methodology tracking, and completion status.',
    '{"survey_type", "survey_date", "facility_id", "inspector_id", "components_surveyed", "leaks_detected", "coverage_pct", "methodology"}',
    '{"survey_id", "survey_result"}',
    '{"survey_types": ["OGI", "METHOD21", "AVO", "CONTINUOUS", "SCREENING"], "tracks_coverage": true, "tracks_methodology": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic fugitive emission calculations using average factor (component count x EF), screening ranges (leak/no-leak EFs by screening value), correlation equation (screening value to mass emission rate), unit-specific (direct measurement per component), mass balance (input-output accounting), engineering estimate, and direct measurement methods. Supports multi-gas GWP weighting for CH4/CO2/N2O/VOC with gas recovery rate corrections and uncertainty quantification.',
    '{"source_type", "calculation_method", "activity_data", "activity_unit", "gwp_source", "ef_source", "facility_id", "reporting_period"}',
    '{"calculation_id", "total_co2e_kg", "per_gas_breakdown", "recovery_applied", "recovery_rate", "uncertainty_pct", "provenance_hash"}',
    '{"methods": ["average_factor", "screening_ranges", "correlation_equation", "unit_specific", "mass_balance", "engineering_estimate", "direct_measurement"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CH4", "CO2", "N2O", "VOC"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'leak_repair_tracking',
    'processing',
    'Track leak repair lifecycle from detection through repair and verification. Record detection dates, screening values (PPM), repair dates and methods, post-repair screening verification, delay-of-repair flags with mandatory justification text, and equipment association.',
    '{"equipment_id", "detection_date", "screening_value_ppm", "repair_date", "repair_method", "post_repair_ppm", "is_verified", "delay_of_repair", "dor_justification"}',
    '{"repair_id", "tracking_result"}',
    '{"supports_dor": true, "requires_dor_justification": true, "supports_verification": true, "tracks_ppm_values": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of fugitive emission calculations against GHG Protocol, EPA (40 CFR Part 98 Subpart W), DEFRA, EU ETS, and ISO 14064 frameworks. Produce named check results with findings detail text and actionable recommendations.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "check_name", "status", "details", "recommendations"}',
    '{"frameworks": ["GHG_PROTOCOL", "EPA", "DEFRA", "EU_ETS", "ISO_14064", "CUSTOM"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-005', '1.0.0',
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
    ('GL-MRV-SCOPE1-005', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage fugitive emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-005', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for activity data quantities, emission factor unit conversions, and mass/volume unit alignment'),
    ('GL-MRV-SCOPE1-005', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-005', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-SCOPE1-005', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, survey events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-005', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of equipment counts and activity data'),
    ('GL-MRV-SCOPE1-005', 'GL-MRV-SCOPE1-002', '>=1.0.0', true, 'Refrigerants & F-Gas Agent for cross-referencing refrigerant fugitive leakage data and shared fluorinated gas emission factors'),
    ('GL-MRV-SCOPE1-005', 'GL-MRV-SCOPE1-004', '>=1.0.0', true, 'Process Emissions Agent for cross-referencing industrial process vent gas fugitive emission data')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-005',
    'Fugitive Emissions Agent',
    'Fugitive emission calculator. Source type registry (equipment_leaks/coal_mining/oil_gas_systems/wastewater/landfill/refrigeration/industrial_processes/other, primary gases CH4/CO2/N2O/VOC/SF6/HFC/PFC, applicable methods). Component type EFs (valve/connector/pump_seal/compressor_seal/pressure_relief/open_ended_line/sampling_connection/flange, gas/light_liquid/heavy_liquid service, leak/no-leak EFs, EPA/IPCC/DEFRA/API/CUSTOM). Emission factor database (source x gas x method, validity dates). Equipment registry (tag numbers, component types, leak status NO_LEAK/LEAK/REPAIRED/EXCLUDED, survey dates). LDAR surveys (OGI/METHOD21/AVO/CONTINUOUS/SCREENING, component counts, leaks detected, coverage). Emission calculations (average_factor/screening_ranges/correlation_equation/unit_specific/mass_balance/engineering_estimate/direct_measurement, multi-gas CH4/CO2/N2O/VOC GWP AR4/AR5/AR6, recovery rate). Per-gas breakdowns with calculation traces. Leak repair tracking (detection/repair/verification, delay-of-repair). Compliance checks (GHG Protocol/EPA/DEFRA/EU ETS/ISO 14064). Audit trail with hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA fugitive_emissions_service IS
    'Fugitive Emissions Agent (AGENT-MRV-005) - fugitive emission source type registry, component type emission factors, emission factor database, equipment registry, LDAR surveys, emission calculations, per-gas breakdowns, leak repair tracking, compliance records, audit trail, provenance chains';

COMMENT ON TABLE fugitive_emissions_service.fe_source_types IS
    'Fugitive emission source type registry: tenant_id, source_type (unique per tenant), category (equipment_leaks/coal_mining/oil_gas_systems/wastewater/landfill/refrigeration/industrial_processes/other), name, description, primary_gases JSONB, applicable_methods JSONB, is_active, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_component_types IS
    'Component type emission factors: tenant_id, component_type (valve/connector/pump_seal/compressor_seal/pressure_relief/open_ended_line/sampling_connection/flange/other), service_type (gas/light_liquid/heavy_liquid), emission_factor, ef_unit, source (EPA/IPCC/DEFRA/API/CUSTOM), leak_ef, no_leak_ef, is_active, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_emission_factors IS
    'Emission factor database: tenant_id, source_type, gas (CH4/CO2/N2O/VOC/SF6/HFC/PFC), factor_value, factor_unit, source (EPA/IPCC/DEFRA/API/CUSTOM), method (average_factor/screening_ranges/correlation_equation/unit_specific/hi_flow_sampler/mass_balance/engineering_estimate/direct_measurement), valid_from/to, reference, is_active, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_equipment_registry IS
    'Facility equipment registry: tenant_id, tag_number, component_type (valve/connector/pump_seal/compressor_seal/pressure_relief/open_ended_line/sampling_connection/flange/other), service_type (gas/light_liquid/heavy_liquid), facility_id, process_unit, location, installation_date, is_active, leak_status (NO_LEAK/LEAK/REPAIRED/EXCLUDED), last_survey_date, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_ldar_surveys IS
    'LDAR survey records: tenant_id, survey_type (OGI/METHOD21/AVO/CONTINUOUS/SCREENING), survey_date, facility_id, inspector_id, components_surveyed, leaks_detected, coverage_pct, methodology, is_complete, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_calculations IS
    'Calculation results: tenant_id, source_type, calculation_method (average_factor/screening_ranges/correlation_equation/unit_specific/mass_balance/engineering_estimate/direct_measurement), activity_data/unit, gwp_source (AR4/AR5/AR6), ef_source (EPA/IPCC/DEFRA/API/CUSTOM), total_co2e_kg, ch4_kg, co2_kg, n2o_kg, voc_kg, recovery_applied/rate, uncertainty_pct, facility_id, reporting_period, provenance_hash, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_calculation_details IS
    'Per-gas calculation breakdown: tenant_id, calculation_id (FK CASCADE), gas (CH4/CO2/N2O/VOC/SF6/HFC/PFC), emission_factor, raw_emissions_kg, gwp_value, co2e_kg, calculation_trace JSONB, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_leak_repairs IS
    'Leak repair tracking: tenant_id, equipment_id (FK), detection_date, screening_value_ppm, repair_date, repair_method, post_repair_ppm, is_verified, delay_of_repair, dor_justification, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_compliance_records IS
    'Regulatory compliance records: tenant_id, calculation_id (FK), framework (GHG_PROTOCOL/EPA/DEFRA/EU_ETS/ISO_14064/CUSTOM), check_name, status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), details, recommendations JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE fugitive_emissions_service.fe_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id, action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT), details JSONB, parent_hash, entry_hash (SHA-256 chain), user_id';

COMMENT ON TABLE fugitive_emissions_service.fe_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, event_type, source_type, method, emissions_kg_co2e, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE fugitive_emissions_service.fe_survey_events IS
    'TimescaleDB hypertable: survey events with tenant_id, survey_type, facility_id, components_surveyed, leaks_detected, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE fugitive_emissions_service.fe_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, event_type, framework, status, check_count, pass_count, fail_count, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW fugitive_emissions_service.fe_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by source_type and method (total calculations, sum emissions kg CO2e, avg emissions kg CO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW fugitive_emissions_service.fe_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by source_type (total calculations, sum emissions kg CO2e per day)';
