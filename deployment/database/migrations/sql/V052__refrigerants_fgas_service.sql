-- =============================================================================
-- V052: Refrigerants & F-Gas Service Schema
-- =============================================================================
-- Component: AGENT-MRV-002 (GL-MRV-SCOPE1-002)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Refrigerants & F-Gas Agent (GL-MRV-SCOPE1-002) with capabilities for
-- refrigerant type registry management (HFC/HFC_BLEND/HFO/PFC/SF6/NF3/
-- HCFC/CFC/NATURAL categories with molecular weight, boiling point, ODP,
-- atmospheric lifetime, phase-out dates, and blend composition tracking),
-- GWP value database (AR4/AR5/AR6/AR6_20YR/CUSTOM sources with 100yr and
-- 20yr timeframes and effective date ranges), equipment registry management
-- (equipment types with refrigerant charge, equipment count, status,
-- installation dates, expected lifetime, climate zone, LDAR level, and
-- custom leak rates), leak rate database (IPCC 2006 default and custom
-- rates per equipment type and lifecycle stage), service event tracking
-- (installation/recharge/repair/decommission events with refrigerant
-- added/recovered quantities and technician records), refrigerant emission
-- calculations (screening/mass-balance/leak-rate methods converting
-- refrigerant loss to CO2e via GWP weighting with uncertainty bounds and
-- data quality scoring), per-gas calculation detail breakdowns (individual
-- refrigerant and blend component losses with applied GWP values),
-- regulatory compliance records (EU F-Gas/Kigali/EPA 608/CARB per-year
-- quota tracking with phase-down targets), step-by-step audit trail
-- entries (input/output data per calculation step with factor references),
-- and SHA-256 provenance chains for zero-hallucination audit trail.
-- =============================================================================
-- Tables (10):
--   1. rf_refrigerant_types        - Refrigerant registry (HFC/HFO/PFC/SF6/NF3/HCFC/CFC/NATURAL, GWP, ODP, phase-out)
--   2. rf_refrigerant_blends       - Blend composition (component refrigerants with weight fractions)
--   3. rf_gwp_values               - GWP value database (AR4/AR5/AR6/AR6_20YR/CUSTOM, 100yr/20yr timeframes)
--   4. rf_equipment_registry       - Equipment registry (type, refrigerant, charge, status, LDAR, climate zone)
--   5. rf_leak_rates               - Leak rate database (IPCC 2006 defaults, custom rates per lifecycle stage)
--   6. rf_service_events           - Service events (install/recharge/repair/decommission, added/recovered kg)
--   7. rf_calculations             - Calculation results (screening/mass-balance/leak-rate to CO2e with provenance)
--   8. rf_calculation_details      - Per-gas breakdown (individual refrigerant losses with GWP and blend flags)
--   9. rf_compliance_records       - Regulatory compliance (EU F-Gas/Kigali/EPA 608/CARB quota tracking)
--  10. rf_audit_entries            - Audit trail (step-by-step calculation trace)
--
-- Hypertables (3):
--  11. rf_calculation_events       - Calculation event time-series (hypertable on created_at)
--  12. rf_service_events_ts        - Service event time-series (hypertable on created_at)
--  13. rf_compliance_events        - Compliance event time-series (hypertable on created_at)
--
-- Continuous Aggregates (2):
--   1. rf_hourly_calculation_stats - Hourly count/sum(emissions)/avg(emissions) by method
--   2. rf_daily_emission_totals    - Daily count/sum(emissions) by refrigerant_code
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (90 days on hypertables), compression policies (7 days),
-- updated_at trigger, security permissions for
-- greenlang_app/greenlang_readonly/greenlang_admin, and agent registry
-- seed data registering GL-MRV-SCOPE1-002.
-- Previous: V051__stationary_combustion_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS refrigerants_fgas_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION refrigerants_fgas_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: refrigerants_fgas_service.rf_refrigerant_types
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_refrigerant_types (
    id                          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID          NOT NULL,
    refrigerant_code            VARCHAR(50)   NOT NULL,
    category                    VARCHAR(30)   NOT NULL,
    name                        VARCHAR(200)  NOT NULL,
    formula                     VARCHAR(100),
    molecular_weight            DECIMAL(10,4),
    boiling_point_c             DECIMAL(8,2),
    odp                         DECIMAL(8,6)  DEFAULT 0,
    atmospheric_lifetime_years  DECIMAL(10,2),
    is_blend                    BOOLEAN       DEFAULT FALSE,
    is_regulated                BOOLEAN       DEFAULT TRUE,
    phase_out_date              DATE,
    is_custom                   BOOLEAN       DEFAULT FALSE,
    created_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_rf_rt_tenant_code UNIQUE (tenant_id, refrigerant_code)
);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_refrigerant_code_not_empty CHECK (LENGTH(TRIM(refrigerant_code)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_category CHECK (category IN (
        'HFC', 'HFC_BLEND', 'HFO', 'PFC', 'SF6', 'NF3', 'HCFC', 'CFC', 'NATURAL'
    ));

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_molecular_weight_positive CHECK (molecular_weight IS NULL OR molecular_weight > 0);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_odp_non_negative CHECK (odp IS NULL OR odp >= 0);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_atmospheric_lifetime_positive CHECK (atmospheric_lifetime_years IS NULL OR atmospheric_lifetime_years > 0);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types
    ADD CONSTRAINT chk_rt_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_rt_updated_at
    BEFORE UPDATE ON refrigerants_fgas_service.rf_refrigerant_types
    FOR EACH ROW EXECUTE FUNCTION refrigerants_fgas_service.set_updated_at();

-- =============================================================================
-- Table 2: refrigerants_fgas_service.rf_refrigerant_blends
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_refrigerant_blends (
    id                          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID          NOT NULL,
    blend_refrigerant_id        UUID          NOT NULL,
    component_refrigerant_id    UUID          NOT NULL,
    weight_fraction             DECIMAL(6,4)  NOT NULL,
    created_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends
    ADD CONSTRAINT fk_rb_blend_refrigerant
        FOREIGN KEY (blend_refrigerant_id)
        REFERENCES refrigerants_fgas_service.rf_refrigerant_types(id)
        ON DELETE CASCADE;

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends
    ADD CONSTRAINT fk_rb_component_refrigerant
        FOREIGN KEY (component_refrigerant_id)
        REFERENCES refrigerants_fgas_service.rf_refrigerant_types(id)
        ON DELETE CASCADE;

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends
    ADD CONSTRAINT chk_rb_weight_fraction_range CHECK (weight_fraction > 0 AND weight_fraction <= 1);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends
    ADD CONSTRAINT chk_rb_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends
    ADD CONSTRAINT chk_rb_blend_ne_component CHECK (blend_refrigerant_id != component_refrigerant_id);

-- =============================================================================
-- Table 3: refrigerants_fgas_service.rf_gwp_values
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_gwp_values (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID          NOT NULL,
    refrigerant_id      UUID          NOT NULL,
    gwp_source          VARCHAR(20)   NOT NULL,
    timeframe           VARCHAR(10)   DEFAULT '100yr',
    value               DECIMAL(12,2) NOT NULL,
    effective_date      DATE,
    source_reference    TEXT,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_rf_gwp_tenant_ref_source_tf UNIQUE (tenant_id, refrigerant_id, gwp_source, timeframe)
);

ALTER TABLE refrigerants_fgas_service.rf_gwp_values
    ADD CONSTRAINT fk_gwp_refrigerant
        FOREIGN KEY (refrigerant_id)
        REFERENCES refrigerants_fgas_service.rf_refrigerant_types(id)
        ON DELETE CASCADE;

ALTER TABLE refrigerants_fgas_service.rf_gwp_values
    ADD CONSTRAINT chk_gwp_source CHECK (gwp_source IN (
        'AR4', 'AR5', 'AR6', 'AR6_20YR', 'CUSTOM'
    ));

ALTER TABLE refrigerants_fgas_service.rf_gwp_values
    ADD CONSTRAINT chk_gwp_timeframe CHECK (timeframe IN ('100yr', '20yr'));

ALTER TABLE refrigerants_fgas_service.rf_gwp_values
    ADD CONSTRAINT chk_gwp_value_non_negative CHECK (value >= 0);

ALTER TABLE refrigerants_fgas_service.rf_gwp_values
    ADD CONSTRAINT chk_gwp_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 4: refrigerants_fgas_service.rf_equipment_registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_equipment_registry (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID          NOT NULL,
    equipment_type          VARCHAR(50)   NOT NULL,
    refrigerant_id          UUID          NOT NULL,
    charge_kg               DECIMAL(12,4) NOT NULL,
    equipment_count         INTEGER       DEFAULT 1,
    status                  VARCHAR(20)   DEFAULT 'ACTIVE',
    installation_date       DATE,
    expected_lifetime_years INTEGER,
    location                VARCHAR(500),
    equipment_identifier    VARCHAR(200),
    custom_leak_rate        DECIMAL(6,4),
    climate_zone            VARCHAR(20),
    ldar_level              VARCHAR(30),
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT fk_er_refrigerant
        FOREIGN KEY (refrigerant_id)
        REFERENCES refrigerants_fgas_service.rf_refrigerant_types(id)
        ON DELETE RESTRICT;

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_equipment_type_not_empty CHECK (LENGTH(TRIM(equipment_type)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_charge_kg_positive CHECK (charge_kg > 0);

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_equipment_count_positive CHECK (equipment_count > 0);

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_status CHECK (status IN (
        'ACTIVE', 'INACTIVE', 'DECOMMISSIONED', 'MAINTENANCE', 'RETIRED'
    ));

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_expected_lifetime_positive CHECK (expected_lifetime_years IS NULL OR expected_lifetime_years > 0);

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_custom_leak_rate_range CHECK (custom_leak_rate IS NULL OR (custom_leak_rate >= 0 AND custom_leak_rate <= 1));

ALTER TABLE refrigerants_fgas_service.rf_equipment_registry
    ADD CONSTRAINT chk_er_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_er_updated_at
    BEFORE UPDATE ON refrigerants_fgas_service.rf_equipment_registry
    FOR EACH ROW EXECUTE FUNCTION refrigerants_fgas_service.set_updated_at();

-- =============================================================================
-- Table 5: refrigerants_fgas_service.rf_leak_rates
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_leak_rates (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID          NOT NULL,
    equipment_type      VARCHAR(50)   NOT NULL,
    lifecycle_stage     VARCHAR(20)   NOT NULL,
    base_rate           DECIMAL(6,4)  NOT NULL,
    source              VARCHAR(100)  DEFAULT 'IPCC_2006',
    is_custom           BOOLEAN       DEFAULT FALSE,
    notes               TEXT,
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_rf_lr_tenant_equip_stage_source UNIQUE (tenant_id, equipment_type, lifecycle_stage, source)
);

ALTER TABLE refrigerants_fgas_service.rf_leak_rates
    ADD CONSTRAINT chk_lr_equipment_type_not_empty CHECK (LENGTH(TRIM(equipment_type)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_leak_rates
    ADD CONSTRAINT chk_lr_lifecycle_stage CHECK (lifecycle_stage IN (
        'INSTALLATION', 'OPERATION', 'SERVICING', 'DISPOSAL', 'STANDBY'
    ));

ALTER TABLE refrigerants_fgas_service.rf_leak_rates
    ADD CONSTRAINT chk_lr_base_rate_range CHECK (base_rate >= 0 AND base_rate <= 1);

ALTER TABLE refrigerants_fgas_service.rf_leak_rates
    ADD CONSTRAINT chk_lr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 6: refrigerants_fgas_service.rf_service_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_service_events (
    id                      UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID          NOT NULL,
    equipment_id            UUID          NOT NULL,
    event_type              VARCHAR(30)   NOT NULL,
    event_date              TIMESTAMPTZ   NOT NULL,
    refrigerant_added_kg    DECIMAL(12,4) DEFAULT 0,
    refrigerant_recovered_kg DECIMAL(12,4) DEFAULT 0,
    technician              VARCHAR(200),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_service_events
    ADD CONSTRAINT fk_se_equipment
        FOREIGN KEY (equipment_id)
        REFERENCES refrigerants_fgas_service.rf_equipment_registry(id)
        ON DELETE CASCADE;

ALTER TABLE refrigerants_fgas_service.rf_service_events
    ADD CONSTRAINT chk_se_event_type CHECK (event_type IN (
        'INSTALLATION', 'RECHARGE', 'REPAIR', 'MAINTENANCE', 'LEAK_CHECK',
        'RECOVERY', 'DECOMMISSION', 'RETROFIT', 'INSPECTION'
    ));

ALTER TABLE refrigerants_fgas_service.rf_service_events
    ADD CONSTRAINT chk_se_refrigerant_added_non_negative CHECK (refrigerant_added_kg >= 0);

ALTER TABLE refrigerants_fgas_service.rf_service_events
    ADD CONSTRAINT chk_se_refrigerant_recovered_non_negative CHECK (refrigerant_recovered_kg >= 0);

ALTER TABLE refrigerants_fgas_service.rf_service_events
    ADD CONSTRAINT chk_se_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 7: refrigerants_fgas_service.rf_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_method      VARCHAR(30)     NOT NULL,
    refrigerant_code        VARCHAR(50),
    equipment_type          VARCHAR(50),
    gwp_source              VARCHAR(20)     DEFAULT 'AR6',
    total_loss_kg           DECIMAL(16,8),
    total_emissions_tco2e   DECIMAL(16,8)   NOT NULL,
    uncertainty_lower       DECIMAL(16,8),
    uncertainty_upper       DECIMAL(16,8),
    data_quality_score      DECIMAL(4,2),
    reporting_period        VARCHAR(20),
    organization_id         VARCHAR(200),
    provenance_hash         VARCHAR(64)     NOT NULL,
    status                  VARCHAR(20)     DEFAULT 'COMPLETED',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'SCREENING', 'MASS_BALANCE', 'LEAK_RATE', 'DIRECT_MEASUREMENT', 'SIMPLIFIED'
    ));

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN (
        'AR4', 'AR5', 'AR6', 'AR6_20YR', 'CUSTOM'
    ));

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_total_loss_kg_non_negative CHECK (total_loss_kg IS NULL OR total_loss_kg >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_total_emissions_non_negative CHECK (total_emissions_tco2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_uncertainty_order CHECK (
        uncertainty_lower IS NULL OR uncertainty_upper IS NULL
        OR uncertainty_upper >= uncertainty_lower
    );

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_data_quality_range CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 5));

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_status CHECK (status IN (
        'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    ));

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON refrigerants_fgas_service.rf_calculations
    FOR EACH ROW EXECUTE FUNCTION refrigerants_fgas_service.set_updated_at();

-- =============================================================================
-- Table 8: refrigerants_fgas_service.rf_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL,
    refrigerant_code        VARCHAR(50)     NOT NULL,
    gas_name                VARCHAR(100),
    loss_kg                 DECIMAL(16,8)   NOT NULL,
    gwp_applied             DECIMAL(12,2)   NOT NULL,
    gwp_source              VARCHAR(20),
    emissions_kg_co2e       DECIMAL(16,8)   NOT NULL,
    emissions_tco2e         DECIMAL(16,8)   NOT NULL,
    is_blend_component      BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT fk_cd_calculation
        FOREIGN KEY (calculation_id)
        REFERENCES refrigerants_fgas_service.rf_calculations(id)
        ON DELETE CASCADE;

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_refrigerant_code_not_empty CHECK (LENGTH(TRIM(refrigerant_code)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_loss_kg_non_negative CHECK (loss_kg >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_gwp_applied_non_negative CHECK (gwp_applied >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_gwp_source CHECK (gwp_source IS NULL OR gwp_source IN (
        'AR4', 'AR5', 'AR6', 'AR6_20YR', 'CUSTOM'
    ));

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_emissions_kg_co2e_non_negative CHECK (emissions_kg_co2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_emissions_tco2e_non_negative CHECK (emissions_tco2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 9: refrigerants_fgas_service.rf_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_compliance_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    organization_id         VARCHAR(200),
    framework               VARCHAR(50)     NOT NULL,
    year                    INTEGER         NOT NULL,
    status                  VARCHAR(20)     NOT NULL,
    quota_co2e              DECIMAL(16,4),
    usage_co2e              DECIMAL(16,4),
    remaining_co2e          DECIMAL(16,4),
    phase_down_target_pct   DECIMAL(6,2),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_rf_cr_tenant_org_framework_year UNIQUE (tenant_id, organization_id, framework, year)
);

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'EU_FGAS', 'KIGALI', 'EPA_608', 'CARB', 'UK_FGAS', 'CUSTOM'
    ));

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_year_reasonable CHECK (year >= 1990 AND year <= 2100);

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_status CHECK (status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_quota_non_negative CHECK (quota_co2e IS NULL OR quota_co2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_usage_non_negative CHECK (usage_co2e IS NULL OR usage_co2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_phase_down_range CHECK (phase_down_target_pct IS NULL OR (phase_down_target_pct >= 0 AND phase_down_target_pct <= 100));

ALTER TABLE refrigerants_fgas_service.rf_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_cr_updated_at
    BEFORE UPDATE ON refrigerants_fgas_service.rf_compliance_records
    FOR EACH ROW EXECUTE FUNCTION refrigerants_fgas_service.set_updated_at();

-- =============================================================================
-- Table 10: refrigerants_fgas_service.rf_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_audit_entries (
    id                  UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID          NOT NULL,
    calculation_id      UUID,
    step_name           VARCHAR(100)  NOT NULL,
    step_order          INTEGER,
    input_data          JSONB         DEFAULT '{}'::jsonb,
    output_data         JSONB         DEFAULT '{}'::jsonb,
    factor_used         VARCHAR(200),
    provenance_hash     VARCHAR(64),
    duration_ms         DECIMAL(10,2),
    created_at          TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

ALTER TABLE refrigerants_fgas_service.rf_audit_entries
    ADD CONSTRAINT chk_ae_step_name_not_empty CHECK (LENGTH(TRIM(step_name)) > 0);

ALTER TABLE refrigerants_fgas_service.rf_audit_entries
    ADD CONSTRAINT chk_ae_step_order_positive CHECK (step_order IS NULL OR step_order > 0);

ALTER TABLE refrigerants_fgas_service.rf_audit_entries
    ADD CONSTRAINT chk_ae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE refrigerants_fgas_service.rf_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: refrigerants_fgas_service.rf_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_calculation_events (
    id              UUID            DEFAULT gen_random_uuid(),
    tenant_id       UUID            NOT NULL,
    event_type      VARCHAR(50),
    method          VARCHAR(30),
    refrigerant_code VARCHAR(50),
    emissions_tco2e DECIMAL(16,8),
    metadata        JSONB           DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'refrigerants_fgas_service.rf_calculation_events',
    'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE refrigerants_fgas_service.rf_calculation_events
    ADD CONSTRAINT chk_ce_emissions_non_negative CHECK (emissions_tco2e IS NULL OR emissions_tco2e >= 0);

ALTER TABLE refrigerants_fgas_service.rf_calculation_events
    ADD CONSTRAINT chk_ce_method CHECK (
        method IS NULL OR method IN ('SCREENING', 'MASS_BALANCE', 'LEAK_RATE', 'DIRECT_MEASUREMENT', 'SIMPLIFIED')
    );

-- =============================================================================
-- Table 12: refrigerants_fgas_service.rf_service_events_ts (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_service_events_ts (
    id              UUID            DEFAULT gen_random_uuid(),
    tenant_id       UUID            NOT NULL,
    equipment_id    UUID,
    event_type      VARCHAR(30),
    charge_delta_kg DECIMAL(12,4),
    metadata        JSONB           DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'refrigerants_fgas_service.rf_service_events_ts',
    'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE refrigerants_fgas_service.rf_service_events_ts
    ADD CONSTRAINT chk_sets_event_type CHECK (
        event_type IS NULL OR event_type IN (
            'INSTALLATION', 'RECHARGE', 'REPAIR', 'MAINTENANCE', 'LEAK_CHECK',
            'RECOVERY', 'DECOMMISSION', 'RETROFIT', 'INSPECTION'
        )
    );

-- =============================================================================
-- Table 13: refrigerants_fgas_service.rf_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS refrigerants_fgas_service.rf_compliance_events (
    id              UUID            DEFAULT gen_random_uuid(),
    tenant_id       UUID            NOT NULL,
    framework       VARCHAR(50),
    status          VARCHAR(20),
    emissions_co2e  DECIMAL(16,4),
    metadata        JSONB           DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

SELECT create_hypertable(
    'refrigerants_fgas_service.rf_compliance_events',
    'created_at',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE refrigerants_fgas_service.rf_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN ('EU_FGAS', 'KIGALI', 'EPA_608', 'CARB', 'UK_FGAS', 'CUSTOM')
    );

ALTER TABLE refrigerants_fgas_service.rf_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE refrigerants_fgas_service.rf_compliance_events
    ADD CONSTRAINT chk_coe_emissions_non_negative CHECK (emissions_co2e IS NULL OR emissions_co2e >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- rf_hourly_calculation_stats: hourly count/sum(emissions)/avg(emissions) by method
CREATE MATERIALIZED VIEW refrigerants_fgas_service.rf_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)  AS bucket,
    method,
    COUNT(*)                            AS total_calculations,
    SUM(emissions_tco2e)                AS sum_emissions_tco2e,
    AVG(emissions_tco2e)                AS avg_emissions_tco2e
FROM refrigerants_fgas_service.rf_calculation_events
WHERE created_at IS NOT NULL
GROUP BY bucket, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'refrigerants_fgas_service.rf_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- rf_daily_emission_totals: daily count/sum(emissions) by refrigerant_code
CREATE MATERIALIZED VIEW refrigerants_fgas_service.rf_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', created_at)   AS bucket,
    refrigerant_code,
    COUNT(*)                            AS total_calculations,
    SUM(emissions_tco2e)                AS sum_emissions_tco2e
FROM refrigerants_fgas_service.rf_calculation_events
WHERE created_at IS NOT NULL
GROUP BY bucket, refrigerant_code
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'refrigerants_fgas_service.rf_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- rf_refrigerant_types indexes (10)
CREATE INDEX IF NOT EXISTS idx_rf_rt_tenant_id              ON refrigerants_fgas_service.rf_refrigerant_types(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rt_refrigerant_code       ON refrigerants_fgas_service.rf_refrigerant_types(refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_rt_category               ON refrigerants_fgas_service.rf_refrigerant_types(category);
CREATE INDEX IF NOT EXISTS idx_rf_rt_name                   ON refrigerants_fgas_service.rf_refrigerant_types(name);
CREATE INDEX IF NOT EXISTS idx_rf_rt_is_blend               ON refrigerants_fgas_service.rf_refrigerant_types(is_blend);
CREATE INDEX IF NOT EXISTS idx_rf_rt_is_regulated           ON refrigerants_fgas_service.rf_refrigerant_types(is_regulated);
CREATE INDEX IF NOT EXISTS idx_rf_rt_phase_out_date         ON refrigerants_fgas_service.rf_refrigerant_types(phase_out_date);
CREATE INDEX IF NOT EXISTS idx_rf_rt_tenant_category        ON refrigerants_fgas_service.rf_refrigerant_types(tenant_id, category);
CREATE INDEX IF NOT EXISTS idx_rf_rt_created_at             ON refrigerants_fgas_service.rf_refrigerant_types(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_rt_updated_at             ON refrigerants_fgas_service.rf_refrigerant_types(updated_at DESC);

-- rf_refrigerant_blends indexes (6)
CREATE INDEX IF NOT EXISTS idx_rf_rb_tenant_id              ON refrigerants_fgas_service.rf_refrigerant_blends(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rb_blend_refrigerant_id   ON refrigerants_fgas_service.rf_refrigerant_blends(blend_refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rb_component_refrigerant_id ON refrigerants_fgas_service.rf_refrigerant_blends(component_refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rb_tenant_blend           ON refrigerants_fgas_service.rf_refrigerant_blends(tenant_id, blend_refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rb_blend_component        ON refrigerants_fgas_service.rf_refrigerant_blends(blend_refrigerant_id, component_refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_rb_created_at             ON refrigerants_fgas_service.rf_refrigerant_blends(created_at DESC);

-- rf_gwp_values indexes (8)
CREATE INDEX IF NOT EXISTS idx_rf_gwp_tenant_id             ON refrigerants_fgas_service.rf_gwp_values(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_refrigerant_id        ON refrigerants_fgas_service.rf_gwp_values(refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_source                ON refrigerants_fgas_service.rf_gwp_values(gwp_source);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_timeframe             ON refrigerants_fgas_service.rf_gwp_values(timeframe);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_effective_date        ON refrigerants_fgas_service.rf_gwp_values(effective_date DESC);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_tenant_ref_source     ON refrigerants_fgas_service.rf_gwp_values(tenant_id, refrigerant_id, gwp_source);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_ref_source_tf         ON refrigerants_fgas_service.rf_gwp_values(refrigerant_id, gwp_source, timeframe);
CREATE INDEX IF NOT EXISTS idx_rf_gwp_created_at            ON refrigerants_fgas_service.rf_gwp_values(created_at DESC);

-- rf_equipment_registry indexes (14)
CREATE INDEX IF NOT EXISTS idx_rf_er_tenant_id              ON refrigerants_fgas_service.rf_equipment_registry(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_er_equipment_type         ON refrigerants_fgas_service.rf_equipment_registry(equipment_type);
CREATE INDEX IF NOT EXISTS idx_rf_er_refrigerant_id         ON refrigerants_fgas_service.rf_equipment_registry(refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_er_status                 ON refrigerants_fgas_service.rf_equipment_registry(status);
CREATE INDEX IF NOT EXISTS idx_rf_er_installation_date      ON refrigerants_fgas_service.rf_equipment_registry(installation_date DESC);
CREATE INDEX IF NOT EXISTS idx_rf_er_equipment_identifier   ON refrigerants_fgas_service.rf_equipment_registry(equipment_identifier);
CREATE INDEX IF NOT EXISTS idx_rf_er_climate_zone           ON refrigerants_fgas_service.rf_equipment_registry(climate_zone);
CREATE INDEX IF NOT EXISTS idx_rf_er_ldar_level             ON refrigerants_fgas_service.rf_equipment_registry(ldar_level);
CREATE INDEX IF NOT EXISTS idx_rf_er_tenant_status          ON refrigerants_fgas_service.rf_equipment_registry(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_rf_er_tenant_type            ON refrigerants_fgas_service.rf_equipment_registry(tenant_id, equipment_type);
CREATE INDEX IF NOT EXISTS idx_rf_er_type_refrigerant       ON refrigerants_fgas_service.rf_equipment_registry(equipment_type, refrigerant_id);
CREATE INDEX IF NOT EXISTS idx_rf_er_location               ON refrigerants_fgas_service.rf_equipment_registry(location);
CREATE INDEX IF NOT EXISTS idx_rf_er_created_at             ON refrigerants_fgas_service.rf_equipment_registry(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_er_updated_at             ON refrigerants_fgas_service.rf_equipment_registry(updated_at DESC);

-- rf_leak_rates indexes (6)
CREATE INDEX IF NOT EXISTS idx_rf_lr_tenant_id              ON refrigerants_fgas_service.rf_leak_rates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_lr_equipment_type         ON refrigerants_fgas_service.rf_leak_rates(equipment_type);
CREATE INDEX IF NOT EXISTS idx_rf_lr_lifecycle_stage        ON refrigerants_fgas_service.rf_leak_rates(lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_rf_lr_source                 ON refrigerants_fgas_service.rf_leak_rates(source);
CREATE INDEX IF NOT EXISTS idx_rf_lr_tenant_equip_stage     ON refrigerants_fgas_service.rf_leak_rates(tenant_id, equipment_type, lifecycle_stage);
CREATE INDEX IF NOT EXISTS idx_rf_lr_created_at             ON refrigerants_fgas_service.rf_leak_rates(created_at DESC);

-- rf_service_events indexes (10)
CREATE INDEX IF NOT EXISTS idx_rf_se_tenant_id              ON refrigerants_fgas_service.rf_service_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_se_equipment_id           ON refrigerants_fgas_service.rf_service_events(equipment_id);
CREATE INDEX IF NOT EXISTS idx_rf_se_event_type             ON refrigerants_fgas_service.rf_service_events(event_type);
CREATE INDEX IF NOT EXISTS idx_rf_se_event_date             ON refrigerants_fgas_service.rf_service_events(event_date DESC);
CREATE INDEX IF NOT EXISTS idx_rf_se_provenance_hash        ON refrigerants_fgas_service.rf_service_events(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_rf_se_technician             ON refrigerants_fgas_service.rf_service_events(technician);
CREATE INDEX IF NOT EXISTS idx_rf_se_tenant_equipment       ON refrigerants_fgas_service.rf_service_events(tenant_id, equipment_id);
CREATE INDEX IF NOT EXISTS idx_rf_se_equipment_date         ON refrigerants_fgas_service.rf_service_events(equipment_id, event_date DESC);
CREATE INDEX IF NOT EXISTS idx_rf_se_tenant_event_type      ON refrigerants_fgas_service.rf_service_events(tenant_id, event_type);
CREATE INDEX IF NOT EXISTS idx_rf_se_created_at             ON refrigerants_fgas_service.rf_service_events(created_at DESC);

-- rf_calculations indexes (14)
CREATE INDEX IF NOT EXISTS idx_rf_calc_tenant_id            ON refrigerants_fgas_service.rf_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_calc_calculation_method   ON refrigerants_fgas_service.rf_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_rf_calc_refrigerant_code     ON refrigerants_fgas_service.rf_calculations(refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_calc_equipment_type       ON refrigerants_fgas_service.rf_calculations(equipment_type);
CREATE INDEX IF NOT EXISTS idx_rf_calc_gwp_source           ON refrigerants_fgas_service.rf_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_rf_calc_total_emissions      ON refrigerants_fgas_service.rf_calculations(total_emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_rf_calc_status               ON refrigerants_fgas_service.rf_calculations(status);
CREATE INDEX IF NOT EXISTS idx_rf_calc_reporting_period     ON refrigerants_fgas_service.rf_calculations(reporting_period);
CREATE INDEX IF NOT EXISTS idx_rf_calc_organization_id      ON refrigerants_fgas_service.rf_calculations(organization_id);
CREATE INDEX IF NOT EXISTS idx_rf_calc_provenance_hash      ON refrigerants_fgas_service.rf_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_rf_calc_tenant_status        ON refrigerants_fgas_service.rf_calculations(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_rf_calc_tenant_method        ON refrigerants_fgas_service.rf_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_rf_calc_tenant_refrigerant   ON refrigerants_fgas_service.rf_calculations(tenant_id, refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_calc_created_at           ON refrigerants_fgas_service.rf_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_calc_updated_at           ON refrigerants_fgas_service.rf_calculations(updated_at DESC);

-- rf_calculation_details indexes (10)
CREATE INDEX IF NOT EXISTS idx_rf_cd_tenant_id              ON refrigerants_fgas_service.rf_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_cd_calculation_id         ON refrigerants_fgas_service.rf_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_rf_cd_refrigerant_code       ON refrigerants_fgas_service.rf_calculation_details(refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_cd_gwp_source             ON refrigerants_fgas_service.rf_calculation_details(gwp_source);
CREATE INDEX IF NOT EXISTS idx_rf_cd_emissions_tco2e        ON refrigerants_fgas_service.rf_calculation_details(emissions_tco2e DESC);
CREATE INDEX IF NOT EXISTS idx_rf_cd_is_blend_component     ON refrigerants_fgas_service.rf_calculation_details(is_blend_component);
CREATE INDEX IF NOT EXISTS idx_rf_cd_tenant_calc            ON refrigerants_fgas_service.rf_calculation_details(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_rf_cd_calc_refrigerant       ON refrigerants_fgas_service.rf_calculation_details(calculation_id, refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_cd_tenant_refrigerant     ON refrigerants_fgas_service.rf_calculation_details(tenant_id, refrigerant_code);
CREATE INDEX IF NOT EXISTS idx_rf_cd_created_at             ON refrigerants_fgas_service.rf_calculation_details(created_at DESC);

-- rf_compliance_records indexes (12)
CREATE INDEX IF NOT EXISTS idx_rf_cr_tenant_id              ON refrigerants_fgas_service.rf_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_cr_organization_id        ON refrigerants_fgas_service.rf_compliance_records(organization_id);
CREATE INDEX IF NOT EXISTS idx_rf_cr_framework              ON refrigerants_fgas_service.rf_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_rf_cr_year                   ON refrigerants_fgas_service.rf_compliance_records(year DESC);
CREATE INDEX IF NOT EXISTS idx_rf_cr_status                 ON refrigerants_fgas_service.rf_compliance_records(status);
CREATE INDEX IF NOT EXISTS idx_rf_cr_tenant_framework       ON refrigerants_fgas_service.rf_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_rf_cr_tenant_year            ON refrigerants_fgas_service.rf_compliance_records(tenant_id, year);
CREATE INDEX IF NOT EXISTS idx_rf_cr_tenant_status          ON refrigerants_fgas_service.rf_compliance_records(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_rf_cr_framework_year         ON refrigerants_fgas_service.rf_compliance_records(framework, year);
CREATE INDEX IF NOT EXISTS idx_rf_cr_org_framework_year     ON refrigerants_fgas_service.rf_compliance_records(organization_id, framework, year);
CREATE INDEX IF NOT EXISTS idx_rf_cr_created_at             ON refrigerants_fgas_service.rf_compliance_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_cr_updated_at             ON refrigerants_fgas_service.rf_compliance_records(updated_at DESC);

-- rf_audit_entries indexes (10)
CREATE INDEX IF NOT EXISTS idx_rf_ae_tenant_id              ON refrigerants_fgas_service.rf_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_rf_ae_calculation_id         ON refrigerants_fgas_service.rf_audit_entries(calculation_id);
CREATE INDEX IF NOT EXISTS idx_rf_ae_step_name              ON refrigerants_fgas_service.rf_audit_entries(step_name);
CREATE INDEX IF NOT EXISTS idx_rf_ae_step_order             ON refrigerants_fgas_service.rf_audit_entries(step_order);
CREATE INDEX IF NOT EXISTS idx_rf_ae_provenance_hash        ON refrigerants_fgas_service.rf_audit_entries(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_rf_ae_created_at             ON refrigerants_fgas_service.rf_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ae_tenant_calc            ON refrigerants_fgas_service.rf_audit_entries(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_rf_ae_calc_step              ON refrigerants_fgas_service.rf_audit_entries(calculation_id, step_order);
CREATE INDEX IF NOT EXISTS idx_rf_ae_input_data             ON refrigerants_fgas_service.rf_audit_entries USING GIN (input_data);
CREATE INDEX IF NOT EXISTS idx_rf_ae_output_data            ON refrigerants_fgas_service.rf_audit_entries USING GIN (output_data);

-- rf_calculation_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_rf_ce_tenant_id              ON refrigerants_fgas_service.rf_calculation_events(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ce_event_type             ON refrigerants_fgas_service.rf_calculation_events(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ce_method                 ON refrigerants_fgas_service.rf_calculation_events(method, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ce_refrigerant_code       ON refrigerants_fgas_service.rf_calculation_events(refrigerant_code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ce_tenant_method          ON refrigerants_fgas_service.rf_calculation_events(tenant_id, method, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_ce_metadata               ON refrigerants_fgas_service.rf_calculation_events USING GIN (metadata);

-- rf_service_events_ts indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_rf_sets_tenant_id            ON refrigerants_fgas_service.rf_service_events_ts(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_sets_equipment_id         ON refrigerants_fgas_service.rf_service_events_ts(equipment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_sets_event_type           ON refrigerants_fgas_service.rf_service_events_ts(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_sets_tenant_equipment     ON refrigerants_fgas_service.rf_service_events_ts(tenant_id, equipment_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_sets_tenant_event_type    ON refrigerants_fgas_service.rf_service_events_ts(tenant_id, event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_sets_metadata             ON refrigerants_fgas_service.rf_service_events_ts USING GIN (metadata);

-- rf_compliance_events indexes (hypertable-aware) (6)
CREATE INDEX IF NOT EXISTS idx_rf_coe_tenant_id             ON refrigerants_fgas_service.rf_compliance_events(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_coe_framework             ON refrigerants_fgas_service.rf_compliance_events(framework, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_coe_status                ON refrigerants_fgas_service.rf_compliance_events(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_coe_tenant_framework      ON refrigerants_fgas_service.rf_compliance_events(tenant_id, framework, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_coe_tenant_status         ON refrigerants_fgas_service.rf_compliance_events(tenant_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rf_coe_metadata              ON refrigerants_fgas_service.rf_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- rf_refrigerant_types: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_refrigerant_types ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_rt_read  ON refrigerants_fgas_service.rf_refrigerant_types FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_rt_write ON refrigerants_fgas_service.rf_refrigerant_types FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_refrigerant_blends: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_refrigerant_blends ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_rb_read  ON refrigerants_fgas_service.rf_refrigerant_blends FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_rb_write ON refrigerants_fgas_service.rf_refrigerant_blends FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_gwp_values: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_gwp_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_gwp_read  ON refrigerants_fgas_service.rf_gwp_values FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_gwp_write ON refrigerants_fgas_service.rf_gwp_values FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_equipment_registry: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_equipment_registry ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_er_read  ON refrigerants_fgas_service.rf_equipment_registry FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_er_write ON refrigerants_fgas_service.rf_equipment_registry FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_leak_rates: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_leak_rates ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_lr_read  ON refrigerants_fgas_service.rf_leak_rates FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_lr_write ON refrigerants_fgas_service.rf_leak_rates FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_service_events: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_service_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_se_read  ON refrigerants_fgas_service.rf_service_events FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_se_write ON refrigerants_fgas_service.rf_service_events FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_calculations: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_calc_read  ON refrigerants_fgas_service.rf_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_calc_write ON refrigerants_fgas_service.rf_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_calculation_details: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_cd_read  ON refrigerants_fgas_service.rf_calculation_details FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_cd_write ON refrigerants_fgas_service.rf_calculation_details FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_compliance_records: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_cr_read  ON refrigerants_fgas_service.rf_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_cr_write ON refrigerants_fgas_service.rf_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_audit_entries: tenant-isolated
ALTER TABLE refrigerants_fgas_service.rf_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_ae_read  ON refrigerants_fgas_service.rf_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY rf_ae_write ON refrigerants_fgas_service.rf_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- rf_calculation_events: open read/write (time-series telemetry)
ALTER TABLE refrigerants_fgas_service.rf_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_ce_read  ON refrigerants_fgas_service.rf_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY rf_ce_write ON refrigerants_fgas_service.rf_calculation_events FOR ALL   USING (TRUE);

-- rf_service_events_ts: open read/write (time-series telemetry)
ALTER TABLE refrigerants_fgas_service.rf_service_events_ts ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_sets_read  ON refrigerants_fgas_service.rf_service_events_ts FOR SELECT USING (TRUE);
CREATE POLICY rf_sets_write ON refrigerants_fgas_service.rf_service_events_ts FOR ALL   USING (TRUE);

-- rf_compliance_events: open read/write (time-series telemetry)
ALTER TABLE refrigerants_fgas_service.rf_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY rf_coe_read  ON refrigerants_fgas_service.rf_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY rf_coe_write ON refrigerants_fgas_service.rf_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA refrigerants_fgas_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA refrigerants_fgas_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA refrigerants_fgas_service TO greenlang_app;
GRANT SELECT ON refrigerants_fgas_service.rf_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON refrigerants_fgas_service.rf_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA refrigerants_fgas_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA refrigerants_fgas_service TO greenlang_readonly;
GRANT SELECT ON refrigerants_fgas_service.rf_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON refrigerants_fgas_service.rf_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA refrigerants_fgas_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA refrigerants_fgas_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA refrigerants_fgas_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'refrigerants-fgas:read',                    'refrigerants-fgas', 'read',                    'View all refrigerants and F-gas service data including types, equipment, calculations, and compliance'),
    (gen_random_uuid(), 'refrigerants-fgas:write',                   'refrigerants-fgas', 'write',                   'Create, update, and manage all refrigerants and F-gas service data'),
    (gen_random_uuid(), 'refrigerants-fgas:execute',                 'refrigerants-fgas', 'execute',                 'Execute refrigerant emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'refrigerants-fgas:admin',                   'refrigerants-fgas', 'admin',                   'Full administrative access to refrigerants and F-gas service including configuration and policy management'),
    (gen_random_uuid(), 'refrigerants-fgas:refrigerants:read',       'refrigerants-fgas', 'refrigerants_read',       'View refrigerant type registry with GWP values, ODP, blend compositions, and phase-out dates'),
    (gen_random_uuid(), 'refrigerants-fgas:refrigerants:write',      'refrigerants-fgas', 'refrigerants_write',      'Create, update, and manage refrigerant type entries including custom refrigerants and blend definitions'),
    (gen_random_uuid(), 'refrigerants-fgas:equipment:read',          'refrigerants-fgas', 'equipment_read',          'View equipment registry with refrigerant charges, status, LDAR levels, and climate zones'),
    (gen_random_uuid(), 'refrigerants-fgas:equipment:write',         'refrigerants-fgas', 'equipment_write',         'Create, update, and manage equipment registry entries with charge and lifecycle data'),
    (gen_random_uuid(), 'refrigerants-fgas:service-events:read',     'refrigerants-fgas', 'service_events_read',     'View service events including installations, recharges, repairs, and decommissions with technician records'),
    (gen_random_uuid(), 'refrigerants-fgas:service-events:write',    'refrigerants-fgas', 'service_events_write',    'Create and manage service event records with refrigerant added/recovered quantities'),
    (gen_random_uuid(), 'refrigerants-fgas:leak-rates:read',         'refrigerants-fgas', 'leak_rates_read',         'View leak rate database with IPCC 2006 defaults and custom rates per equipment type and lifecycle stage'),
    (gen_random_uuid(), 'refrigerants-fgas:leak-rates:write',        'refrigerants-fgas', 'leak_rates_write',        'Create, update, and manage leak rate entries including custom rates with source references'),
    (gen_random_uuid(), 'refrigerants-fgas:compliance:read',         'refrigerants-fgas', 'compliance_read',         'View regulatory compliance records for EU F-Gas, Kigali, EPA 608, and CARB with quota tracking'),
    (gen_random_uuid(), 'refrigerants-fgas:compliance:write',        'refrigerants-fgas', 'compliance_write',        'Create, update, and manage compliance records with phase-down targets and quota allocations'),
    (gen_random_uuid(), 'refrigerants-fgas:calculations:read',       'refrigerants-fgas', 'calculations_read',       'View calculation results with CO2e totals, per-gas breakdowns, uncertainty bounds, and provenance hashes'),
    (gen_random_uuid(), 'refrigerants-fgas:audit:read',              'refrigerants-fgas', 'audit_read',              'View step-by-step audit trail entries for calculations with factor references and provenance hashes'),
    (gen_random_uuid(), 'refrigerants-fgas:uncertainty:execute',     'refrigerants-fgas', 'uncertainty_execute',     'Execute uncertainty quantification analysis on refrigerant emission calculation results'),
    (gen_random_uuid(), 'refrigerants-fgas:stats:read',              'refrigerants-fgas', 'stats_read',              'View refrigerants and F-gas service statistics, continuous aggregates, and time-series telemetry')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('refrigerants_fgas_service.rf_calculation_events', INTERVAL '90 days');
SELECT add_retention_policy('refrigerants_fgas_service.rf_service_events_ts',  INTERVAL '90 days');
SELECT add_retention_policy('refrigerants_fgas_service.rf_compliance_events',  INTERVAL '90 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE refrigerants_fgas_service.rf_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'method',
         timescaledb.compress_orderby   = 'created_at DESC');
SELECT add_compression_policy('refrigerants_fgas_service.rf_calculation_events', INTERVAL '7 days');

ALTER TABLE refrigerants_fgas_service.rf_service_events_ts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'event_type',
         timescaledb.compress_orderby   = 'created_at DESC');
SELECT add_compression_policy('refrigerants_fgas_service.rf_service_events_ts', INTERVAL '7 days');

ALTER TABLE refrigerants_fgas_service.rf_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'created_at DESC');
SELECT add_compression_policy('refrigerants_fgas_service.rf_compliance_events', INTERVAL '7 days');

-- =============================================================================
-- Seed: Register the Refrigerants & F-Gas Agent (GL-MRV-SCOPE1-002)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-002',
    'Refrigerants & F-Gas Agent',
    'Refrigerants and fluorinated gas emission calculator for GreenLang Climate OS. Manages refrigerant type registry with HFC/HFC_BLEND/HFO/PFC/SF6/NF3/HCFC/CFC/NATURAL categories including molecular weight, boiling point, ozone depletion potential (ODP), atmospheric lifetime, phase-out dates, and blend composition tracking with weight fractions. Maintains GWP value database from AR4/AR5/AR6/AR6_20YR/CUSTOM sources with 100yr and 20yr timeframes and effective date ranges. Registers equipment with refrigerant type, charge (kg), equipment count, installation dates, expected lifetime, climate zone, LDAR (Leak Detection and Repair) level, and custom leak rates. Stores leak rate database with IPCC 2006 default rates and custom rates per equipment type and lifecycle stage (installation/operation/servicing/disposal/standby). Tracks service events including installations, recharges, repairs, maintenance, leak checks, recovery, decommissions, retrofits, and inspections with refrigerant added/recovered quantities and technician records. Executes deterministic refrigerant emission calculations using screening, mass-balance, and leak-rate methods converting refrigerant loss (kg) to CO2e via GWP weighting with uncertainty bounds and data quality scoring. Produces per-gas calculation detail breakdowns with individual refrigerant and blend component losses, applied GWP values, and blend component flags. Tracks regulatory compliance against EU F-Gas Regulation, Kigali Amendment, EPA Section 608, CARB, and UK F-Gas frameworks with per-year quota, usage, remaining CO2e, and phase-down target percentages. Generates step-by-step audit trail entries with input/output data, factor references, and duration metrics. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/refrigerants-fgas',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-002', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/refrigerants-fgas-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"refrigerants", "f-gas", "scope-1", "ghg-protocol", "hfc", "pfc", "sf6", "kigali", "eu-fgas", "mrv"}',
    '{"cross-sector", "hvac", "refrigeration", "manufacturing", "commercial-buildings", "food-processing", "cold-chain"}',
    'c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'refrigerant_database',
    'configuration',
    'Register and manage refrigerant type entries with category classification (HFC/HFC_BLEND/HFO/PFC/SF6/NF3/HCFC/CFC/NATURAL), molecular weight, boiling point, ODP, atmospheric lifetime, phase-out dates, blend composition, and GWP values from multiple assessment reports.',
    '{"refrigerant_code", "category", "name", "formula", "molecular_weight", "boiling_point_c", "odp", "atmospheric_lifetime_years", "is_blend", "phase_out_date", "gwp_values"}',
    '{"refrigerant_id", "registration_result"}',
    '{"categories": ["HFC", "HFC_BLEND", "HFO", "PFC", "SF6", "NF3", "HCFC", "CFC", "NATURAL"], "gwp_sources": ["AR4", "AR5", "AR6", "AR6_20YR", "CUSTOM"], "supports_blends": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic refrigerant emission calculations using screening method (equipment count x charge x leak rate x GWP), mass-balance method (purchased - recovered - remaining stock to CO2e), or leak-rate method (service event data to annual loss to CO2e). Supports blend decomposition with per-component GWP weighting.',
    '{"calculation_method", "refrigerant_code", "equipment_type", "gwp_source", "charge_kg", "leak_rate", "purchased_kg", "recovered_kg", "remaining_stock_kg", "service_events"}',
    '{"calculation_id", "total_loss_kg", "total_emissions_tco2e", "per_gas_breakdown", "uncertainty_bounds", "data_quality_score", "provenance_hash"}',
    '{"methods": ["SCREENING", "MASS_BALANCE", "LEAK_RATE", "DIRECT_MEASUREMENT", "SIMPLIFIED"], "gwp_sources": ["AR4", "AR5", "AR6", "AR6_20YR", "CUSTOM"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'equipment_management',
    'configuration',
    'Register and manage equipment entries with type classification, refrigerant assignment, charge (kg), equipment count, status tracking, installation dates, expected lifetime, location, climate zone, LDAR level, and custom leak rates.',
    '{"equipment_type", "refrigerant_code", "charge_kg", "equipment_count", "status", "installation_date", "expected_lifetime_years", "location", "climate_zone", "ldar_level", "custom_leak_rate"}',
    '{"equipment_id", "registration_result"}',
    '{"statuses": ["ACTIVE", "INACTIVE", "DECOMMISSIONED", "MAINTENANCE", "RETIRED"], "ldar_levels": ["NONE", "BASIC", "ENHANCED", "CONTINUOUS"], "supports_custom_leak_rates": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'service_event_tracking',
    'processing',
    'Record and track service events including installations, recharges, repairs, maintenance, leak checks, recovery, decommissions, retrofits, and inspections with refrigerant added/recovered quantities, technician records, and provenance hashing.',
    '{"equipment_id", "event_type", "event_date", "refrigerant_added_kg", "refrigerant_recovered_kg", "technician", "notes"}',
    '{"event_id", "charge_delta_kg", "provenance_hash"}',
    '{"event_types": ["INSTALLATION", "RECHARGE", "REPAIR", "MAINTENANCE", "LEAK_CHECK", "RECOVERY", "DECOMMISSION", "RETROFIT", "INSPECTION"], "tracks_provenance": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'compliance_tracking',
    'reporting',
    'Track regulatory compliance against EU F-Gas Regulation, Kigali Amendment, EPA Section 608, CARB, and UK F-Gas frameworks with per-year quota, usage, remaining CO2e, phase-down target percentages, and compliance status assessment.',
    '{"organization_id", "framework", "year", "quota_co2e", "usage_co2e", "phase_down_target_pct"}',
    '{"compliance_id", "status", "remaining_co2e", "compliance_assessment"}',
    '{"frameworks": ["EU_FGAS", "KIGALI", "EPA_608", "CARB", "UK_FGAS", "CUSTOM"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'uncertainty_quantification',
    'processing',
    'Quantify uncertainty in refrigerant emission calculations based on calculation method, GWP source confidence, leak rate estimation confidence, measurement precision, and equipment data quality.',
    '{"calculation_id", "calculation_method", "gwp_source", "leak_rate_source"}',
    '{"uncertainty_pct", "confidence_interval", "data_quality_score", "methodology_reference"}',
    '{"method_uncertainty_ranges": {"SCREENING": "+-50%", "MASS_BALANCE": "+-25%", "LEAK_RATE": "+-30%", "DIRECT_MEASUREMENT": "+-10%"}, "ipcc_2006_guidelines": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-002', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive step-by-step audit trail for each calculation with input/output data per step, GWP factors used, leak rates applied, blend decomposition steps, and SHA-256 provenance hashes.',
    '{"calculation_id"}',
    '{"audit_entries", "provenance_chain", "factor_references", "total_duration_ms"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_step": true, "includes_factor_refs": true, "includes_timing": true}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-002', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage refrigerant emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-002', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for refrigerant charges, GWP values, and emission unit conversions'),
    ('GL-MRV-SCOPE1-002', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for GWP sources, IPCC references, and regulatory framework citations'),
    ('GL-MRV-SCOPE1-002', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-SCOPE1-002', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, service events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-002', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of refrigerant charge records'),
    ('GL-MRV-SCOPE1-002', 'GL-MRV-X-001',   '>=1.0.0', true,  'Stationary Combustion Calculator for cross-referencing equipment facilities and facility-level aggregations')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-002',
    'Refrigerants & F-Gas Agent',
    'Refrigerants and fluorinated gas emission calculator. Refrigerant type registry (HFC/HFC_BLEND/HFO/PFC/SF6/NF3/HCFC/CFC/NATURAL, molecular weight, boiling point, ODP, atmospheric lifetime, phase-out dates). Blend composition tracking (weight fractions). GWP value database (AR4/AR5/AR6/AR6_20YR/CUSTOM, 100yr/20yr timeframes). Equipment registry (type, refrigerant, charge kg, count, status, LDAR level, climate zone). Leak rate database (IPCC 2006 defaults, custom rates per lifecycle stage). Service event tracking (install/recharge/repair/decommission, added/recovered kg, technician records). Emission calculations (screening/mass-balance/leak-rate methods, blend decomposition, uncertainty bounds). Regulatory compliance (EU F-Gas/Kigali/EPA 608/CARB quota and phase-down tracking). Step-by-step audit trail. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA refrigerants_fgas_service IS
    'Refrigerants & F-Gas Agent (AGENT-MRV-002) - refrigerant type registry, GWP value database, blend compositions, equipment registry, leak rate database, service event tracking, emission calculations, per-gas breakdowns, regulatory compliance, audit trail, provenance chains';

COMMENT ON TABLE refrigerants_fgas_service.rf_refrigerant_types IS
    'Refrigerant type registry: tenant_id, refrigerant_code (unique per tenant), category (HFC/HFC_BLEND/HFO/PFC/SF6/NF3/HCFC/CFC/NATURAL), name, formula, molecular_weight, boiling_point_c, odp, atmospheric_lifetime_years, is_blend, is_regulated, phase_out_date, is_custom';

COMMENT ON TABLE refrigerants_fgas_service.rf_refrigerant_blends IS
    'Blend composition: tenant_id, blend_refrigerant_id FK, component_refrigerant_id FK, weight_fraction (0-1), self-reference prevented';

COMMENT ON TABLE refrigerants_fgas_service.rf_gwp_values IS
    'GWP value database: tenant_id, refrigerant_id FK, gwp_source (AR4/AR5/AR6/AR6_20YR/CUSTOM), timeframe (100yr/20yr), value, effective_date, source_reference, unique per (tenant, refrigerant, source, timeframe)';

COMMENT ON TABLE refrigerants_fgas_service.rf_equipment_registry IS
    'Equipment registry: tenant_id, equipment_type, refrigerant_id FK, charge_kg, equipment_count, status (ACTIVE/INACTIVE/DECOMMISSIONED/MAINTENANCE/RETIRED), installation_date, expected_lifetime_years, location, equipment_identifier, custom_leak_rate, climate_zone, ldar_level';

COMMENT ON TABLE refrigerants_fgas_service.rf_leak_rates IS
    'Leak rate database: tenant_id, equipment_type, lifecycle_stage (INSTALLATION/OPERATION/SERVICING/DISPOSAL/STANDBY), base_rate (0-1), source (default IPCC_2006), is_custom, notes, unique per (tenant, equipment_type, lifecycle_stage, source)';

COMMENT ON TABLE refrigerants_fgas_service.rf_service_events IS
    'Service events: tenant_id, equipment_id FK, event_type (INSTALLATION/RECHARGE/REPAIR/MAINTENANCE/LEAK_CHECK/RECOVERY/DECOMMISSION/RETROFIT/INSPECTION), event_date, refrigerant_added_kg, refrigerant_recovered_kg, technician, notes, provenance_hash';

COMMENT ON TABLE refrigerants_fgas_service.rf_calculations IS
    'Calculation results: tenant_id, calculation_method (SCREENING/MASS_BALANCE/LEAK_RATE/DIRECT_MEASUREMENT/SIMPLIFIED), refrigerant_code, equipment_type, gwp_source, total_loss_kg, total_emissions_tco2e, uncertainty_lower/upper, data_quality_score, reporting_period, organization_id, provenance_hash, status';

COMMENT ON TABLE refrigerants_fgas_service.rf_calculation_details IS
    'Per-gas calculation breakdown: tenant_id, calculation_id FK (CASCADE), refrigerant_code, gas_name, loss_kg, gwp_applied, gwp_source, emissions_kg_co2e, emissions_tco2e, is_blend_component';

COMMENT ON TABLE refrigerants_fgas_service.rf_compliance_records IS
    'Regulatory compliance records: tenant_id, organization_id, framework (EU_FGAS/KIGALI/EPA_608/CARB/UK_FGAS/CUSTOM), year, status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), quota_co2e, usage_co2e, remaining_co2e, phase_down_target_pct, notes, unique per (tenant, org, framework, year)';

COMMENT ON TABLE refrigerants_fgas_service.rf_audit_entries IS
    'Audit trail entries: tenant_id, calculation_id, step_name, step_order, input_data JSONB, output_data JSONB, factor_used, provenance_hash, duration_ms';

COMMENT ON TABLE refrigerants_fgas_service.rf_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, event_type, method, refrigerant_code, emissions_tco2e, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE refrigerants_fgas_service.rf_service_events_ts IS
    'TimescaleDB hypertable: service event time-series with tenant_id, equipment_id, event_type, charge_delta_kg, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON TABLE refrigerants_fgas_service.rf_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, framework, status, emissions_co2e, metadata JSONB (7-day chunks, 90-day retention)';

COMMENT ON MATERIALIZED VIEW refrigerants_fgas_service.rf_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by method (total calculations, sum emissions tCO2e, avg emissions tCO2e per hour)';

COMMENT ON MATERIALIZED VIEW refrigerants_fgas_service.rf_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by refrigerant_code (total calculations, sum emissions tCO2e per day)';
