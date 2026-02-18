-- =============================================================================
-- V056: Flaring Service Schema
-- =============================================================================
-- Component: AGENT-MRV-006 (GL-MRV-SCOPE1-006)
-- Date:      February 2026
--
-- This migration creates the complete database schema for the
-- Flaring Agent (GL-MRV-SCOPE1-006) with capabilities for flare system
-- registry management (elevated_steam_assisted, elevated_air_assisted,
-- elevated_unassisted, enclosed_ground, multi_point_ground, offshore_marine,
-- candlestick, low_pressure types with facility assignment, tip diameter,
-- height, capacity, assist type, pilot/purge gas configuration, and
-- installation/inspection tracking), gas composition analysis records
-- (15-component mole fraction analysis for CH4, C2H6, C3H8, n-C4H10,
-- i-C4H10, C5H12, C6+, CO2, N2, H2S, H2, CO, C2H4, C3H6, H2O with
-- trigger-computed total fraction, HHV/LHV heating values, specific gravity,
-- molecular weight, lab reference, and analysis date), emission factor
-- database (gas x flare_type factors with EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM
-- sources, factor units kg/MMBtu or kg/GJ or kg/scf or kg/Nm3, year/region
-- scoping, and references), flaring event tracking (routine, non_routine,
-- emergency, maintenance, pilot_purge, well_completion categories with
-- flare system FK, start/end times, duration, gas volumes scf/Nm3,
-- composition FK, flow rate measurement, estimation methods), emission
-- calculation records (gas_composition, default_ef, engineering_estimate,
-- direct_measurement methods with combustion efficiency, multi-gas GWP
-- weighting CO2/CH4/N2O/BLACK_CARBON using AR4/AR5/AR6 values, pilot/purge
-- CO2e, uncertainty quantification, data quality scoring, and provenance
-- hashing), per-gas calculation detail breakdowns (individual emission
-- factors, GWP values, raw and CO2e emissions with source attribution),
-- combustion efficiency test records (base/adjusted CE, wind speed, tip
-- velocity Mach, LHV, steam/air ratios, DRE, test method/reference),
-- pilot and purge gas tracking (period-based pilot/purge volume recording,
-- composition FK, operating hours, active pilot count, CO2e emissions),
-- regulatory compliance records (GHG_PROTOCOL, EPA_SUBPART_W, CSRD_ESRS,
-- ISO_14064, EU_ETS_MRR, EU_METHANE_REG, WORLD_BANK_ZRF, OGMP_2_0
-- framework checks with requirements met/checked counts, findings, and
-- recommendations), and step-by-step audit trail entries (entity-level
-- action trace with input/output/prev hash chaining, formula text,
-- parameters/result JSONB). SHA-256 provenance chains for zero-hallucination
-- audit trail.
-- =============================================================================
-- Tables (10):
--   1. fl_flare_systems            - Flare system registry (8 types, capacity, assist, pilot/purge config)
--   2. fl_gas_compositions         - Gas composition analysis (15 components, trigger-computed total, HHV/LHV)
--   3. fl_emission_factors         - Emission factors by gas, flare type, source (EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM)
--   4. fl_flaring_events           - Flaring event records (6 categories, volumes, flow rates, estimation methods)
--   5. fl_calculations             - Emission calculation results (multi-gas CO2e with CE, pilot/purge, provenance)
--   6. fl_calculation_details      - Per-gas breakdown (EF, GWP, raw and CO2e emissions, source)
--   7. fl_combustion_efficiency    - CE test records (base/adjusted CE, wind, tip velocity, DRE)
--   8. fl_pilot_purge_records      - Pilot and purge gas tracking (period volumes, operating hours, CO2e)
--   9. fl_compliance_records       - Regulatory compliance (8 frameworks with requirements tracking)
--  10. fl_audit_entries            - Audit trail (entity-level with input/output/prev hash chaining)
--
-- Hypertables (3):
--  11. fl_calculation_events       - Calculation event time-series (hypertable on time)
--  12. fl_flaring_event_ts         - Flaring event time-series (hypertable on time)
--  13. fl_compliance_events        - Compliance event time-series (hypertable on time)
--
-- Continuous Aggregates (2):
--   1. fl_hourly_calculation_stats - Hourly count/sum(co2e)/avg(co2e) by flare_type and method
--   2. fl_daily_emission_totals    - Daily count/sum(co2e) by flare_type and event_category
--
-- Also includes: indexes (B-tree, GIN, partial, composite),
-- CHECK constraints, RLS policies per tenant, retention
-- policies (365 days on hypertables), compression policies (30 days),
-- updated_at trigger, total_fraction computation trigger,
-- security permissions for greenlang_app/greenlang_readonly/greenlang_admin,
-- and agent registry seed data registering GL-MRV-SCOPE1-006.
-- Previous: V055__fugitive_emissions_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS flaring_service;

-- TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION flaring_service.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Function: Compute total_fraction from gas composition components
-- =============================================================================
-- PostgreSQL GENERATED ALWAYS AS (STORED) supports COALESCE with column
-- references, but to ensure maximum compatibility and to allow future
-- extensibility (e.g., additional components), we use a trigger instead.
-- =============================================================================

CREATE OR REPLACE FUNCTION flaring_service.compute_total_fraction()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.total_fraction :=
        COALESCE(NEW.ch4_fraction, 0) +
        COALESCE(NEW.c2h6_fraction, 0) +
        COALESCE(NEW.c3h8_fraction, 0) +
        COALESCE(NEW.n_c4h10_fraction, 0) +
        COALESCE(NEW.i_c4h10_fraction, 0) +
        COALESCE(NEW.c5h12_fraction, 0) +
        COALESCE(NEW.c6_plus_fraction, 0) +
        COALESCE(NEW.co2_fraction, 0) +
        COALESCE(NEW.n2_fraction, 0) +
        COALESCE(NEW.h2s_fraction, 0) +
        COALESCE(NEW.h2_fraction, 0) +
        COALESCE(NEW.co_fraction, 0) +
        COALESCE(NEW.c2h4_fraction, 0) +
        COALESCE(NEW.c3h6_fraction, 0) +
        COALESCE(NEW.h2o_fraction, 0);
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: flaring_service.fl_flare_systems
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_flare_systems (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    flare_type              VARCHAR(50)     NOT NULL,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    facility_id             UUID,
    location_description    TEXT,
    tip_diameter_inches     DECIMAL(10,4),
    height_feet             DECIMAL(10,2),
    capacity_mmbtu_hr       DECIMAL(14,4),
    assist_type             VARCHAR(20)     DEFAULT 'NONE',
    num_pilots              INTEGER         DEFAULT 1,
    pilot_gas_flow_mmbtu_hr DECIMAL(10,6),
    purge_gas_flow_scfh     DECIMAL(12,4),
    purge_gas_type          VARCHAR(30)     DEFAULT 'NATURAL_GAS',
    installation_date       DATE,
    last_inspection_date    DATE,
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_flare_type CHECK (flare_type IN (
        'ELEVATED_STEAM_ASSISTED', 'ELEVATED_AIR_ASSISTED', 'ELEVATED_UNASSISTED',
        'ENCLOSED_GROUND', 'MULTI_POINT_GROUND', 'OFFSHORE_MARINE',
        'CANDLESTICK', 'LOW_PRESSURE'
    ));

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_status CHECK (status IN (
        'ACTIVE', 'INACTIVE', 'DECOMMISSIONED', 'MAINTENANCE', 'STANDBY'
    ));

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_assist_type CHECK (assist_type IN ('STEAM', 'AIR', 'NONE'));

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_tip_diameter_positive CHECK (tip_diameter_inches IS NULL OR tip_diameter_inches > 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_height_positive CHECK (height_feet IS NULL OR height_feet > 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_capacity_positive CHECK (capacity_mmbtu_hr IS NULL OR capacity_mmbtu_hr > 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_num_pilots_positive CHECK (num_pilots IS NULL OR num_pilots >= 1);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_pilot_gas_flow_non_negative CHECK (pilot_gas_flow_mmbtu_hr IS NULL OR pilot_gas_flow_mmbtu_hr >= 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_purge_gas_flow_non_negative CHECK (purge_gas_flow_scfh IS NULL OR purge_gas_flow_scfh >= 0);

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_purge_gas_type CHECK (purge_gas_type IS NULL OR purge_gas_type IN (
        'NATURAL_GAS', 'NITROGEN', 'CO2', 'INERT'
    ));

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_inspection_after_installation CHECK (
        last_inspection_date IS NULL OR installation_date IS NULL OR last_inspection_date >= installation_date
    );

ALTER TABLE flaring_service.fl_flare_systems
    ADD CONSTRAINT chk_fs_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_fs_updated_at
    BEFORE UPDATE ON flaring_service.fl_flare_systems
    FOR EACH ROW EXECUTE FUNCTION flaring_service.set_updated_at();

-- =============================================================================
-- Table 2: flaring_service.fl_gas_compositions
-- =============================================================================
-- 15-component gas composition analysis with trigger-computed total_fraction.
-- Components: CH4, C2H6, C3H8, n-C4H10, i-C4H10, C5H12, C6+, CO2, N2,
--             H2S, H2, CO, C2H4, C3H6, H2O
-- All fractions are mole fractions (0.0 to 1.0).
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_gas_compositions (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    name                VARCHAR(255)    NOT NULL,
    description         TEXT,
    ch4_fraction        DECIMAL(10,8),
    c2h6_fraction       DECIMAL(10,8),
    c3h8_fraction       DECIMAL(10,8),
    n_c4h10_fraction    DECIMAL(10,8),
    i_c4h10_fraction    DECIMAL(10,8),
    c5h12_fraction      DECIMAL(10,8),
    c6_plus_fraction    DECIMAL(10,8),
    co2_fraction        DECIMAL(10,8),
    n2_fraction         DECIMAL(10,8),
    h2s_fraction        DECIMAL(10,8),
    h2_fraction         DECIMAL(10,8),
    co_fraction         DECIMAL(10,8),
    c2h4_fraction       DECIMAL(10,8),
    c3h6_fraction       DECIMAL(10,8),
    h2o_fraction        DECIMAL(10,8),
    total_fraction      DECIMAL(10,8),
    hhv_btu_scf         DECIMAL(12,4),
    lhv_btu_scf         DECIMAL(12,4),
    specific_gravity    DECIMAL(8,6),
    molecular_weight    DECIMAL(10,4),
    analysis_date       DATE,
    lab_reference       VARCHAR(100),
    source              VARCHAR(50)     DEFAULT 'LAB_ANALYSIS',
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_name_not_empty CHECK (LENGTH(TRIM(name)) > 0);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_ch4_fraction_range CHECK (ch4_fraction IS NULL OR (ch4_fraction >= 0 AND ch4_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c2h6_fraction_range CHECK (c2h6_fraction IS NULL OR (c2h6_fraction >= 0 AND c2h6_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c3h8_fraction_range CHECK (c3h8_fraction IS NULL OR (c3h8_fraction >= 0 AND c3h8_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_n_c4h10_fraction_range CHECK (n_c4h10_fraction IS NULL OR (n_c4h10_fraction >= 0 AND n_c4h10_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_i_c4h10_fraction_range CHECK (i_c4h10_fraction IS NULL OR (i_c4h10_fraction >= 0 AND i_c4h10_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c5h12_fraction_range CHECK (c5h12_fraction IS NULL OR (c5h12_fraction >= 0 AND c5h12_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c6_plus_fraction_range CHECK (c6_plus_fraction IS NULL OR (c6_plus_fraction >= 0 AND c6_plus_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_co2_fraction_range CHECK (co2_fraction IS NULL OR (co2_fraction >= 0 AND co2_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_n2_fraction_range CHECK (n2_fraction IS NULL OR (n2_fraction >= 0 AND n2_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_h2s_fraction_range CHECK (h2s_fraction IS NULL OR (h2s_fraction >= 0 AND h2s_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_h2_fraction_range CHECK (h2_fraction IS NULL OR (h2_fraction >= 0 AND h2_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_co_fraction_range CHECK (co_fraction IS NULL OR (co_fraction >= 0 AND co_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c2h4_fraction_range CHECK (c2h4_fraction IS NULL OR (c2h4_fraction >= 0 AND c2h4_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_c3h6_fraction_range CHECK (c3h6_fraction IS NULL OR (c3h6_fraction >= 0 AND c3h6_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_h2o_fraction_range CHECK (h2o_fraction IS NULL OR (h2o_fraction >= 0 AND h2o_fraction <= 1));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_total_fraction_range CHECK (total_fraction IS NULL OR (total_fraction >= 0 AND total_fraction <= 1.01));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_hhv_positive CHECK (hhv_btu_scf IS NULL OR hhv_btu_scf > 0);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_lhv_positive CHECK (lhv_btu_scf IS NULL OR lhv_btu_scf > 0);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_specific_gravity_positive CHECK (specific_gravity IS NULL OR specific_gravity > 0);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_molecular_weight_positive CHECK (molecular_weight IS NULL OR molecular_weight > 0);

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_source CHECK (source IS NULL OR source IN (
        'LAB_ANALYSIS', 'FIELD_MEASUREMENT', 'DEFAULT', 'ENGINEERING_ESTIMATE', 'HISTORICAL'
    ));

ALTER TABLE flaring_service.fl_gas_compositions
    ADD CONSTRAINT chk_gc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_gc_compute_total_fraction
    BEFORE INSERT OR UPDATE ON flaring_service.fl_gas_compositions
    FOR EACH ROW EXECUTE FUNCTION flaring_service.compute_total_fraction();

CREATE TRIGGER trg_gc_updated_at
    BEFORE UPDATE ON flaring_service.fl_gas_compositions
    FOR EACH ROW EXECUTE FUNCTION flaring_service.set_updated_at();

-- =============================================================================
-- Table 3: flaring_service.fl_emission_factors
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_emission_factors (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    gas                 VARCHAR(20)     NOT NULL,
    flare_type          VARCHAR(50),
    source              VARCHAR(30)     NOT NULL,
    factor_value        DECIMAL(18,10)  NOT NULL,
    factor_unit         VARCHAR(50)     NOT NULL,
    year                INTEGER,
    region              VARCHAR(50),
    description         TEXT,
    reference           TEXT,
    is_active           BOOLEAN         DEFAULT TRUE,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'BLACK_CARBON'
    ));

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_flare_type CHECK (flare_type IS NULL OR flare_type IN (
        'ELEVATED_STEAM_ASSISTED', 'ELEVATED_AIR_ASSISTED', 'ELEVATED_UNASSISTED',
        'ENCLOSED_GROUND', 'MULTI_POINT_GROUND', 'OFFSHORE_MARINE',
        'CANDLESTICK', 'LOW_PRESSURE'
    ));

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_source CHECK (source IN (
        'EPA', 'IPCC', 'API', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_factor_value_non_negative CHECK (factor_value >= 0);

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_factor_unit_not_empty CHECK (LENGTH(TRIM(factor_unit)) > 0);

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_year_range CHECK (year IS NULL OR (year >= 1990 AND year <= 2100));

ALTER TABLE flaring_service.fl_emission_factors
    ADD CONSTRAINT chk_ef_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_ef_updated_at
    BEFORE UPDATE ON flaring_service.fl_emission_factors
    FOR EACH ROW EXECUTE FUNCTION flaring_service.set_updated_at();

-- =============================================================================
-- Table 4: flaring_service.fl_flaring_events
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_flaring_events (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    flare_id                    UUID            NOT NULL REFERENCES flaring_service.fl_flare_systems(id),
    event_category              VARCHAR(30)     NOT NULL,
    start_time                  TIMESTAMPTZ     NOT NULL,
    end_time                    TIMESTAMPTZ,
    duration_hours              DECIMAL(12,4),
    gas_volume_scf              DECIMAL(18,4),
    gas_volume_nm3              DECIMAL(18,4),
    composition_id              UUID            REFERENCES flaring_service.fl_gas_compositions(id),
    measured_flow_rate_scfh     DECIMAL(14,4),
    flow_measurement_method     VARCHAR(50),
    is_estimated                BOOLEAN         DEFAULT FALSE,
    estimation_method           VARCHAR(100),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_event_category CHECK (event_category IN (
        'ROUTINE', 'NON_ROUTINE', 'EMERGENCY', 'MAINTENANCE',
        'PILOT_PURGE', 'WELL_COMPLETION'
    ));

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_end_after_start CHECK (end_time IS NULL OR end_time >= start_time);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_duration_non_negative CHECK (duration_hours IS NULL OR duration_hours >= 0);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_gas_volume_scf_non_negative CHECK (gas_volume_scf IS NULL OR gas_volume_scf >= 0);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_gas_volume_nm3_non_negative CHECK (gas_volume_nm3 IS NULL OR gas_volume_nm3 >= 0);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_flow_rate_non_negative CHECK (measured_flow_rate_scfh IS NULL OR measured_flow_rate_scfh >= 0);

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_flow_measurement_method CHECK (flow_measurement_method IS NULL OR flow_measurement_method IN (
        'ULTRASONIC', 'ORIFICE_PLATE', 'VORTEX', 'THERMAL_MASS', 'PITOT_TUBE', 'CORIOLIS', 'ESTIMATED'
    ));

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_estimation_method_required CHECK (
        is_estimated = FALSE OR (is_estimated = TRUE AND estimation_method IS NOT NULL AND LENGTH(TRIM(estimation_method)) > 0)
    );

ALTER TABLE flaring_service.fl_flaring_events
    ADD CONSTRAINT chk_fe_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_fe_updated_at
    BEFORE UPDATE ON flaring_service.fl_flaring_events
    FOR EACH ROW EXECUTE FUNCTION flaring_service.set_updated_at();

-- =============================================================================
-- Table 5: flaring_service.fl_calculations
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    flare_id                UUID            REFERENCES flaring_service.fl_flare_systems(id),
    event_id                UUID            REFERENCES flaring_service.fl_flaring_events(id),
    calculation_method      VARCHAR(30)     NOT NULL,
    gas_volume_scf          DECIMAL(18,4),
    composition_id          UUID            REFERENCES flaring_service.fl_gas_compositions(id),
    combustion_efficiency   DECIMAL(8,6),
    gwp_source              VARCHAR(20)     NOT NULL DEFAULT 'AR6',
    total_co2e_kg           DECIMAL(18,8),
    co2_kg                  DECIMAL(18,8),
    ch4_kg                  DECIMAL(18,8),
    n2o_kg                  DECIMAL(18,8),
    black_carbon_kg         DECIMAL(18,8),
    pilot_co2e_kg           DECIMAL(18,8),
    purge_co2e_kg           DECIMAL(18,8),
    uncertainty_percent     DECIMAL(8,4),
    data_quality_score      DECIMAL(5,3),
    provenance_hash         VARCHAR(64)     NOT NULL,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'COMPLETED',
    calculated_at           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_calculation_method CHECK (calculation_method IN (
        'GAS_COMPOSITION', 'DEFAULT_EF', 'ENGINEERING_ESTIMATE', 'DIRECT_MEASUREMENT'
    ));

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_gas_volume_non_negative CHECK (gas_volume_scf IS NULL OR gas_volume_scf >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_combustion_efficiency_range CHECK (combustion_efficiency IS NULL OR (combustion_efficiency >= 0 AND combustion_efficiency <= 1));

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_gwp_source CHECK (gwp_source IN ('AR4', 'AR5', 'AR6'));

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_total_co2e_kg_non_negative CHECK (total_co2e_kg IS NULL OR total_co2e_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_co2_kg_non_negative CHECK (co2_kg IS NULL OR co2_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_ch4_kg_non_negative CHECK (ch4_kg IS NULL OR ch4_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_n2o_kg_non_negative CHECK (n2o_kg IS NULL OR n2o_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_black_carbon_kg_non_negative CHECK (black_carbon_kg IS NULL OR black_carbon_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_pilot_co2e_kg_non_negative CHECK (pilot_co2e_kg IS NULL OR pilot_co2e_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_purge_co2e_kg_non_negative CHECK (purge_co2e_kg IS NULL OR purge_co2e_kg >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_uncertainty_non_negative CHECK (uncertainty_percent IS NULL OR uncertainty_percent >= 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_data_quality_score_range CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 5));

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_provenance_hash_not_empty CHECK (LENGTH(TRIM(provenance_hash)) > 0);

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_status CHECK (status IN (
        'COMPLETED', 'PENDING', 'FAILED', 'DRAFT', 'REVIEWED', 'APPROVED'
    ));

ALTER TABLE flaring_service.fl_calculations
    ADD CONSTRAINT chk_calc_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

CREATE TRIGGER trg_calc_updated_at
    BEFORE UPDATE ON flaring_service.fl_calculations
    FOR EACH ROW EXECUTE FUNCTION flaring_service.set_updated_at();

-- =============================================================================
-- Table 6: flaring_service.fl_calculation_details
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_calculation_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            NOT NULL REFERENCES flaring_service.fl_calculations(id) ON DELETE CASCADE,
    gas                     VARCHAR(20)     NOT NULL,
    emission_kg             DECIMAL(18,8)   NOT NULL,
    co2e_kg                 DECIMAL(18,8)   NOT NULL,
    gwp_value               DECIMAL(12,4)   NOT NULL,
    emission_factor_value   DECIMAL(18,10),
    emission_factor_source  VARCHAR(30),
    emission_factor_unit    VARCHAR(50),
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_gas CHECK (gas IN (
        'CO2', 'CH4', 'N2O', 'BLACK_CARBON'
    ));

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_emission_kg_non_negative CHECK (emission_kg >= 0);

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_co2e_kg_non_negative CHECK (co2e_kg >= 0);

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_gwp_value_positive CHECK (gwp_value > 0);

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_non_negative CHECK (emission_factor_value IS NULL OR emission_factor_value >= 0);

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_emission_factor_source CHECK (emission_factor_source IS NULL OR emission_factor_source IN (
        'EPA', 'IPCC', 'API', 'DEFRA', 'EU_ETS', 'CUSTOM'
    ));

ALTER TABLE flaring_service.fl_calculation_details
    ADD CONSTRAINT chk_cd_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 7: flaring_service.fl_combustion_efficiency
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_combustion_efficiency (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    flare_id            UUID            NOT NULL REFERENCES flaring_service.fl_flare_systems(id),
    test_date           TIMESTAMPTZ     NOT NULL,
    base_ce             DECIMAL(8,6)    NOT NULL,
    adjusted_ce         DECIMAL(8,6),
    wind_speed_ms       DECIMAL(8,4),
    tip_velocity_mach   DECIMAL(8,6),
    lhv_btu_scf         DECIMAL(12,4),
    steam_ratio         DECIMAL(8,4),
    air_ratio           DECIMAL(8,4),
    dre                 DECIMAL(8,6),
    test_method         VARCHAR(100),
    test_reference      VARCHAR(100),
    notes               TEXT,
    metadata            JSONB           DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_base_ce_range CHECK (base_ce >= 0 AND base_ce <= 1);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_adjusted_ce_range CHECK (adjusted_ce IS NULL OR (adjusted_ce >= 0 AND adjusted_ce <= 1));

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_wind_speed_non_negative CHECK (wind_speed_ms IS NULL OR wind_speed_ms >= 0);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_tip_velocity_non_negative CHECK (tip_velocity_mach IS NULL OR tip_velocity_mach >= 0);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_lhv_positive CHECK (lhv_btu_scf IS NULL OR lhv_btu_scf > 0);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_steam_ratio_non_negative CHECK (steam_ratio IS NULL OR steam_ratio >= 0);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_air_ratio_non_negative CHECK (air_ratio IS NULL OR air_ratio >= 0);

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_dre_range CHECK (dre IS NULL OR (dre >= 0 AND dre <= 1));

ALTER TABLE flaring_service.fl_combustion_efficiency
    ADD CONSTRAINT chk_ce_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 8: flaring_service.fl_pilot_purge_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_pilot_purge_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    flare_id                    UUID            NOT NULL REFERENCES flaring_service.fl_flare_systems(id),
    record_period_start         TIMESTAMPTZ     NOT NULL,
    record_period_end           TIMESTAMPTZ     NOT NULL,
    pilot_gas_volume_scf        DECIMAL(18,4),
    pilot_gas_composition_id    UUID            REFERENCES flaring_service.fl_gas_compositions(id),
    purge_gas_volume_scf        DECIMAL(18,4),
    purge_gas_type              VARCHAR(30),
    pilot_operating_hours       DECIMAL(10,2),
    num_pilots_active           INTEGER,
    pilot_co2e_kg               DECIMAL(18,8),
    purge_co2e_kg               DECIMAL(18,8),
    metadata                    JSONB           DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_period_end_after_start CHECK (record_period_end >= record_period_start);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_pilot_gas_volume_non_negative CHECK (pilot_gas_volume_scf IS NULL OR pilot_gas_volume_scf >= 0);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_purge_gas_volume_non_negative CHECK (purge_gas_volume_scf IS NULL OR purge_gas_volume_scf >= 0);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_purge_gas_type CHECK (purge_gas_type IS NULL OR purge_gas_type IN (
        'NATURAL_GAS', 'NITROGEN', 'CO2', 'INERT'
    ));

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_operating_hours_non_negative CHECK (pilot_operating_hours IS NULL OR pilot_operating_hours >= 0);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_num_pilots_positive CHECK (num_pilots_active IS NULL OR num_pilots_active >= 1);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_pilot_co2e_non_negative CHECK (pilot_co2e_kg IS NULL OR pilot_co2e_kg >= 0);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_purge_co2e_non_negative CHECK (purge_co2e_kg IS NULL OR purge_co2e_kg >= 0);

ALTER TABLE flaring_service.fl_pilot_purge_records
    ADD CONSTRAINT chk_pp_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 9: flaring_service.fl_compliance_records
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_compliance_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    calculation_id          UUID            REFERENCES flaring_service.fl_calculations(id),
    framework               VARCHAR(30)     NOT NULL,
    compliance_status       VARCHAR(20)     NOT NULL,
    requirements_checked    INTEGER         NOT NULL DEFAULT 0,
    requirements_met        INTEGER         NOT NULL DEFAULT 0,
    findings                JSONB           DEFAULT '[]'::jsonb,
    recommendations         JSONB           DEFAULT '[]'::jsonb,
    checked_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                JSONB           DEFAULT '{}'::jsonb,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_framework CHECK (framework IN (
        'GHG_PROTOCOL', 'EPA_SUBPART_W', 'CSRD_ESRS', 'ISO_14064',
        'EU_ETS_MRR', 'EU_METHANE_REG', 'WORLD_BANK_ZRF', 'OGMP_2_0'
    ));

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_compliance_status CHECK (compliance_status IN (
        'COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT'
    ));

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_requirements_checked_non_negative CHECK (requirements_checked >= 0);

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_requirements_met_non_negative CHECK (requirements_met >= 0);

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_requirements_met_not_exceed_checked CHECK (requirements_met <= requirements_checked);

ALTER TABLE flaring_service.fl_compliance_records
    ADD CONSTRAINT chk_cr_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 10: flaring_service.fl_audit_entries
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_audit_entries (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID            NOT NULL,
    entity_type         VARCHAR(50)     NOT NULL,
    entity_id           UUID            NOT NULL,
    action              VARCHAR(30)     NOT NULL,
    step_name           VARCHAR(100),
    input_hash          VARCHAR(64),
    output_hash         VARCHAR(64),
    prev_hash           VARCHAR(64),
    formula             TEXT,
    parameters          JSONB           DEFAULT '{}'::jsonb,
    result              JSONB           DEFAULT '{}'::jsonb,
    actor_id            UUID,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

ALTER TABLE flaring_service.fl_audit_entries
    ADD CONSTRAINT chk_ae_entity_type_not_empty CHECK (LENGTH(TRIM(entity_type)) > 0);

ALTER TABLE flaring_service.fl_audit_entries
    ADD CONSTRAINT chk_ae_action CHECK (action IN (
        'CREATE', 'UPDATE', 'DELETE', 'CALCULATE', 'AGGREGATE',
        'VALIDATE', 'APPROVE', 'REJECT', 'IMPORT', 'EXPORT'
    ));

ALTER TABLE flaring_service.fl_audit_entries
    ADD CONSTRAINT chk_ae_tenant_id_not_null CHECK (tenant_id IS NOT NULL);

-- =============================================================================
-- Table 11: flaring_service.fl_calculation_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_calculation_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    flare_type          VARCHAR(50),
    method              VARCHAR(30),
    emissions_kg_co2e   DECIMAL(18,8),
    duration_ms         DECIMAL(12,2),
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'flaring_service.fl_calculation_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE flaring_service.fl_calculation_events
    ADD CONSTRAINT chk_cae_emissions_non_negative CHECK (emissions_kg_co2e IS NULL OR emissions_kg_co2e >= 0);

ALTER TABLE flaring_service.fl_calculation_events
    ADD CONSTRAINT chk_cae_duration_non_negative CHECK (duration_ms IS NULL OR duration_ms >= 0);

ALTER TABLE flaring_service.fl_calculation_events
    ADD CONSTRAINT chk_cae_flare_type CHECK (
        flare_type IS NULL OR flare_type IN (
            'ELEVATED_STEAM_ASSISTED', 'ELEVATED_AIR_ASSISTED', 'ELEVATED_UNASSISTED',
            'ENCLOSED_GROUND', 'MULTI_POINT_GROUND', 'OFFSHORE_MARINE',
            'CANDLESTICK', 'LOW_PRESSURE'
        )
    );

ALTER TABLE flaring_service.fl_calculation_events
    ADD CONSTRAINT chk_cae_method CHECK (
        method IS NULL OR method IN (
            'GAS_COMPOSITION', 'DEFAULT_EF', 'ENGINEERING_ESTIMATE', 'DIRECT_MEASUREMENT'
        )
    );

-- =============================================================================
-- Table 12: flaring_service.fl_flaring_event_ts (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_flaring_event_ts (
    time                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id               UUID            NOT NULL,
    event_type              VARCHAR(50),
    event_category          VARCHAR(30),
    flare_type              VARCHAR(50),
    gas_volume_scf          DECIMAL(18,4),
    duration_hours          DECIMAL(12,4),
    metadata                JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'flaring_service.fl_flaring_event_ts',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE flaring_service.fl_flaring_event_ts
    ADD CONSTRAINT chk_fets_event_category CHECK (
        event_category IS NULL OR event_category IN (
            'ROUTINE', 'NON_ROUTINE', 'EMERGENCY', 'MAINTENANCE',
            'PILOT_PURGE', 'WELL_COMPLETION'
        )
    );

ALTER TABLE flaring_service.fl_flaring_event_ts
    ADD CONSTRAINT chk_fets_flare_type CHECK (
        flare_type IS NULL OR flare_type IN (
            'ELEVATED_STEAM_ASSISTED', 'ELEVATED_AIR_ASSISTED', 'ELEVATED_UNASSISTED',
            'ENCLOSED_GROUND', 'MULTI_POINT_GROUND', 'OFFSHORE_MARINE',
            'CANDLESTICK', 'LOW_PRESSURE'
        )
    );

ALTER TABLE flaring_service.fl_flaring_event_ts
    ADD CONSTRAINT chk_fets_gas_volume_non_negative CHECK (gas_volume_scf IS NULL OR gas_volume_scf >= 0);

ALTER TABLE flaring_service.fl_flaring_event_ts
    ADD CONSTRAINT chk_fets_duration_non_negative CHECK (duration_hours IS NULL OR duration_hours >= 0);

-- =============================================================================
-- Table 13: flaring_service.fl_compliance_events (hypertable)
-- =============================================================================

CREATE TABLE IF NOT EXISTS flaring_service.fl_compliance_events (
    time                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tenant_id           UUID            NOT NULL,
    event_type          VARCHAR(50),
    framework           VARCHAR(30),
    status              VARCHAR(20),
    check_count         INTEGER,
    pass_count          INTEGER,
    fail_count          INTEGER,
    metadata            JSONB           DEFAULT '{}'::jsonb
);

SELECT create_hypertable(
    'flaring_service.fl_compliance_events',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

ALTER TABLE flaring_service.fl_compliance_events
    ADD CONSTRAINT chk_coe_framework CHECK (
        framework IS NULL OR framework IN (
            'GHG_PROTOCOL', 'EPA_SUBPART_W', 'CSRD_ESRS', 'ISO_14064',
            'EU_ETS_MRR', 'EU_METHANE_REG', 'WORLD_BANK_ZRF', 'OGMP_2_0'
        )
    );

ALTER TABLE flaring_service.fl_compliance_events
    ADD CONSTRAINT chk_coe_status CHECK (
        status IS NULL OR status IN ('COMPLIANT', 'NON_COMPLIANT', 'PENDING', 'UNDER_REVIEW', 'EXEMPT')
    );

ALTER TABLE flaring_service.fl_compliance_events
    ADD CONSTRAINT chk_coe_check_count_non_negative CHECK (check_count IS NULL OR check_count >= 0);

ALTER TABLE flaring_service.fl_compliance_events
    ADD CONSTRAINT chk_coe_pass_count_non_negative CHECK (pass_count IS NULL OR pass_count >= 0);

ALTER TABLE flaring_service.fl_compliance_events
    ADD CONSTRAINT chk_coe_fail_count_non_negative CHECK (fail_count IS NULL OR fail_count >= 0);

-- =============================================================================
-- Continuous Aggregates
-- =============================================================================

-- fl_hourly_calculation_stats: hourly count/sum(co2e)/avg(co2e) by flare_type and method
CREATE MATERIALIZED VIEW flaring_service.fl_hourly_calculation_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time)     AS bucket,
    flare_type,
    method,
    COUNT(*)                        AS total_calculations,
    SUM(emissions_kg_co2e)          AS sum_emissions_kg_co2e,
    AVG(emissions_kg_co2e)          AS avg_emissions_kg_co2e,
    AVG(duration_ms)                AS avg_duration_ms,
    MAX(duration_ms)                AS max_duration_ms
FROM flaring_service.fl_calculation_events
WHERE time IS NOT NULL
GROUP BY bucket, flare_type, method
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'flaring_service.fl_hourly_calculation_stats',
    start_offset      => INTERVAL '2 hours',
    end_offset        => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '15 minutes'
);

-- fl_daily_emission_totals: daily count/sum(co2e) by flare_type and event_category
CREATE MATERIALIZED VIEW flaring_service.fl_daily_emission_totals
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time)      AS bucket,
    flare_type,
    event_category,
    COUNT(*)                        AS total_events,
    SUM(gas_volume_scf)             AS sum_gas_volume_scf,
    AVG(duration_hours)             AS avg_duration_hours
FROM flaring_service.fl_flaring_event_ts
WHERE time IS NOT NULL
GROUP BY bucket, flare_type, event_category
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'flaring_service.fl_daily_emission_totals',
    start_offset      => INTERVAL '2 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Indexes
-- =============================================================================

-- fl_flare_systems indexes
CREATE INDEX IF NOT EXISTS idx_fl_fs_tenant_id              ON flaring_service.fl_flare_systems(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_fs_name                   ON flaring_service.fl_flare_systems(name);
CREATE INDEX IF NOT EXISTS idx_fl_fs_flare_type             ON flaring_service.fl_flare_systems(flare_type);
CREATE INDEX IF NOT EXISTS idx_fl_fs_status                 ON flaring_service.fl_flare_systems(status);
CREATE INDEX IF NOT EXISTS idx_fl_fs_facility_id            ON flaring_service.fl_flare_systems(facility_id);
CREATE INDEX IF NOT EXISTS idx_fl_fs_assist_type            ON flaring_service.fl_flare_systems(assist_type);
CREATE INDEX IF NOT EXISTS idx_fl_fs_purge_gas_type         ON flaring_service.fl_flare_systems(purge_gas_type);
CREATE INDEX IF NOT EXISTS idx_fl_fs_installation_date      ON flaring_service.fl_flare_systems(installation_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fs_last_inspection_date   ON flaring_service.fl_flare_systems(last_inspection_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fs_tenant_flare_type      ON flaring_service.fl_flare_systems(tenant_id, flare_type);
CREATE INDEX IF NOT EXISTS idx_fl_fs_tenant_status          ON flaring_service.fl_flare_systems(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_fl_fs_tenant_facility        ON flaring_service.fl_flare_systems(tenant_id, facility_id);
CREATE INDEX IF NOT EXISTS idx_fl_fs_created_at             ON flaring_service.fl_flare_systems(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fs_updated_at             ON flaring_service.fl_flare_systems(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fs_metadata               ON flaring_service.fl_flare_systems USING GIN (metadata);

-- fl_gas_compositions indexes
CREATE INDEX IF NOT EXISTS idx_fl_gc_tenant_id              ON flaring_service.fl_gas_compositions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_gc_name                   ON flaring_service.fl_gas_compositions(name);
CREATE INDEX IF NOT EXISTS idx_fl_gc_source                 ON flaring_service.fl_gas_compositions(source);
CREATE INDEX IF NOT EXISTS idx_fl_gc_analysis_date          ON flaring_service.fl_gas_compositions(analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_lab_reference          ON flaring_service.fl_gas_compositions(lab_reference);
CREATE INDEX IF NOT EXISTS idx_fl_gc_hhv_btu_scf            ON flaring_service.fl_gas_compositions(hhv_btu_scf DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_lhv_btu_scf            ON flaring_service.fl_gas_compositions(lhv_btu_scf DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_tenant_source          ON flaring_service.fl_gas_compositions(tenant_id, source);
CREATE INDEX IF NOT EXISTS idx_fl_gc_tenant_analysis_date   ON flaring_service.fl_gas_compositions(tenant_id, analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_created_at             ON flaring_service.fl_gas_compositions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_updated_at             ON flaring_service.fl_gas_compositions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_gc_metadata               ON flaring_service.fl_gas_compositions USING GIN (metadata);

-- fl_emission_factors indexes
CREATE INDEX IF NOT EXISTS idx_fl_ef_tenant_id              ON flaring_service.fl_emission_factors(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_ef_gas                    ON flaring_service.fl_emission_factors(gas);
CREATE INDEX IF NOT EXISTS idx_fl_ef_flare_type             ON flaring_service.fl_emission_factors(flare_type);
CREATE INDEX IF NOT EXISTS idx_fl_ef_source                 ON flaring_service.fl_emission_factors(source);
CREATE INDEX IF NOT EXISTS idx_fl_ef_year                   ON flaring_service.fl_emission_factors(year DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ef_region                 ON flaring_service.fl_emission_factors(region);
CREATE INDEX IF NOT EXISTS idx_fl_ef_is_active              ON flaring_service.fl_emission_factors(is_active);
CREATE INDEX IF NOT EXISTS idx_fl_ef_gas_flare_type         ON flaring_service.fl_emission_factors(gas, flare_type);
CREATE INDEX IF NOT EXISTS idx_fl_ef_gas_source             ON flaring_service.fl_emission_factors(gas, source);
CREATE INDEX IF NOT EXISTS idx_fl_ef_tenant_gas             ON flaring_service.fl_emission_factors(tenant_id, gas);
CREATE INDEX IF NOT EXISTS idx_fl_ef_tenant_gas_source      ON flaring_service.fl_emission_factors(tenant_id, gas, source);
CREATE INDEX IF NOT EXISTS idx_fl_ef_tenant_gas_flare       ON flaring_service.fl_emission_factors(tenant_id, gas, flare_type);
CREATE INDEX IF NOT EXISTS idx_fl_ef_tenant_active          ON flaring_service.fl_emission_factors(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_fl_ef_created_at             ON flaring_service.fl_emission_factors(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ef_updated_at             ON flaring_service.fl_emission_factors(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ef_metadata               ON flaring_service.fl_emission_factors USING GIN (metadata);

-- fl_flaring_events indexes
CREATE INDEX IF NOT EXISTS idx_fl_fe_tenant_id              ON flaring_service.fl_flaring_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_fe_flare_id               ON flaring_service.fl_flaring_events(flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_fe_event_category         ON flaring_service.fl_flaring_events(event_category);
CREATE INDEX IF NOT EXISTS idx_fl_fe_start_time             ON flaring_service.fl_flaring_events(start_time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_end_time               ON flaring_service.fl_flaring_events(end_time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_composition_id         ON flaring_service.fl_flaring_events(composition_id);
CREATE INDEX IF NOT EXISTS idx_fl_fe_flow_measurement       ON flaring_service.fl_flaring_events(flow_measurement_method);
CREATE INDEX IF NOT EXISTS idx_fl_fe_is_estimated           ON flaring_service.fl_flaring_events(is_estimated);
CREATE INDEX IF NOT EXISTS idx_fl_fe_tenant_flare           ON flaring_service.fl_flaring_events(tenant_id, flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_fe_tenant_category        ON flaring_service.fl_flaring_events(tenant_id, event_category);
CREATE INDEX IF NOT EXISTS idx_fl_fe_flare_category         ON flaring_service.fl_flaring_events(flare_id, event_category);
CREATE INDEX IF NOT EXISTS idx_fl_fe_tenant_start_time      ON flaring_service.fl_flaring_events(tenant_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_flare_start_time       ON flaring_service.fl_flaring_events(flare_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_created_at             ON flaring_service.fl_flaring_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_updated_at             ON flaring_service.fl_flaring_events(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fe_metadata               ON flaring_service.fl_flaring_events USING GIN (metadata);

-- fl_calculations indexes
CREATE INDEX IF NOT EXISTS idx_fl_calc_tenant_id            ON flaring_service.fl_calculations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_calc_flare_id             ON flaring_service.fl_calculations(flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_calc_event_id             ON flaring_service.fl_calculations(event_id);
CREATE INDEX IF NOT EXISTS idx_fl_calc_method               ON flaring_service.fl_calculations(calculation_method);
CREATE INDEX IF NOT EXISTS idx_fl_calc_composition_id       ON flaring_service.fl_calculations(composition_id);
CREATE INDEX IF NOT EXISTS idx_fl_calc_gwp_source           ON flaring_service.fl_calculations(gwp_source);
CREATE INDEX IF NOT EXISTS idx_fl_calc_total_co2e_kg        ON flaring_service.fl_calculations(total_co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fl_calc_status               ON flaring_service.fl_calculations(status);
CREATE INDEX IF NOT EXISTS idx_fl_calc_provenance_hash      ON flaring_service.fl_calculations(provenance_hash);
CREATE INDEX IF NOT EXISTS idx_fl_calc_calculated_at        ON flaring_service.fl_calculations(calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_calc_tenant_flare         ON flaring_service.fl_calculations(tenant_id, flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_calc_tenant_method        ON flaring_service.fl_calculations(tenant_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_fl_calc_tenant_status        ON flaring_service.fl_calculations(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_fl_calc_flare_method         ON flaring_service.fl_calculations(flare_id, calculation_method);
CREATE INDEX IF NOT EXISTS idx_fl_calc_created_at           ON flaring_service.fl_calculations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_calc_updated_at           ON flaring_service.fl_calculations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_calc_metadata             ON flaring_service.fl_calculations USING GIN (metadata);

-- fl_calculation_details indexes
CREATE INDEX IF NOT EXISTS idx_fl_cd_tenant_id              ON flaring_service.fl_calculation_details(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_cd_calculation_id         ON flaring_service.fl_calculation_details(calculation_id);
CREATE INDEX IF NOT EXISTS idx_fl_cd_gas                    ON flaring_service.fl_calculation_details(gas);
CREATE INDEX IF NOT EXISTS idx_fl_cd_emission_kg            ON flaring_service.fl_calculation_details(emission_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cd_co2e_kg                ON flaring_service.fl_calculation_details(co2e_kg DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cd_emission_factor_source ON flaring_service.fl_calculation_details(emission_factor_source);
CREATE INDEX IF NOT EXISTS idx_fl_cd_tenant_calc            ON flaring_service.fl_calculation_details(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_fl_cd_calc_gas               ON flaring_service.fl_calculation_details(calculation_id, gas);
CREATE INDEX IF NOT EXISTS idx_fl_cd_tenant_gas             ON flaring_service.fl_calculation_details(tenant_id, gas);
CREATE INDEX IF NOT EXISTS idx_fl_cd_created_at             ON flaring_service.fl_calculation_details(created_at DESC);

-- fl_combustion_efficiency indexes
CREATE INDEX IF NOT EXISTS idx_fl_ce_tenant_id              ON flaring_service.fl_combustion_efficiency(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_ce_flare_id               ON flaring_service.fl_combustion_efficiency(flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_ce_test_date              ON flaring_service.fl_combustion_efficiency(test_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ce_test_method            ON flaring_service.fl_combustion_efficiency(test_method);
CREATE INDEX IF NOT EXISTS idx_fl_ce_base_ce                ON flaring_service.fl_combustion_efficiency(base_ce DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ce_tenant_flare           ON flaring_service.fl_combustion_efficiency(tenant_id, flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_ce_flare_date             ON flaring_service.fl_combustion_efficiency(flare_id, test_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ce_tenant_date            ON flaring_service.fl_combustion_efficiency(tenant_id, test_date DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ce_created_at             ON flaring_service.fl_combustion_efficiency(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ce_metadata               ON flaring_service.fl_combustion_efficiency USING GIN (metadata);

-- fl_pilot_purge_records indexes
CREATE INDEX IF NOT EXISTS idx_fl_pp_tenant_id              ON flaring_service.fl_pilot_purge_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_pp_flare_id               ON flaring_service.fl_pilot_purge_records(flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_pp_period_start           ON flaring_service.fl_pilot_purge_records(record_period_start DESC);
CREATE INDEX IF NOT EXISTS idx_fl_pp_period_end             ON flaring_service.fl_pilot_purge_records(record_period_end DESC);
CREATE INDEX IF NOT EXISTS idx_fl_pp_purge_gas_type         ON flaring_service.fl_pilot_purge_records(purge_gas_type);
CREATE INDEX IF NOT EXISTS idx_fl_pp_composition_id         ON flaring_service.fl_pilot_purge_records(pilot_gas_composition_id);
CREATE INDEX IF NOT EXISTS idx_fl_pp_tenant_flare           ON flaring_service.fl_pilot_purge_records(tenant_id, flare_id);
CREATE INDEX IF NOT EXISTS idx_fl_pp_flare_period           ON flaring_service.fl_pilot_purge_records(flare_id, record_period_start DESC);
CREATE INDEX IF NOT EXISTS idx_fl_pp_tenant_period          ON flaring_service.fl_pilot_purge_records(tenant_id, record_period_start DESC);
CREATE INDEX IF NOT EXISTS idx_fl_pp_created_at             ON flaring_service.fl_pilot_purge_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_pp_metadata               ON flaring_service.fl_pilot_purge_records USING GIN (metadata);

-- fl_compliance_records indexes
CREATE INDEX IF NOT EXISTS idx_fl_cr_tenant_id              ON flaring_service.fl_compliance_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_cr_calculation_id         ON flaring_service.fl_compliance_records(calculation_id);
CREATE INDEX IF NOT EXISTS idx_fl_cr_framework              ON flaring_service.fl_compliance_records(framework);
CREATE INDEX IF NOT EXISTS idx_fl_cr_compliance_status      ON flaring_service.fl_compliance_records(compliance_status);
CREATE INDEX IF NOT EXISTS idx_fl_cr_checked_at             ON flaring_service.fl_compliance_records(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cr_tenant_framework       ON flaring_service.fl_compliance_records(tenant_id, framework);
CREATE INDEX IF NOT EXISTS idx_fl_cr_tenant_status          ON flaring_service.fl_compliance_records(tenant_id, compliance_status);
CREATE INDEX IF NOT EXISTS idx_fl_cr_framework_status       ON flaring_service.fl_compliance_records(framework, compliance_status);
CREATE INDEX IF NOT EXISTS idx_fl_cr_tenant_calculation     ON flaring_service.fl_compliance_records(tenant_id, calculation_id);
CREATE INDEX IF NOT EXISTS idx_fl_cr_created_at             ON flaring_service.fl_compliance_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cr_findings               ON flaring_service.fl_compliance_records USING GIN (findings);
CREATE INDEX IF NOT EXISTS idx_fl_cr_recommendations        ON flaring_service.fl_compliance_records USING GIN (recommendations);
CREATE INDEX IF NOT EXISTS idx_fl_cr_metadata               ON flaring_service.fl_compliance_records USING GIN (metadata);

-- fl_audit_entries indexes
CREATE INDEX IF NOT EXISTS idx_fl_ae_tenant_id              ON flaring_service.fl_audit_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_fl_ae_entity_type            ON flaring_service.fl_audit_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_fl_ae_entity_id              ON flaring_service.fl_audit_entries(entity_id);
CREATE INDEX IF NOT EXISTS idx_fl_ae_action                 ON flaring_service.fl_audit_entries(action);
CREATE INDEX IF NOT EXISTS idx_fl_ae_step_name              ON flaring_service.fl_audit_entries(step_name);
CREATE INDEX IF NOT EXISTS idx_fl_ae_input_hash             ON flaring_service.fl_audit_entries(input_hash);
CREATE INDEX IF NOT EXISTS idx_fl_ae_output_hash            ON flaring_service.fl_audit_entries(output_hash);
CREATE INDEX IF NOT EXISTS idx_fl_ae_prev_hash              ON flaring_service.fl_audit_entries(prev_hash);
CREATE INDEX IF NOT EXISTS idx_fl_ae_actor_id               ON flaring_service.fl_audit_entries(actor_id);
CREATE INDEX IF NOT EXISTS idx_fl_ae_tenant_entity          ON flaring_service.fl_audit_entries(tenant_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_fl_ae_tenant_action          ON flaring_service.fl_audit_entries(tenant_id, action);
CREATE INDEX IF NOT EXISTS idx_fl_ae_created_at             ON flaring_service.fl_audit_entries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fl_ae_parameters             ON flaring_service.fl_audit_entries USING GIN (parameters);
CREATE INDEX IF NOT EXISTS idx_fl_ae_result                 ON flaring_service.fl_audit_entries USING GIN (result);

-- fl_calculation_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fl_cae_tenant_id             ON flaring_service.fl_calculation_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_event_type            ON flaring_service.fl_calculation_events(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_flare_type            ON flaring_service.fl_calculation_events(flare_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_method                ON flaring_service.fl_calculation_events(method, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_tenant_flare_type     ON flaring_service.fl_calculation_events(tenant_id, flare_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_tenant_method         ON flaring_service.fl_calculation_events(tenant_id, method, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_cae_metadata              ON flaring_service.fl_calculation_events USING GIN (metadata);

-- fl_flaring_event_ts indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fl_fets_tenant_id            ON flaring_service.fl_flaring_event_ts(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_event_type           ON flaring_service.fl_flaring_event_ts(event_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_event_category       ON flaring_service.fl_flaring_event_ts(event_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_flare_type           ON flaring_service.fl_flaring_event_ts(flare_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_tenant_category      ON flaring_service.fl_flaring_event_ts(tenant_id, event_category, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_tenant_flare_type    ON flaring_service.fl_flaring_event_ts(tenant_id, flare_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_fets_metadata             ON flaring_service.fl_flaring_event_ts USING GIN (metadata);

-- fl_compliance_events indexes (hypertable-aware)
CREATE INDEX IF NOT EXISTS idx_fl_coe_tenant_id             ON flaring_service.fl_compliance_events(tenant_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_coe_framework             ON flaring_service.fl_compliance_events(framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_coe_status                ON flaring_service.fl_compliance_events(status, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_coe_tenant_framework      ON flaring_service.fl_compliance_events(tenant_id, framework, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_coe_tenant_status         ON flaring_service.fl_compliance_events(tenant_id, status, time DESC);
CREATE INDEX IF NOT EXISTS idx_fl_coe_metadata              ON flaring_service.fl_compliance_events USING GIN (metadata);

-- =============================================================================
-- Row-Level Security
-- =============================================================================

-- fl_flare_systems: tenant-isolated
ALTER TABLE flaring_service.fl_flare_systems ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_fs_read  ON flaring_service.fl_flare_systems FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_fs_write ON flaring_service.fl_flare_systems FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_gas_compositions: tenant-isolated
ALTER TABLE flaring_service.fl_gas_compositions ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_gc_read  ON flaring_service.fl_gas_compositions FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_gc_write ON flaring_service.fl_gas_compositions FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_emission_factors: tenant-isolated
ALTER TABLE flaring_service.fl_emission_factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_ef_read  ON flaring_service.fl_emission_factors FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_ef_write ON flaring_service.fl_emission_factors FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_flaring_events: tenant-isolated
ALTER TABLE flaring_service.fl_flaring_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_fe_read  ON flaring_service.fl_flaring_events FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_fe_write ON flaring_service.fl_flaring_events FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_calculations: tenant-isolated
ALTER TABLE flaring_service.fl_calculations ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_calc_read  ON flaring_service.fl_calculations FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_calc_write ON flaring_service.fl_calculations FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_calculation_details: tenant-isolated
ALTER TABLE flaring_service.fl_calculation_details ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_cd_read  ON flaring_service.fl_calculation_details FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_cd_write ON flaring_service.fl_calculation_details FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_combustion_efficiency: tenant-isolated
ALTER TABLE flaring_service.fl_combustion_efficiency ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_ce_read  ON flaring_service.fl_combustion_efficiency FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_ce_write ON flaring_service.fl_combustion_efficiency FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_pilot_purge_records: tenant-isolated
ALTER TABLE flaring_service.fl_pilot_purge_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_pp_read  ON flaring_service.fl_pilot_purge_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_pp_write ON flaring_service.fl_pilot_purge_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_compliance_records: tenant-isolated
ALTER TABLE flaring_service.fl_compliance_records ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_cr_read  ON flaring_service.fl_compliance_records FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_cr_write ON flaring_service.fl_compliance_records FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_audit_entries: tenant-isolated
ALTER TABLE flaring_service.fl_audit_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_ae_read  ON flaring_service.fl_audit_entries FOR SELECT USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);
CREATE POLICY fl_ae_write ON flaring_service.fl_audit_entries FOR ALL USING (
    tenant_id::text = current_setting('app.current_tenant', true)
    OR current_setting('app.is_admin', true) = 'true'
);

-- fl_calculation_events: open read/write (time-series telemetry)
ALTER TABLE flaring_service.fl_calculation_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_cae_read  ON flaring_service.fl_calculation_events FOR SELECT USING (TRUE);
CREATE POLICY fl_cae_write ON flaring_service.fl_calculation_events FOR ALL   USING (TRUE);

-- fl_flaring_event_ts: open read/write (time-series telemetry)
ALTER TABLE flaring_service.fl_flaring_event_ts ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_fets_read  ON flaring_service.fl_flaring_event_ts FOR SELECT USING (TRUE);
CREATE POLICY fl_fets_write ON flaring_service.fl_flaring_event_ts FOR ALL   USING (TRUE);

-- fl_compliance_events: open read/write (time-series telemetry)
ALTER TABLE flaring_service.fl_compliance_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY fl_coe_read  ON flaring_service.fl_compliance_events FOR SELECT USING (TRUE);
CREATE POLICY fl_coe_write ON flaring_service.fl_compliance_events FOR ALL   USING (TRUE);

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA flaring_service TO greenlang_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA flaring_service TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA flaring_service TO greenlang_app;
GRANT SELECT ON flaring_service.fl_hourly_calculation_stats TO greenlang_app;
GRANT SELECT ON flaring_service.fl_daily_emission_totals TO greenlang_app;

GRANT USAGE ON SCHEMA flaring_service TO greenlang_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA flaring_service TO greenlang_readonly;
GRANT SELECT ON flaring_service.fl_hourly_calculation_stats TO greenlang_readonly;
GRANT SELECT ON flaring_service.fl_daily_emission_totals TO greenlang_readonly;

GRANT ALL ON SCHEMA flaring_service TO greenlang_admin;
GRANT ALL ON ALL TABLES IN SCHEMA flaring_service TO greenlang_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA flaring_service TO greenlang_admin;

INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'flaring:read',                         'flaring', 'read',                         'View all flaring service data including flare systems, gas compositions, calculations, events, and compliance records'),
    (gen_random_uuid(), 'flaring:write',                        'flaring', 'write',                        'Create, update, and manage all flaring service data'),
    (gen_random_uuid(), 'flaring:execute',                      'flaring', 'execute',                      'Execute flaring emission calculations with full audit trail and provenance'),
    (gen_random_uuid(), 'flaring:flares:read',                  'flaring', 'flares_read',                  'View flare system registry with 8 flare types, capacity, assist type, pilot/purge configuration, installation and inspection dates'),
    (gen_random_uuid(), 'flaring:flares:write',                 'flaring', 'flares_write',                 'Create, update, and manage flare system registry entries'),
    (gen_random_uuid(), 'flaring:compositions:read',            'flaring', 'compositions_read',            'View gas composition analysis records with 15-component mole fractions, HHV/LHV, specific gravity, molecular weight, and lab references'),
    (gen_random_uuid(), 'flaring:compositions:write',           'flaring', 'compositions_write',           'Create, update, and manage gas composition analysis records'),
    (gen_random_uuid(), 'flaring:factors:read',                 'flaring', 'factors_read',                 'View emission factors by gas type (CO2/CH4/N2O/BLACK_CARBON), flare type, and source (EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM)'),
    (gen_random_uuid(), 'flaring:factors:write',                'flaring', 'factors_write',                'Create, update, and manage emission factor entries with source, year, and region data'),
    (gen_random_uuid(), 'flaring:events:read',                  'flaring', 'events_read',                  'View flaring event records with 6 categories (ROUTINE/NON_ROUTINE/EMERGENCY/MAINTENANCE/PILOT_PURGE/WELL_COMPLETION), volumes, and flow rates'),
    (gen_random_uuid(), 'flaring:events:write',                 'flaring', 'events_write',                 'Create, update, and manage flaring event records'),
    (gen_random_uuid(), 'flaring:calculations:read',            'flaring', 'calculations_read',            'View flaring emission calculation results with multi-gas breakdown (CO2/CH4/N2O/BLACK_CARBON), combustion efficiency, pilot/purge CO2e, and provenance data'),
    (gen_random_uuid(), 'flaring:calculations:write',           'flaring', 'calculations_write',           'Create and manage flaring emission calculation records'),
    (gen_random_uuid(), 'flaring:efficiency:read',              'flaring', 'efficiency_read',              'View combustion efficiency test records with base/adjusted CE, wind speed, tip velocity, LHV, steam/air ratios, and DRE values'),
    (gen_random_uuid(), 'flaring:efficiency:write',             'flaring', 'efficiency_write',             'Create and manage combustion efficiency test records'),
    (gen_random_uuid(), 'flaring:compliance:read',              'flaring', 'compliance_read',              'View regulatory compliance records for GHG Protocol, EPA Subpart W, CSRD/ESRS, ISO 14064, EU ETS MRR, EU Methane Reg, World Bank ZRF, and OGMP 2.0'),
    (gen_random_uuid(), 'flaring:compliance:execute',           'flaring', 'compliance_execute',           'Execute regulatory compliance checks against 8 frameworks (GHG Protocol, EPA Subpart W, CSRD/ESRS, ISO 14064, EU ETS MRR, EU Methane Reg, World Bank ZRF, OGMP 2.0)'),
    (gen_random_uuid(), 'flaring:admin',                        'flaring', 'admin',                        'Full administrative access to flaring service including configuration, diagnostics, and bulk operations')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Retention Policies
-- =============================================================================

SELECT add_retention_policy('flaring_service.fl_calculation_events', INTERVAL '365 days');
SELECT add_retention_policy('flaring_service.fl_flaring_event_ts',   INTERVAL '365 days');
SELECT add_retention_policy('flaring_service.fl_compliance_events',  INTERVAL '365 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

ALTER TABLE flaring_service.fl_calculation_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'flare_type',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('flaring_service.fl_calculation_events', INTERVAL '30 days');

ALTER TABLE flaring_service.fl_flaring_event_ts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'event_category',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('flaring_service.fl_flaring_event_ts', INTERVAL '30 days');

ALTER TABLE flaring_service.fl_compliance_events
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'framework',
         timescaledb.compress_orderby   = 'time DESC');
SELECT add_compression_policy('flaring_service.fl_compliance_events', INTERVAL '30 days');

-- =============================================================================
-- Seed: Register the Flaring Agent (GL-MRV-SCOPE1-006)
-- =============================================================================

INSERT INTO agent_registry_service.agents (
    agent_id, name, description, layer, execution_mode,
    idempotency_support, deterministic, max_concurrent_runs,
    glip_version, supports_checkpointing, author,
    documentation_url, enabled, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-006',
    'Flaring Agent',
    'Flaring emission calculator for GreenLang Climate OS. Manages flare system registry with 8 flare types (ELEVATED_STEAM_ASSISTED, ELEVATED_AIR_ASSISTED, ELEVATED_UNASSISTED, ENCLOSED_GROUND, MULTI_POINT_GROUND, OFFSHORE_MARINE, CANDLESTICK, LOW_PRESSURE) including tip diameter, height, capacity, assist type (STEAM/AIR/NONE), pilot/purge gas configuration, and installation/inspection tracking. Maintains 15-component gas composition analysis records (CH4, C2H6, C3H8, n-C4H10, i-C4H10, C5H12, C6+, CO2, N2, H2S, H2, CO, C2H4, C3H6, H2O) with trigger-computed total mole fraction, HHV/LHV heating values, specific gravity, molecular weight, and lab references. Stores emission factor database with gas x flare_type factors from EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM sources with year/region scoping. Tracks flaring events across 6 categories (ROUTINE, NON_ROUTINE, EMERGENCY, MAINTENANCE, PILOT_PURGE, WELL_COMPLETION) with start/end times, gas volumes (scf/Nm3), composition references, flow rate measurements (7 methods), and estimation method tracking. Executes deterministic flaring emission calculations using gas composition, default emission factor, engineering estimate, and direct measurement methods with combustion efficiency modeling, multi-gas GWP weighting (CO2/CH4/N2O/BLACK_CARBON) using AR4/AR5/AR6 values, pilot/purge gas CO2e accounting, uncertainty quantification, and data quality scoring. Produces per-gas calculation detail breakdowns with individual emission factors, GWP values, and CO2e emissions. Records combustion efficiency tests with base/adjusted CE, wind speed effects, tip velocity Mach, LHV thresholds, steam/air ratios, and DRE. Tracks pilot and purge gas consumption with period-based volume recording, composition references, operating hours, and CO2e emissions. Checks regulatory compliance against 8 frameworks (GHG Protocol, EPA Subpart W Sec. W.23, CSRD/ESRS E1, ISO 14064, EU ETS MRR, EU Methane Regulation 2024/1787, World Bank Zero Routine Flaring 2030, OGMP 2.0) with requirements tracking, findings, and recommendations. Generates entity-level audit trail entries with step-by-step action tracking, input/output/prev SHA-256 hash chaining for tamper-evident provenance, formula text, parameters/result JSONB, and actor attribution. SHA-256 provenance chains for zero-hallucination audit trail.',
    2, 'async', true, true, 10, '1.0.0', true,
    'GreenLang MRV Team',
    'https://docs.greenlang.ai/agents/flaring',
    true, 'default'
) ON CONFLICT (agent_id) DO NOTHING;

INSERT INTO agent_registry_service.agent_versions (
    agent_id, version, resource_profile, container_spec,
    tags, sectors, provenance_hash
) VALUES (
    'GL-MRV-SCOPE1-006', '1.0.0',
    '{"cpu_request": "500m", "cpu_limit": "2000m", "memory_request": "512Mi", "memory_limit": "2Gi", "gpu": false}'::jsonb,
    '{"image": "greenlang/flaring-service", "tag": "1.0.0", "port": 8000}'::jsonb,
    '{"flaring", "scope-1", "flare-stack", "combustion-efficiency", "ghg-protocol", "epa-subpart-w", "ogmp", "mrv"}',
    '{"oil-and-gas", "petrochemical", "refining", "upstream", "midstream", "downstream", "lng", "cross-sector"}',
    'a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8'
) ON CONFLICT (agent_id, version) DO NOTHING;

INSERT INTO agent_registry_service.agent_capabilities (
    agent_id, version, name, category,
    description, input_types, output_types, parameters
) VALUES
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'flare_system_registry',
    'configuration',
    'Register and manage flare system entries with 8 flare types (ELEVATED_STEAM_ASSISTED/ELEVATED_AIR_ASSISTED/ELEVATED_UNASSISTED/ENCLOSED_GROUND/MULTI_POINT_GROUND/OFFSHORE_MARINE/CANDLESTICK/LOW_PRESSURE), tip diameter, height, capacity, assist type (STEAM/AIR/NONE), pilot/purge gas configuration, and facility assignment.',
    '{"name", "flare_type", "facility_id", "tip_diameter_inches", "height_feet", "capacity_mmbtu_hr", "assist_type", "num_pilots", "pilot_gas_flow_mmbtu_hr", "purge_gas_flow_scfh", "purge_gas_type"}',
    '{"flare_id", "registration_result"}',
    '{"flare_types": ["ELEVATED_STEAM_ASSISTED", "ELEVATED_AIR_ASSISTED", "ELEVATED_UNASSISTED", "ENCLOSED_GROUND", "MULTI_POINT_GROUND", "OFFSHORE_MARINE", "CANDLESTICK", "LOW_PRESSURE"], "assist_types": ["STEAM", "AIR", "NONE"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'gas_composition_analysis',
    'configuration',
    'Register and manage gas composition analysis records with 15-component mole fractions (CH4, C2H6, C3H8, n-C4H10, i-C4H10, C5H12, C6+, CO2, N2, H2S, H2, CO, C2H4, C3H6, H2O), trigger-computed total fraction, HHV/LHV heating values in BTU/scf, specific gravity, molecular weight, lab reference, and analysis date.',
    '{"name", "ch4_fraction", "c2h6_fraction", "c3h8_fraction", "n_c4h10_fraction", "i_c4h10_fraction", "c5h12_fraction", "c6_plus_fraction", "co2_fraction", "n2_fraction", "h2s_fraction", "h2_fraction", "co_fraction", "c2h4_fraction", "c3h6_fraction", "h2o_fraction", "hhv_btu_scf", "lhv_btu_scf", "specific_gravity", "molecular_weight", "analysis_date", "lab_reference"}',
    '{"composition_id", "total_fraction", "registration_result"}',
    '{"components": 15, "auto_total_fraction": true, "supports_heating_values": true, "sources": ["LAB_ANALYSIS", "FIELD_MEASUREMENT", "DEFAULT", "ENGINEERING_ESTIMATE", "HISTORICAL"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'flaring_event_tracking',
    'processing',
    'Record and manage flaring events with 6 categories (ROUTINE/NON_ROUTINE/EMERGENCY/MAINTENANCE/PILOT_PURGE/WELL_COMPLETION), flare system association, start/end times, duration, gas volumes (scf/Nm3), composition reference, flow rate measurements (7 methods: ULTRASONIC/ORIFICE_PLATE/VORTEX/THERMAL_MASS/PITOT_TUBE/CORIOLIS/ESTIMATED), and estimation method tracking.',
    '{"flare_id", "event_category", "start_time", "end_time", "gas_volume_scf", "gas_volume_nm3", "composition_id", "measured_flow_rate_scfh", "flow_measurement_method", "is_estimated", "estimation_method"}',
    '{"event_id", "tracking_result"}',
    '{"event_categories": ["ROUTINE", "NON_ROUTINE", "EMERGENCY", "MAINTENANCE", "PILOT_PURGE", "WELL_COMPLETION"], "flow_methods": ["ULTRASONIC", "ORIFICE_PLATE", "VORTEX", "THERMAL_MASS", "PITOT_TUBE", "CORIOLIS", "ESTIMATED"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'emission_calculation',
    'processing',
    'Execute deterministic flaring emission calculations using gas composition (flow x gas_fraction x stoichiometric ratios x CE), default emission factor (volume x EF), engineering estimate (mass balance), and direct measurement (CEMS) methods. Supports combustion efficiency modeling, multi-gas GWP weighting for CO2/CH4/N2O/BLACK_CARBON with AR4/AR5/AR6 values, pilot/purge gas CO2e accounting, uncertainty quantification via Monte Carlo, and data quality scoring.',
    '{"flare_id", "event_id", "calculation_method", "gas_volume_scf", "composition_id", "combustion_efficiency", "gwp_source"}',
    '{"calculation_id", "total_co2e_kg", "per_gas_breakdown", "pilot_co2e_kg", "purge_co2e_kg", "uncertainty_percent", "data_quality_score", "provenance_hash"}',
    '{"methods": ["GAS_COMPOSITION", "DEFAULT_EF", "ENGINEERING_ESTIMATE", "DIRECT_MEASUREMENT"], "gwp_sources": ["AR4", "AR5", "AR6"], "gases": ["CO2", "CH4", "N2O", "BLACK_CARBON"], "zero_hallucination": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'combustion_efficiency_management',
    'processing',
    'Record and manage combustion efficiency test results with base CE (default 98% EPA/IPCC, 99% enclosed), adjusted CE accounting for wind speed (degradation above 10 m/s), tip velocity (optimal Mach 0.2-0.5), LHV threshold (instability below 200 BTU/scf), steam-to-gas ratio (optimal 0.3-0.5), air-to-gas ratio, and compound-specific destruction and removal efficiency (DRE 98-99.9%).',
    '{"flare_id", "test_date", "base_ce", "wind_speed_ms", "tip_velocity_mach", "lhv_btu_scf", "steam_ratio", "air_ratio", "dre", "test_method"}',
    '{"efficiency_id", "adjusted_ce", "test_result"}',
    '{"default_ce_elevated": 0.98, "default_ce_enclosed": 0.99, "wind_threshold_ms": 10, "optimal_mach_range": [0.2, 0.5], "min_lhv_btu_scf": 200}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'pilot_purge_tracking',
    'processing',
    'Track pilot and purge gas consumption with period-based recording, pilot gas volume and composition reference, purge gas volume and type (NATURAL_GAS/NITROGEN/CO2/INERT), pilot operating hours, active pilot count, and calculated pilot/purge CO2e emissions. N2 purge = zero emissions; natural gas purge = CH4 + CO2 emissions.',
    '{"flare_id", "record_period_start", "record_period_end", "pilot_gas_volume_scf", "pilot_gas_composition_id", "purge_gas_volume_scf", "purge_gas_type", "pilot_operating_hours", "num_pilots_active"}',
    '{"record_id", "pilot_co2e_kg", "purge_co2e_kg"}',
    '{"purge_gas_types": ["NATURAL_GAS", "NITROGEN", "CO2", "INERT"], "n2_purge_zero_emissions": true}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'compliance_checking',
    'reporting',
    'Check regulatory compliance of flaring emission calculations against 8 frameworks: GHG Protocol Corporate Standard Ch.5, EPA 40 CFR Part 98 Subpart W Sec. W.23, CSRD/ESRS E1, ISO 14064-1, EU ETS MRR, EU Methane Regulation 2024/1787, World Bank Zero Routine Flaring 2030, and OGMP 2.0. Produce compliance status with requirements checked/met counts, findings, and recommendations.',
    '{"calculation_id", "framework"}',
    '{"compliance_id", "compliance_status", "requirements_checked", "requirements_met", "findings", "recommendations"}',
    '{"frameworks": ["GHG_PROTOCOL", "EPA_SUBPART_W", "CSRD_ESRS", "ISO_14064", "EU_ETS_MRR", "EU_METHANE_REG", "WORLD_BANK_ZRF", "OGMP_2_0"], "statuses": ["COMPLIANT", "NON_COMPLIANT", "PENDING", "UNDER_REVIEW", "EXEMPT"]}'::jsonb
),
(
    'GL-MRV-SCOPE1-006', '1.0.0',
    'audit_trail',
    'reporting',
    'Generate comprehensive entity-level audit trail entries with step-by-step action tracking, input/output/prev SHA-256 hash chaining for tamper-evident provenance, formula text recording, parameters/result JSONB payloads, and actor attribution.',
    '{"entity_type", "entity_id"}',
    '{"audit_entries", "hash_chain", "total_actions"}',
    '{"hash_algorithm": "SHA-256", "tracks_every_action": true, "supports_hash_chaining": true, "records_formula": true, "actions": ["CREATE", "UPDATE", "DELETE", "CALCULATE", "AGGREGATE", "VALIDATE", "APPROVE", "REJECT", "IMPORT", "EXPORT"]}'::jsonb
)
ON CONFLICT DO NOTHING;

INSERT INTO agent_registry_service.agent_dependencies (
    agent_id, depends_on_agent_id, version_constraint, optional, reason
) VALUES
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-001', '>=1.0.0', false, 'DAG orchestration for multi-stage flaring emission calculation pipeline execution ordering'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-003', '>=1.0.0', false, 'Unit and reference normalization for gas volume conversions (scf/Nm3), heating value units (BTU/GJ), and flow rate alignment'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-005', '>=1.0.0', false, 'Citations and evidence tracking for emission factor sources, methodology references, and regulatory citations'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-008', '>=1.0.0', false, 'Reproducibility verification for deterministic calculation result hashing and drift detection'),
    ('GL-MRV-SCOPE1-006', 'GL-FOUND-X-010', '>=1.0.0', false, 'Observability metrics for calculation events, flaring events, and compliance event telemetry'),
    ('GL-MRV-SCOPE1-006', 'GL-DATA-Q-010',  '>=1.0.0', true,  'Data Quality Profiler for input data validation and quality scoring of gas compositions and flow rate measurements'),
    ('GL-MRV-SCOPE1-006', 'GL-MRV-X-001',   '>=1.0.0', true,  'Stationary Combustion Calculator for shared Scope 1 infrastructure, fuel heating values, and emission factor database'),
    ('GL-MRV-SCOPE1-006', 'GL-MRV-SCOPE1-005', '>=1.0.0', true, 'Fugitive Emissions Agent for shared gas system references, venting vs flaring classification, and cross-referencing leak/flare data')
ON CONFLICT (agent_id, depends_on_agent_id) DO NOTHING;

INSERT INTO agent_registry_service.service_catalog (
    agent_id, display_name, summary, category, status, tenant_id
) VALUES (
    'GL-MRV-SCOPE1-006',
    'Flaring Agent',
    'Flaring emission calculator. Flare system registry (8 types: ELEVATED_STEAM_ASSISTED/ELEVATED_AIR_ASSISTED/ELEVATED_UNASSISTED/ENCLOSED_GROUND/MULTI_POINT_GROUND/OFFSHORE_MARINE/CANDLESTICK/LOW_PRESSURE, capacity, assist STEAM/AIR/NONE, pilot/purge config). Gas composition analysis (15 components CH4/C2H6/C3H8/n-C4H10/i-C4H10/C5H12/C6+/CO2/N2/H2S/H2/CO/C2H4/C3H6/H2O, trigger-computed total fraction, HHV/LHV, specific gravity, molecular weight). Emission factor database (CO2/CH4/N2O/BLACK_CARBON x flare_type, EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM). Flaring events (6 categories: ROUTINE/NON_ROUTINE/EMERGENCY/MAINTENANCE/PILOT_PURGE/WELL_COMPLETION, volumes scf/Nm3, flow rates). Emission calculations (GAS_COMPOSITION/DEFAULT_EF/ENGINEERING_ESTIMATE/DIRECT_MEASUREMENT, combustion efficiency, multi-gas CO2/CH4/N2O/BLACK_CARBON GWP AR4/AR5/AR6, pilot/purge CO2e). Per-gas breakdowns. CE tests (base/adjusted, wind, tip velocity, LHV, steam/air, DRE). Pilot/purge tracking. Compliance checks (8 frameworks: GHG Protocol/EPA Subpart W/CSRD-ESRS/ISO 14064/EU ETS MRR/EU Methane Reg/World Bank ZRF/OGMP 2.0). Audit trail with input/output/prev hash chaining. SHA-256 provenance chains.',
    'mrv', 'active', 'default'
) ON CONFLICT (agent_id, tenant_id) DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA flaring_service IS
    'Flaring Agent (AGENT-MRV-006) - flare system registry (8 types), gas composition analysis (15 components), emission factor database, flaring event tracking (6 categories), emission calculations (4 methods), per-gas breakdowns, combustion efficiency testing, pilot/purge gas tracking, compliance records (8 frameworks), audit trail, provenance chains';

COMMENT ON TABLE flaring_service.fl_flare_systems IS
    'Flare system registry: tenant_id, name, flare_type (ELEVATED_STEAM_ASSISTED/ELEVATED_AIR_ASSISTED/ELEVATED_UNASSISTED/ENCLOSED_GROUND/MULTI_POINT_GROUND/OFFSHORE_MARINE/CANDLESTICK/LOW_PRESSURE), status (ACTIVE/INACTIVE/DECOMMISSIONED/MAINTENANCE/STANDBY), facility_id, location_description, tip_diameter_inches, height_feet, capacity_mmbtu_hr, assist_type (STEAM/AIR/NONE), num_pilots, pilot_gas_flow_mmbtu_hr, purge_gas_flow_scfh, purge_gas_type (NATURAL_GAS/NITROGEN/CO2/INERT), installation_date, last_inspection_date, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_gas_compositions IS
    'Gas composition analysis records: tenant_id, name, 15-component mole fractions (ch4/c2h6/c3h8/n_c4h10/i_c4h10/c5h12/c6_plus/co2/n2/h2s/h2/co/c2h4/c3h6/h2o, all DECIMAL(10,8) range 0-1), total_fraction (trigger-computed sum), hhv_btu_scf, lhv_btu_scf, specific_gravity, molecular_weight, analysis_date, lab_reference, source (LAB_ANALYSIS/FIELD_MEASUREMENT/DEFAULT/ENGINEERING_ESTIMATE/HISTORICAL), metadata JSONB';

COMMENT ON TABLE flaring_service.fl_emission_factors IS
    'Emission factor database: tenant_id, gas (CO2/CH4/N2O/BLACK_CARBON), flare_type (nullable, 8 types), source (EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM), factor_value DECIMAL(18,10), factor_unit (kg/MMBtu, kg/GJ, kg/scf, kg/Nm3), year, region, description, reference, is_active, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_flaring_events IS
    'Flaring event records: tenant_id, flare_id (FK), event_category (ROUTINE/NON_ROUTINE/EMERGENCY/MAINTENANCE/PILOT_PURGE/WELL_COMPLETION), start_time, end_time, duration_hours, gas_volume_scf, gas_volume_nm3, composition_id (FK), measured_flow_rate_scfh, flow_measurement_method (ULTRASONIC/ORIFICE_PLATE/VORTEX/THERMAL_MASS/PITOT_TUBE/CORIOLIS/ESTIMATED), is_estimated, estimation_method, notes, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_calculations IS
    'Calculation results: tenant_id, flare_id (FK), event_id (FK), calculation_method (GAS_COMPOSITION/DEFAULT_EF/ENGINEERING_ESTIMATE/DIRECT_MEASUREMENT), gas_volume_scf, composition_id (FK), combustion_efficiency (0-1), gwp_source (AR4/AR5/AR6), total_co2e_kg, co2_kg, ch4_kg, n2o_kg, black_carbon_kg, pilot_co2e_kg, purge_co2e_kg, uncertainty_percent, data_quality_score (0-5), provenance_hash (SHA-256), status (COMPLETED/PENDING/FAILED/DRAFT/REVIEWED/APPROVED), calculated_at, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_calculation_details IS
    'Per-gas calculation breakdown: tenant_id, calculation_id (FK CASCADE), gas (CO2/CH4/N2O/BLACK_CARBON), emission_kg, co2e_kg, gwp_value, emission_factor_value, emission_factor_source (EPA/IPCC/API/DEFRA/EU_ETS/CUSTOM), emission_factor_unit, notes';

COMMENT ON TABLE flaring_service.fl_combustion_efficiency IS
    'Combustion efficiency test records: tenant_id, flare_id (FK), test_date, base_ce (0-1), adjusted_ce (0-1), wind_speed_ms, tip_velocity_mach, lhv_btu_scf, steam_ratio, air_ratio, dre (0-1), test_method, test_reference, notes, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_pilot_purge_records IS
    'Pilot and purge gas tracking: tenant_id, flare_id (FK), record_period_start/end, pilot_gas_volume_scf, pilot_gas_composition_id (FK), purge_gas_volume_scf, purge_gas_type (NATURAL_GAS/NITROGEN/CO2/INERT), pilot_operating_hours, num_pilots_active, pilot_co2e_kg, purge_co2e_kg, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_compliance_records IS
    'Regulatory compliance records: tenant_id, calculation_id (FK), framework (GHG_PROTOCOL/EPA_SUBPART_W/CSRD_ESRS/ISO_14064/EU_ETS_MRR/EU_METHANE_REG/WORLD_BANK_ZRF/OGMP_2_0), compliance_status (COMPLIANT/NON_COMPLIANT/PENDING/UNDER_REVIEW/EXEMPT), requirements_checked, requirements_met, findings JSONB, recommendations JSONB, checked_at, metadata JSONB';

COMMENT ON TABLE flaring_service.fl_audit_entries IS
    'Audit trail entries: tenant_id, entity_type, entity_id (UUID), action (CREATE/UPDATE/DELETE/CALCULATE/AGGREGATE/VALIDATE/APPROVE/REJECT/IMPORT/EXPORT), step_name, input_hash, output_hash, prev_hash (SHA-256 chain), formula, parameters JSONB, result JSONB, actor_id';

COMMENT ON TABLE flaring_service.fl_calculation_events IS
    'TimescaleDB hypertable: calculation events with tenant_id, event_type, flare_type, method, emissions_kg_co2e, duration_ms, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE flaring_service.fl_flaring_event_ts IS
    'TimescaleDB hypertable: flaring events with tenant_id, event_type, event_category, flare_type, gas_volume_scf, duration_hours, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON TABLE flaring_service.fl_compliance_events IS
    'TimescaleDB hypertable: compliance events with tenant_id, event_type, framework, status, check_count, pass_count, fail_count, metadata JSONB (7-day chunks, 365-day retention, 30-day compression)';

COMMENT ON MATERIALIZED VIEW flaring_service.fl_hourly_calculation_stats IS
    'Continuous aggregate: hourly calculation stats by flare_type and method (total calculations, sum emissions kg CO2e, avg emissions kg CO2e, avg duration ms, max duration ms per hour)';

COMMENT ON MATERIALIZED VIEW flaring_service.fl_daily_emission_totals IS
    'Continuous aggregate: daily emission totals by flare_type and event_category (total events, sum gas volume scf, avg duration hours per day)';
