-- =============================================================================
-- V316: PACK-040 M&V Pack - Projects, ECM Registry, Measurement Boundaries
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the pack040_mv schema and foundational tables for M&V project
-- management. Tracks M&V projects, energy conservation measures (ECMs),
-- measurement boundaries, project-ECM mappings, and project configuration.
-- These tables form the core entity model for all downstream M&V operations
-- including baseline development, savings verification, and reporting.
--
-- Tables (5):
--   1. pack040_mv.mv_projects
--   2. pack040_mv.mv_ecms
--   3. pack040_mv.mv_measurement_boundaries
--   4. pack040_mv.mv_project_ecm_map
--   5. pack040_mv.mv_project_config
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V315__pack039_energy_monitoring_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack040_mv;

SET search_path TO pack040_mv, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack040_mv.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack040_mv.mv_projects
-- =============================================================================
-- Central registry of M&V projects. Each project represents a complete M&V
-- engagement for a facility or group of ECMs. Projects track lifecycle status
-- from planning through long-term persistence monitoring. A project defines
-- the baseline period, reporting period, IPMVP option, and overall M&V plan.

CREATE TABLE pack040_mv.mv_projects (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    facility_id                 UUID            NOT NULL,
    project_name                VARCHAR(255)    NOT NULL,
    project_code                VARCHAR(50),
    project_description         TEXT,
    project_type                VARCHAR(50)     NOT NULL DEFAULT 'STANDARD_MV',
    project_status              VARCHAR(30)     NOT NULL DEFAULT 'PLANNING',
    ipmvp_option                VARCHAR(10),
    compliance_framework        VARCHAR(50)     NOT NULL DEFAULT 'IPMVP',
    baseline_period_start       DATE,
    baseline_period_end         DATE,
    reporting_period_start      DATE,
    reporting_period_end        DATE,
    installation_date           DATE,
    commissioning_date          DATE,
    contract_start_date         DATE,
    contract_end_date           DATE,
    contract_term_years         INTEGER,
    guaranteed_savings_kwh      NUMERIC(18,3),
    guaranteed_savings_pct      NUMERIC(7,4),
    guaranteed_cost_savings     NUMERIC(18,2),
    currency_code               VARCHAR(3)      NOT NULL DEFAULT 'USD',
    energy_type                 VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    measurement_unit            VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    data_granularity            VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    weather_station_id          VARCHAR(50),
    weather_normalization       BOOLEAN         NOT NULL DEFAULT true,
    production_normalization    BOOLEAN         NOT NULL DEFAULT false,
    occupancy_normalization     BOOLEAN         NOT NULL DEFAULT false,
    mv_practitioner             VARCHAR(255),
    mv_practitioner_cert        VARCHAR(100),
    client_name                 VARCHAR(255),
    esco_name                   VARCHAR(255),
    performance_contract_id     VARCHAR(100),
    total_project_cost          NUMERIC(18,2),
    expected_payback_years      NUMERIC(5,2),
    confidence_level_pct        NUMERIC(5,2)    DEFAULT 90.0,
    precision_pct               NUMERIC(5,2)    DEFAULT 10.0,
    notes                       TEXT,
    tags                        JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p040_prj_type CHECK (
        project_type IN (
            'STANDARD_MV', 'PERFORMANCE_CONTRACT', 'ESCO_EPC',
            'UTILITY_PROGRAM', 'VOLUNTARY', 'REGULATORY',
            'COMMISSIONING', 'RETRO_COMMISSIONING', 'PORTFOLIO',
            'DEMONSTRATION', 'PILOT'
        )
    ),
    CONSTRAINT chk_p040_prj_status CHECK (
        project_status IN (
            'PLANNING', 'BASELINE_DEVELOPMENT', 'PRE_INSTALLATION',
            'INSTALLATION', 'POST_INSTALLATION', 'REPORTING',
            'ONGOING_MV', 'PERSISTENCE_TRACKING', 'COMPLETED',
            'SUSPENDED', 'CANCELLED', 'ARCHIVED'
        )
    ),
    CONSTRAINT chk_p040_prj_option CHECK (
        ipmvp_option IS NULL OR ipmvp_option IN (
            'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D', 'MIXED'
        )
    ),
    CONSTRAINT chk_p040_prj_framework CHECK (
        compliance_framework IN (
            'IPMVP', 'ASHRAE_14', 'ISO_50015', 'ISO_50001',
            'FEMP_4_0', 'EU_EED', 'EU_EPC', 'BPA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_prj_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'COMPRESSED_AIR', 'DIESEL', 'PROPANE',
            'FUEL_OIL', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
            'MIXED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_prj_unit CHECK (
        measurement_unit IN (
            'kWh', 'MWh', 'GJ', 'therms', 'CCF', 'MCF',
            'MMBtu', 'kW', 'MW', 'tonnes_steam', 'ton_hours',
            'gallons', 'liters', 'Nm3', 'kg', 'BTU'
        )
    ),
    CONSTRAINT chk_p040_prj_granularity CHECK (
        data_granularity IN (
            'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'BILLING_PERIOD'
        )
    ),
    CONSTRAINT chk_p040_prj_baseline_dates CHECK (
        baseline_period_start IS NULL OR baseline_period_end IS NULL OR
        baseline_period_start < baseline_period_end
    ),
    CONSTRAINT chk_p040_prj_reporting_dates CHECK (
        reporting_period_start IS NULL OR reporting_period_end IS NULL OR
        reporting_period_start < reporting_period_end
    ),
    CONSTRAINT chk_p040_prj_contract_dates CHECK (
        contract_start_date IS NULL OR contract_end_date IS NULL OR
        contract_start_date <= contract_end_date
    ),
    CONSTRAINT chk_p040_prj_contract_term CHECK (
        contract_term_years IS NULL OR (contract_term_years >= 1 AND contract_term_years <= 30)
    ),
    CONSTRAINT chk_p040_prj_confidence CHECK (
        confidence_level_pct IS NULL OR (confidence_level_pct >= 50 AND confidence_level_pct <= 99)
    ),
    CONSTRAINT chk_p040_prj_precision CHECK (
        precision_pct IS NULL OR (precision_pct > 0 AND precision_pct <= 50)
    ),
    CONSTRAINT chk_p040_prj_guaranteed_pct CHECK (
        guaranteed_savings_pct IS NULL OR (guaranteed_savings_pct >= 0 AND guaranteed_savings_pct <= 100)
    ),
    CONSTRAINT chk_p040_prj_payback CHECK (
        expected_payback_years IS NULL OR expected_payback_years > 0
    ),
    CONSTRAINT uq_p040_prj_tenant_code UNIQUE (tenant_id, project_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_prj_tenant           ON pack040_mv.mv_projects(tenant_id);
CREATE INDEX idx_p040_prj_facility         ON pack040_mv.mv_projects(facility_id);
CREATE INDEX idx_p040_prj_code             ON pack040_mv.mv_projects(project_code);
CREATE INDEX idx_p040_prj_type             ON pack040_mv.mv_projects(project_type);
CREATE INDEX idx_p040_prj_status           ON pack040_mv.mv_projects(project_status);
CREATE INDEX idx_p040_prj_option           ON pack040_mv.mv_projects(ipmvp_option);
CREATE INDEX idx_p040_prj_framework        ON pack040_mv.mv_projects(compliance_framework);
CREATE INDEX idx_p040_prj_energy_type      ON pack040_mv.mv_projects(energy_type);
CREATE INDEX idx_p040_prj_baseline_start   ON pack040_mv.mv_projects(baseline_period_start);
CREATE INDEX idx_p040_prj_reporting_start  ON pack040_mv.mv_projects(reporting_period_start);
CREATE INDEX idx_p040_prj_created          ON pack040_mv.mv_projects(created_at DESC);
CREATE INDEX idx_p040_prj_metadata         ON pack040_mv.mv_projects USING GIN(metadata);
CREATE INDEX idx_p040_prj_tags             ON pack040_mv.mv_projects USING GIN(tags);

-- Composite: tenant + active projects
CREATE INDEX idx_p040_prj_tenant_active    ON pack040_mv.mv_projects(tenant_id, facility_id)
    WHERE project_status NOT IN ('COMPLETED', 'CANCELLED', 'ARCHIVED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_prj_updated
    BEFORE UPDATE ON pack040_mv.mv_projects
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_ecms
-- =============================================================================
-- Registry of Energy Conservation Measures (ECMs) implemented as part of
-- M&V projects. Each ECM describes a specific energy efficiency improvement
-- with its expected savings, cost, and technical characteristics. ECMs are
-- the fundamental savings-generating units that M&V quantifies and verifies.

CREATE TABLE pack040_mv.mv_ecms (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    facility_id                 UUID            NOT NULL,
    ecm_name                    VARCHAR(255)    NOT NULL,
    ecm_code                    VARCHAR(50),
    ecm_description             TEXT,
    ecm_category                VARCHAR(50)     NOT NULL DEFAULT 'HVAC',
    ecm_subcategory             VARCHAR(100),
    ecm_status                  VARCHAR(30)     NOT NULL DEFAULT 'PROPOSED',
    implementation_date         DATE,
    commissioning_date          DATE,
    useful_life_years           INTEGER,
    affected_energy_type        VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    affected_end_use            VARCHAR(50)     NOT NULL DEFAULT 'HVAC',
    estimated_annual_savings_kwh NUMERIC(18,3),
    estimated_annual_savings_pct NUMERIC(7,4),
    estimated_demand_savings_kw  NUMERIC(12,3),
    estimated_annual_cost_savings NUMERIC(18,2),
    implementation_cost         NUMERIC(18,2),
    simple_payback_years        NUMERIC(6,2),
    npv                         NUMERIC(18,2),
    irr_pct                     NUMERIC(7,4),
    incentive_amount            NUMERIC(18,2),
    equipment_manufacturer      VARCHAR(255),
    equipment_model             VARCHAR(255),
    equipment_specs             JSONB           DEFAULT '{}',
    baseline_equipment          TEXT,
    retrofit_equipment          TEXT,
    interactive_effects         BOOLEAN         NOT NULL DEFAULT false,
    interactive_effect_description TEXT,
    ipmvp_option_recommended    VARCHAR(10),
    measurement_boundary        VARCHAR(50)     DEFAULT 'RETROFIT_ISOLATION',
    requires_sub_metering       BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    tags                        JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p040_ecm_category CHECK (
        ecm_category IN (
            'HVAC', 'LIGHTING', 'ENVELOPE', 'CONTROLS', 'MOTORS',
            'VFD', 'COMPRESSED_AIR', 'STEAM', 'BOILER', 'CHILLER',
            'PUMP', 'FAN', 'REFRIGERATION', 'PROCESS', 'PLUG_LOAD',
            'RENEWABLE', 'COGENERATION', 'ENERGY_STORAGE', 'WATER',
            'COMMISSIONING', 'RETRO_COMMISSIONING', 'BEHAVIORAL',
            'BUILDING_AUTOMATION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_ecm_status CHECK (
        ecm_status IN (
            'PROPOSED', 'APPROVED', 'DESIGN', 'PROCUREMENT',
            'INSTALLATION', 'COMMISSIONED', 'OPERATIONAL',
            'VERIFIED', 'DECOMMISSIONED', 'REJECTED', 'DEFERRED'
        )
    ),
    CONSTRAINT chk_p040_ecm_energy_type CHECK (
        affected_energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'COMPRESSED_AIR', 'DIESEL', 'PROPANE',
            'FUEL_OIL', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
            'MIXED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_ecm_end_use CHECK (
        affected_end_use IN (
            'HVAC', 'LIGHTING', 'PLUG_LOAD', 'PROCESS', 'DOMESTIC_HW',
            'REFRIGERATION', 'COOKING', 'LAUNDRY', 'ELEVATOR',
            'IT_EQUIPMENT', 'COMPRESSED_AIR', 'PUMPING', 'VENTILATION',
            'WHOLE_BUILDING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_ecm_option CHECK (
        ipmvp_option_recommended IS NULL OR ipmvp_option_recommended IN (
            'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D'
        )
    ),
    CONSTRAINT chk_p040_ecm_boundary CHECK (
        measurement_boundary IS NULL OR measurement_boundary IN (
            'RETROFIT_ISOLATION', 'WHOLE_FACILITY', 'SYSTEM_LEVEL',
            'COMPONENT_LEVEL', 'BUILDING_LEVEL', 'CAMPUS_LEVEL'
        )
    ),
    CONSTRAINT chk_p040_ecm_useful_life CHECK (
        useful_life_years IS NULL OR (useful_life_years >= 1 AND useful_life_years <= 50)
    ),
    CONSTRAINT chk_p040_ecm_savings_pct CHECK (
        estimated_annual_savings_pct IS NULL OR
        (estimated_annual_savings_pct >= 0 AND estimated_annual_savings_pct <= 100)
    ),
    CONSTRAINT chk_p040_ecm_payback CHECK (
        simple_payback_years IS NULL OR simple_payback_years >= 0
    ),
    CONSTRAINT uq_p040_ecm_tenant_code UNIQUE (tenant_id, ecm_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ecm_tenant           ON pack040_mv.mv_ecms(tenant_id);
CREATE INDEX idx_p040_ecm_facility         ON pack040_mv.mv_ecms(facility_id);
CREATE INDEX idx_p040_ecm_code             ON pack040_mv.mv_ecms(ecm_code);
CREATE INDEX idx_p040_ecm_category         ON pack040_mv.mv_ecms(ecm_category);
CREATE INDEX idx_p040_ecm_status           ON pack040_mv.mv_ecms(ecm_status);
CREATE INDEX idx_p040_ecm_energy_type      ON pack040_mv.mv_ecms(affected_energy_type);
CREATE INDEX idx_p040_ecm_end_use          ON pack040_mv.mv_ecms(affected_end_use);
CREATE INDEX idx_p040_ecm_option           ON pack040_mv.mv_ecms(ipmvp_option_recommended);
CREATE INDEX idx_p040_ecm_impl_date        ON pack040_mv.mv_ecms(implementation_date);
CREATE INDEX idx_p040_ecm_created          ON pack040_mv.mv_ecms(created_at DESC);
CREATE INDEX idx_p040_ecm_metadata         ON pack040_mv.mv_ecms USING GIN(metadata);
CREATE INDEX idx_p040_ecm_tags             ON pack040_mv.mv_ecms USING GIN(tags);

-- Composite: tenant + operational ECMs
CREATE INDEX idx_p040_ecm_tenant_oper      ON pack040_mv.mv_ecms(tenant_id, facility_id)
    WHERE ecm_status IN ('OPERATIONAL', 'VERIFIED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ecm_updated
    BEFORE UPDATE ON pack040_mv.mv_ecms
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_measurement_boundaries
-- =============================================================================
-- Defines the measurement boundary for each M&V project or ECM per IPMVP.
-- The measurement boundary determines which energy flows are measured and
-- which are excluded from savings calculations. Boundaries can be at the
-- retrofit isolation level (Options A/B) or whole facility level (Option C).

CREATE TABLE pack040_mv.mv_measurement_boundaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    boundary_name               VARCHAR(255)    NOT NULL,
    boundary_type               VARCHAR(50)     NOT NULL DEFAULT 'RETROFIT_ISOLATION',
    boundary_description        TEXT,
    ipmvp_option                VARCHAR(10)     NOT NULL DEFAULT 'OPTION_C',
    included_systems            JSONB           DEFAULT '[]',
    excluded_systems            JSONB           DEFAULT '[]',
    included_meters             UUID[]          DEFAULT '{}',
    excluded_meters             UUID[]          DEFAULT '{}',
    energy_types_included       VARCHAR(50)[]   DEFAULT '{ELECTRICITY}',
    affected_floor_area_m2      NUMERIC(12,2),
    total_facility_area_m2      NUMERIC(12,2),
    affected_area_pct           NUMERIC(7,4),
    baseline_annual_energy_kwh  NUMERIC(18,3),
    ecm_share_of_facility_pct   NUMERIC(7,4),
    independent_variables       JSONB           DEFAULT '[]',
    static_factors              JSONB           DEFAULT '[]',
    measurement_points          JSONB           DEFAULT '[]',
    boundary_diagram_ref        VARCHAR(255),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_mb_type CHECK (
        boundary_type IN (
            'RETROFIT_ISOLATION', 'WHOLE_FACILITY', 'SYSTEM_LEVEL',
            'COMPONENT_LEVEL', 'BUILDING_LEVEL', 'CAMPUS_LEVEL',
            'SIMULATED'
        )
    ),
    CONSTRAINT chk_p040_mb_option CHECK (
        ipmvp_option IN (
            'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D'
        )
    ),
    CONSTRAINT chk_p040_mb_area CHECK (
        affected_floor_area_m2 IS NULL OR affected_floor_area_m2 > 0
    ),
    CONSTRAINT chk_p040_mb_total_area CHECK (
        total_facility_area_m2 IS NULL OR total_facility_area_m2 > 0
    ),
    CONSTRAINT chk_p040_mb_area_pct CHECK (
        affected_area_pct IS NULL OR
        (affected_area_pct >= 0 AND affected_area_pct <= 100)
    ),
    CONSTRAINT chk_p040_mb_ecm_share CHECK (
        ecm_share_of_facility_pct IS NULL OR
        (ecm_share_of_facility_pct >= 0 AND ecm_share_of_facility_pct <= 100)
    ),
    CONSTRAINT chk_p040_mb_baseline_energy CHECK (
        baseline_annual_energy_kwh IS NULL OR baseline_annual_energy_kwh >= 0
    ),
    CONSTRAINT uq_p040_mb_project_name UNIQUE (project_id, boundary_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_mb_tenant            ON pack040_mv.mv_measurement_boundaries(tenant_id);
CREATE INDEX idx_p040_mb_project           ON pack040_mv.mv_measurement_boundaries(project_id);
CREATE INDEX idx_p040_mb_ecm               ON pack040_mv.mv_measurement_boundaries(ecm_id);
CREATE INDEX idx_p040_mb_type              ON pack040_mv.mv_measurement_boundaries(boundary_type);
CREATE INDEX idx_p040_mb_option            ON pack040_mv.mv_measurement_boundaries(ipmvp_option);
CREATE INDEX idx_p040_mb_active            ON pack040_mv.mv_measurement_boundaries(is_active) WHERE is_active = true;
CREATE INDEX idx_p040_mb_created           ON pack040_mv.mv_measurement_boundaries(created_at DESC);
CREATE INDEX idx_p040_mb_meters            ON pack040_mv.mv_measurement_boundaries USING GIN(included_meters);

-- Composite: project + active boundaries
CREATE INDEX idx_p040_mb_project_active    ON pack040_mv.mv_measurement_boundaries(project_id, boundary_type)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_mb_updated
    BEFORE UPDATE ON pack040_mv.mv_measurement_boundaries
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_project_ecm_map
-- =============================================================================
-- Many-to-many mapping between M&V projects and ECMs. A project may include
-- multiple ECMs, and an ECM may participate in multiple M&V projects (e.g.,
-- re-verification). Tracks the IPMVP option assigned per ECM within a project
-- and the associated measurement boundary.

CREATE TABLE pack040_mv.mv_project_ecm_map (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            NOT NULL REFERENCES pack040_mv.mv_ecms(id) ON DELETE CASCADE,
    boundary_id                 UUID            REFERENCES pack040_mv.mv_measurement_boundaries(id) ON DELETE SET NULL,
    ipmvp_option                VARCHAR(10)     NOT NULL DEFAULT 'OPTION_C',
    ecm_priority                INTEGER         NOT NULL DEFAULT 1,
    ecm_weight_pct              NUMERIC(7,4)    DEFAULT 100.0,
    is_interactive              BOOLEAN         NOT NULL DEFAULT false,
    interactive_with_ecm_id     UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    interactive_effect_pct      NUMERIC(7,4),
    verification_status         VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    verified_savings_kwh        NUMERIC(18,3),
    verified_savings_pct        NUMERIC(7,4),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_pem_option CHECK (
        ipmvp_option IN (
            'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D'
        )
    ),
    CONSTRAINT chk_p040_pem_priority CHECK (
        ecm_priority >= 1 AND ecm_priority <= 100
    ),
    CONSTRAINT chk_p040_pem_weight CHECK (
        ecm_weight_pct IS NULL OR
        (ecm_weight_pct >= 0 AND ecm_weight_pct <= 100)
    ),
    CONSTRAINT chk_p040_pem_interactive_pct CHECK (
        interactive_effect_pct IS NULL OR
        (interactive_effect_pct >= -100 AND interactive_effect_pct <= 100)
    ),
    CONSTRAINT chk_p040_pem_verif_status CHECK (
        verification_status IN (
            'PENDING', 'IN_PROGRESS', 'VERIFIED', 'PARTIALLY_VERIFIED',
            'DISPUTED', 'FAILED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p040_pem_verified_pct CHECK (
        verified_savings_pct IS NULL OR
        (verified_savings_pct >= -100 AND verified_savings_pct <= 200)
    ),
    CONSTRAINT chk_p040_pem_no_self_interact CHECK (
        interactive_with_ecm_id IS NULL OR interactive_with_ecm_id != ecm_id
    ),
    CONSTRAINT uq_p040_pem_project_ecm UNIQUE (project_id, ecm_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_pem_tenant           ON pack040_mv.mv_project_ecm_map(tenant_id);
CREATE INDEX idx_p040_pem_project          ON pack040_mv.mv_project_ecm_map(project_id);
CREATE INDEX idx_p040_pem_ecm              ON pack040_mv.mv_project_ecm_map(ecm_id);
CREATE INDEX idx_p040_pem_boundary         ON pack040_mv.mv_project_ecm_map(boundary_id);
CREATE INDEX idx_p040_pem_option           ON pack040_mv.mv_project_ecm_map(ipmvp_option);
CREATE INDEX idx_p040_pem_verif            ON pack040_mv.mv_project_ecm_map(verification_status);
CREATE INDEX idx_p040_pem_interactive      ON pack040_mv.mv_project_ecm_map(interactive_with_ecm_id);
CREATE INDEX idx_p040_pem_created          ON pack040_mv.mv_project_ecm_map(created_at DESC);

-- Composite: project + verified ECMs
CREATE INDEX idx_p040_pem_project_verif    ON pack040_mv.mv_project_ecm_map(project_id, ecm_id)
    WHERE verification_status = 'VERIFIED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_pem_updated
    BEFORE UPDATE ON pack040_mv.mv_project_ecm_map
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_project_config
-- =============================================================================
-- Project-level configuration parameters that control M&V engine behavior.
-- Stores ASHRAE 14 validation thresholds, regression settings, uncertainty
-- parameters, reporting preferences, and integration configuration. Each
-- project may have multiple configuration versions for audit tracking.

CREATE TABLE pack040_mv.mv_project_config (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    config_version              INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    -- ASHRAE 14 criteria
    max_cvrmse_monthly_pct      NUMERIC(6,2)    NOT NULL DEFAULT 25.0,
    max_cvrmse_daily_pct        NUMERIC(6,2)    NOT NULL DEFAULT 30.0,
    max_cvrmse_hourly_pct       NUMERIC(6,2)    NOT NULL DEFAULT 30.0,
    max_nmbe_monthly_pct        NUMERIC(6,2)    NOT NULL DEFAULT 5.0,
    max_nmbe_daily_pct          NUMERIC(6,2)    NOT NULL DEFAULT 10.0,
    max_nmbe_hourly_pct         NUMERIC(6,2)    NOT NULL DEFAULT 10.0,
    min_r_squared_monthly       NUMERIC(5,4)    NOT NULL DEFAULT 0.70,
    min_r_squared_daily         NUMERIC(5,4)    NOT NULL DEFAULT 0.50,
    -- Regression settings
    preferred_model_type        VARCHAR(30)     NOT NULL DEFAULT 'AUTO_SELECT',
    max_independent_vars        INTEGER         NOT NULL DEFAULT 5,
    auto_balance_point          BOOLEAN         NOT NULL DEFAULT true,
    balance_point_heating_f     NUMERIC(6,2)    DEFAULT 65.0,
    balance_point_cooling_f     NUMERIC(6,2)    DEFAULT 65.0,
    min_data_points             INTEGER         NOT NULL DEFAULT 12,
    outlier_removal_method      VARCHAR(30)     NOT NULL DEFAULT 'COOKS_DISTANCE',
    outlier_threshold           NUMERIC(6,3)    NOT NULL DEFAULT 4.0,
    -- Uncertainty settings
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    max_fsu_pct                 NUMERIC(6,2)    NOT NULL DEFAULT 50.0,
    include_measurement_unc     BOOLEAN         NOT NULL DEFAULT true,
    include_sampling_unc        BOOLEAN         NOT NULL DEFAULT false,
    -- Persistence settings
    persistence_check_frequency VARCHAR(20)     NOT NULL DEFAULT 'QUARTERLY',
    degradation_alert_threshold NUMERIC(6,2)    NOT NULL DEFAULT 20.0,
    recom_trigger_factor        NUMERIC(5,3)    NOT NULL DEFAULT 0.80,
    -- Reporting settings
    default_report_format       VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    auto_report_generation      BOOLEAN         NOT NULL DEFAULT true,
    report_frequency            VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    include_residual_plots      BOOLEAN         NOT NULL DEFAULT true,
    include_cusum_chart         BOOLEAN         NOT NULL DEFAULT true,
    -- Integration
    weather_data_source         VARCHAR(50)     NOT NULL DEFAULT 'NOAA_ISD',
    utility_data_source         VARCHAR(50)     DEFAULT 'MANUAL',
    pack039_integration         BOOLEAN         NOT NULL DEFAULT false,
    pack031_integration         BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cfg_model_type CHECK (
        preferred_model_type IN (
            'OLS', '3P_COOLING', '3P_HEATING', '4P', '5P',
            'TOWT', 'MULTIVARIATE', 'AUTO_SELECT'
        )
    ),
    CONSTRAINT chk_p040_cfg_outlier_method CHECK (
        outlier_removal_method IN (
            'COOKS_DISTANCE', 'STUDENTIZED_RESIDUAL', 'LEVERAGE',
            'DFFITS', 'NONE'
        )
    ),
    CONSTRAINT chk_p040_cfg_persistence_freq CHECK (
        persistence_check_frequency IN (
            'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'ANNUALLY'
        )
    ),
    CONSTRAINT chk_p040_cfg_report_format CHECK (
        default_report_format IN ('PDF', 'HTML', 'MARKDOWN', 'JSON', 'EXCEL')
    ),
    CONSTRAINT chk_p040_cfg_report_freq CHECK (
        report_frequency IN (
            'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI_ANNUALLY', 'ANNUALLY'
        )
    ),
    CONSTRAINT chk_p040_cfg_weather_src CHECK (
        weather_data_source IN (
            'NOAA_ISD', 'NOAA_LCD', 'WEATHER_API', 'TMY3', 'ON_SITE',
            'PACK_039', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_cfg_cvrmse_monthly CHECK (
        max_cvrmse_monthly_pct > 0 AND max_cvrmse_monthly_pct <= 100
    ),
    CONSTRAINT chk_p040_cfg_cvrmse_daily CHECK (
        max_cvrmse_daily_pct > 0 AND max_cvrmse_daily_pct <= 100
    ),
    CONSTRAINT chk_p040_cfg_nmbe_monthly CHECK (
        max_nmbe_monthly_pct > 0 AND max_nmbe_monthly_pct <= 100
    ),
    CONSTRAINT chk_p040_cfg_r_squared CHECK (
        min_r_squared_monthly >= 0 AND min_r_squared_monthly <= 1
    ),
    CONSTRAINT chk_p040_cfg_max_vars CHECK (
        max_independent_vars >= 1 AND max_independent_vars <= 20
    ),
    CONSTRAINT chk_p040_cfg_min_points CHECK (
        min_data_points >= 6 AND min_data_points <= 365
    ),
    CONSTRAINT chk_p040_cfg_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99
    ),
    CONSTRAINT chk_p040_cfg_fsu CHECK (
        max_fsu_pct > 0 AND max_fsu_pct <= 200
    ),
    CONSTRAINT chk_p040_cfg_degrad CHECK (
        degradation_alert_threshold > 0 AND degradation_alert_threshold <= 100
    ),
    CONSTRAINT chk_p040_cfg_recom CHECK (
        recom_trigger_factor > 0 AND recom_trigger_factor <= 1
    ),
    CONSTRAINT chk_p040_cfg_version CHECK (
        config_version >= 1
    ),
    CONSTRAINT uq_p040_cfg_project_version UNIQUE (project_id, config_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cfg_tenant           ON pack040_mv.mv_project_config(tenant_id);
CREATE INDEX idx_p040_cfg_project          ON pack040_mv.mv_project_config(project_id);
CREATE INDEX idx_p040_cfg_current          ON pack040_mv.mv_project_config(is_current) WHERE is_current = true;
CREATE INDEX idx_p040_cfg_model_type       ON pack040_mv.mv_project_config(preferred_model_type);
CREATE INDEX idx_p040_cfg_created          ON pack040_mv.mv_project_config(created_at DESC);

-- Composite: project + current config
CREATE INDEX idx_p040_cfg_project_current  ON pack040_mv.mv_project_config(project_id, config_version DESC)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cfg_updated
    BEFORE UPDATE ON pack040_mv.mv_project_config
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_ecms ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_measurement_boundaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_project_ecm_map ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_project_config ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_prj_tenant_isolation
    ON pack040_mv.mv_projects
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_prj_service_bypass
    ON pack040_mv.mv_projects
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ecm_tenant_isolation
    ON pack040_mv.mv_ecms
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ecm_service_bypass
    ON pack040_mv.mv_ecms
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_mb_tenant_isolation
    ON pack040_mv.mv_measurement_boundaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_mb_service_bypass
    ON pack040_mv.mv_measurement_boundaries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_pem_tenant_isolation
    ON pack040_mv.mv_project_ecm_map
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_pem_service_bypass
    ON pack040_mv.mv_project_ecm_map
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cfg_tenant_isolation
    ON pack040_mv.mv_project_config
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cfg_service_bypass
    ON pack040_mv.mv_project_config
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack040_mv TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_projects TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_ecms TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_measurement_boundaries TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_project_ecm_map TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_project_config TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack040_mv.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack040_mv IS
    'PACK-040 M&V Pack - Measurement & Verification per IPMVP, ASHRAE 14, ISO 50015, FEMP 4.0. Baseline development, savings verification, uncertainty analysis, persistence tracking, and compliance reporting.';

COMMENT ON TABLE pack040_mv.mv_projects IS
    'Central registry of M&V projects tracking lifecycle from planning through persistence monitoring with IPMVP option, baseline/reporting periods, and contract terms.';
COMMENT ON TABLE pack040_mv.mv_ecms IS
    'Energy Conservation Measures registry with expected savings, costs, technical specs, and IPMVP option recommendations.';
COMMENT ON TABLE pack040_mv.mv_measurement_boundaries IS
    'IPMVP measurement boundary definitions specifying included/excluded systems, meters, and energy flows for each project or ECM.';
COMMENT ON TABLE pack040_mv.mv_project_ecm_map IS
    'Many-to-many mapping between projects and ECMs with per-ECM IPMVP option, interactive effects, and verification status.';
COMMENT ON TABLE pack040_mv.mv_project_config IS
    'Project configuration controlling ASHRAE 14 thresholds, regression settings, uncertainty parameters, and reporting preferences.';

COMMENT ON COLUMN pack040_mv.mv_projects.id IS 'Unique identifier for the M&V project.';
COMMENT ON COLUMN pack040_mv.mv_projects.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack040_mv.mv_projects.facility_id IS 'Reference to the facility in the core facility registry.';
COMMENT ON COLUMN pack040_mv.mv_projects.ipmvp_option IS 'Primary IPMVP option: OPTION_A (key parameter), OPTION_B (all parameter), OPTION_C (whole facility), OPTION_D (simulation).';
COMMENT ON COLUMN pack040_mv.mv_projects.compliance_framework IS 'Primary compliance standard: IPMVP, ASHRAE_14, ISO_50015, FEMP_4_0, EU_EED.';
COMMENT ON COLUMN pack040_mv.mv_projects.guaranteed_savings_kwh IS 'Contractually guaranteed annual energy savings in kWh for performance contracts.';
COMMENT ON COLUMN pack040_mv.mv_projects.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_ecms.ecm_category IS 'ECM technology category: HVAC, LIGHTING, ENVELOPE, CONTROLS, VFD, etc.';
COMMENT ON COLUMN pack040_mv.mv_ecms.interactive_effects IS 'Whether this ECM has interactive effects with other ECMs (e.g., lighting reduction affects HVAC cooling load).';
COMMENT ON COLUMN pack040_mv.mv_ecms.measurement_boundary IS 'Recommended measurement boundary level for M&V of this ECM.';

COMMENT ON COLUMN pack040_mv.mv_measurement_boundaries.boundary_type IS 'Measurement boundary scope: RETROFIT_ISOLATION, WHOLE_FACILITY, SYSTEM_LEVEL, etc.';
COMMENT ON COLUMN pack040_mv.mv_measurement_boundaries.ecm_share_of_facility_pct IS 'Percentage of whole-facility energy affected by the ECM within this boundary.';
COMMENT ON COLUMN pack040_mv.mv_measurement_boundaries.independent_variables IS 'JSON array of independent variables used in baseline regression within this boundary.';

COMMENT ON COLUMN pack040_mv.mv_project_ecm_map.ecm_weight_pct IS 'Weighting factor for apportioning shared savings among multiple ECMs (0-100).';
COMMENT ON COLUMN pack040_mv.mv_project_ecm_map.interactive_effect_pct IS 'Interactive effect magnitude as percentage adjustment to standalone ECM savings.';

COMMENT ON COLUMN pack040_mv.mv_project_config.max_cvrmse_monthly_pct IS 'ASHRAE 14 maximum allowable CVRMSE for monthly data models (default 25%).';
COMMENT ON COLUMN pack040_mv.mv_project_config.max_nmbe_monthly_pct IS 'ASHRAE 14 maximum allowable NMBE for monthly data models (default +/-5%).';
COMMENT ON COLUMN pack040_mv.mv_project_config.preferred_model_type IS 'Preferred regression model: OLS, 3P_COOLING, 3P_HEATING, 4P, 5P, TOWT, AUTO_SELECT.';
COMMENT ON COLUMN pack040_mv.mv_project_config.recom_trigger_factor IS 'Persistence factor threshold below which re-commissioning is recommended (default 0.80).';
