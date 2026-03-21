-- =============================================================================
-- V187: PACK-031 Industrial Energy Audit - Steam Systems
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    007 of 010
-- Date:         March 2026
--
-- Steam system audit tables including system-level overview, boiler
-- inventory, steam trap surveys, insulation assessments, and flue
-- gas analysis records for combustion efficiency monitoring.
--
-- Tables (5):
--   1. pack031_energy_audit.steam_systems
--   2. pack031_energy_audit.steam_boilers
--   3. pack031_energy_audit.steam_trap_surveys
--   4. pack031_energy_audit.insulation_assessments
--   5. pack031_energy_audit.flue_gas_analyses
--
-- Previous: V186__pack031_energy_audit_006.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.steam_systems
-- =============================================================================
-- System-level steam overview with total capacity, operating pressure,
-- and condensate return percentage.

CREATE TABLE pack031_energy_audit.steam_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    DEFAULT 'Main Steam System',
    total_capacity_kg_h     NUMERIC(12,4),
    operating_pressure_bar  NUMERIC(8,4),
    condensate_return_pct   NUMERIC(5,2),
    makeup_water_pct        NUMERIC(5,2),
    deaerator_type          VARCHAR(50),
    water_treatment_type    VARCHAR(100),
    distribution_length_m   NUMERIC(10,2),
    annual_steam_kwh        NUMERIC(18,6),
    annual_fuel_cost_eur    NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_ss_capacity CHECK (
        total_capacity_kg_h IS NULL OR total_capacity_kg_h >= 0
    ),
    CONSTRAINT chk_p031_ss_pressure CHECK (
        operating_pressure_bar IS NULL OR operating_pressure_bar >= 0
    ),
    CONSTRAINT chk_p031_ss_condensate CHECK (
        condensate_return_pct IS NULL OR (condensate_return_pct >= 0 AND condensate_return_pct <= 100)
    ),
    CONSTRAINT chk_p031_ss_makeup CHECK (
        makeup_water_pct IS NULL OR (makeup_water_pct >= 0 AND makeup_water_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_ss_facility      ON pack031_energy_audit.steam_systems(facility_id);
CREATE INDEX idx_p031_ss_tenant        ON pack031_energy_audit.steam_systems(tenant_id);

-- Trigger
CREATE TRIGGER trg_p031_ss_updated
    BEFORE UPDATE ON pack031_energy_audit.steam_systems
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.steam_boilers
-- =============================================================================
-- Individual steam boiler inventory within a steam system, linking to
-- the equipment registry with boiler-specific operating parameters.

CREATE TABLE pack031_energy_audit.steam_boilers (
    boiler_inv_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack031_energy_audit.steam_systems(system_id) ON DELETE CASCADE,
    equipment_id            UUID            REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    type                    VARCHAR(50),
    capacity_kg_h           NUMERIC(12,4),
    fuel_type               VARCHAR(50)     NOT NULL,
    design_pressure_bar     NUMERIC(8,4),
    operating_pressure_bar  NUMERIC(8,4),
    feed_water_temp_c       NUMERIC(8,2),
    steam_temp_c            NUMERIC(8,2),
    efficiency_pct          NUMERIC(5,2),
    annual_operating_hours  INTEGER,
    annual_fuel_consumption NUMERIC(14,4),
    annual_fuel_unit        VARCHAR(30),
    role                    VARCHAR(30)     DEFAULT 'base_load',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_sb_type CHECK (
        type IS NULL OR type IN ('FIRE_TUBE', 'WATER_TUBE', 'CAST_IRON', 'ELECTRIC',
                                  'WASTE_HEAT', 'BIOMASS', 'CHP', 'ONCE_THROUGH')
    ),
    CONSTRAINT chk_p031_sb_capacity CHECK (
        capacity_kg_h IS NULL OR capacity_kg_h >= 0
    ),
    CONSTRAINT chk_p031_sb_efficiency CHECK (
        efficiency_pct IS NULL OR (efficiency_pct >= 0 AND efficiency_pct <= 110)
    ),
    CONSTRAINT chk_p031_sb_role CHECK (
        role IN ('base_load', 'peak', 'standby', 'backup')
    )
);

-- Indexes
CREATE INDEX idx_p031_sb_system        ON pack031_energy_audit.steam_boilers(system_id);
CREATE INDEX idx_p031_sb_equip         ON pack031_energy_audit.steam_boilers(equipment_id);
CREATE INDEX idx_p031_sb_tenant        ON pack031_energy_audit.steam_boilers(tenant_id);
CREATE INDEX idx_p031_sb_type          ON pack031_energy_audit.steam_boilers(type);
CREATE INDEX idx_p031_sb_fuel          ON pack031_energy_audit.steam_boilers(fuel_type);

-- =============================================================================
-- Table 3: pack031_energy_audit.steam_trap_surveys
-- =============================================================================
-- Steam trap condition survey records with failure rate, estimated
-- steam loss, and cost of failed traps.

CREATE TABLE pack031_energy_audit.steam_trap_surveys (
    trap_survey_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack031_energy_audit.steam_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    survey_date             DATE            NOT NULL,
    survey_method           VARCHAR(50)     DEFAULT 'ultrasonic',
    total_traps             INTEGER         NOT NULL DEFAULT 0,
    failed_traps            INTEGER         DEFAULT 0,
    leaking_traps           INTEGER         DEFAULT 0,
    blocked_traps           INTEGER         DEFAULT 0,
    failure_rate_pct        NUMERIC(5,2),
    estimated_steam_loss_kg_h NUMERIC(10,4),
    estimated_cost_eur      NUMERIC(12,4),
    traps_replaced          INTEGER         DEFAULT 0,
    replacement_cost_eur    NUMERIC(12,4),
    next_survey_date        DATE,
    surveyor_name           VARCHAR(255),
    status                  VARCHAR(30)     DEFAULT 'completed',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_trap_total CHECK (total_traps >= 0),
    CONSTRAINT chk_p031_trap_failed CHECK (
        failed_traps IS NULL OR failed_traps >= 0
    ),
    CONSTRAINT chk_p031_trap_leaking CHECK (
        leaking_traps IS NULL OR leaking_traps >= 0
    ),
    CONSTRAINT chk_p031_trap_blocked CHECK (
        blocked_traps IS NULL OR blocked_traps >= 0
    ),
    CONSTRAINT chk_p031_trap_failure_rate CHECK (
        failure_rate_pct IS NULL OR (failure_rate_pct >= 0 AND failure_rate_pct <= 100)
    ),
    CONSTRAINT chk_p031_trap_loss CHECK (
        estimated_steam_loss_kg_h IS NULL OR estimated_steam_loss_kg_h >= 0
    ),
    CONSTRAINT chk_p031_trap_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled')
    )
);

-- Indexes
CREATE INDEX idx_p031_trap_system      ON pack031_energy_audit.steam_trap_surveys(system_id);
CREATE INDEX idx_p031_trap_tenant      ON pack031_energy_audit.steam_trap_surveys(tenant_id);
CREATE INDEX idx_p031_trap_date        ON pack031_energy_audit.steam_trap_surveys(survey_date);
CREATE INDEX idx_p031_trap_status      ON pack031_energy_audit.steam_trap_surveys(status);

-- =============================================================================
-- Table 4: pack031_energy_audit.insulation_assessments
-- =============================================================================
-- Thermal insulation condition assessment for steam/hot water piping,
-- valves, flanges, and vessels with heat loss calculations.

CREATE TABLE pack031_energy_audit.insulation_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack031_energy_audit.steam_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    assessment_date         DATE            NOT NULL,
    total_sections          INTEGER         NOT NULL DEFAULT 0,
    insulated_sections      INTEGER         DEFAULT 0,
    uninsulated             INTEGER         DEFAULT 0,
    damaged                 INTEGER         DEFAULT 0,
    bare_pipe_heat_loss_kw  NUMERIC(10,4),
    insulated_heat_loss_kw  NUMERIC(10,4),
    savings_kw              NUMERIC(10,4),
    annual_savings_eur      NUMERIC(12,4),
    insulation_cost_eur     NUMERIC(12,4),
    payback_months          NUMERIC(8,2),
    survey_method           VARCHAR(50)     DEFAULT 'infrared_thermography',
    assessor_name           VARCHAR(255),
    report_url              TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_insul_sections CHECK (total_sections >= 0),
    CONSTRAINT chk_p031_insul_uninsulated CHECK (
        uninsulated IS NULL OR uninsulated >= 0
    ),
    CONSTRAINT chk_p031_insul_damaged CHECK (
        damaged IS NULL OR damaged >= 0
    ),
    CONSTRAINT chk_p031_insul_loss CHECK (
        bare_pipe_heat_loss_kw IS NULL OR bare_pipe_heat_loss_kw >= 0
    ),
    CONSTRAINT chk_p031_insul_savings CHECK (
        savings_kw IS NULL OR savings_kw >= 0
    )
);

-- Indexes
CREATE INDEX idx_p031_insul_system     ON pack031_energy_audit.insulation_assessments(system_id);
CREATE INDEX idx_p031_insul_tenant     ON pack031_energy_audit.insulation_assessments(tenant_id);
CREATE INDEX idx_p031_insul_date       ON pack031_energy_audit.insulation_assessments(assessment_date);

-- =============================================================================
-- Table 5: pack031_energy_audit.flue_gas_analyses
-- =============================================================================
-- Flue gas analysis records for combustion efficiency monitoring
-- including O2, CO2, CO, stack temperature, and excess air.

CREATE TABLE pack031_energy_audit.flue_gas_analyses (
    analysis_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    boiler_inv_id           UUID            NOT NULL REFERENCES pack031_energy_audit.steam_boilers(boiler_inv_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    analysis_date           DATE            NOT NULL,
    co2_pct                 NUMERIC(5,2),
    o2_pct                  NUMERIC(5,2),
    co_ppm                  NUMERIC(8,2),
    nox_ppm                 NUMERIC(8,2),
    so2_ppm                 NUMERIC(8,2),
    stack_temp_c            NUMERIC(8,2),
    ambient_temp_c          NUMERIC(8,2),
    excess_air_pct          NUMERIC(5,2),
    combustion_efficiency_pct NUMERIC(5,2),
    net_stack_temp_c        NUMERIC(8,2)   GENERATED ALWAYS AS (
        CASE WHEN stack_temp_c IS NOT NULL AND ambient_temp_c IS NOT NULL
             THEN stack_temp_c - ambient_temp_c
             ELSE NULL END
    ) STORED,
    instrument_type         VARCHAR(100),
    calibration_date        DATE,
    analyst_name            VARCHAR(255),
    recommendations         TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_flue_co2 CHECK (
        co2_pct IS NULL OR (co2_pct >= 0 AND co2_pct <= 25)
    ),
    CONSTRAINT chk_p031_flue_o2 CHECK (
        o2_pct IS NULL OR (o2_pct >= 0 AND o2_pct <= 21)
    ),
    CONSTRAINT chk_p031_flue_co CHECK (
        co_ppm IS NULL OR co_ppm >= 0
    ),
    CONSTRAINT chk_p031_flue_excess_air CHECK (
        excess_air_pct IS NULL OR excess_air_pct >= 0
    ),
    CONSTRAINT chk_p031_flue_efficiency CHECK (
        combustion_efficiency_pct IS NULL OR (combustion_efficiency_pct >= 0 AND combustion_efficiency_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_flue_boiler      ON pack031_energy_audit.flue_gas_analyses(boiler_inv_id);
CREATE INDEX idx_p031_flue_tenant      ON pack031_energy_audit.flue_gas_analyses(tenant_id);
CREATE INDEX idx_p031_flue_date        ON pack031_energy_audit.flue_gas_analyses(analysis_date);
CREATE INDEX idx_p031_flue_efficiency  ON pack031_energy_audit.flue_gas_analyses(combustion_efficiency_pct);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.steam_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.steam_boilers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.steam_trap_surveys ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.insulation_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.flue_gas_analyses ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_ss_tenant_isolation ON pack031_energy_audit.steam_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_ss_service_bypass ON pack031_energy_audit.steam_systems
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_sb_tenant_isolation ON pack031_energy_audit.steam_boilers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_sb_service_bypass ON pack031_energy_audit.steam_boilers
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_trap_tenant_isolation ON pack031_energy_audit.steam_trap_surveys
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_trap_service_bypass ON pack031_energy_audit.steam_trap_surveys
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_insul_tenant_isolation ON pack031_energy_audit.insulation_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_insul_service_bypass ON pack031_energy_audit.insulation_assessments
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_flue_tenant_isolation ON pack031_energy_audit.flue_gas_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_flue_service_bypass ON pack031_energy_audit.flue_gas_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.steam_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.steam_boilers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.steam_trap_surveys TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.insulation_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.flue_gas_analyses TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.steam_systems IS
    'System-level steam overview with total capacity, operating pressure, condensate return, and distribution metrics.';
COMMENT ON TABLE pack031_energy_audit.steam_boilers IS
    'Individual steam boiler inventory with capacity, fuel type, efficiency, and operating parameters.';
COMMENT ON TABLE pack031_energy_audit.steam_trap_surveys IS
    'Steam trap condition surveys with failure rate, estimated steam loss, and cost of failed traps.';
COMMENT ON TABLE pack031_energy_audit.insulation_assessments IS
    'Thermal insulation assessment for piping, valves, and vessels with heat loss and savings calculations.';
COMMENT ON TABLE pack031_energy_audit.flue_gas_analyses IS
    'Flue gas analysis records for combustion efficiency monitoring (O2, CO2, CO, stack temp, excess air).';

COMMENT ON COLUMN pack031_energy_audit.steam_trap_surveys.failure_rate_pct IS
    'Steam trap failure rate as percentage of total (target < 5% for well-maintained systems).';
COMMENT ON COLUMN pack031_energy_audit.flue_gas_analyses.net_stack_temp_c IS
    'Generated column: stack temperature minus ambient temperature for heat loss assessment.';
