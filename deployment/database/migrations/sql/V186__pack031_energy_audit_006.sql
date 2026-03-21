-- =============================================================================
-- V186: PACK-031 Industrial Energy Audit - Compressed Air Systems
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Compressed air system audit tables including system-level overview,
-- compressor inventory with operating parameters, leak survey records,
-- and pressure/flow profiling (TimescaleDB hypertable).
--
-- Tables (4):
--   1. pack031_energy_audit.compressed_air_systems
--   2. pack031_energy_audit.compressor_inventory
--   3. pack031_energy_audit.leak_surveys
--   4. pack031_energy_audit.pressure_profiles  (hypertable)
--
-- Previous: V185__pack031_energy_audit_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.compressed_air_systems
-- =============================================================================
-- System-level compressed air overview with aggregate power, FAD,
-- and system-specific power metrics.

CREATE TABLE pack031_energy_audit.compressed_air_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    DEFAULT 'Main Compressed Air System',
    total_compressor_power_kw NUMERIC(12,4),
    system_pressure_bar     NUMERIC(8,4),
    total_fad_m3min         NUMERIC(10,4),
    system_specific_power   NUMERIC(8,4),
    distribution_length_m   NUMERIC(10,2),
    receiver_volume_m3      NUMERIC(10,4),
    air_quality_class       VARCHAR(30),
    annual_energy_kwh       NUMERIC(14,4),
    annual_cost_eur         NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_cas_power CHECK (
        total_compressor_power_kw IS NULL OR total_compressor_power_kw >= 0
    ),
    CONSTRAINT chk_p031_cas_pressure CHECK (
        system_pressure_bar IS NULL OR system_pressure_bar >= 0
    ),
    CONSTRAINT chk_p031_cas_fad CHECK (
        total_fad_m3min IS NULL OR total_fad_m3min >= 0
    ),
    CONSTRAINT chk_p031_cas_sp CHECK (
        system_specific_power IS NULL OR system_specific_power >= 0
    )
);

-- Indexes
CREATE INDEX idx_p031_cas_facility     ON pack031_energy_audit.compressed_air_systems(facility_id);
CREATE INDEX idx_p031_cas_tenant       ON pack031_energy_audit.compressed_air_systems(tenant_id);

-- Trigger
CREATE TRIGGER trg_p031_cas_updated
    BEFORE UPDATE ON pack031_energy_audit.compressed_air_systems
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.compressor_inventory
-- =============================================================================
-- Individual compressor inventory within a compressed air system,
-- linking to the equipment registry with operating parameters.

CREATE TABLE pack031_energy_audit.compressor_inventory (
    compressor_inv_id       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack031_energy_audit.compressed_air_systems(system_id) ON DELETE CASCADE,
    equipment_id            UUID            REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    type                    VARCHAR(50),
    rated_power_kw          NUMERIC(12,4),
    fad_m3min               NUMERIC(10,4),
    pressure_bar            NUMERIC(8,4),
    control_type            VARCHAR(50),
    has_vsd                 BOOLEAN         DEFAULT FALSE,
    specific_power          NUMERIC(8,4),
    load_pct                NUMERIC(5,2),
    operating_hours         INTEGER,
    sequencer_priority      INTEGER,
    role                    VARCHAR(30)     DEFAULT 'base_load',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_cinv_type CHECK (
        type IS NULL OR type IN ('SCREW', 'RECIPROCATING', 'SCROLL', 'CENTRIFUGAL', 'ROTARY_VANE')
    ),
    CONSTRAINT chk_p031_cinv_power CHECK (
        rated_power_kw IS NULL OR rated_power_kw >= 0
    ),
    CONSTRAINT chk_p031_cinv_load CHECK (
        load_pct IS NULL OR (load_pct >= 0 AND load_pct <= 100)
    ),
    CONSTRAINT chk_p031_cinv_control CHECK (
        control_type IS NULL OR control_type IN ('LOAD_UNLOAD', 'MODULATING', 'VSD',
                                                  'ON_OFF', 'INLET_MODULATION', 'MULTI_STEP')
    ),
    CONSTRAINT chk_p031_cinv_role CHECK (
        role IN ('base_load', 'trim', 'standby', 'backup')
    )
);

-- Indexes
CREATE INDEX idx_p031_cinv_system      ON pack031_energy_audit.compressor_inventory(system_id);
CREATE INDEX idx_p031_cinv_equip       ON pack031_energy_audit.compressor_inventory(equipment_id);
CREATE INDEX idx_p031_cinv_tenant      ON pack031_energy_audit.compressor_inventory(tenant_id);
CREATE INDEX idx_p031_cinv_type        ON pack031_energy_audit.compressor_inventory(type);

-- =============================================================================
-- Table 3: pack031_energy_audit.leak_surveys
-- =============================================================================
-- Compressed air leak survey records with leak count, estimated leak
-- flow, cost impact, and repair status tracking.

CREATE TABLE pack031_energy_audit.leak_surveys (
    survey_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack031_energy_audit.compressed_air_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    survey_date             DATE            NOT NULL,
    survey_method           VARCHAR(50)     DEFAULT 'ultrasonic',
    total_leaks_found       INTEGER         NOT NULL DEFAULT 0,
    estimated_leak_flow_m3min NUMERIC(10,4),
    leak_percentage         NUMERIC(5,2),
    cost_of_leaks_eur       NUMERIC(12,4),
    leaks_repaired          INTEGER         DEFAULT 0,
    repair_cost_eur         NUMERIC(12,4),
    post_repair_leak_pct    NUMERIC(5,2),
    next_survey_date        DATE,
    status                  VARCHAR(30)     DEFAULT 'completed',
    surveyor_name           VARCHAR(255),
    report_url              TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_leak_count CHECK (total_leaks_found >= 0),
    CONSTRAINT chk_p031_leak_flow CHECK (
        estimated_leak_flow_m3min IS NULL OR estimated_leak_flow_m3min >= 0
    ),
    CONSTRAINT chk_p031_leak_pct CHECK (
        leak_percentage IS NULL OR (leak_percentage >= 0 AND leak_percentage <= 100)
    ),
    CONSTRAINT chk_p031_leak_cost CHECK (
        cost_of_leaks_eur IS NULL OR cost_of_leaks_eur >= 0
    ),
    CONSTRAINT chk_p031_leak_repaired CHECK (
        leaks_repaired IS NULL OR leaks_repaired >= 0
    ),
    CONSTRAINT chk_p031_leak_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled')
    )
);

-- Indexes
CREATE INDEX idx_p031_leak_system      ON pack031_energy_audit.leak_surveys(system_id);
CREATE INDEX idx_p031_leak_tenant      ON pack031_energy_audit.leak_surveys(tenant_id);
CREATE INDEX idx_p031_leak_date        ON pack031_energy_audit.leak_surveys(survey_date);
CREATE INDEX idx_p031_leak_status      ON pack031_energy_audit.leak_surveys(status);

-- =============================================================================
-- Table 4: pack031_energy_audit.pressure_profiles
-- =============================================================================
-- Time-series pressure, flow, and power profiles for compressed air
-- systems (TimescaleDB hypertable) for demand pattern analysis.

CREATE TABLE pack031_energy_audit.pressure_profiles (
    profile_id              UUID            DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    timestamp               TIMESTAMPTZ     NOT NULL,
    pressure_bar            NUMERIC(8,4),
    flow_m3min              NUMERIC(10,4),
    power_kw                NUMERIC(10,4),
    compressors_running     INTEGER,
    metadata                JSONB           DEFAULT '{}',
    -- Constraints
    CONSTRAINT chk_p031_pp_pressure CHECK (
        pressure_bar IS NULL OR pressure_bar >= 0
    ),
    CONSTRAINT chk_p031_pp_flow CHECK (
        flow_m3min IS NULL OR flow_m3min >= 0
    ),
    CONSTRAINT chk_p031_pp_power CHECK (
        power_kw IS NULL OR power_kw >= 0
    )
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'pack031_energy_audit.pressure_profiles',
    'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

-- Indexes
CREATE INDEX idx_p031_pp_system        ON pack031_energy_audit.pressure_profiles(system_id, timestamp DESC);
CREATE INDEX idx_p031_pp_tenant        ON pack031_energy_audit.pressure_profiles(tenant_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.compressed_air_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.compressor_inventory ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.leak_surveys ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.pressure_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_cas_tenant_isolation ON pack031_energy_audit.compressed_air_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_cas_service_bypass ON pack031_energy_audit.compressed_air_systems
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_cinv_tenant_isolation ON pack031_energy_audit.compressor_inventory
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_cinv_service_bypass ON pack031_energy_audit.compressor_inventory
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_leak_tenant_isolation ON pack031_energy_audit.leak_surveys
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_leak_service_bypass ON pack031_energy_audit.leak_surveys
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_pp_tenant_isolation ON pack031_energy_audit.pressure_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_pp_service_bypass ON pack031_energy_audit.pressure_profiles
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.compressed_air_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.compressor_inventory TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.leak_surveys TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.pressure_profiles TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.compressed_air_systems IS
    'System-level compressed air overview with aggregate power, FAD, specific power, and distribution metrics.';
COMMENT ON TABLE pack031_energy_audit.compressor_inventory IS
    'Individual compressor inventory within a compressed air system with operating parameters and sequencer roles.';
COMMENT ON TABLE pack031_energy_audit.leak_surveys IS
    'Compressed air leak survey records with leak count, estimated flow loss, cost impact, and repair tracking.';
COMMENT ON TABLE pack031_energy_audit.pressure_profiles IS
    'Time-series pressure/flow/power profiles (TimescaleDB hypertable) for compressed air demand pattern analysis.';

COMMENT ON COLUMN pack031_energy_audit.compressed_air_systems.system_specific_power IS
    'System-level specific power in kW/(m3/min) - key efficiency KPI for compressed air.';
COMMENT ON COLUMN pack031_energy_audit.leak_surveys.leak_percentage IS
    'Leak rate as percentage of total system capacity (target < 10%).';
