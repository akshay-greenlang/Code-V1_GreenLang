-- =============================================================================
-- V184: PACK-031 Industrial Energy Audit - Equipment Registry
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Equipment registry with specialized data tables for motors, pumps,
-- compressors, boilers, and HVAC systems. Each equipment type has
-- performance characteristics needed for energy audit calculations.
--
-- Tables (6):
--   1. pack031_energy_audit.equipment
--   2. pack031_energy_audit.motor_data
--   3. pack031_energy_audit.pump_data
--   4. pack031_energy_audit.compressor_data
--   5. pack031_energy_audit.boiler_data
--   6. pack031_energy_audit.hvac_data
--
-- Previous: V183__pack031_energy_audit_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.equipment
-- =============================================================================
-- Master equipment registry with physical characteristics, operating
-- parameters, and hierarchical system grouping.

CREATE TABLE pack031_energy_audit.equipment (
    equipment_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    type                    VARCHAR(100)    NOT NULL,
    manufacturer            VARCHAR(255),
    model                   VARCHAR(255),
    serial_number           VARCHAR(100),
    year_installed          INTEGER,
    rated_power_kw          NUMERIC(12,4),
    operating_hours         INTEGER,
    load_factor_pct         NUMERIC(5,2),
    location                VARCHAR(255),
    parent_system_id        UUID            REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE SET NULL,
    criticality             VARCHAR(20),
    condition_rating        VARCHAR(20),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_equip_type CHECK (
        type IN ('MOTOR', 'PUMP', 'FAN', 'COMPRESSOR', 'BOILER', 'HVAC',
                 'CHILLER', 'COOLING_TOWER', 'HEAT_EXCHANGER', 'FURNACE',
                 'OVEN', 'DRYER', 'TRANSFORMER', 'LIGHTING', 'CONVEYOR',
                 'HYDRAULIC_PRESS', 'CNC_MACHINE', 'OTHER')
    ),
    CONSTRAINT chk_p031_equip_year CHECK (
        year_installed IS NULL OR (year_installed >= 1950 AND year_installed <= 2100)
    ),
    CONSTRAINT chk_p031_equip_power CHECK (
        rated_power_kw IS NULL OR rated_power_kw >= 0
    ),
    CONSTRAINT chk_p031_equip_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p031_equip_load CHECK (
        load_factor_pct IS NULL OR (load_factor_pct >= 0 AND load_factor_pct <= 100)
    ),
    CONSTRAINT chk_p031_equip_criticality CHECK (
        criticality IS NULL OR criticality IN ('critical', 'important', 'standard', 'non_essential')
    ),
    CONSTRAINT chk_p031_equip_condition CHECK (
        condition_rating IS NULL OR condition_rating IN ('excellent', 'good', 'fair', 'poor', 'end_of_life')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_equip_facility   ON pack031_energy_audit.equipment(facility_id);
CREATE INDEX idx_p031_equip_tenant     ON pack031_energy_audit.equipment(tenant_id);
CREATE INDEX idx_p031_equip_type       ON pack031_energy_audit.equipment(type);
CREATE INDEX idx_p031_equip_parent     ON pack031_energy_audit.equipment(parent_system_id);
CREATE INDEX idx_p031_equip_power      ON pack031_energy_audit.equipment(rated_power_kw);
CREATE INDEX idx_p031_equip_year       ON pack031_energy_audit.equipment(year_installed);
CREATE INDEX idx_p031_equip_created    ON pack031_energy_audit.equipment(created_at DESC);

-- Trigger
CREATE TRIGGER trg_p031_equip_updated
    BEFORE UPDATE ON pack031_energy_audit.equipment
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.motor_data
-- =============================================================================
-- Motor-specific performance data including IE efficiency class,
-- actual load percentage, variable speed drive status, and annual energy.

CREATE TABLE pack031_energy_audit.motor_data (
    motor_data_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            UUID            NOT NULL REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    efficiency_class        VARCHAR(10),
    rated_power_kw          NUMERIC(12,4),
    poles                   INTEGER,
    voltage                 INTEGER,
    frequency               INTEGER         DEFAULT 50,
    actual_load_pct         NUMERIC(5,2),
    has_vsd                 BOOLEAN         DEFAULT FALSE,
    vsd_type                VARCHAR(50),
    power_factor            NUMERIC(4,3),
    annual_energy_kwh       NUMERIC(14,4),
    replacement_candidate   BOOLEAN         DEFAULT FALSE,
    savings_with_ie4_kwh    NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_motor_class CHECK (
        efficiency_class IS NULL OR efficiency_class IN ('IE1', 'IE2', 'IE3', 'IE4', 'IE5', 'UNKNOWN')
    ),
    CONSTRAINT chk_p031_motor_load CHECK (
        actual_load_pct IS NULL OR (actual_load_pct >= 0 AND actual_load_pct <= 150)
    ),
    CONSTRAINT chk_p031_motor_poles CHECK (
        poles IS NULL OR poles IN (2, 4, 6, 8, 10, 12)
    ),
    CONSTRAINT chk_p031_motor_energy CHECK (
        annual_energy_kwh IS NULL OR annual_energy_kwh >= 0
    )
);

-- Indexes
CREATE INDEX idx_p031_motor_equip      ON pack031_energy_audit.motor_data(equipment_id);
CREATE INDEX idx_p031_motor_tenant     ON pack031_energy_audit.motor_data(tenant_id);
CREATE INDEX idx_p031_motor_class      ON pack031_energy_audit.motor_data(efficiency_class);
CREATE INDEX idx_p031_motor_vsd        ON pack031_energy_audit.motor_data(has_vsd);

-- =============================================================================
-- Table 3: pack031_energy_audit.pump_data
-- =============================================================================
-- Pump-specific performance data including flow, head, efficiency,
-- and system curve characteristics.

CREATE TABLE pack031_energy_audit.pump_data (
    pump_data_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            UUID            NOT NULL REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    pump_type               VARCHAR(50),
    flow_m3h                NUMERIC(10,4),
    head_m                  NUMERIC(10,4),
    efficiency_pct          NUMERIC(5,2),
    bep_flow_m3h            NUMERIC(10,4),
    operating_pct_of_bep    NUMERIC(5,2),
    system_curve            JSONB           DEFAULT '{}',
    throttle_controlled     BOOLEAN         DEFAULT FALSE,
    annual_energy_kwh       NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_pump_flow CHECK (
        flow_m3h IS NULL OR flow_m3h >= 0
    ),
    CONSTRAINT chk_p031_pump_head CHECK (
        head_m IS NULL OR head_m >= 0
    ),
    CONSTRAINT chk_p031_pump_eff CHECK (
        efficiency_pct IS NULL OR (efficiency_pct >= 0 AND efficiency_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_pump_equip       ON pack031_energy_audit.pump_data(equipment_id);
CREATE INDEX idx_p031_pump_tenant      ON pack031_energy_audit.pump_data(tenant_id);

-- =============================================================================
-- Table 4: pack031_energy_audit.compressor_data
-- =============================================================================
-- Compressor-specific performance data including free air delivery,
-- pressure, specific power, and load profile.

CREATE TABLE pack031_energy_audit.compressor_data (
    compressor_data_id      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            UUID            NOT NULL REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    type                    VARCHAR(50),
    fad_m3min               NUMERIC(10,4),
    pressure_bar            NUMERIC(8,4),
    specific_power          NUMERIC(8,4),
    control_type            VARCHAR(50),
    has_vsd                 BOOLEAN         DEFAULT FALSE,
    load_profile            JSONB           DEFAULT '{}',
    annual_energy_kwh       NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_comp_type CHECK (
        type IS NULL OR type IN ('SCREW', 'RECIPROCATING', 'SCROLL', 'CENTRIFUGAL', 'ROTARY_VANE')
    ),
    CONSTRAINT chk_p031_comp_fad CHECK (
        fad_m3min IS NULL OR fad_m3min >= 0
    ),
    CONSTRAINT chk_p031_comp_pressure CHECK (
        pressure_bar IS NULL OR pressure_bar >= 0
    ),
    CONSTRAINT chk_p031_comp_sp CHECK (
        specific_power IS NULL OR specific_power >= 0
    )
);

-- Indexes
CREATE INDEX idx_p031_comp_equip       ON pack031_energy_audit.compressor_data(equipment_id);
CREATE INDEX idx_p031_comp_tenant      ON pack031_energy_audit.compressor_data(tenant_id);
CREATE INDEX idx_p031_comp_type        ON pack031_energy_audit.compressor_data(type);

-- =============================================================================
-- Table 5: pack031_energy_audit.boiler_data
-- =============================================================================
-- Boiler-specific performance data including capacity, fuel type,
-- stack temperature, excess air, and blowdown percentage.

CREATE TABLE pack031_energy_audit.boiler_data (
    boiler_data_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            UUID            NOT NULL REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    type                    VARCHAR(50),
    capacity_kw             NUMERIC(12,4),
    fuel_type               VARCHAR(50),
    design_efficiency_pct   NUMERIC(5,2),
    measured_efficiency_pct NUMERIC(5,2),
    stack_temp_c            NUMERIC(8,2),
    excess_air_pct          NUMERIC(5,2),
    blowdown_pct            NUMERIC(5,2),
    condensing              BOOLEAN         DEFAULT FALSE,
    economizer_fitted       BOOLEAN         DEFAULT FALSE,
    annual_fuel_kwh         NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_boiler_type CHECK (
        type IS NULL OR type IN ('FIRE_TUBE', 'WATER_TUBE', 'CAST_IRON', 'CONDENSING',
                                  'ELECTRIC', 'WASTE_HEAT', 'BIOMASS', 'CHP')
    ),
    CONSTRAINT chk_p031_boiler_capacity CHECK (
        capacity_kw IS NULL OR capacity_kw >= 0
    ),
    CONSTRAINT chk_p031_boiler_eff CHECK (
        measured_efficiency_pct IS NULL OR (measured_efficiency_pct >= 0 AND measured_efficiency_pct <= 110)
    ),
    CONSTRAINT chk_p031_boiler_blowdown CHECK (
        blowdown_pct IS NULL OR (blowdown_pct >= 0 AND blowdown_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_boiler_equip     ON pack031_energy_audit.boiler_data(equipment_id);
CREATE INDEX idx_p031_boiler_tenant    ON pack031_energy_audit.boiler_data(tenant_id);
CREATE INDEX idx_p031_boiler_type      ON pack031_energy_audit.boiler_data(type);
CREATE INDEX idx_p031_boiler_fuel      ON pack031_energy_audit.boiler_data(fuel_type);

-- =============================================================================
-- Table 6: pack031_energy_audit.hvac_data
-- =============================================================================
-- HVAC system performance data including cooling/heating capacity,
-- COP, EER, and refrigerant type.

CREATE TABLE pack031_energy_audit.hvac_data (
    hvac_data_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id            UUID            NOT NULL REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    type                    VARCHAR(50),
    cooling_capacity_kw     NUMERIC(12,4),
    heating_capacity_kw     NUMERIC(12,4),
    cop                     NUMERIC(6,3),
    eer                     NUMERIC(6,3),
    seer                    NUMERIC(6,3),
    scop                    NUMERIC(6,3),
    refrigerant             VARCHAR(20),
    refrigerant_gwp         NUMERIC(8,2),
    refrigerant_charge_kg   NUMERIC(8,2),
    annual_energy_kwh       NUMERIC(14,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_hvac_type CHECK (
        type IS NULL OR type IN ('SPLIT', 'VRF', 'CHILLER_AIR', 'CHILLER_WATER',
                                  'HEAT_PUMP', 'ROOFTOP', 'AHU', 'FCU',
                                  'ABSORPTION', 'DISTRICT', 'OTHER')
    ),
    CONSTRAINT chk_p031_hvac_cop CHECK (
        cop IS NULL OR cop >= 0
    ),
    CONSTRAINT chk_p031_hvac_eer CHECK (
        eer IS NULL OR eer >= 0
    ),
    CONSTRAINT chk_p031_hvac_cooling CHECK (
        cooling_capacity_kw IS NULL OR cooling_capacity_kw >= 0
    ),
    CONSTRAINT chk_p031_hvac_heating CHECK (
        heating_capacity_kw IS NULL OR heating_capacity_kw >= 0
    )
);

-- Indexes
CREATE INDEX idx_p031_hvac_equip       ON pack031_energy_audit.hvac_data(equipment_id);
CREATE INDEX idx_p031_hvac_tenant      ON pack031_energy_audit.hvac_data(tenant_id);
CREATE INDEX idx_p031_hvac_type        ON pack031_energy_audit.hvac_data(type);
CREATE INDEX idx_p031_hvac_refrigerant ON pack031_energy_audit.hvac_data(refrigerant);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.equipment ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.motor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.pump_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.compressor_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.boiler_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.hvac_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_equip_tenant_isolation ON pack031_energy_audit.equipment
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_equip_service_bypass ON pack031_energy_audit.equipment
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_motor_tenant_isolation ON pack031_energy_audit.motor_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_motor_service_bypass ON pack031_energy_audit.motor_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_pump_tenant_isolation ON pack031_energy_audit.pump_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_pump_service_bypass ON pack031_energy_audit.pump_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_compressor_tenant_isolation ON pack031_energy_audit.compressor_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_compressor_service_bypass ON pack031_energy_audit.compressor_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_boiler_tenant_isolation ON pack031_energy_audit.boiler_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_boiler_service_bypass ON pack031_energy_audit.boiler_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_hvac_tenant_isolation ON pack031_energy_audit.hvac_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_hvac_service_bypass ON pack031_energy_audit.hvac_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.equipment TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.motor_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.pump_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.compressor_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.boiler_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.hvac_data TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.equipment IS
    'Master equipment registry with physical characteristics, operating parameters, and hierarchical system grouping.';
COMMENT ON TABLE pack031_energy_audit.motor_data IS
    'Motor-specific data: IE efficiency class, VSD status, actual load, annual energy consumption.';
COMMENT ON TABLE pack031_energy_audit.pump_data IS
    'Pump-specific data: flow, head, efficiency, BEP, system curve, throttle control status.';
COMMENT ON TABLE pack031_energy_audit.compressor_data IS
    'Compressor-specific data: FAD, pressure, specific power, control type, load profile.';
COMMENT ON TABLE pack031_energy_audit.boiler_data IS
    'Boiler-specific data: capacity, fuel type, efficiency, stack temp, excess air, blowdown.';
COMMENT ON TABLE pack031_energy_audit.hvac_data IS
    'HVAC-specific data: cooling/heating capacity, COP, EER, SEER, SCOP, refrigerant type and GWP.';
