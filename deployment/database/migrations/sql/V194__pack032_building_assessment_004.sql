-- =============================================================================
-- V194: PACK-032 Building Energy Assessment - DHW & Lighting
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    004 of 010
-- Date:         March 2026
--
-- Creates domestic hot water (DHW) and lighting system tables for building
-- energy assessment including LENI calculations and daylight analysis.
--
-- Tables (3):
--   1. pack032_building_assessment.dhw_systems
--   2. pack032_building_assessment.lighting_zones
--   3. pack032_building_assessment.lighting_assessments
--
-- Previous: V193__pack032_building_assessment_003.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.dhw_systems
-- =============================================================================
-- Domestic hot water generation and distribution with efficiency, storage,
-- solar thermal contribution, and legionella management regime.

CREATE TABLE pack032_building_assessment.dhw_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system_type             VARCHAR(100)    NOT NULL,
    fuel_type               VARCHAR(100)    NOT NULL,
    efficiency_pct          NUMERIC(6,2),
    storage_volume_litres   NUMERIC(10,2),
    storage_insulation_mm   NUMERIC(8,2),
    distribution_length_m   NUMERIC(10,2),
    solar_thermal_contribution_pct NUMERIC(6,2),
    legionella_regime       VARCHAR(100),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_dhw_efficiency CHECK (
        efficiency_pct IS NULL OR (efficiency_pct > 0 AND efficiency_pct <= 500)
    ),
    CONSTRAINT chk_p032_dhw_storage_vol CHECK (
        storage_volume_litres IS NULL OR storage_volume_litres >= 0
    ),
    CONSTRAINT chk_p032_dhw_storage_ins CHECK (
        storage_insulation_mm IS NULL OR storage_insulation_mm >= 0
    ),
    CONSTRAINT chk_p032_dhw_dist_len CHECK (
        distribution_length_m IS NULL OR distribution_length_m >= 0
    ),
    CONSTRAINT chk_p032_dhw_solar_pct CHECK (
        solar_thermal_contribution_pct IS NULL OR
        (solar_thermal_contribution_pct >= 0 AND solar_thermal_contribution_pct <= 100)
    ),
    CONSTRAINT chk_p032_dhw_system_type CHECK (
        system_type IN ('INSTANTANEOUS_COMBI', 'STORAGE_CYLINDER', 'POINT_OF_USE',
                         'CENTRALIZED', 'DISTRICT_HOT_WATER', 'HEAT_PUMP_DHW',
                         'SOLAR_THERMAL', 'ELECTRIC_IMMERSION', 'OTHER')
    ),
    CONSTRAINT chk_p032_dhw_fuel_type CHECK (
        fuel_type IN ('NATURAL_GAS', 'LPG', 'OIL', 'ELECTRICITY', 'BIOMASS',
                       'DISTRICT_HEAT', 'SOLAR', 'HYDROGEN', 'OTHER')
    ),
    CONSTRAINT chk_p032_dhw_legionella CHECK (
        legionella_regime IS NULL OR legionella_regime IN ('PASTEURISATION', 'CHLORINATION',
                                                            'COPPER_SILVER_IONISATION',
                                                            'TEMPERATURE_CONTROL', 'NONE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_dhw_building    ON pack032_building_assessment.dhw_systems(building_id);
CREATE INDEX idx_p032_dhw_tenant      ON pack032_building_assessment.dhw_systems(tenant_id);
CREATE INDEX idx_p032_dhw_system_type ON pack032_building_assessment.dhw_systems(system_type);
CREATE INDEX idx_p032_dhw_fuel_type   ON pack032_building_assessment.dhw_systems(fuel_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_dhw_updated
    BEFORE UPDATE ON pack032_building_assessment.dhw_systems
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.lighting_zones
-- =============================================================================
-- Per-zone lighting inventory with fixture count, wattage, LPD, lux levels,
-- control strategy, and operating hours.

CREATE TABLE pack032_building_assessment.lighting_zones (
    zone_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    space_type              VARCHAR(100)    NOT NULL,
    area_m2                 NUMERIC(12,2),
    fixture_count           INTEGER,
    fixture_type            VARCHAR(100),
    wattage_per_fixture     NUMERIC(8,2),
    lpd_w_m2                NUMERIC(8,2),
    lux_level               NUMERIC(8,2),
    control_type            VARCHAR(100),
    operating_hours         INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_lz_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_lz_fixture_count CHECK (
        fixture_count IS NULL OR fixture_count >= 0
    ),
    CONSTRAINT chk_p032_lz_wattage CHECK (
        wattage_per_fixture IS NULL OR wattage_per_fixture >= 0
    ),
    CONSTRAINT chk_p032_lz_lpd CHECK (
        lpd_w_m2 IS NULL OR lpd_w_m2 >= 0
    ),
    CONSTRAINT chk_p032_lz_lux CHECK (
        lux_level IS NULL OR lux_level >= 0
    ),
    CONSTRAINT chk_p032_lz_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p032_lz_fixture_type CHECK (
        fixture_type IS NULL OR fixture_type IN ('LED', 'T5_FLUORESCENT', 'T8_FLUORESCENT',
                                                    'CFL', 'HALOGEN', 'METAL_HALIDE',
                                                    'SODIUM', 'INCANDESCENT', 'OTHER')
    ),
    CONSTRAINT chk_p032_lz_control CHECK (
        control_type IS NULL OR control_type IN ('MANUAL_SWITCH', 'PIR_OCCUPANCY',
                                                    'DAYLIGHT_DIMMING', 'DAYLIGHT_SWITCHING',
                                                    'DALI', 'TIMER', 'COMBINED', 'NONE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_lz_building   ON pack032_building_assessment.lighting_zones(building_id);
CREATE INDEX idx_p032_lz_tenant     ON pack032_building_assessment.lighting_zones(tenant_id);
CREATE INDEX idx_p032_lz_space_type ON pack032_building_assessment.lighting_zones(space_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_lz_updated
    BEFORE UPDATE ON pack032_building_assessment.lighting_zones
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.lighting_assessments
-- =============================================================================
-- Building-level lighting assessment summary with LENI, daylight factor,
-- LED penetration, and control coverage.

CREATE TABLE pack032_building_assessment.lighting_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    total_lpd_w_m2          NUMERIC(8,2),
    leni_kwh_m2             NUMERIC(10,2),
    daylight_factor_avg     NUMERIC(6,2),
    led_percentage          NUMERIC(6,2),
    control_coverage_pct    NUMERIC(6,2),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_la_lpd CHECK (
        total_lpd_w_m2 IS NULL OR total_lpd_w_m2 >= 0
    ),
    CONSTRAINT chk_p032_la_leni CHECK (
        leni_kwh_m2 IS NULL OR leni_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p032_la_daylight CHECK (
        daylight_factor_avg IS NULL OR daylight_factor_avg >= 0
    ),
    CONSTRAINT chk_p032_la_led_pct CHECK (
        led_percentage IS NULL OR (led_percentage >= 0 AND led_percentage <= 100)
    ),
    CONSTRAINT chk_p032_la_control_pct CHECK (
        control_coverage_pct IS NULL OR (control_coverage_pct >= 0 AND control_coverage_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_la_building ON pack032_building_assessment.lighting_assessments(building_id);
CREATE INDEX idx_p032_la_tenant   ON pack032_building_assessment.lighting_assessments(tenant_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_la_updated
    BEFORE UPDATE ON pack032_building_assessment.lighting_assessments
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.dhw_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.lighting_zones ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.lighting_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_dhw_tenant_isolation
    ON pack032_building_assessment.dhw_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_dhw_service_bypass
    ON pack032_building_assessment.dhw_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_lz_tenant_isolation
    ON pack032_building_assessment.lighting_zones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_lz_service_bypass
    ON pack032_building_assessment.lighting_zones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_la_tenant_isolation
    ON pack032_building_assessment.lighting_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_la_service_bypass
    ON pack032_building_assessment.lighting_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.dhw_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.lighting_zones TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.lighting_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.dhw_systems IS
    'Domestic hot water generation and distribution with efficiency, storage, solar thermal contribution, and legionella regime.';

COMMENT ON TABLE pack032_building_assessment.lighting_zones IS
    'Per-zone lighting inventory with fixture count, wattage, lighting power density, lux levels, and control strategy.';

COMMENT ON TABLE pack032_building_assessment.lighting_assessments IS
    'Building-level lighting assessment summary with LENI, daylight factor, LED penetration, and control coverage.';

COMMENT ON COLUMN pack032_building_assessment.lighting_zones.lpd_w_m2 IS
    'Lighting Power Density in W/m2.';
COMMENT ON COLUMN pack032_building_assessment.lighting_assessments.leni_kwh_m2 IS
    'Lighting Energy Numeric Indicator in kWh/m2 per year (EN 15193).';
COMMENT ON COLUMN pack032_building_assessment.dhw_systems.legionella_regime IS
    'Legionella prevention regime applied to the DHW system.';
