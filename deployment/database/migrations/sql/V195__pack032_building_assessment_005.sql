-- =============================================================================
-- V195: PACK-032 Building Energy Assessment - Renewable Systems
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Creates renewable energy system tables for on-site generation including
-- photovoltaic panels and solar thermal collectors.
--
-- Tables (3):
--   1. pack032_building_assessment.renewable_systems
--   2. pack032_building_assessment.pv_panels
--   3. pack032_building_assessment.solar_thermal_systems
--
-- Previous: V194__pack032_building_assessment_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.renewable_systems
-- =============================================================================
-- On-site renewable energy generation systems with capacity, annual output,
-- self-consumption ratio, and installation details.

CREATE TABLE pack032_building_assessment.renewable_systems (
    system_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    system_type             VARCHAR(100)    NOT NULL,
    capacity_kwp            NUMERIC(12,2),
    annual_generation_kwh   NUMERIC(14,2),
    self_consumption_pct    NUMERIC(6,2),
    orientation             VARCHAR(20),
    tilt_angle              NUMERIC(6,2),
    area_m2                 NUMERIC(12,2),
    installation_date       DATE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_rs_capacity CHECK (
        capacity_kwp IS NULL OR capacity_kwp > 0
    ),
    CONSTRAINT chk_p032_rs_generation CHECK (
        annual_generation_kwh IS NULL OR annual_generation_kwh >= 0
    ),
    CONSTRAINT chk_p032_rs_self_consumption CHECK (
        self_consumption_pct IS NULL OR (self_consumption_pct >= 0 AND self_consumption_pct <= 100)
    ),
    CONSTRAINT chk_p032_rs_tilt CHECK (
        tilt_angle IS NULL OR (tilt_angle >= 0 AND tilt_angle <= 90)
    ),
    CONSTRAINT chk_p032_rs_area CHECK (
        area_m2 IS NULL OR area_m2 > 0
    ),
    CONSTRAINT chk_p032_rs_orientation CHECK (
        orientation IS NULL OR orientation IN ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'FLAT')
    ),
    CONSTRAINT chk_p032_rs_system_type CHECK (
        system_type IN ('SOLAR_PV', 'SOLAR_THERMAL', 'WIND_TURBINE', 'MICRO_CHP',
                         'BIOMASS_CHP', 'HEAT_PUMP_RENEWABLE', 'MICRO_HYDRO', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_rs_building    ON pack032_building_assessment.renewable_systems(building_id);
CREATE INDEX idx_p032_rs_tenant      ON pack032_building_assessment.renewable_systems(tenant_id);
CREATE INDEX idx_p032_rs_system_type ON pack032_building_assessment.renewable_systems(system_type);
CREATE INDEX idx_p032_rs_install     ON pack032_building_assessment.renewable_systems(installation_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_rs_updated
    BEFORE UPDATE ON pack032_building_assessment.renewable_systems
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.pv_panels
-- =============================================================================
-- Photovoltaic panel details including module specifications, inverter capacity,
-- and performance ratio.

CREATE TABLE pack032_building_assessment.pv_panels (
    panel_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id               UUID            NOT NULL REFERENCES pack032_building_assessment.renewable_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    module_type             VARCHAR(100),
    module_count            INTEGER,
    module_watt             NUMERIC(8,2),
    inverter_capacity_kw    NUMERIC(12,2),
    performance_ratio       NUMERIC(6,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_pv_module_count CHECK (
        module_count IS NULL OR module_count > 0
    ),
    CONSTRAINT chk_p032_pv_module_watt CHECK (
        module_watt IS NULL OR module_watt > 0
    ),
    CONSTRAINT chk_p032_pv_inverter CHECK (
        inverter_capacity_kw IS NULL OR inverter_capacity_kw > 0
    ),
    CONSTRAINT chk_p032_pv_perf_ratio CHECK (
        performance_ratio IS NULL OR (performance_ratio > 0 AND performance_ratio <= 1)
    ),
    CONSTRAINT chk_p032_pv_module_type CHECK (
        module_type IS NULL OR module_type IN ('MONOCRYSTALLINE', 'POLYCRYSTALLINE',
                                                 'THIN_FILM_CDTE', 'THIN_FILM_CIGS',
                                                 'BIFACIAL', 'PEROVSKITE', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_pv_system ON pack032_building_assessment.pv_panels(system_id);
CREATE INDEX idx_p032_pv_tenant ON pack032_building_assessment.pv_panels(tenant_id);

-- =============================================================================
-- Table 3: pack032_building_assessment.solar_thermal_systems
-- =============================================================================
-- Solar thermal collector details with collector area, storage, and solar fraction.

CREATE TABLE pack032_building_assessment.solar_thermal_systems (
    system_id               UUID            NOT NULL REFERENCES pack032_building_assessment.renewable_systems(system_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    collector_type          VARCHAR(100),
    collector_area_m2       NUMERIC(10,2),
    storage_volume_litres   NUMERIC(10,2),
    solar_fraction_pct      NUMERIC(6,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    PRIMARY KEY (system_id),
    CONSTRAINT chk_p032_st_area CHECK (
        collector_area_m2 IS NULL OR collector_area_m2 > 0
    ),
    CONSTRAINT chk_p032_st_storage CHECK (
        storage_volume_litres IS NULL OR storage_volume_litres > 0
    ),
    CONSTRAINT chk_p032_st_fraction CHECK (
        solar_fraction_pct IS NULL OR (solar_fraction_pct >= 0 AND solar_fraction_pct <= 100)
    ),
    CONSTRAINT chk_p032_st_collector_type CHECK (
        collector_type IS NULL OR collector_type IN ('FLAT_PLATE', 'EVACUATED_TUBE',
                                                       'UNGLAZED', 'CONCENTRATING', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_st_system ON pack032_building_assessment.solar_thermal_systems(system_id);
CREATE INDEX idx_p032_st_tenant ON pack032_building_assessment.solar_thermal_systems(tenant_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.renewable_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.pv_panels ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.solar_thermal_systems ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_rs_tenant_isolation
    ON pack032_building_assessment.renewable_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_rs_service_bypass
    ON pack032_building_assessment.renewable_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_pv_tenant_isolation
    ON pack032_building_assessment.pv_panels
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_pv_service_bypass
    ON pack032_building_assessment.pv_panels
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_st_tenant_isolation
    ON pack032_building_assessment.solar_thermal_systems
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_st_service_bypass
    ON pack032_building_assessment.solar_thermal_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.renewable_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.pv_panels TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.solar_thermal_systems TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.renewable_systems IS
    'On-site renewable energy generation systems with capacity, annual output, self-consumption ratio, and installation details.';

COMMENT ON TABLE pack032_building_assessment.pv_panels IS
    'Photovoltaic panel details including module specifications, inverter capacity, and performance ratio.';

COMMENT ON TABLE pack032_building_assessment.solar_thermal_systems IS
    'Solar thermal collector details with collector type, area, storage volume, and solar fraction.';

COMMENT ON COLUMN pack032_building_assessment.renewable_systems.capacity_kwp IS
    'Peak capacity in kilowatts peak (kWp).';
COMMENT ON COLUMN pack032_building_assessment.renewable_systems.self_consumption_pct IS
    'Percentage of generated energy consumed on-site (0-100).';
COMMENT ON COLUMN pack032_building_assessment.pv_panels.performance_ratio IS
    'Ratio of actual to theoretical energy output (0-1), accounting for losses.';
COMMENT ON COLUMN pack032_building_assessment.solar_thermal_systems.solar_fraction_pct IS
    'Percentage of DHW demand met by the solar thermal system.';
