-- =============================================================================
-- V266: PACK-035 Energy Benchmark Pack - Core Schema & Facility Profiles
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack035_energy_benchmark schema and foundational tables for
-- energy benchmarking of buildings and facilities. Tracks facility profiles,
-- building types, floor areas, and metering points.
--
-- Tables (2):
--   1. pack035_energy_benchmark.facility_profiles
--   2. pack035_energy_benchmark.metering_points
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V265__pack034_iso50001_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack035_energy_benchmark;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack035_energy_benchmark.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack035_energy_benchmark.facility_profiles
-- =============================================================================
-- Building and facility profiles for energy benchmarking scope including
-- building type classification, climate zone, floor areas, operational
-- parameters, and HVAC system metadata.

CREATE TABLE pack035_energy_benchmark.facility_profiles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_name           VARCHAR(255)    NOT NULL,
    building_type           VARCHAR(50)     NOT NULL,
    climate_zone            VARCHAR(20),
    country_code            CHAR(2)         NOT NULL,
    region                  VARCHAR(100),
    city                    VARCHAR(100),
    postal_code             VARCHAR(20),
    latitude                DECIMAL(10, 7),
    longitude               DECIMAL(10, 7),
    year_built              INTEGER,
    year_renovated          INTEGER,
    gross_internal_area_m2  DECIMAL(12, 2)  NOT NULL,
    net_internal_area_m2    DECIMAL(12, 2),
    gross_lettable_area_m2  DECIMAL(12, 2),
    treated_floor_area_m2   DECIMAL(12, 2),
    floors_above_ground     INTEGER,
    floors_below_ground     INTEGER         DEFAULT 0,
    typical_occupancy       INTEGER,
    operating_hours_per_week DECIMAL(5, 1),
    hvac_system_type        VARCHAR(100),
    heating_fuel            VARCHAR(50),
    cooling_system          VARCHAR(50),
    has_sub_metering        BOOLEAN         DEFAULT false,
    energy_star_property_id VARCHAR(50),
    epc_certificate_id      VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p035_building_type CHECK (
        building_type IN (
            'OFFICE', 'RETAIL', 'WAREHOUSE', 'MANUFACTURING', 'HEALTHCARE',
            'EDUCATION', 'DATA_CENTER', 'HOTEL', 'RESTAURANT', 'MIXED_USE',
            'RESIDENTIAL_MULTIFAMILY', 'LABORATORY', 'LIBRARY', 'WORSHIP',
            'ENTERTAINMENT', 'SPORTS', 'PARKING', 'SME'
        )
    ),
    CONSTRAINT chk_p035_country_code CHECK (
        LENGTH(country_code) = 2
    ),
    CONSTRAINT chk_p035_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p035_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p035_gia_positive CHECK (
        gross_internal_area_m2 > 0
    ),
    CONSTRAINT chk_p035_nia_non_neg CHECK (
        net_internal_area_m2 IS NULL OR net_internal_area_m2 >= 0
    ),
    CONSTRAINT chk_p035_gla_non_neg CHECK (
        gross_lettable_area_m2 IS NULL OR gross_lettable_area_m2 >= 0
    ),
    CONSTRAINT chk_p035_tfa_non_neg CHECK (
        treated_floor_area_m2 IS NULL OR treated_floor_area_m2 >= 0
    ),
    CONSTRAINT chk_p035_year_built CHECK (
        year_built IS NULL OR (year_built >= 1800 AND year_built <= 2100)
    ),
    CONSTRAINT chk_p035_year_renovated CHECK (
        year_renovated IS NULL OR (year_renovated >= 1800 AND year_renovated <= 2100)
    ),
    CONSTRAINT chk_p035_floors_above CHECK (
        floors_above_ground IS NULL OR floors_above_ground >= 0
    ),
    CONSTRAINT chk_p035_floors_below CHECK (
        floors_below_ground IS NULL OR floors_below_ground >= 0
    ),
    CONSTRAINT chk_p035_occupancy CHECK (
        typical_occupancy IS NULL OR typical_occupancy >= 0
    ),
    CONSTRAINT chk_p035_operating_hours CHECK (
        operating_hours_per_week IS NULL OR (operating_hours_per_week >= 0 AND operating_hours_per_week <= 168)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p035_fp_tenant          ON pack035_energy_benchmark.facility_profiles(tenant_id);
CREATE INDEX idx_p035_fp_building_type   ON pack035_energy_benchmark.facility_profiles(building_type);
CREATE INDEX idx_p035_fp_country         ON pack035_energy_benchmark.facility_profiles(country_code);
CREATE INDEX idx_p035_fp_climate_zone    ON pack035_energy_benchmark.facility_profiles(climate_zone);
CREATE INDEX idx_p035_fp_city            ON pack035_energy_benchmark.facility_profiles(city);
CREATE INDEX idx_p035_fp_year_built      ON pack035_energy_benchmark.facility_profiles(year_built);
CREATE INDEX idx_p035_fp_gia             ON pack035_energy_benchmark.facility_profiles(gross_internal_area_m2);
CREATE INDEX idx_p035_fp_created         ON pack035_energy_benchmark.facility_profiles(created_at DESC);
CREATE INDEX idx_p035_fp_metadata        ON pack035_energy_benchmark.facility_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p035_fp_updated
    BEFORE UPDATE ON pack035_energy_benchmark.facility_profiles
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack035_energy_benchmark.metering_points
-- =============================================================================
-- Metering point definitions per facility including energy carrier type,
-- meter hierarchy (main/sub), and end-use categorisation.

CREATE TABLE pack035_energy_benchmark.metering_points (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    meter_name              VARCHAR(255)    NOT NULL,
    energy_carrier          VARCHAR(50)     NOT NULL,
    meter_type              VARCHAR(50)     NOT NULL DEFAULT 'MAIN',
    unit                    VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    is_sub_meter            BOOLEAN         DEFAULT false,
    parent_meter_id         UUID            REFERENCES pack035_energy_benchmark.metering_points(id) ON DELETE SET NULL,
    serves_end_use          VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_carrier CHECK (
        energy_carrier IN (
            'ELECTRICITY', 'NATURAL_GAS', 'FUEL_OIL', 'LPG',
            'DISTRICT_HEATING', 'DISTRICT_COOLING', 'BIOMASS',
            'SOLAR_THERMAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_meter_type CHECK (
        meter_type IN ('MAIN', 'SUB', 'CHECK', 'VIRTUAL', 'FISCAL')
    ),
    CONSTRAINT chk_p035_end_use CHECK (
        serves_end_use IS NULL OR serves_end_use IN (
            'HEATING', 'COOLING', 'LIGHTING', 'VENTILATION', 'DHW',
            'PLUG_LOADS', 'PROCESS', 'CATERING', 'IT_EQUIPMENT',
            'VERTICAL_TRANSPORT', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p035_mp_facility        ON pack035_energy_benchmark.metering_points(facility_id);
CREATE INDEX idx_p035_mp_tenant          ON pack035_energy_benchmark.metering_points(tenant_id);
CREATE INDEX idx_p035_mp_carrier         ON pack035_energy_benchmark.metering_points(energy_carrier);
CREATE INDEX idx_p035_mp_type            ON pack035_energy_benchmark.metering_points(meter_type);
CREATE INDEX idx_p035_mp_parent          ON pack035_energy_benchmark.metering_points(parent_meter_id);
CREATE INDEX idx_p035_mp_end_use         ON pack035_energy_benchmark.metering_points(serves_end_use);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.facility_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.metering_points ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_fp_tenant_isolation
    ON pack035_energy_benchmark.facility_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p035_fp_service_bypass
    ON pack035_energy_benchmark.facility_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_mp_tenant_isolation
    ON pack035_energy_benchmark.metering_points
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p035_mp_service_bypass
    ON pack035_energy_benchmark.metering_points
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack035_energy_benchmark TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.facility_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.metering_points TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack035_energy_benchmark.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack035_energy_benchmark IS
    'PACK-035 Energy Benchmark Pack - energy benchmarking, EUI calculation, peer comparison, and performance rating for buildings and facilities.';

COMMENT ON TABLE pack035_energy_benchmark.facility_profiles IS
    'Building and facility profiles for energy benchmarking with building type, climate zone, floor areas, and operational parameters.';

COMMENT ON TABLE pack035_energy_benchmark.metering_points IS
    'Metering point definitions per facility with energy carrier type, meter hierarchy, and end-use categorisation.';

COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.id IS
    'Unique identifier for the facility profile.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.building_type IS
    'Building type classification aligned with CIBSE TM46/ENERGY STAR categories.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.climate_zone IS
    'Climate zone code (e.g., Koppen-Geiger or ASHRAE climate zone).';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.gross_internal_area_m2 IS
    'Total gross internal floor area in square metres, primary normalisation denominator for EUI.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.treated_floor_area_m2 IS
    'Treated (conditioned) floor area for EPC/DEC calculations per EPBD methodology.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.energy_star_property_id IS
    'ENERGY STAR Portfolio Manager property ID if registered.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.epc_certificate_id IS
    'Energy Performance Certificate identifier for EU buildings.';
COMMENT ON COLUMN pack035_energy_benchmark.facility_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack035_energy_benchmark.metering_points.energy_carrier IS
    'Energy carrier type: ELECTRICITY, NATURAL_GAS, FUEL_OIL, LPG, DISTRICT_HEATING, DISTRICT_COOLING, BIOMASS, SOLAR_THERMAL, OTHER.';
COMMENT ON COLUMN pack035_energy_benchmark.metering_points.serves_end_use IS
    'End-use category served by this meter for sub-metered disaggregation.';
