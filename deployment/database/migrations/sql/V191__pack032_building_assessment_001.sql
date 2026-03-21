-- =============================================================================
-- V191: PACK-032 Building Energy Assessment - Core Schema & Building Profiles
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack032_building_assessment schema and foundational tables for
-- building energy assessment management. Tracks building metadata, thermal
-- zones, and building contact personnel.
--
-- Tables (3):
--   1. pack032_building_assessment.building_profiles
--   2. pack032_building_assessment.building_zones
--   3. pack032_building_assessment.building_contacts
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V190__pack031_energy_audit_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack032_building_assessment;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack032_building_assessment.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack032_building_assessment.building_profiles
-- =============================================================================
-- Building profiles for energy assessment scope including EPC/DEC ratings,
-- Energy Star score, CRREM alignment, and physical characteristics.

CREATE TABLE pack032_building_assessment.building_profiles (
    building_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    building_name           VARCHAR(500)    NOT NULL,
    building_type           VARCHAR(100)    NOT NULL,
    address                 TEXT,
    postcode                VARCHAR(20),
    country                 VARCHAR(3)      NOT NULL,
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    year_built              INTEGER,
    gross_floor_area_m2     NUMERIC(14,2),
    net_floor_area_m2       NUMERIC(14,2),
    heated_floor_area_m2    NUMERIC(14,2),
    building_volume_m3      NUMERIC(14,2),
    floors_above_ground     INTEGER,
    floors_below_ground     INTEGER,
    occupancy_type          VARCHAR(100),
    operating_hours         INTEGER,
    occupant_count          INTEGER,
    epc_rating              VARCHAR(5),
    dec_rating              VARCHAR(5),
    energy_star_score       INTEGER,
    crrem_aligned           BOOLEAN         DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_bp_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p032_bp_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p032_bp_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p032_bp_year_built CHECK (
        year_built IS NULL OR (year_built >= 1800 AND year_built <= 2100)
    ),
    CONSTRAINT chk_p032_bp_gross_floor CHECK (
        gross_floor_area_m2 IS NULL OR gross_floor_area_m2 > 0
    ),
    CONSTRAINT chk_p032_bp_net_floor CHECK (
        net_floor_area_m2 IS NULL OR net_floor_area_m2 > 0
    ),
    CONSTRAINT chk_p032_bp_heated_floor CHECK (
        heated_floor_area_m2 IS NULL OR heated_floor_area_m2 > 0
    ),
    CONSTRAINT chk_p032_bp_volume CHECK (
        building_volume_m3 IS NULL OR building_volume_m3 > 0
    ),
    CONSTRAINT chk_p032_bp_floors_above CHECK (
        floors_above_ground IS NULL OR floors_above_ground >= 0
    ),
    CONSTRAINT chk_p032_bp_floors_below CHECK (
        floors_below_ground IS NULL OR floors_below_ground >= 0
    ),
    CONSTRAINT chk_p032_bp_operating_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p032_bp_occupant_count CHECK (
        occupant_count IS NULL OR occupant_count >= 0
    ),
    CONSTRAINT chk_p032_bp_energy_star CHECK (
        energy_star_score IS NULL OR (energy_star_score >= 1 AND energy_star_score <= 100)
    ),
    CONSTRAINT chk_p032_bp_building_type CHECK (
        building_type IN ('OFFICE', 'RETAIL', 'WAREHOUSE', 'INDUSTRIAL', 'RESIDENTIAL',
                          'HOTEL', 'HOSPITAL', 'SCHOOL', 'UNIVERSITY', 'DATA_CENTER',
                          'MIXED_USE', 'LABORATORY', 'LEISURE', 'PUBLIC_BUILDING', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_bp_tenant          ON pack032_building_assessment.building_profiles(tenant_id);
CREATE INDEX idx_p032_bp_building_type   ON pack032_building_assessment.building_profiles(building_type);
CREATE INDEX idx_p032_bp_country         ON pack032_building_assessment.building_profiles(country);
CREATE INDEX idx_p032_bp_postcode        ON pack032_building_assessment.building_profiles(postcode);
CREATE INDEX idx_p032_bp_epc_rating      ON pack032_building_assessment.building_profiles(epc_rating);
CREATE INDEX idx_p032_bp_crrem           ON pack032_building_assessment.building_profiles(crrem_aligned);
CREATE INDEX idx_p032_bp_year_built      ON pack032_building_assessment.building_profiles(year_built);
CREATE INDEX idx_p032_bp_created         ON pack032_building_assessment.building_profiles(created_at DESC);
CREATE INDEX idx_p032_bp_metadata        ON pack032_building_assessment.building_profiles USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_bp_updated
    BEFORE UPDATE ON pack032_building_assessment.building_profiles
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.building_zones
-- =============================================================================
-- Thermal zones within a building for granular energy analysis including
-- setpoint temperatures and occupancy schedules.

CREATE TABLE pack032_building_assessment.building_zones (
    zone_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    zone_name               VARCHAR(255)    NOT NULL,
    zone_type               VARCHAR(100)    NOT NULL,
    floor_area_m2           NUMERIC(12,2),
    volume_m3               NUMERIC(12,2),
    occupancy_hours         INTEGER,
    setpoint_heating_c      NUMERIC(5,2),
    setpoint_cooling_c      NUMERIC(5,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_bz_floor_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT chk_p032_bz_volume CHECK (
        volume_m3 IS NULL OR volume_m3 > 0
    ),
    CONSTRAINT chk_p032_bz_occupancy_hours CHECK (
        occupancy_hours IS NULL OR (occupancy_hours >= 0 AND occupancy_hours <= 8784)
    ),
    CONSTRAINT chk_p032_bz_heating_sp CHECK (
        setpoint_heating_c IS NULL OR (setpoint_heating_c >= -10 AND setpoint_heating_c <= 40)
    ),
    CONSTRAINT chk_p032_bz_cooling_sp CHECK (
        setpoint_cooling_c IS NULL OR (setpoint_cooling_c >= 10 AND setpoint_cooling_c <= 50)
    ),
    CONSTRAINT chk_p032_bz_zone_type CHECK (
        zone_type IN ('HEATED', 'COOLED', 'HEATED_COOLED', 'UNCONDITIONED', 'SEMI_CONDITIONED',
                      'SERVER_ROOM', 'STORAGE', 'CIRCULATION', 'PLANT_ROOM', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_bz_building   ON pack032_building_assessment.building_zones(building_id);
CREATE INDEX idx_p032_bz_tenant     ON pack032_building_assessment.building_zones(tenant_id);
CREATE INDEX idx_p032_bz_zone_type  ON pack032_building_assessment.building_zones(zone_type);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_bz_updated
    BEFORE UPDATE ON pack032_building_assessment.building_zones
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.building_contacts
-- =============================================================================
-- Contact personnel linked to each building for assessment coordination.

CREATE TABLE pack032_building_assessment.building_contacts (
    contact_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    role                    VARCHAR(100),
    name                    VARCHAR(255)    NOT NULL,
    email                   VARCHAR(255),
    phone                   VARCHAR(50),
    is_primary              BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_bc_building   ON pack032_building_assessment.building_contacts(building_id);
CREATE INDEX idx_p032_bc_tenant     ON pack032_building_assessment.building_contacts(tenant_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.building_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.building_zones ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.building_contacts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_bp_tenant_isolation
    ON pack032_building_assessment.building_profiles
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p032_bp_service_bypass
    ON pack032_building_assessment.building_profiles
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_bz_tenant_isolation
    ON pack032_building_assessment.building_zones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p032_bz_service_bypass
    ON pack032_building_assessment.building_zones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_bc_tenant_isolation
    ON pack032_building_assessment.building_contacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p032_bc_service_bypass
    ON pack032_building_assessment.building_contacts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack032_building_assessment TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.building_profiles TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.building_zones TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.building_contacts TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack032_building_assessment.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack032_building_assessment IS
    'PACK-032 Building Energy Assessment Pack - building energy performance assessment, EPC/DEC compliance, CRREM alignment, and retrofit planning.';

COMMENT ON TABLE pack032_building_assessment.building_profiles IS
    'Building profiles for energy assessment scope with EPC/DEC ratings, Energy Star score, CRREM alignment, and physical characteristics.';

COMMENT ON TABLE pack032_building_assessment.building_zones IS
    'Thermal zones within a building for granular energy analysis including setpoint temperatures and occupancy schedules.';

COMMENT ON TABLE pack032_building_assessment.building_contacts IS
    'Contact personnel linked to each building for assessment coordination and communication.';

COMMENT ON COLUMN pack032_building_assessment.building_profiles.building_id IS
    'Unique identifier for the building.';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.building_name IS
    'Human-readable name of the building.';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.building_type IS
    'Classification of the building (OFFICE, RETAIL, WAREHOUSE, etc.).';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.epc_rating IS
    'Energy Performance Certificate rating (A+ to G).';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.dec_rating IS
    'Display Energy Certificate operational rating (A to G).';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.energy_star_score IS
    'US EPA Energy Star score (1-100).';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.crrem_aligned IS
    'Whether the building is aligned with CRREM decarbonisation pathway.';
COMMENT ON COLUMN pack032_building_assessment.building_profiles.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
