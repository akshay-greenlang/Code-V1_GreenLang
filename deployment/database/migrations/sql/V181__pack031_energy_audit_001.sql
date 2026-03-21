-- =============================================================================
-- V181: PACK-031 Industrial Energy Audit - Core Schema & Facility Profiles
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack031_energy_audit schema and foundational tables for
-- industrial facility energy audit management. Tracks facility metadata,
-- energy carriers, and facility contact personnel.
--
-- Tables (3):
--   1. pack031_energy_audit.energy_audit_facilities
--   2. pack031_energy_audit.energy_carriers
--   3. pack031_energy_audit.facility_contacts
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V180__PACK027_views_and_indexes.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack031_energy_audit;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack031_energy_audit.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack031_energy_audit.energy_audit_facilities
-- =============================================================================
-- Industrial facility profiles for energy audit scope including EED obligation,
-- ISO 50001 certification, EU ETS coverage, and physical characteristics.

CREATE TABLE pack031_energy_audit.energy_audit_facilities (
    facility_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(500)    NOT NULL,
    sector                  VARCHAR(100)    NOT NULL,
    sub_sector              VARCHAR(100),
    address                 TEXT,
    country                 VARCHAR(3)      NOT NULL,
    latitude                NUMERIC(10,7),
    longitude               NUMERIC(10,7),
    area_sqm                NUMERIC(14,2),
    production_capacity     VARCHAR(255),
    operating_hours_per_year INTEGER,
    employees               INTEGER,
    revenue_eur             NUMERIC(18,2),
    eed_obligation          BOOLEAN         DEFAULT FALSE,
    iso_50001_certified     BOOLEAN         DEFAULT FALSE,
    eu_ets_covered          BOOLEAN         DEFAULT FALSE,
    facility_status         VARCHAR(30)     DEFAULT 'active',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_fac_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p031_fac_latitude CHECK (
        latitude IS NULL OR (latitude >= -90 AND latitude <= 90)
    ),
    CONSTRAINT chk_p031_fac_longitude CHECK (
        longitude IS NULL OR (longitude >= -180 AND longitude <= 180)
    ),
    CONSTRAINT chk_p031_fac_area_non_neg CHECK (
        area_sqm IS NULL OR area_sqm >= 0
    ),
    CONSTRAINT chk_p031_fac_hours CHECK (
        operating_hours_per_year IS NULL OR (operating_hours_per_year >= 0 AND operating_hours_per_year <= 8784)
    ),
    CONSTRAINT chk_p031_fac_employees CHECK (
        employees IS NULL OR employees >= 0
    ),
    CONSTRAINT chk_p031_fac_revenue CHECK (
        revenue_eur IS NULL OR revenue_eur >= 0
    ),
    CONSTRAINT chk_p031_fac_status CHECK (
        facility_status IN ('active', 'inactive', 'decommissioned', 'pending')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_fac_org          ON pack031_energy_audit.energy_audit_facilities(org_id);
CREATE INDEX idx_p031_fac_tenant       ON pack031_energy_audit.energy_audit_facilities(tenant_id);
CREATE INDEX idx_p031_fac_sector       ON pack031_energy_audit.energy_audit_facilities(sector);
CREATE INDEX idx_p031_fac_sub_sector   ON pack031_energy_audit.energy_audit_facilities(sub_sector);
CREATE INDEX idx_p031_fac_country      ON pack031_energy_audit.energy_audit_facilities(country);
CREATE INDEX idx_p031_fac_eed          ON pack031_energy_audit.energy_audit_facilities(eed_obligation);
CREATE INDEX idx_p031_fac_iso50001     ON pack031_energy_audit.energy_audit_facilities(iso_50001_certified);
CREATE INDEX idx_p031_fac_eu_ets       ON pack031_energy_audit.energy_audit_facilities(eu_ets_covered);
CREATE INDEX idx_p031_fac_status       ON pack031_energy_audit.energy_audit_facilities(facility_status);
CREATE INDEX idx_p031_fac_created      ON pack031_energy_audit.energy_audit_facilities(created_at DESC);
CREATE INDEX idx_p031_fac_metadata     ON pack031_energy_audit.energy_audit_facilities USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p031_fac_updated
    BEFORE UPDATE ON pack031_energy_audit.energy_audit_facilities
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.energy_carriers
-- =============================================================================
-- Energy carrier definitions per facility (electricity, natural gas, diesel,
-- steam, etc.) with cost and CO2 emission factors.

CREATE TABLE pack031_energy_audit.energy_carriers (
    carrier_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    carrier_type            VARCHAR(100)    NOT NULL,
    unit                    VARCHAR(30)     NOT NULL,
    cost_per_unit_eur       NUMERIC(12,4),
    co2_factor_kg_per_kwh   NUMERIC(10,6),
    calorific_value_kwh     NUMERIC(12,6),
    source_description      VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_carrier_type CHECK (
        carrier_type IN ('ELECTRICITY', 'NATURAL_GAS', 'DIESEL', 'HEAVY_FUEL_OIL',
                         'LPG', 'COAL', 'BIOMASS', 'DISTRICT_HEATING', 'DISTRICT_COOLING',
                         'STEAM', 'COMPRESSED_AIR', 'SOLAR_THERMAL', 'GEOTHERMAL', 'OTHER')
    ),
    CONSTRAINT chk_p031_carrier_cost CHECK (
        cost_per_unit_eur IS NULL OR cost_per_unit_eur >= 0
    ),
    CONSTRAINT chk_p031_carrier_co2 CHECK (
        co2_factor_kg_per_kwh IS NULL OR co2_factor_kg_per_kwh >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_carrier_facility ON pack031_energy_audit.energy_carriers(facility_id);
CREATE INDEX idx_p031_carrier_tenant   ON pack031_energy_audit.energy_carriers(tenant_id);
CREATE INDEX idx_p031_carrier_type     ON pack031_energy_audit.energy_carriers(carrier_type);

-- =============================================================================
-- Table 3: pack031_energy_audit.facility_contacts
-- =============================================================================
-- Contact personnel linked to each facility for audit coordination.

CREATE TABLE pack031_energy_audit.facility_contacts (
    contact_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    role                    VARCHAR(100),
    email                   VARCHAR(255),
    phone                   VARCHAR(50),
    is_primary              BOOLEAN         DEFAULT FALSE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p031_contact_facility ON pack031_energy_audit.facility_contacts(facility_id);
CREATE INDEX idx_p031_contact_tenant   ON pack031_energy_audit.facility_contacts(tenant_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.energy_audit_facilities ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.energy_carriers ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.facility_contacts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_fac_tenant_isolation
    ON pack031_energy_audit.energy_audit_facilities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p031_fac_service_bypass
    ON pack031_energy_audit.energy_audit_facilities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_carrier_tenant_isolation
    ON pack031_energy_audit.energy_carriers
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p031_carrier_service_bypass
    ON pack031_energy_audit.energy_carriers
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_contact_tenant_isolation
    ON pack031_energy_audit.facility_contacts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p031_contact_service_bypass
    ON pack031_energy_audit.facility_contacts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack031_energy_audit TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_audit_facilities TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.energy_carriers TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.facility_contacts TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack031_energy_audit.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack031_energy_audit IS
    'PACK-031 Industrial Energy Audit Pack - industrial energy audit management, EN 16247 compliance, and energy savings verification.';

COMMENT ON TABLE pack031_energy_audit.energy_audit_facilities IS
    'Industrial facility profiles for energy audit scope with EED obligation, ISO 50001 certification, EU ETS coverage, and physical characteristics.';

COMMENT ON TABLE pack031_energy_audit.energy_carriers IS
    'Energy carrier definitions per facility (electricity, gas, diesel, steam, etc.) with cost and CO2 emission factors.';

COMMENT ON TABLE pack031_energy_audit.facility_contacts IS
    'Contact personnel linked to each facility for audit coordination and communication.';

COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.facility_id IS
    'Unique identifier for the facility.';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.org_id IS
    'Organization that owns this facility.';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.tenant_id IS
    'Multi-tenant isolation key.';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.eed_obligation IS
    'Whether the facility falls under the EU Energy Efficiency Directive audit obligation.';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.iso_50001_certified IS
    'Whether the facility holds ISO 50001 certification (exempt from EED audits).';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.eu_ets_covered IS
    'Whether the facility is covered by the EU Emissions Trading System.';
COMMENT ON COLUMN pack031_energy_audit.energy_audit_facilities.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
