-- =============================================================================
-- V256: PACK-034 ISO 50001 Energy Management System - Core EnMS Schema
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    001 of 010
-- Date:         March 2026
--
-- Creates the pack034_iso50001 schema and foundational tables for ISO 50001
-- Energy Management System compliance. Tracks EnMS lifecycle, scope boundaries,
-- and organizational commitment.
--
-- Tables (3):
--   1. pack034_iso50001.energy_management_systems
--   2. pack034_iso50001.enms_scope
--   3. pack034_iso50001.enms_boundaries
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V255__pack033_quick_wins_010.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack034_iso50001;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack034_iso50001.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack034_iso50001.energy_management_systems
-- =============================================================================
-- Core EnMS registry tracking organizational energy management systems,
-- their lifecycle status, certification details, and PDCA cycle tracking.

CREATE TABLE pack034_iso50001.energy_management_systems (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id             UUID            NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    scope_description           TEXT,
    boundaries_json             JSONB           DEFAULT '{}',
    policy_statement            TEXT,
    top_management_commitment   TEXT,
    enms_status                 VARCHAR(30)     NOT NULL DEFAULT 'planning',
    certification_body          VARCHAR(255),
    certification_date          DATE,
    next_surveillance_date      DATE,
    pdca_cycle_count            INTEGER         NOT NULL DEFAULT 0,
    created_by                  UUID,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ems_status CHECK (
        enms_status IN ('planning', 'implementing', 'operational', 'certified')
    ),
    CONSTRAINT chk_p034_ems_pdca CHECK (
        pdca_cycle_count >= 0
    ),
    CONSTRAINT chk_p034_ems_cert_date CHECK (
        certification_date IS NULL OR certification_date <= next_surveillance_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ems_org            ON pack034_iso50001.energy_management_systems(organization_id);
CREATE INDEX idx_p034_ems_status         ON pack034_iso50001.energy_management_systems(enms_status);
CREATE INDEX idx_p034_ems_cert_body      ON pack034_iso50001.energy_management_systems(certification_body);
CREATE INDEX idx_p034_ems_cert_date      ON pack034_iso50001.energy_management_systems(certification_date);
CREATE INDEX idx_p034_ems_surveillance   ON pack034_iso50001.energy_management_systems(next_surveillance_date);
CREATE INDEX idx_p034_ems_created        ON pack034_iso50001.energy_management_systems(created_at DESC);
CREATE INDEX idx_p034_ems_boundaries     ON pack034_iso50001.energy_management_systems USING GIN(boundaries_json);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ems_updated
    BEFORE UPDATE ON pack034_iso50001.energy_management_systems
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.enms_scope
-- =============================================================================
-- Defines the scope of each EnMS, identifying what is included and excluded
-- at the site, process, and equipment boundary levels.

CREATE TABLE pack034_iso50001.enms_scope (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    boundary_type               VARCHAR(30)     NOT NULL,
    name                        VARCHAR(500)    NOT NULL,
    description                 TEXT,
    included                    BOOLEAN         NOT NULL DEFAULT TRUE,
    justification               TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_scope_type CHECK (
        boundary_type IN ('site', 'process', 'equipment')
    ),
    CONSTRAINT chk_p034_scope_justification CHECK (
        included = TRUE OR justification IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_scope_enms         ON pack034_iso50001.enms_scope(enms_id);
CREATE INDEX idx_p034_scope_type         ON pack034_iso50001.enms_scope(boundary_type);
CREATE INDEX idx_p034_scope_included     ON pack034_iso50001.enms_scope(included);
CREATE INDEX idx_p034_scope_created      ON pack034_iso50001.enms_scope(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_scope_updated
    BEFORE UPDATE ON pack034_iso50001.enms_scope
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.enms_boundaries
-- =============================================================================
-- Physical and organizational boundary definitions for each EnMS, including
-- the energy types consumed and any excluded areas with justification.

CREATE TABLE pack034_iso50001.enms_boundaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    physical_boundary           TEXT            NOT NULL,
    organizational_boundary     TEXT            NOT NULL,
    energy_types_json           JSONB           NOT NULL DEFAULT '[]',
    excluded_areas_json         JSONB           DEFAULT '[]',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_bounds_enms        ON pack034_iso50001.enms_boundaries(enms_id);
CREATE INDEX idx_p034_bounds_energy      ON pack034_iso50001.enms_boundaries USING GIN(energy_types_json);
CREATE INDEX idx_p034_bounds_excluded    ON pack034_iso50001.enms_boundaries USING GIN(excluded_areas_json);
CREATE INDEX idx_p034_bounds_created     ON pack034_iso50001.enms_boundaries(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_bounds_updated
    BEFORE UPDATE ON pack034_iso50001.enms_boundaries
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.energy_management_systems ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.enms_scope ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.enms_boundaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_ems_tenant_isolation
    ON pack034_iso50001.energy_management_systems
    USING (organization_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p034_ems_service_bypass
    ON pack034_iso50001.energy_management_systems
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_scope_tenant_isolation
    ON pack034_iso50001.enms_scope
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_scope_service_bypass
    ON pack034_iso50001.enms_scope
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_bounds_tenant_isolation
    ON pack034_iso50001.enms_boundaries
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_bounds_service_bypass
    ON pack034_iso50001.enms_boundaries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack034_iso50001 TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.energy_management_systems TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.enms_scope TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.enms_boundaries TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack034_iso50001.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack034_iso50001 IS
    'PACK-034 ISO 50001 Energy Management System Pack - implements ISO 50001:2018 compliant energy management with PDCA cycle tracking, SEU analysis, EnPI monitoring, CUSUM analytics, and certification readiness.';

COMMENT ON TABLE pack034_iso50001.energy_management_systems IS
    'Core EnMS registry tracking organizational energy management systems, lifecycle status, certification details, and PDCA cycle counts.';

COMMENT ON TABLE pack034_iso50001.enms_scope IS
    'Scope definitions identifying included/excluded sites, processes, and equipment with justification for exclusions.';

COMMENT ON TABLE pack034_iso50001.enms_boundaries IS
    'Physical and organizational boundary definitions with energy types and excluded areas.';

COMMENT ON COLUMN pack034_iso50001.energy_management_systems.id IS
    'Unique identifier for the energy management system.';
COMMENT ON COLUMN pack034_iso50001.energy_management_systems.organization_id IS
    'Multi-tenant isolation key referencing the organization.';
COMMENT ON COLUMN pack034_iso50001.energy_management_systems.enms_status IS
    'Current lifecycle status: planning, implementing, operational, or certified.';
COMMENT ON COLUMN pack034_iso50001.energy_management_systems.pdca_cycle_count IS
    'Number of completed Plan-Do-Check-Act improvement cycles.';
COMMENT ON COLUMN pack034_iso50001.energy_management_systems.certification_body IS
    'Name of the third-party certification body (e.g., BSI, TUV, SGS).';
COMMENT ON COLUMN pack034_iso50001.energy_management_systems.next_surveillance_date IS
    'Date of next surveillance audit by the certification body.';
COMMENT ON COLUMN pack034_iso50001.enms_scope.boundary_type IS
    'Type of boundary: site (physical location), process (operational), or equipment (asset-level).';
COMMENT ON COLUMN pack034_iso50001.enms_scope.included IS
    'Whether this item is included in the EnMS scope. Exclusions require justification.';
COMMENT ON COLUMN pack034_iso50001.enms_boundaries.energy_types_json IS
    'JSON array of energy types consumed (e.g., ["electricity", "natural_gas", "diesel"]).';
COMMENT ON COLUMN pack034_iso50001.enms_boundaries.excluded_areas_json IS
    'JSON array of excluded areas with justification for each exclusion.';
