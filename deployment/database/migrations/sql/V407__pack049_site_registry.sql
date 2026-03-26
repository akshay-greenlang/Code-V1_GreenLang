-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V407 - Site Registry
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates site registry tables for multi-site portfolio management. Sites
-- are the fundamental unit of data collection and consolidation. Each site
-- has a unique code, classification, location, lifecycle status, and
-- operational characteristics. Sites can be organised into groups for
-- peer comparison and reporting.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_sites
--   2. ghg_multisite.gl_ms_site_characteristics
--   3. ghg_multisite.gl_ms_site_groups
--   4. ghg_multisite.gl_ms_site_group_members
--
-- Also includes: indexes, RLS, comments.
-- Previous: V406__pack049_core_schema.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_sites
-- =============================================================================

CREATE TABLE ghg_multisite.gl_ms_sites (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    site_code                   VARCHAR(50)     NOT NULL,
    site_name                   VARCHAR(255)    NOT NULL,
    facility_type               VARCHAR(50)     NOT NULL DEFAULT 'OFFICE',
    legal_entity_id             UUID,
    business_unit               VARCHAR(100),
    country                     VARCHAR(3)      NOT NULL,
    region                      VARCHAR(100),
    city                        VARCHAR(100),
    postal_code                 VARCHAR(20),
    latitude                    NUMERIC(10,7),
    longitude                   NUMERIC(10,7),
    lifecycle_status             VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    acquisition_date            DATE,
    commissioning_date          DATE,
    decommissioning_date        DATE,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    tags                        JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_site_type CHECK (
        facility_type IN (
            'MANUFACTURING', 'OFFICE', 'WAREHOUSE', 'RETAIL',
            'DATA_CENTER', 'LABORATORY', 'HOSPITAL', 'HOTEL',
            'SCHOOL', 'UNIVERSITY', 'TRANSPORT_HUB', 'PORT',
            'AIRPORT', 'MINE', 'REFINERY', 'POWER_PLANT',
            'FARM', 'MIXED_USE', 'CONSTRUCTION_SITE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_site_lifecycle CHECK (
        lifecycle_status IN (
            'PLANNED', 'UNDER_CONSTRUCTION', 'ACTIVE',
            'MOTHBALLED', 'DECOMMISSIONING', 'DECOMMISSIONED'
        )
    ),
    CONSTRAINT chk_p049_site_lat CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90)),
    CONSTRAINT chk_p049_site_lon CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180)),
    CONSTRAINT uq_p049_site_tenant_code UNIQUE (tenant_id, config_id, site_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_site_tenant         ON ghg_multisite.gl_ms_sites(tenant_id);
CREATE INDEX idx_p049_site_config         ON ghg_multisite.gl_ms_sites(config_id);
CREATE INDEX idx_p049_site_code           ON ghg_multisite.gl_ms_sites(site_code);
CREATE INDEX idx_p049_site_type           ON ghg_multisite.gl_ms_sites(facility_type);
CREATE INDEX idx_p049_site_country        ON ghg_multisite.gl_ms_sites(country);
CREATE INDEX idx_p049_site_bu             ON ghg_multisite.gl_ms_sites(business_unit)
    WHERE business_unit IS NOT NULL;
CREATE INDEX idx_p049_site_active         ON ghg_multisite.gl_ms_sites(tenant_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p049_site_lifecycle      ON ghg_multisite.gl_ms_sites(lifecycle_status);
CREATE INDEX idx_p049_site_entity         ON ghg_multisite.gl_ms_sites(legal_entity_id)
    WHERE legal_entity_id IS NOT NULL;
CREATE INDEX idx_p049_site_tags           ON ghg_multisite.gl_ms_sites USING gin(tags);
CREATE INDEX idx_p049_site_meta           ON ghg_multisite.gl_ms_sites USING gin(metadata);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_sites ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_site_tenant_isolation ON ghg_multisite.gl_ms_sites
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_site_characteristics
-- =============================================================================

CREATE TABLE ghg_multisite.gl_ms_site_characteristics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    floor_area_m2               NUMERIC(14,2),
    headcount                   INTEGER,
    operating_hours_per_year    NUMERIC(8,2),
    production_output           NUMERIC(20,6),
    production_unit             VARCHAR(50),
    grid_region                 VARCHAR(100),
    climate_zone                VARCHAR(50),
    electricity_provider        VARCHAR(255),
    gas_provider                VARCHAR(255),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_sc_area CHECK (floor_area_m2 IS NULL OR floor_area_m2 >= 0),
    CONSTRAINT chk_p049_sc_hc CHECK (headcount IS NULL OR headcount >= 0),
    CONSTRAINT chk_p049_sc_hours CHECK (operating_hours_per_year IS NULL OR operating_hours_per_year >= 0),
    CONSTRAINT uq_p049_sc_site UNIQUE (site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_sc_tenant           ON ghg_multisite.gl_ms_site_characteristics(tenant_id);
CREATE INDEX idx_p049_sc_site             ON ghg_multisite.gl_ms_site_characteristics(site_id);
CREATE INDEX idx_p049_sc_grid             ON ghg_multisite.gl_ms_site_characteristics(grid_region)
    WHERE grid_region IS NOT NULL;
CREATE INDEX idx_p049_sc_climate          ON ghg_multisite.gl_ms_site_characteristics(climate_zone)
    WHERE climate_zone IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_characteristics ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_sc_tenant_isolation ON ghg_multisite.gl_ms_site_characteristics
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_site_groups
-- =============================================================================

CREATE TABLE ghg_multisite.gl_ms_site_groups (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    group_name                  VARCHAR(255)    NOT NULL,
    group_type                  VARCHAR(30)     NOT NULL DEFAULT 'PEER_GROUP',
    description                 TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_sg_type CHECK (
        group_type IN (
            'PEER_GROUP', 'BUSINESS_UNIT', 'REGION', 'FACILITY_TYPE',
            'REPORTING_SEGMENT', 'CUSTOM'
        )
    ),
    CONSTRAINT uq_p049_sg_cfg_name UNIQUE (config_id, group_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_sg_tenant           ON ghg_multisite.gl_ms_site_groups(tenant_id);
CREATE INDEX idx_p049_sg_config           ON ghg_multisite.gl_ms_site_groups(config_id);
CREATE INDEX idx_p049_sg_type             ON ghg_multisite.gl_ms_site_groups(group_type);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_groups ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_sg_tenant_isolation ON ghg_multisite.gl_ms_site_groups
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_site_group_members
-- =============================================================================

CREATE TABLE ghg_multisite.gl_ms_site_group_members (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    group_id                    UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_site_groups(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    added_at                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT uq_p049_sgm_group_site UNIQUE (group_id, site_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_sgm_tenant          ON ghg_multisite.gl_ms_site_group_members(tenant_id);
CREATE INDEX idx_p049_sgm_group           ON ghg_multisite.gl_ms_site_group_members(group_id);
CREATE INDEX idx_p049_sgm_site            ON ghg_multisite.gl_ms_site_group_members(site_id);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_site_group_members ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_sgm_tenant_isolation ON ghg_multisite.gl_ms_site_group_members
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_sites IS
    'PACK-049: Site registry with classification (20 types), location, lifecycle, and metadata.';
COMMENT ON TABLE ghg_multisite.gl_ms_site_characteristics IS
    'PACK-049: Operational characteristics per site (area, headcount, hours, production, grid/climate).';
COMMENT ON TABLE ghg_multisite.gl_ms_site_groups IS
    'PACK-049: Site grouping for peer comparison, BU rollup, and regional aggregation.';
COMMENT ON TABLE ghg_multisite.gl_ms_site_group_members IS
    'PACK-049: Many-to-many site-to-group membership with dedup constraint.';
