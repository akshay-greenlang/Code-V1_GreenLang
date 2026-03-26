-- =============================================================================
-- PACK-049 GHG Multi-Site Management Pack
-- Migration: V410 - Regional Factors
-- =============================================================================
-- Pack:         PACK-049 (GHG Multi-Site Management Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates regional emission factor management tables. Multi-site portfolios
-- span multiple countries and grid regions, each with distinct emission
-- factors. This migration provides: factor assignments per site, manual
-- overrides with approval workflow, grid region definitions, and climate
-- zone classifications.
--
-- Tables (4):
--   1. ghg_multisite.gl_ms_factor_assignments
--   2. ghg_multisite.gl_ms_factor_overrides
--   3. ghg_multisite.gl_ms_grid_regions
--   4. ghg_multisite.gl_ms_climate_zones
--
-- Also includes: indexes, RLS, comments.
-- Previous: V409__pack049_boundary.sql
-- =============================================================================

SET search_path TO ghg_multisite, public;

-- =============================================================================
-- Table 1: ghg_multisite.gl_ms_factor_assignments
-- =============================================================================
-- Links sites to their applicable emission factor sets based on location,
-- grid region, and regulatory jurisdiction. Each assignment specifies the
-- factor source, tier level, and vintage year.

CREATE TABLE ghg_multisite.gl_ms_factor_assignments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    factor_scope                VARCHAR(10)     NOT NULL,
    factor_category             VARCHAR(100)    NOT NULL,
    factor_source               VARCHAR(100)    NOT NULL,
    factor_tier                 INTEGER         NOT NULL DEFAULT 2,
    factor_value                NUMERIC(20,10)  NOT NULL,
    factor_unit                 VARCHAR(100)    NOT NULL,
    grid_region_id              UUID,
    country                     VARCHAR(3)      NOT NULL,
    vintage_year                INTEGER         NOT NULL,
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to                DATE,
    is_default                  BOOLEAN         NOT NULL DEFAULT false,
    source_reference            VARCHAR(500),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p049_fa_scope CHECK (
        factor_scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3')
    ),
    CONSTRAINT chk_p049_fa_tier CHECK (
        factor_tier >= 1 AND factor_tier <= 4
    ),
    CONSTRAINT chk_p049_fa_value CHECK (factor_value > 0),
    CONSTRAINT chk_p049_fa_vintage CHECK (
        vintage_year >= 1990 AND vintage_year <= 2100
    ),
    CONSTRAINT chk_p049_fa_dates CHECK (
        effective_to IS NULL OR effective_to > effective_from
    ),
    CONSTRAINT uq_p049_fa_site_scope_cat UNIQUE (
        site_id, factor_scope, factor_category, effective_from
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_fa_tenant          ON ghg_multisite.gl_ms_factor_assignments(tenant_id);
CREATE INDEX idx_p049_fa_config          ON ghg_multisite.gl_ms_factor_assignments(config_id);
CREATE INDEX idx_p049_fa_site            ON ghg_multisite.gl_ms_factor_assignments(site_id);
CREATE INDEX idx_p049_fa_scope           ON ghg_multisite.gl_ms_factor_assignments(factor_scope);
CREATE INDEX idx_p049_fa_source          ON ghg_multisite.gl_ms_factor_assignments(factor_source);
CREATE INDEX idx_p049_fa_tier            ON ghg_multisite.gl_ms_factor_assignments(factor_tier);
CREATE INDEX idx_p049_fa_country         ON ghg_multisite.gl_ms_factor_assignments(country);
CREATE INDEX idx_p049_fa_vintage         ON ghg_multisite.gl_ms_factor_assignments(vintage_year);
CREATE INDEX idx_p049_fa_grid            ON ghg_multisite.gl_ms_factor_assignments(grid_region_id)
    WHERE grid_region_id IS NOT NULL;
CREATE INDEX idx_p049_fa_default         ON ghg_multisite.gl_ms_factor_assignments(config_id, is_default)
    WHERE is_default = true;
CREATE INDEX idx_p049_fa_active          ON ghg_multisite.gl_ms_factor_assignments(site_id)
    WHERE effective_to IS NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_factor_assignments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_fa_tenant_isolation ON ghg_multisite.gl_ms_factor_assignments
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 2: ghg_multisite.gl_ms_factor_overrides
-- =============================================================================
-- Manual overrides of default emission factors for specific sites. Overrides
-- require justification and approval, and maintain a link to the original
-- assignment for audit trail purposes.

CREATE TABLE ghg_multisite.gl_ms_factor_overrides (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    assignment_id               UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_factor_assignments(id) ON DELETE CASCADE,
    site_id                     UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_sites(id) ON DELETE CASCADE,
    original_value              NUMERIC(20,10)  NOT NULL,
    override_value              NUMERIC(20,10)  NOT NULL,
    override_reason             VARCHAR(50)     NOT NULL,
    justification               TEXT            NOT NULL,
    supporting_evidence         VARCHAR(500),
    status                      VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    requested_by                UUID,
    requested_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    rejection_reason            TEXT,
    effective_from              DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to                DATE,
    provenance_hash             VARCHAR(64),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_fo_reason CHECK (
        override_reason IN (
            'SUPPLIER_SPECIFIC', 'MEASURED_VALUE', 'REGULATORY_REQUIREMENT',
            'INDUSTRY_SPECIFIC', 'TECHNOLOGY_UPDATE', 'LOCAL_GRID_DATA',
            'VERIFIED_FACTOR', 'CORRECTION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p049_fo_status CHECK (
        status IN ('PENDING', 'APPROVED', 'REJECTED', 'EXPIRED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p049_fo_orig CHECK (original_value > 0),
    CONSTRAINT chk_p049_fo_over CHECK (override_value > 0),
    CONSTRAINT chk_p049_fo_dates CHECK (
        effective_to IS NULL OR effective_to > effective_from
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_fo_tenant          ON ghg_multisite.gl_ms_factor_overrides(tenant_id);
CREATE INDEX idx_p049_fo_assignment      ON ghg_multisite.gl_ms_factor_overrides(assignment_id);
CREATE INDEX idx_p049_fo_site            ON ghg_multisite.gl_ms_factor_overrides(site_id);
CREATE INDEX idx_p049_fo_status          ON ghg_multisite.gl_ms_factor_overrides(status);
CREATE INDEX idx_p049_fo_reason          ON ghg_multisite.gl_ms_factor_overrides(override_reason);
CREATE INDEX idx_p049_fo_pending         ON ghg_multisite.gl_ms_factor_overrides(status)
    WHERE status = 'PENDING';
CREATE INDEX idx_p049_fo_approved        ON ghg_multisite.gl_ms_factor_overrides(site_id, status)
    WHERE status = 'APPROVED';
CREATE INDEX idx_p049_fo_active          ON ghg_multisite.gl_ms_factor_overrides(site_id)
    WHERE status = 'APPROVED' AND effective_to IS NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_factor_overrides ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_fo_tenant_isolation ON ghg_multisite.gl_ms_factor_overrides
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 3: ghg_multisite.gl_ms_grid_regions
-- =============================================================================
-- Reference table for electricity grid regions. Each grid region has a
-- location-based and (optionally) residual emission factor, plus metadata
-- about the grid mix.

CREATE TABLE ghg_multisite.gl_ms_grid_regions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    region_code                 VARCHAR(50)     NOT NULL,
    region_name                 VARCHAR(255)    NOT NULL,
    country                     VARCHAR(3)      NOT NULL,
    grid_operator               VARCHAR(255),
    location_factor_kgco2_kwh   NUMERIC(12,8),
    residual_factor_kgco2_kwh   NUMERIC(12,8),
    t_and_d_loss_pct            NUMERIC(10,4),
    renewable_share_pct         NUMERIC(10,4),
    factor_source               VARCHAR(255),
    vintage_year                INTEGER,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_gr_loc_factor CHECK (
        location_factor_kgco2_kwh IS NULL OR location_factor_kgco2_kwh >= 0
    ),
    CONSTRAINT chk_p049_gr_res_factor CHECK (
        residual_factor_kgco2_kwh IS NULL OR residual_factor_kgco2_kwh >= 0
    ),
    CONSTRAINT chk_p049_gr_td_loss CHECK (
        t_and_d_loss_pct IS NULL OR (t_and_d_loss_pct >= 0 AND t_and_d_loss_pct <= 100)
    ),
    CONSTRAINT chk_p049_gr_renewable CHECK (
        renewable_share_pct IS NULL OR (renewable_share_pct >= 0 AND renewable_share_pct <= 100)
    ),
    CONSTRAINT chk_p049_gr_vintage CHECK (
        vintage_year IS NULL OR (vintage_year >= 1990 AND vintage_year <= 2100)
    ),
    CONSTRAINT uq_p049_gr_cfg_code UNIQUE (config_id, region_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_gr_tenant          ON ghg_multisite.gl_ms_grid_regions(tenant_id);
CREATE INDEX idx_p049_gr_config          ON ghg_multisite.gl_ms_grid_regions(config_id);
CREATE INDEX idx_p049_gr_code            ON ghg_multisite.gl_ms_grid_regions(region_code);
CREATE INDEX idx_p049_gr_country         ON ghg_multisite.gl_ms_grid_regions(country);
CREATE INDEX idx_p049_gr_active          ON ghg_multisite.gl_ms_grid_regions(config_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p049_gr_vintage         ON ghg_multisite.gl_ms_grid_regions(vintage_year)
    WHERE vintage_year IS NOT NULL;

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_grid_regions ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_gr_tenant_isolation ON ghg_multisite.gl_ms_grid_regions
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- =============================================================================
-- Table 4: ghg_multisite.gl_ms_climate_zones
-- =============================================================================
-- Climate zone classifications affecting energy use patterns and
-- normalisation. Sites in different climate zones are normalised before
-- comparison using heating/cooling degree day adjustments.

CREATE TABLE ghg_multisite.gl_ms_climate_zones (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_multisite.gl_ms_configurations(id) ON DELETE CASCADE,
    zone_code                   VARCHAR(30)     NOT NULL,
    zone_name                   VARCHAR(255)    NOT NULL,
    zone_classification         VARCHAR(30)     NOT NULL DEFAULT 'KOPPEN',
    heating_degree_days         NUMERIC(10,2),
    cooling_degree_days         NUMERIC(10,2),
    base_temperature_c          NUMERIC(6,2)    NOT NULL DEFAULT 18.00,
    annual_avg_temperature_c    NUMERIC(6,2),
    annual_precipitation_mm     NUMERIC(10,2),
    solar_irradiance_kwh_m2     NUMERIC(10,2),
    countries                   JSONB           DEFAULT '[]',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p049_cz_class CHECK (
        zone_classification IN (
            'KOPPEN', 'ASHRAE', 'CEN', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p049_cz_hdd CHECK (
        heating_degree_days IS NULL OR heating_degree_days >= 0
    ),
    CONSTRAINT chk_p049_cz_cdd CHECK (
        cooling_degree_days IS NULL OR cooling_degree_days >= 0
    ),
    CONSTRAINT chk_p049_cz_base_temp CHECK (
        base_temperature_c >= -50 AND base_temperature_c <= 50
    ),
    CONSTRAINT chk_p049_cz_avg_temp CHECK (
        annual_avg_temperature_c IS NULL OR (annual_avg_temperature_c >= -60 AND annual_avg_temperature_c <= 60)
    ),
    CONSTRAINT chk_p049_cz_precip CHECK (
        annual_precipitation_mm IS NULL OR annual_precipitation_mm >= 0
    ),
    CONSTRAINT chk_p049_cz_solar CHECK (
        solar_irradiance_kwh_m2 IS NULL OR solar_irradiance_kwh_m2 >= 0
    ),
    CONSTRAINT uq_p049_cz_cfg_code UNIQUE (config_id, zone_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p049_cz_tenant          ON ghg_multisite.gl_ms_climate_zones(tenant_id);
CREATE INDEX idx_p049_cz_config          ON ghg_multisite.gl_ms_climate_zones(config_id);
CREATE INDEX idx_p049_cz_code            ON ghg_multisite.gl_ms_climate_zones(zone_code);
CREATE INDEX idx_p049_cz_class           ON ghg_multisite.gl_ms_climate_zones(zone_classification);
CREATE INDEX idx_p049_cz_active          ON ghg_multisite.gl_ms_climate_zones(config_id, is_active)
    WHERE is_active = true;
CREATE INDEX idx_p049_cz_countries       ON ghg_multisite.gl_ms_climate_zones USING gin(countries);

-- ---------------------------------------------------------------------------
-- RLS
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_multisite.gl_ms_climate_zones ENABLE ROW LEVEL SECURITY;

CREATE POLICY p049_cz_tenant_isolation ON ghg_multisite.gl_ms_climate_zones
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_multisite.gl_ms_factor_assignments IS
    'PACK-049: Site-level emission factor assignments with tier, source, vintage, and grid region linkage.';
COMMENT ON TABLE ghg_multisite.gl_ms_factor_overrides IS
    'PACK-049: Manual factor overrides (9 reasons) with approval workflow and audit provenance.';
COMMENT ON TABLE ghg_multisite.gl_ms_grid_regions IS
    'PACK-049: Grid region definitions with location/residual factors, T&D losses, and renewable share.';
COMMENT ON TABLE ghg_multisite.gl_ms_climate_zones IS
    'PACK-049: Climate zone classifications (4 systems) with HDD/CDD, temperature, and solar data.';
