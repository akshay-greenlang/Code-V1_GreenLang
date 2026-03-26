-- =============================================================================
-- V386: PACK-047 GHG Emissions Benchmark Pack - Core Schema
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    001 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the ghg_benchmark schema and foundational tables for GHG emissions
-- benchmarking. Tracks organisation-level benchmark configurations (sector
-- classification, peer size band, scope alignment, pathway enablement) and
-- reporting periods for which benchmark analyses are performed. Supports
-- multi-pathway alignment (IEA NZE, IPCC AR6, SBTi SDA, OECM, TPI, CRREM),
-- portfolio-level carbon benchmarking, and implied temperature rating (ITR).
--
-- Tables (2):
--   1. ghg_benchmark.gl_bm_configurations
--   2. ghg_benchmark.gl_bm_reporting_periods
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V385__pack046_views_indexes_seed.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS ghg_benchmark;

SET search_path TO ghg_benchmark, public;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ghg_benchmark.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_configurations
-- =============================================================================
-- Pack-level benchmark configuration per organisation. Defines the sector
-- context, peer size band selection, scope alignment for comparisons,
-- enabled pathways for alignment analysis, and configuration versioning.
-- Each organisation may have multiple configurations (e.g., corporate-level
-- vs portfolio-level), but each config_name must be unique within the tenant.

CREATE TABLE ghg_benchmark.gl_bm_configurations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_name                 VARCHAR(255)    NOT NULL,
    config_version              INTEGER         NOT NULL DEFAULT 1,
    sector_classification       VARCHAR(100)    NOT NULL,
    peer_size_band              VARCHAR(50)     NOT NULL DEFAULT 'ALL',
    scope_alignment             VARCHAR(50)     NOT NULL DEFAULT 'SCOPE_1_2_LOCATION',
    pathways_enabled            JSONB           NOT NULL DEFAULT '["IEA_NZE","SBTI_SDA"]',
    base_currency               VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    gwp_version                 VARCHAR(10)     NOT NULL DEFAULT 'AR6',
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    config_hash                 TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p047_cfg_sector CHECK (
        sector_classification IN (
            'MULTI_SECTOR', 'MANUFACTURING', 'REAL_ESTATE', 'POWER_GENERATION',
            'TRANSPORT', 'FREIGHT_TRANSPORT', 'PASSENGER_TRANSPORT',
            'CEMENT', 'STEEL', 'ALUMINIUM', 'PULP_PAPER', 'CHEMICALS',
            'FOOD', 'BEVERAGE', 'MINING', 'OIL_GAS', 'UTILITIES',
            'BANKING', 'INSURANCE', 'ASSET_MANAGEMENT', 'HEALTHCARE',
            'HOSPITALITY', 'RETAIL', 'DATA_CENTER', 'ICT', 'AGRICULTURE',
            'FORESTRY', 'SERVICES', 'OFFICE', 'COMMERCIAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p047_cfg_size_band CHECK (
        peer_size_band IN (
            'ALL', 'MICRO', 'SMALL', 'MEDIUM', 'LARGE', 'ENTERPRISE',
            'MEGA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p047_cfg_scope CHECK (
        scope_alignment IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p047_cfg_gwp CHECK (
        gwp_version IN ('AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p047_cfg_currency CHECK (
        LENGTH(base_currency) = 3
    ),
    CONSTRAINT chk_p047_cfg_version CHECK (
        config_version >= 1
    ),
    CONSTRAINT uq_p047_cfg_tenant_name UNIQUE (tenant_id, config_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_cfg_tenant           ON ghg_benchmark.gl_bm_configurations(tenant_id);
CREATE INDEX idx_p047_cfg_sector           ON ghg_benchmark.gl_bm_configurations(sector_classification);
CREATE INDEX idx_p047_cfg_size_band        ON ghg_benchmark.gl_bm_configurations(peer_size_band);
CREATE INDEX idx_p047_cfg_scope            ON ghg_benchmark.gl_bm_configurations(scope_alignment);
CREATE INDEX idx_p047_cfg_gwp              ON ghg_benchmark.gl_bm_configurations(gwp_version);
CREATE INDEX idx_p047_cfg_active           ON ghg_benchmark.gl_bm_configurations(is_active) WHERE is_active = true;
CREATE INDEX idx_p047_cfg_created          ON ghg_benchmark.gl_bm_configurations(created_at DESC);
CREATE INDEX idx_p047_cfg_pathways         ON ghg_benchmark.gl_bm_configurations USING GIN(pathways_enabled);
CREATE INDEX idx_p047_cfg_metadata         ON ghg_benchmark.gl_bm_configurations USING GIN(metadata);

-- Composite: tenant + active for filtered listing
CREATE INDEX idx_p047_cfg_tenant_active    ON ghg_benchmark.gl_bm_configurations(tenant_id, is_active);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_cfg_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_configurations
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_reporting_periods
-- =============================================================================
-- Reporting periods for which benchmark analyses are executed. Each period
-- is linked to a configuration and has a label (e.g., 'FY2025', 'CY2025'),
-- start/end dates, optional fiscal year end date, and a status tracking the
-- benchmark lifecycle from DRAFT through ACTIVE to CLOSED.

CREATE TABLE ghg_benchmark.gl_bm_reporting_periods (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    period_name                 VARCHAR(100)    NOT NULL,
    start_date                  DATE            NOT NULL,
    end_date                    DATE            NOT NULL,
    fiscal_year_end             DATE,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_rp_dates CHECK (
        end_date > start_date
    ),
    CONSTRAINT chk_p047_rp_status CHECK (
        status IN ('DRAFT', 'ACTIVE', 'CLOSED')
    ),
    CONSTRAINT uq_p047_rp_config_name UNIQUE (config_id, period_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_rp_tenant            ON ghg_benchmark.gl_bm_reporting_periods(tenant_id);
CREATE INDEX idx_p047_rp_config            ON ghg_benchmark.gl_bm_reporting_periods(config_id);
CREATE INDEX idx_p047_rp_name              ON ghg_benchmark.gl_bm_reporting_periods(period_name);
CREATE INDEX idx_p047_rp_status            ON ghg_benchmark.gl_bm_reporting_periods(status);
CREATE INDEX idx_p047_rp_dates             ON ghg_benchmark.gl_bm_reporting_periods(start_date, end_date);
CREATE INDEX idx_p047_rp_created           ON ghg_benchmark.gl_bm_reporting_periods(created_at DESC);

-- Composite: tenant + config for filtered queries
CREATE INDEX idx_p047_rp_tenant_config     ON ghg_benchmark.gl_bm_reporting_periods(tenant_id, config_id);

-- Composite: config + status for lifecycle queries
CREATE INDEX idx_p047_rp_config_status     ON ghg_benchmark.gl_bm_reporting_periods(config_id, status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_rp_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_reporting_periods
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_reporting_periods ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_cfg_tenant_isolation
    ON ghg_benchmark.gl_bm_configurations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_cfg_service_bypass
    ON ghg_benchmark.gl_bm_configurations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_rp_tenant_isolation
    ON ghg_benchmark.gl_bm_reporting_periods
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_rp_service_bypass
    ON ghg_benchmark.gl_bm_reporting_periods
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA ghg_benchmark TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_configurations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_reporting_periods TO PUBLIC;
GRANT EXECUTE ON FUNCTION ghg_benchmark.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA ghg_benchmark IS
    'PACK-047 GHG Emissions Benchmark Pack - Complete benchmarking lifecycle including peer groups, normalisation, external datasets, pathway alignment, implied temperature rating, trajectory analysis, portfolio carbon benchmarking, transition risk, and data quality.';

COMMENT ON TABLE ghg_benchmark.gl_bm_configurations IS
    'Organisation-level benchmark configuration defining sector context, peer size band, scope alignment, enabled pathways, and GWP version for benchmark analyses.';
COMMENT ON TABLE ghg_benchmark.gl_bm_reporting_periods IS
    'Reporting periods for benchmark analyses with lifecycle status tracking (DRAFT, ACTIVE, CLOSED).';

COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.sector_classification IS 'Sector classification for peer group selection and pathway alignment mapping.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.peer_size_band IS 'Size band filter for peer selection: MICRO (<10M), SMALL (10-50M), MEDIUM (50-250M), LARGE (250M-1B), ENTERPRISE (1B+), MEGA (10B+).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.scope_alignment IS 'Which emission scopes to include in benchmark comparisons: SCOPE_1_ONLY through ALL_SCOPES.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.pathways_enabled IS 'JSON array of enabled pathway types for alignment analysis, e.g. ["IEA_NZE","SBTI_SDA","IPCC_AR6_C1"].';
COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.config_hash IS 'SHA-256 hash of configuration parameters for change detection and versioning.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_configurations.gwp_version IS 'Global Warming Potential version used: AR4, AR5, or AR6 (IPCC Assessment Report).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_reporting_periods.period_name IS 'Human-readable period identifier, e.g. FY2025, CY2025, Q4-2025.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_reporting_periods.fiscal_year_end IS 'Fiscal year end date for alignment with corporate reporting calendars.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_reporting_periods.status IS 'Benchmark lifecycle: DRAFT (configuration), ACTIVE (analysis in progress), CLOSED (finalised).';
