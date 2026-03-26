-- =============================================================================
-- V387: PACK-047 GHG Emissions Benchmark Pack - Peer Groups
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates peer group management tables: peer group definitions, individual
-- peer entity definitions, and peer emissions data. Peer groups define the
-- comparison set by sector, size band, and geography. Peer definitions hold
-- individual entities with similarity scoring. Peer data stores annual
-- emissions and intensity values per entity with PCAF quality scoring and
-- GWP version tracking for normalisation.
--
-- Tables (3):
--   1. ghg_benchmark.gl_bm_peer_groups
--   2. ghg_benchmark.gl_bm_peer_definitions
--   3. ghg_benchmark.gl_bm_peer_data
--
-- Also includes: indexes, RLS, comments.
-- Previous: V386__pack047_core_schema.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_peer_groups
-- =============================================================================
-- Peer group definitions for benchmark comparisons. Each group has a sector
-- code, sector classification system (NACE, SIC, GICS, ICB), size band,
-- minimum/maximum peer thresholds, geographic weighting, quality scoring,
-- and provenance tracking. An organisation may have multiple peer groups for
-- different perspectives (e.g., global sector vs regional sub-sector).

CREATE TABLE ghg_benchmark.gl_bm_peer_groups (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    group_name                  VARCHAR(255)    NOT NULL,
    sector_code                 VARCHAR(50)     NOT NULL,
    sector_system               VARCHAR(30)     NOT NULL DEFAULT 'NACE',
    size_band                   VARCHAR(50),
    min_peers                   INTEGER         NOT NULL DEFAULT 5,
    max_peers                   INTEGER         NOT NULL DEFAULT 100,
    geographic_weight           NUMERIC(5,3)    NOT NULL DEFAULT 1.000,
    peer_count                  INTEGER         NOT NULL DEFAULT 0,
    quality_score               NUMERIC(5,3),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p047_pg_sector_system CHECK (
        sector_system IN ('NACE', 'SIC', 'GICS', 'ICB', 'ISIC', 'NAICS', 'CUSTOM')
    ),
    CONSTRAINT chk_p047_pg_size_band CHECK (
        size_band IS NULL OR size_band IN (
            'ALL', 'MICRO', 'SMALL', 'MEDIUM', 'LARGE', 'ENTERPRISE',
            'MEGA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p047_pg_min_peers CHECK (
        min_peers >= 1
    ),
    CONSTRAINT chk_p047_pg_max_peers CHECK (
        max_peers >= min_peers
    ),
    CONSTRAINT chk_p047_pg_geo_weight CHECK (
        geographic_weight >= 0 AND geographic_weight <= 10
    ),
    CONSTRAINT chk_p047_pg_peer_count CHECK (
        peer_count >= 0
    ),
    CONSTRAINT chk_p047_pg_quality CHECK (
        quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 5)
    ),
    CONSTRAINT uq_p047_pg_config_name UNIQUE (config_id, group_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pg_tenant            ON ghg_benchmark.gl_bm_peer_groups(tenant_id);
CREATE INDEX idx_p047_pg_config            ON ghg_benchmark.gl_bm_peer_groups(config_id);
CREATE INDEX idx_p047_pg_sector_code       ON ghg_benchmark.gl_bm_peer_groups(sector_code);
CREATE INDEX idx_p047_pg_sector_system     ON ghg_benchmark.gl_bm_peer_groups(sector_system);
CREATE INDEX idx_p047_pg_size_band         ON ghg_benchmark.gl_bm_peer_groups(size_band);
CREATE INDEX idx_p047_pg_active            ON ghg_benchmark.gl_bm_peer_groups(is_active) WHERE is_active = true;
CREATE INDEX idx_p047_pg_quality           ON ghg_benchmark.gl_bm_peer_groups(quality_score);
CREATE INDEX idx_p047_pg_created           ON ghg_benchmark.gl_bm_peer_groups(created_at DESC);
CREATE INDEX idx_p047_pg_metadata          ON ghg_benchmark.gl_bm_peer_groups USING GIN(metadata);

-- Composite: tenant + config for listing
CREATE INDEX idx_p047_pg_tenant_config     ON ghg_benchmark.gl_bm_peer_groups(tenant_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_pg_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_peer_groups
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_peer_definitions
-- =============================================================================
-- Individual peer entity definitions within a peer group. Each record
-- represents one comparable entity with its sector code, revenue (for size
-- classification), country, emission factor, similarity score against the
-- reference organisation, data quality scoring, and outlier flagging.
-- Similarity scoring uses cosine distance across normalised features.

CREATE TABLE ghg_benchmark.gl_bm_peer_definitions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    peer_group_id               UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_peer_groups(id) ON DELETE CASCADE,
    peer_name                   VARCHAR(255)    NOT NULL,
    peer_identifier             VARCHAR(100),
    sector_code                 VARCHAR(50),
    revenue                     NUMERIC(20,2),
    country_code                VARCHAR(3),
    emission_factor             NUMERIC(20,10),
    similarity_score            NUMERIC(5,4),
    data_quality_score          NUMERIC(3,1),
    is_outlier                  BOOLEAN         NOT NULL DEFAULT false,
    source_dataset              VARCHAR(100),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_pd_revenue CHECK (
        revenue IS NULL OR revenue >= 0
    ),
    CONSTRAINT chk_p047_pd_country CHECK (
        country_code IS NULL OR LENGTH(country_code) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p047_pd_emission_factor CHECK (
        emission_factor IS NULL OR emission_factor >= 0
    ),
    CONSTRAINT chk_p047_pd_similarity CHECK (
        similarity_score IS NULL OR (similarity_score >= 0 AND similarity_score <= 1)
    ),
    CONSTRAINT chk_p047_pd_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 1 AND data_quality_score <= 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pd_tenant            ON ghg_benchmark.gl_bm_peer_definitions(tenant_id);
CREATE INDEX idx_p047_pd_peer_group        ON ghg_benchmark.gl_bm_peer_definitions(peer_group_id);
CREATE INDEX idx_p047_pd_peer_name         ON ghg_benchmark.gl_bm_peer_definitions(peer_name);
CREATE INDEX idx_p047_pd_sector_code       ON ghg_benchmark.gl_bm_peer_definitions(sector_code);
CREATE INDEX idx_p047_pd_country           ON ghg_benchmark.gl_bm_peer_definitions(country_code);
CREATE INDEX idx_p047_pd_similarity        ON ghg_benchmark.gl_bm_peer_definitions(similarity_score DESC);
CREATE INDEX idx_p047_pd_outlier           ON ghg_benchmark.gl_bm_peer_definitions(is_outlier) WHERE is_outlier = true;
CREATE INDEX idx_p047_pd_source            ON ghg_benchmark.gl_bm_peer_definitions(source_dataset);
CREATE INDEX idx_p047_pd_created           ON ghg_benchmark.gl_bm_peer_definitions(created_at DESC);
CREATE INDEX idx_p047_pd_metadata          ON ghg_benchmark.gl_bm_peer_definitions USING GIN(metadata);

-- Composite: peer group + similarity for ranked retrieval
CREATE INDEX idx_p047_pd_group_sim         ON ghg_benchmark.gl_bm_peer_definitions(peer_group_id, similarity_score DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_pd_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_peer_definitions
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_benchmark.gl_bm_peer_data
-- =============================================================================
-- Annual emissions and intensity data per peer entity. Each record holds one
-- reporting year's emissions broken down by scope (scope1, scope2 location/
-- market, scope3), total emissions, denominator value with unit, and derived
-- intensity. Includes data source reference, verification status, GWP
-- version, and PCAF quality scoring (1-5) for PCAF-aligned analyses.

CREATE TABLE ghg_benchmark.gl_bm_peer_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    peer_definition_id          UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_peer_definitions(id) ON DELETE CASCADE,
    reporting_year              INTEGER         NOT NULL,
    scope1                      NUMERIC(20,6),
    scope2_location             NUMERIC(20,6),
    scope2_market               NUMERIC(20,6),
    scope3                      NUMERIC(20,6),
    total_emissions             NUMERIC(20,6),
    denominator_value           NUMERIC(20,6),
    denominator_unit            VARCHAR(50),
    intensity                   NUMERIC(20,10),
    data_source                 VARCHAR(100),
    verification_status         VARCHAR(30),
    gwp_version                 VARCHAR(10),
    pcaf_score                  INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_pdata_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p047_pdata_scope1 CHECK (
        scope1 IS NULL OR scope1 >= 0
    ),
    CONSTRAINT chk_p047_pdata_scope2l CHECK (
        scope2_location IS NULL OR scope2_location >= 0
    ),
    CONSTRAINT chk_p047_pdata_scope2m CHECK (
        scope2_market IS NULL OR scope2_market >= 0
    ),
    CONSTRAINT chk_p047_pdata_scope3 CHECK (
        scope3 IS NULL OR scope3 >= 0
    ),
    CONSTRAINT chk_p047_pdata_total CHECK (
        total_emissions IS NULL OR total_emissions >= 0
    ),
    CONSTRAINT chk_p047_pdata_denominator CHECK (
        denominator_value IS NULL OR denominator_value >= 0
    ),
    CONSTRAINT chk_p047_pdata_intensity CHECK (
        intensity IS NULL OR intensity >= 0
    ),
    CONSTRAINT chk_p047_pdata_verification CHECK (
        verification_status IS NULL OR verification_status IN (
            'VERIFIED', 'LIMITED_ASSURANCE', 'REASONABLE_ASSURANCE',
            'SELF_REPORTED', 'ESTIMATED', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p047_pdata_gwp CHECK (
        gwp_version IS NULL OR gwp_version IN ('AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p047_pdata_pcaf CHECK (
        pcaf_score IS NULL OR (pcaf_score >= 1 AND pcaf_score <= 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pdata_tenant         ON ghg_benchmark.gl_bm_peer_data(tenant_id);
CREATE INDEX idx_p047_pdata_peer_def       ON ghg_benchmark.gl_bm_peer_data(peer_definition_id);
CREATE INDEX idx_p047_pdata_year           ON ghg_benchmark.gl_bm_peer_data(reporting_year);
CREATE INDEX idx_p047_pdata_source         ON ghg_benchmark.gl_bm_peer_data(data_source);
CREATE INDEX idx_p047_pdata_verification   ON ghg_benchmark.gl_bm_peer_data(verification_status);
CREATE INDEX idx_p047_pdata_gwp            ON ghg_benchmark.gl_bm_peer_data(gwp_version);
CREATE INDEX idx_p047_pdata_pcaf           ON ghg_benchmark.gl_bm_peer_data(pcaf_score);
CREATE INDEX idx_p047_pdata_created        ON ghg_benchmark.gl_bm_peer_data(created_at DESC);
CREATE INDEX idx_p047_pdata_metadata       ON ghg_benchmark.gl_bm_peer_data USING GIN(metadata);

-- Composite: peer definition + year for time series retrieval
CREATE INDEX idx_p047_pdata_def_year       ON ghg_benchmark.gl_bm_peer_data(peer_definition_id, reporting_year);

-- Composite: reporting year + verification for quality filtering
CREATE INDEX idx_p047_pdata_year_verif     ON ghg_benchmark.gl_bm_peer_data(reporting_year, verification_status);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_peer_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_peer_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_peer_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_pg_tenant_isolation
    ON ghg_benchmark.gl_bm_peer_groups
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pg_service_bypass
    ON ghg_benchmark.gl_bm_peer_groups
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_pd_tenant_isolation
    ON ghg_benchmark.gl_bm_peer_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pd_service_bypass
    ON ghg_benchmark.gl_bm_peer_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_pdata_tenant_isolation
    ON ghg_benchmark.gl_bm_peer_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pdata_service_bypass
    ON ghg_benchmark.gl_bm_peer_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_peer_groups TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_peer_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_peer_data TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_groups IS
    'Peer group definitions for benchmark comparisons with sector, size band, geographic weighting, and quality scoring.';
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_definitions IS
    'Individual peer entity definitions with similarity scoring, outlier flagging, and data quality assessment.';
COMMENT ON TABLE ghg_benchmark.gl_bm_peer_data IS
    'Annual emissions and intensity data per peer entity with scope breakdown, PCAF scoring, and GWP version tracking.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_groups.sector_system IS 'Sector classification system: NACE, SIC, GICS, ICB, ISIC, NAICS, or CUSTOM.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_groups.min_peers IS 'Minimum number of peers required for statistically valid benchmarking (default 5).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_groups.geographic_weight IS 'Weight multiplier for geographic proximity in peer selection (1.0 = neutral).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_groups.quality_score IS 'Aggregate data quality score for the peer group (0-5 scale, higher = better).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_definitions.similarity_score IS 'Cosine similarity score (0-1) between peer and reference organisation across normalised features.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_definitions.is_outlier IS 'Whether this peer has been flagged as a statistical outlier (IQR or Z-score method).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_definitions.source_dataset IS 'Source dataset from which the peer was drawn, e.g. CDP_2024, TPI_2025, GRESB_2024.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_data.pcaf_score IS 'PCAF data quality score (1-5): 1=verified emissions, 2=reported emissions, 3=physical activity, 4=revenue-based, 5=estimated.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_data.gwp_version IS 'IPCC Assessment Report GWP values used: AR4, AR5, or AR6.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_peer_data.verification_status IS 'Emissions verification level: VERIFIED, LIMITED_ASSURANCE, REASONABLE_ASSURANCE, SELF_REPORTED, ESTIMATED, UNKNOWN.';
