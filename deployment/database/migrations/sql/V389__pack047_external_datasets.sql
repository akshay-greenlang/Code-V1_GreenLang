-- =============================================================================
-- V389: PACK-047 GHG Emissions Benchmark Pack - External Datasets
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for external benchmark dataset management: source registry,
-- ingested data records, and data caching. External sources include CDP
-- questionnaire responses, TPI management quality assessments, GRESB real
-- estate benchmarks, CRREM decarbonisation pathways, ISS ESG analytics,
-- and custom datasets. A cache layer reduces API cost and latency for
-- frequently accessed external data with TTL-based expiry and hit counting.
--
-- Tables (3):
--   1. ghg_benchmark.gl_bm_external_sources
--   2. ghg_benchmark.gl_bm_external_data
--   3. ghg_benchmark.gl_bm_data_cache
--
-- Also includes: indexes, RLS, comments.
-- Previous: V388__pack047_normalisation.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_external_sources
-- =============================================================================
-- Registry of external benchmark data sources. Each source has a type
-- classification (CDP, TPI, GRESB, CRREM, ISS_ESG, CUSTOM), version
-- tracking, data count, cache TTL configuration, and operational status.
-- Metadata stores source-specific configuration (API endpoints, credentials
-- reference, rate limits, data formats).

CREATE TABLE ghg_benchmark.gl_bm_external_sources (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    source_type                 VARCHAR(30)     NOT NULL,
    source_name                 VARCHAR(255)    NOT NULL,
    source_version              VARCHAR(50),
    source_url                  TEXT,
    last_updated                TIMESTAMPTZ,
    data_count                  INTEGER         NOT NULL DEFAULT 0,
    cache_ttl_hours             INTEGER         NOT NULL DEFAULT 24,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'ACTIVE',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p047_es_type CHECK (
        source_type IN (
            'CDP', 'TPI', 'GRESB', 'CRREM', 'ISS_ESG', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p047_es_data_count CHECK (
        data_count >= 0
    ),
    CONSTRAINT chk_p047_es_ttl CHECK (
        cache_ttl_hours >= 0 AND cache_ttl_hours <= 8760
    ),
    CONSTRAINT chk_p047_es_status CHECK (
        status IN ('ACTIVE', 'INACTIVE', 'DEPRECATED', 'ERROR')
    ),
    CONSTRAINT uq_p047_es_tenant_name UNIQUE (tenant_id, source_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_es_tenant            ON ghg_benchmark.gl_bm_external_sources(tenant_id);
CREATE INDEX idx_p047_es_type              ON ghg_benchmark.gl_bm_external_sources(source_type);
CREATE INDEX idx_p047_es_status            ON ghg_benchmark.gl_bm_external_sources(status);
CREATE INDEX idx_p047_es_last_updated      ON ghg_benchmark.gl_bm_external_sources(last_updated DESC);
CREATE INDEX idx_p047_es_created           ON ghg_benchmark.gl_bm_external_sources(created_at DESC);
CREATE INDEX idx_p047_es_metadata          ON ghg_benchmark.gl_bm_external_sources USING GIN(metadata);

-- Composite: tenant + type for filtered listing
CREATE INDEX idx_p047_es_tenant_type       ON ghg_benchmark.gl_bm_external_sources(tenant_id, source_type);

-- Composite: type + status for operational queries
CREATE INDEX idx_p047_es_type_status       ON ghg_benchmark.gl_bm_external_sources(source_type, status);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_es_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_external_sources
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_external_data
-- =============================================================================
-- Ingested data records from external benchmark sources. Each record holds
-- one entity's metric for a specific reporting year. Entity identification
-- supports multiple schemes (ISIN, LEI, ticker, etc.). Metrics can be any
-- named measurement (total_emissions, scope1_intensity, management_quality,
-- carbon_risk_rating, etc.) with numeric value and unit. Raw JSON preserves
-- the complete original record from the source for audit.

CREATE TABLE ghg_benchmark.gl_bm_external_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    source_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_external_sources(id) ON DELETE CASCADE,
    entity_identifier           VARCHAR(100),
    entity_name                 VARCHAR(255)    NOT NULL,
    sector_code                 VARCHAR(50),
    country_code                VARCHAR(3),
    reporting_year              INTEGER         NOT NULL,
    metric_name                 VARCHAR(100)    NOT NULL,
    metric_value                NUMERIC(20,10)  NOT NULL,
    metric_unit                 VARCHAR(50),
    data_quality                INTEGER,
    raw_data                    JSONB,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    ingested_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_ed_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p047_ed_country CHECK (
        country_code IS NULL OR LENGTH(country_code) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p047_ed_quality CHECK (
        data_quality IS NULL OR (data_quality >= 1 AND data_quality <= 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_ed_tenant            ON ghg_benchmark.gl_bm_external_data(tenant_id);
CREATE INDEX idx_p047_ed_source            ON ghg_benchmark.gl_bm_external_data(source_id);
CREATE INDEX idx_p047_ed_entity_id         ON ghg_benchmark.gl_bm_external_data(entity_identifier);
CREATE INDEX idx_p047_ed_entity_name       ON ghg_benchmark.gl_bm_external_data(entity_name);
CREATE INDEX idx_p047_ed_sector            ON ghg_benchmark.gl_bm_external_data(sector_code);
CREATE INDEX idx_p047_ed_country           ON ghg_benchmark.gl_bm_external_data(country_code);
CREATE INDEX idx_p047_ed_year              ON ghg_benchmark.gl_bm_external_data(reporting_year);
CREATE INDEX idx_p047_ed_metric            ON ghg_benchmark.gl_bm_external_data(metric_name);
CREATE INDEX idx_p047_ed_quality           ON ghg_benchmark.gl_bm_external_data(data_quality);
CREATE INDEX idx_p047_ed_ingested          ON ghg_benchmark.gl_bm_external_data(ingested_at DESC);
CREATE INDEX idx_p047_ed_created           ON ghg_benchmark.gl_bm_external_data(created_at DESC);
CREATE INDEX idx_p047_ed_raw_data          ON ghg_benchmark.gl_bm_external_data USING GIN(raw_data);

-- Composite: source + year for batch retrieval
CREATE INDEX idx_p047_ed_source_year       ON ghg_benchmark.gl_bm_external_data(source_id, reporting_year);

-- Composite: entity + metric for entity-level analysis
CREATE INDEX idx_p047_ed_entity_metric     ON ghg_benchmark.gl_bm_external_data(entity_identifier, metric_name);

-- Composite: sector + year for sector-level aggregation
CREATE INDEX idx_p047_ed_sector_year       ON ghg_benchmark.gl_bm_external_data(sector_code, reporting_year);

-- =============================================================================
-- Table 3: ghg_benchmark.gl_bm_data_cache
-- =============================================================================
-- Cache layer for external data API responses. Reduces cost and latency for
-- frequently accessed benchmark data. Each cache entry has a unique key
-- (typically hash of source + query parameters), TTL-based expiry, and hit
-- counting for access pattern analysis. Expired entries are cleaned by
-- background processes.

CREATE TABLE ghg_benchmark.gl_bm_data_cache (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    cache_key                   TEXT            NOT NULL,
    source_id                   UUID            REFERENCES ghg_benchmark.gl_bm_external_sources(id) ON DELETE SET NULL,
    cached_data                 JSONB           NOT NULL,
    cached_at                   TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    expires_at                  TIMESTAMPTZ     NOT NULL,
    hit_count                   INTEGER         NOT NULL DEFAULT 0,
    last_hit_at                 TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_dc_expires CHECK (
        expires_at > cached_at
    ),
    CONSTRAINT chk_p047_dc_hits CHECK (
        hit_count >= 0
    ),
    CONSTRAINT uq_p047_dc_tenant_key UNIQUE (tenant_id, cache_key)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_dc_tenant            ON ghg_benchmark.gl_bm_data_cache(tenant_id);
CREATE INDEX idx_p047_dc_cache_key         ON ghg_benchmark.gl_bm_data_cache(cache_key);
CREATE INDEX idx_p047_dc_source            ON ghg_benchmark.gl_bm_data_cache(source_id);
CREATE INDEX idx_p047_dc_expires           ON ghg_benchmark.gl_bm_data_cache(expires_at);
CREATE INDEX idx_p047_dc_cached            ON ghg_benchmark.gl_bm_data_cache(cached_at DESC);
CREATE INDEX idx_p047_dc_hit_count         ON ghg_benchmark.gl_bm_data_cache(hit_count DESC);
CREATE INDEX idx_p047_dc_created           ON ghg_benchmark.gl_bm_data_cache(created_at DESC);

-- Partial: unexpired cache entries for active lookup
CREATE INDEX idx_p047_dc_active            ON ghg_benchmark.gl_bm_data_cache(tenant_id, cache_key) WHERE expires_at > NOW();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_external_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_external_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_data_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_es_tenant_isolation
    ON ghg_benchmark.gl_bm_external_sources
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_es_service_bypass
    ON ghg_benchmark.gl_bm_external_sources
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_ed_tenant_isolation
    ON ghg_benchmark.gl_bm_external_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_ed_service_bypass
    ON ghg_benchmark.gl_bm_external_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_dc_tenant_isolation
    ON ghg_benchmark.gl_bm_data_cache
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_dc_service_bypass
    ON ghg_benchmark.gl_bm_data_cache
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_external_sources TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_external_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_data_cache TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_external_sources IS
    'Registry of external benchmark data sources (CDP, TPI, GRESB, CRREM, ISS ESG, Custom) with version tracking and cache TTL configuration.';
COMMENT ON TABLE ghg_benchmark.gl_bm_external_data IS
    'Ingested data records from external sources with entity identification, metric values, quality scoring, and raw JSON preservation.';
COMMENT ON TABLE ghg_benchmark.gl_bm_data_cache IS
    'Cache layer for external API responses with TTL-based expiry, hit counting, and background cleanup support.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_external_sources.source_type IS 'Source classification: CDP (questionnaires), TPI (management quality), GRESB (real estate), CRREM (pathways), ISS_ESG (analytics), CUSTOM.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_sources.cache_ttl_hours IS 'Cache time-to-live in hours (0 = no caching, max 8760 = 1 year). Default 24 hours.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_sources.data_count IS 'Number of data records ingested from this source. Updated on each ingestion run.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_data.entity_identifier IS 'External identifier (ISIN, LEI, ticker, DUNS, etc.) for entity matching across sources.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_data.metric_name IS 'Named metric: total_emissions, scope1_intensity, management_quality, carbon_risk_rating, pathway_alignment, etc.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_data.raw_data IS 'Complete original JSON record from the source for audit trail and reprocessing.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_external_data.data_quality IS 'Data quality indicator (1=highest/verified, 5=lowest/estimated). Aligns with PCAF scoring.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_cache.cache_key IS 'Unique cache key, typically SHA-256 of (source_type + query_parameters + tenant_id).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_cache.hit_count IS 'Number of cache hits since creation. Used for access pattern analysis and eviction prioritisation.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_data_cache.expires_at IS 'Expiry timestamp calculated as cached_at + source TTL. Entries past expiry are cleaned by background jobs.';
