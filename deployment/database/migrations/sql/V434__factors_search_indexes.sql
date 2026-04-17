-- V434: Factors catalog search indexes and regulatory tags (Phase 10 - Scale)
-- Adds optimized indexes, regulatory tag column, and search cache metadata.

-- Full-text search vector column + GIN index
ALTER TABLE factors_catalog.factors
    ADD COLUMN IF NOT EXISTS search_vector tsvector;

CREATE INDEX IF NOT EXISTS idx_factors_search_gin
    ON factors_catalog.factors USING gin (search_vector);

-- Compound lookup index
CREATE INDEX IF NOT EXISTS idx_factors_compound_lookup
    ON factors_catalog.factors (edition_id, source_id, category, geography);

-- Certified-only partial index
CREATE INDEX IF NOT EXISTS idx_factors_certified
    ON factors_catalog.factors (factor_id) WHERE status = 'certified';

-- Geography + year for location-based queries
CREATE INDEX IF NOT EXISTS idx_factors_geo_year
    ON factors_catalog.factors (geography, year);

-- Regulatory tags (JSONB array of framework codes)
ALTER TABLE factors_catalog.factors
    ADD COLUMN IF NOT EXISTS regulatory_tags jsonb DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_factors_regulatory_tags
    ON factors_catalog.factors USING gin (regulatory_tags);

-- Search cache metadata table
CREATE TABLE IF NOT EXISTS factors_catalog.search_cache_stats (
    id              BIGSERIAL PRIMARY KEY,
    cache_key       VARCHAR(64) NOT NULL,
    query_params    JSONB NOT NULL,
    hit_count       INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_hit_at     TIMESTAMPTZ,
    ttl_seconds     INTEGER DEFAULT 3600
);

CREATE INDEX IF NOT EXISTS idx_search_cache_key
    ON factors_catalog.search_cache_stats (cache_key);
