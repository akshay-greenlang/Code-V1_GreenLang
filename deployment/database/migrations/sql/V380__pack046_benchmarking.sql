-- =============================================================================
-- V380: PACK-046 Intensity Metrics Pack - Benchmarking
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for peer benchmarking and comparison. Peer groups define
-- the comparison set (by sector, geography, size). Peer data stores
-- external intensity values from CDP, TPI, GRESB, CRREM, and custom
-- sources. Benchmark results calculate the organisation's percentile rank,
-- gap-to-average, gap-to-best, and gap-to-target against the peer group.
-- Normalisation adjustments (PPP, climate, etc.) are tracked as JSON.
--
-- Tables (3):
--   1. ghg_intensity.gl_im_peer_groups
--   2. ghg_intensity.gl_im_peer_data
--   3. ghg_intensity.gl_im_benchmark_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V379__pack046_decomposition.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_peer_groups
-- =============================================================================
-- Peer group definitions for benchmarking. Each group has selection criteria
-- (sector, sub-sector, geography, size range) and tracks the number of
-- peers. An organisation may have multiple peer groups for different
-- comparison perspectives (e.g., global sector vs regional peers).

CREATE TABLE ghg_intensity.gl_im_peer_groups (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    group_name                  VARCHAR(255)    NOT NULL,
    sector                      VARCHAR(100)    NOT NULL,
    sub_sector                  VARCHAR(100),
    geography                   VARCHAR(100),
    size_range                  VARCHAR(100),
    selection_criteria          JSONB           NOT NULL DEFAULT '{}',
    peer_count                  INTEGER         NOT NULL DEFAULT 0,
    data_vintage_year           INTEGER,
    source_description          TEXT,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_pg_peer_count CHECK (
        peer_count >= 0
    ),
    CONSTRAINT chk_p046_pg_vintage CHECK (
        data_vintage_year IS NULL OR (data_vintage_year >= 2000 AND data_vintage_year <= 2100)
    ),
    CONSTRAINT uq_p046_pg_org_name UNIQUE (org_id, config_id, group_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_pg_tenant            ON ghg_intensity.gl_im_peer_groups(tenant_id);
CREATE INDEX idx_p046_pg_org               ON ghg_intensity.gl_im_peer_groups(org_id);
CREATE INDEX idx_p046_pg_config            ON ghg_intensity.gl_im_peer_groups(config_id);
CREATE INDEX idx_p046_pg_sector            ON ghg_intensity.gl_im_peer_groups(sector);
CREATE INDEX idx_p046_pg_active            ON ghg_intensity.gl_im_peer_groups(is_active) WHERE is_active = true;
CREATE INDEX idx_p046_pg_created           ON ghg_intensity.gl_im_peer_groups(created_at DESC);
CREATE INDEX idx_p046_pg_criteria          ON ghg_intensity.gl_im_peer_groups USING GIN(selection_criteria);

-- Composite: org + config for listing
CREATE INDEX idx_p046_pg_org_config        ON ghg_intensity.gl_im_peer_groups(org_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_pg_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_peer_groups
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_peer_data
-- =============================================================================
-- External peer intensity data from public and proprietary sources. Each
-- record represents one peer's intensity value for a specific reporting
-- year, denominator, and scope. Sources include CDP questionnaire responses,
-- TPI management quality scores, GRESB assessments, CRREM pathways, and
-- manually entered custom data.

CREATE TABLE ghg_intensity.gl_im_peer_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    peer_group_id               UUID            NOT NULL REFERENCES ghg_intensity.gl_im_peer_groups(id) ON DELETE CASCADE,
    peer_name                   VARCHAR(255)    NOT NULL,
    peer_identifier             VARCHAR(100),
    identifier_type             VARCHAR(30),
    source                      VARCHAR(50)     NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    intensity_value             NUMERIC(20,10)  NOT NULL,
    intensity_unit              VARCHAR(100)    NOT NULL,
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50),
    emissions_tco2e             NUMERIC(20,6),
    denominator_value           NUMERIC(20,6),
    data_quality_indicator      VARCHAR(30),
    raw_data                    JSONB,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_pd_source CHECK (
        source IN (
            'CDP', 'TPI', 'GRESB', 'CRREM', 'PCAF', 'SBTI',
            'ANNUAL_REPORT', 'SUSTAINABILITY_REPORT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_pd_identifier_type CHECK (
        identifier_type IS NULL OR identifier_type IN (
            'ISIN', 'LEI', 'TICKER', 'CIK', 'DUNS', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_pd_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p046_pd_intensity CHECK (
        intensity_value >= 0
    ),
    CONSTRAINT chk_p046_pd_scope CHECK (
        scope_inclusion IS NULL OR scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_pd_quality CHECK (
        data_quality_indicator IS NULL OR data_quality_indicator IN (
            'VERIFIED', 'REPORTED', 'ESTIMATED', 'MODELLED', 'UNKNOWN'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_pd_peer_group        ON ghg_intensity.gl_im_peer_data(peer_group_id);
CREATE INDEX idx_p046_pd_peer_name         ON ghg_intensity.gl_im_peer_data(peer_name);
CREATE INDEX idx_p046_pd_source            ON ghg_intensity.gl_im_peer_data(source);
CREATE INDEX idx_p046_pd_year              ON ghg_intensity.gl_im_peer_data(reporting_year);
CREATE INDEX idx_p046_pd_denom             ON ghg_intensity.gl_im_peer_data(denominator_code);
CREATE INDEX idx_p046_pd_intensity         ON ghg_intensity.gl_im_peer_data(intensity_value);
CREATE INDEX idx_p046_pd_created           ON ghg_intensity.gl_im_peer_data(created_at DESC);
CREATE INDEX idx_p046_pd_raw_data          ON ghg_intensity.gl_im_peer_data USING GIN(raw_data);

-- Composite: group + year for period-based retrieval
CREATE INDEX idx_p046_pd_group_year        ON ghg_intensity.gl_im_peer_data(peer_group_id, reporting_year);

-- =============================================================================
-- Table 3: ghg_intensity.gl_im_benchmark_results
-- =============================================================================
-- Benchmark comparison results for the organisation against a peer group.
-- Calculates percentile rank, summary statistics (mean, median, P25, P75,
-- best-in-class), and gap metrics (gap-to-average, gap-to-best, gap-to-
-- target). Normalisation adjustments (PPP, climate zone, product mix) are
-- tracked as JSON for transparency.

CREATE TABLE ghg_intensity.gl_im_benchmark_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id),
    peer_group_id               UUID            NOT NULL REFERENCES ghg_intensity.gl_im_peer_groups(id),
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL,
    org_intensity               NUMERIC(20,10)  NOT NULL,
    percentile_rank             NUMERIC(10,6)   NOT NULL,
    peer_mean                   NUMERIC(20,10),
    peer_median                 NUMERIC(20,10),
    peer_p10                    NUMERIC(20,10),
    peer_p25                    NUMERIC(20,10),
    peer_p75                    NUMERIC(20,10),
    peer_p90                    NUMERIC(20,10),
    peer_best                   NUMERIC(20,10),
    peer_worst                  NUMERIC(20,10),
    gap_to_average              NUMERIC(20,10),
    gap_to_best                 NUMERIC(20,10),
    gap_to_target               NUMERIC(20,10),
    gap_to_average_pct          NUMERIC(10,6),
    gap_to_best_pct             NUMERIC(10,6),
    normalisation_adjustments   JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_br_percentile CHECK (
        percentile_rank >= 0 AND percentile_rank <= 100
    ),
    CONSTRAINT chk_p046_br_org_intensity CHECK (
        org_intensity >= 0
    ),
    CONSTRAINT chk_p046_br_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_br_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_br_tenant            ON ghg_intensity.gl_im_benchmark_results(tenant_id);
CREATE INDEX idx_p046_br_org               ON ghg_intensity.gl_im_benchmark_results(org_id);
CREATE INDEX idx_p046_br_config            ON ghg_intensity.gl_im_benchmark_results(config_id);
CREATE INDEX idx_p046_br_period            ON ghg_intensity.gl_im_benchmark_results(period_id);
CREATE INDEX idx_p046_br_peer_group        ON ghg_intensity.gl_im_benchmark_results(peer_group_id);
CREATE INDEX idx_p046_br_denom             ON ghg_intensity.gl_im_benchmark_results(denominator_code);
CREATE INDEX idx_p046_br_scope             ON ghg_intensity.gl_im_benchmark_results(scope_inclusion);
CREATE INDEX idx_p046_br_percentile        ON ghg_intensity.gl_im_benchmark_results(percentile_rank);
CREATE INDEX idx_p046_br_calculated        ON ghg_intensity.gl_im_benchmark_results(calculated_at DESC);
CREATE INDEX idx_p046_br_created           ON ghg_intensity.gl_im_benchmark_results(created_at DESC);
CREATE INDEX idx_p046_br_provenance        ON ghg_intensity.gl_im_benchmark_results(provenance_hash);

-- Composite: org + period + peer group for dashboard queries
CREATE INDEX idx_p046_br_org_period_pg     ON ghg_intensity.gl_im_benchmark_results(org_id, period_id, peer_group_id);

-- Composite: denominator + scope for filtered analysis
CREATE INDEX idx_p046_br_denom_scope       ON ghg_intensity.gl_im_benchmark_results(denominator_code, scope_inclusion);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_peer_groups ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_peer_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_benchmark_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_pg_tenant_isolation
    ON ghg_intensity.gl_im_peer_groups
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_pg_service_bypass
    ON ghg_intensity.gl_im_peer_groups
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Peer data inherits access via peer_group FK
CREATE POLICY p046_pd_service_bypass
    ON ghg_intensity.gl_im_peer_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_br_tenant_isolation
    ON ghg_intensity.gl_im_benchmark_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_br_service_bypass
    ON ghg_intensity.gl_im_benchmark_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_peer_groups TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_peer_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_benchmark_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_peer_groups IS
    'Peer group definitions for benchmarking with sector, geography, and size-based selection criteria.';
COMMENT ON TABLE ghg_intensity.gl_im_peer_data IS
    'External peer intensity data from CDP, TPI, GRESB, CRREM, and custom sources for benchmarking.';
COMMENT ON TABLE ghg_intensity.gl_im_benchmark_results IS
    'Benchmark comparison results with percentile rank, summary statistics, and gap-to-average/best/target metrics.';

COMMENT ON COLUMN ghg_intensity.gl_im_peer_groups.selection_criteria IS 'JSON criteria for peer selection: {"revenue_range": [100, 500], "geography": "EU", "sub_sector": "cement"}.';
COMMENT ON COLUMN ghg_intensity.gl_im_peer_groups.peer_count IS 'Number of peers in the group. Updated when peer data is added/removed.';
COMMENT ON COLUMN ghg_intensity.gl_im_peer_data.peer_identifier IS 'External identifier for the peer entity (ISIN, LEI, ticker, etc.).';
COMMENT ON COLUMN ghg_intensity.gl_im_peer_data.source IS 'Data source: CDP, TPI, GRESB, CRREM, PCAF, SBTI, ANNUAL_REPORT, SUSTAINABILITY_REPORT, or CUSTOM.';
COMMENT ON COLUMN ghg_intensity.gl_im_benchmark_results.percentile_rank IS 'Organisation percentile rank within peer group (0-100). Lower is better for intensity metrics.';
COMMENT ON COLUMN ghg_intensity.gl_im_benchmark_results.gap_to_average IS 'Difference between org intensity and peer group mean. Negative means better than average.';
COMMENT ON COLUMN ghg_intensity.gl_im_benchmark_results.gap_to_best IS 'Difference between org intensity and best-in-class peer. Zero means best-in-class.';
COMMENT ON COLUMN ghg_intensity.gl_im_benchmark_results.normalisation_adjustments IS 'JSON tracking normalisation adjustments applied: PPP, climate zone, product mix, etc.';
