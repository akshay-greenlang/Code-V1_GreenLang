-- =============================================================================
-- V164: PACK-026 SME Net Zero - Peer Benchmarking
-- =============================================================================
-- Pack:         PACK-026 (SME Net Zero Pack)
-- Migration:    007 of 008
-- Date:         March 2026
--
-- Anonymized peer group benchmarking with aggregated emission intensity
-- statistics by industry NACE code, size tier, and country. Individual
-- SME peer rankings with percentile positioning and improvement areas.
--
-- Tables (2):
--   1. pack026_sme_net_zero.peer_groups
--   2. pack026_sme_net_zero.peer_rankings
--
-- Previous: V163__PACK026_progress_tracking.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack026_sme_net_zero.peer_groups
-- =============================================================================
-- Anonymized, aggregated peer group benchmarks by industry NACE code, size
-- tier, and country. Contains statistical distributions of emission intensity
-- for comparison against individual SME performance.

CREATE TABLE pack026_sme_net_zero.peer_groups (
    group_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Group dimensions
    industry_nace           VARCHAR(10)     NOT NULL,
    industry_description    VARCHAR(255),
    size_tier               VARCHAR(10)     NOT NULL,
    country                 VARCHAR(3)      NOT NULL,
    region                  VARCHAR(100),
    -- Intensity statistics (tCO2e per employee)
    avg_intensity_per_employee      DECIMAL(18,6),
    median_intensity_per_employee   DECIMAL(18,6),
    p25_intensity_per_employee      DECIMAL(18,6),
    p75_intensity_per_employee      DECIMAL(18,6),
    top_quartile_intensity          DECIMAL(18,6),
    bottom_quartile_intensity       DECIMAL(18,6),
    min_intensity_per_employee      DECIMAL(18,6),
    max_intensity_per_employee      DECIMAL(18,6),
    std_dev_intensity               DECIMAL(18,6),
    -- Revenue intensity (tCO2e per EUR revenue)
    avg_intensity_per_revenue       DECIMAL(18,10),
    median_intensity_per_revenue    DECIMAL(18,10),
    -- Reduction rate
    median_reduction_rate           DECIMAL(8,3),
    avg_reduction_rate              DECIMAL(8,3),
    top_quartile_reduction_rate     DECIMAL(8,3),
    -- Scope breakdown (averages)
    avg_scope1_pct                  DECIMAL(6,2),
    avg_scope2_pct                  DECIMAL(6,2),
    avg_scope3_pct                  DECIMAL(6,2),
    -- Sample
    sample_size                     INTEGER         NOT NULL DEFAULT 0,
    data_year                       INTEGER         NOT NULL,
    -- Common actions
    top_actions                     TEXT[]          DEFAULT '{}',
    avg_actions_completed           DECIMAL(6,2),
    avg_payback_years               DECIMAL(6,2),
    -- Certification rates
    climate_hub_pct                 DECIMAL(6,2),
    iso14001_pct                    DECIMAL(6,2),
    bcorp_pct                       DECIMAL(6,2),
    -- Data quality
    data_source                     VARCHAR(100)    DEFAULT 'GREENLANG_AGGREGATE',
    confidence_level                VARCHAR(20)     DEFAULT 'MODERATE',
    last_updated                    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Metadata
    metadata                        JSONB           DEFAULT '{}',
    created_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_pg_size_tier CHECK (
        size_tier IN ('MICRO', 'SMALL', 'MEDIUM', 'ALL')
    ),
    CONSTRAINT chk_p026_pg_country_len CHECK (
        LENGTH(country) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p026_pg_sample CHECK (
        sample_size >= 0
    ),
    CONSTRAINT chk_p026_pg_data_year CHECK (
        data_year >= 2020 AND data_year <= 2100
    ),
    CONSTRAINT chk_p026_pg_confidence CHECK (
        confidence_level IN ('HIGH', 'MODERATE', 'LOW', 'INSUFFICIENT')
    ),
    CONSTRAINT chk_p026_pg_scope_pcts CHECK (
        (avg_scope1_pct IS NULL OR (avg_scope1_pct >= 0 AND avg_scope1_pct <= 100))
        AND (avg_scope2_pct IS NULL OR (avg_scope2_pct >= 0 AND avg_scope2_pct <= 100))
        AND (avg_scope3_pct IS NULL OR (avg_scope3_pct >= 0 AND avg_scope3_pct <= 100))
    ),
    CONSTRAINT chk_p026_pg_intensity_non_neg CHECK (
        avg_intensity_per_employee IS NULL OR avg_intensity_per_employee >= 0
    ),
    CONSTRAINT uq_p026_pg_nace_size_country_year UNIQUE (industry_nace, size_tier, country, data_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for peer_groups
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_pg_nace             ON pack026_sme_net_zero.peer_groups(industry_nace);
CREATE INDEX idx_p026_pg_size_tier        ON pack026_sme_net_zero.peer_groups(size_tier);
CREATE INDEX idx_p026_pg_country          ON pack026_sme_net_zero.peer_groups(country);
CREATE INDEX idx_p026_pg_nace_size        ON pack026_sme_net_zero.peer_groups(industry_nace, size_tier);
CREATE INDEX idx_p026_pg_nace_country     ON pack026_sme_net_zero.peer_groups(industry_nace, country);
CREATE INDEX idx_p026_pg_data_year        ON pack026_sme_net_zero.peer_groups(data_year);
CREATE INDEX idx_p026_pg_sample_size      ON pack026_sme_net_zero.peer_groups(sample_size);
CREATE INDEX idx_p026_pg_avg_intensity    ON pack026_sme_net_zero.peer_groups(avg_intensity_per_employee);
CREATE INDEX idx_p026_pg_confidence       ON pack026_sme_net_zero.peer_groups(confidence_level);
CREATE INDEX idx_p026_pg_created          ON pack026_sme_net_zero.peer_groups(created_at DESC);
CREATE INDEX idx_p026_pg_top_actions      ON pack026_sme_net_zero.peer_groups USING GIN(top_actions);
CREATE INDEX idx_p026_pg_metadata         ON pack026_sme_net_zero.peer_groups USING GIN(metadata);

-- =============================================================================
-- Table 2: pack026_sme_net_zero.peer_rankings
-- =============================================================================
-- Individual SME peer rankings within their peer group with percentile
-- positioning, intensity comparison vs average, and improvement areas.

CREATE TABLE pack026_sme_net_zero.peer_rankings (
    ranking_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sme_id                  UUID            NOT NULL REFERENCES pack026_sme_net_zero.sme_profiles(sme_id) ON DELETE CASCADE,
    group_id                UUID            NOT NULL REFERENCES pack026_sme_net_zero.peer_groups(group_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    -- Ranking
    percentile              DECIMAL(6,2)    NOT NULL,
    rank_position           INTEGER,
    total_in_group          INTEGER,
    -- Intensity comparison
    sme_intensity_per_employee      DECIMAL(18,6),
    intensity_vs_avg_pct            DECIMAL(8,3),
    intensity_vs_median_pct         DECIMAL(8,3),
    intensity_vs_top_quartile_pct   DECIMAL(8,3),
    -- Scope comparison
    scope1_vs_avg_pct       DECIMAL(8,3),
    scope2_vs_avg_pct       DECIMAL(8,3),
    scope3_vs_avg_pct       DECIMAL(8,3),
    -- Reduction rate comparison
    reduction_rate_vs_avg   DECIMAL(8,3),
    -- Improvement areas
    areas_for_improvement   TEXT[]          DEFAULT '{}',
    recommended_actions     TEXT[]          DEFAULT '{}',
    potential_reduction_tco2e DECIMAL(14,4),
    -- Rating
    peer_performance_tier   VARCHAR(20),
    trend                   VARCHAR(20),
    -- Dates
    ranking_date            DATE            NOT NULL,
    data_year               INTEGER         NOT NULL,
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p026_pr_percentile CHECK (
        percentile >= 0 AND percentile <= 100
    ),
    CONSTRAINT chk_p026_pr_tier CHECK (
        peer_performance_tier IS NULL OR peer_performance_tier IN (
            'LEADER', 'ABOVE_AVERAGE', 'AVERAGE', 'BELOW_AVERAGE', 'LAGGARD'
        )
    ),
    CONSTRAINT chk_p026_pr_trend CHECK (
        trend IS NULL OR trend IN ('IMPROVING', 'STABLE', 'DECLINING', 'NEW')
    ),
    CONSTRAINT chk_p026_pr_data_year CHECK (
        data_year >= 2020 AND data_year <= 2100
    ),
    CONSTRAINT chk_p026_pr_rank_positive CHECK (
        rank_position IS NULL OR rank_position >= 1
    ),
    CONSTRAINT chk_p026_pr_intensity_non_neg CHECK (
        sme_intensity_per_employee IS NULL OR sme_intensity_per_employee >= 0
    ),
    CONSTRAINT uq_p026_pr_sme_group_year UNIQUE (sme_id, group_id, data_year)
);

-- ---------------------------------------------------------------------------
-- Indexes for peer_rankings
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p026_pr_sme              ON pack026_sme_net_zero.peer_rankings(sme_id);
CREATE INDEX idx_p026_pr_group            ON pack026_sme_net_zero.peer_rankings(group_id);
CREATE INDEX idx_p026_pr_tenant           ON pack026_sme_net_zero.peer_rankings(tenant_id);
CREATE INDEX idx_p026_pr_percentile       ON pack026_sme_net_zero.peer_rankings(percentile DESC);
CREATE INDEX idx_p026_pr_sme_year         ON pack026_sme_net_zero.peer_rankings(sme_id, data_year);
CREATE INDEX idx_p026_pr_tier             ON pack026_sme_net_zero.peer_rankings(peer_performance_tier);
CREATE INDEX idx_p026_pr_trend            ON pack026_sme_net_zero.peer_rankings(trend);
CREATE INDEX idx_p026_pr_ranking_date     ON pack026_sme_net_zero.peer_rankings(ranking_date);
CREATE INDEX idx_p026_pr_data_year        ON pack026_sme_net_zero.peer_rankings(data_year);
CREATE INDEX idx_p026_pr_created          ON pack026_sme_net_zero.peer_rankings(created_at DESC);
CREATE INDEX idx_p026_pr_improvement      ON pack026_sme_net_zero.peer_rankings USING GIN(areas_for_improvement);
CREATE INDEX idx_p026_pr_actions          ON pack026_sme_net_zero.peer_rankings USING GIN(recommended_actions);
CREATE INDEX idx_p026_pr_metadata         ON pack026_sme_net_zero.peer_rankings USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p026_peer_groups_updated
    BEFORE UPDATE ON pack026_sme_net_zero.peer_groups
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p026_peer_rankings_updated
    BEFORE UPDATE ON pack026_sme_net_zero.peer_rankings
    FOR EACH ROW EXECUTE FUNCTION pack026_sme_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
-- peer_groups is anonymized aggregated data, accessible to all (no tenant_id)
ALTER TABLE pack026_sme_net_zero.peer_rankings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p026_pr_tenant_isolation
    ON pack026_sme_net_zero.peer_rankings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p026_pr_service_bypass
    ON pack026_sme_net_zero.peer_rankings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT ON pack026_sme_net_zero.peer_groups TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack026_sme_net_zero.peer_rankings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack026_sme_net_zero.peer_groups IS
    'Anonymized, aggregated peer group benchmarks by NACE industry, size tier, and country with emission intensity statistics for SME comparison.';
COMMENT ON TABLE pack026_sme_net_zero.peer_rankings IS
    'Individual SME peer rankings within their peer group with percentile positioning, intensity comparison, and improvement recommendations.';

COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.group_id IS 'Unique peer group identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.industry_nace IS 'NACE Rev.2 industry classification code for peer grouping.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.size_tier IS 'SME size tier: MICRO, SMALL, MEDIUM, or ALL for combined groups.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.avg_intensity_per_employee IS 'Average emission intensity per employee (tCO2e/FTE) in the peer group.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.median_reduction_rate IS 'Median annual emission reduction rate in the peer group (%).';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.top_quartile_intensity IS 'Top quartile (best 25%) emission intensity threshold.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.sample_size IS 'Number of SMEs in the anonymized peer group (min 5 for privacy).';
COMMENT ON COLUMN pack026_sme_net_zero.peer_groups.confidence_level IS 'Statistical confidence based on sample size: HIGH (50+), MODERATE (20-49), LOW (5-19), INSUFFICIENT (<5).';

COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.ranking_id IS 'Unique ranking identifier.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.percentile IS 'Percentile rank within peer group (100 = best, 0 = worst).';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.intensity_vs_avg_pct IS 'Percentage difference vs peer group average (negative = better than average).';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.areas_for_improvement IS 'Array of identified improvement areas based on peer comparison.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.peer_performance_tier IS 'Performance tier: LEADER, ABOVE_AVERAGE, AVERAGE, BELOW_AVERAGE, LAGGARD.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.trend IS 'Year-over-year trend: IMPROVING, STABLE, DECLINING, NEW.';
COMMENT ON COLUMN pack026_sme_net_zero.peer_rankings.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
