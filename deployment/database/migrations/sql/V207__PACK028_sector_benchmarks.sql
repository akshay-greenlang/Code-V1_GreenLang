-- =============================================================================
-- V187: PACK-028 Sector Pathway Pack - Sector Benchmarks
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    007 of 015
-- Date:         March 2026
--
-- Multi-dimensional sector benchmarking with peer, leader, IEA pathway,
-- and SBTi-validated company comparisons. Includes percentile rankings,
-- gap metrics, and trend analysis for sector intensity benchmarking.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_benchmarks
--
-- Previous: V186__PACK028_abatement_waterfall.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_benchmarks
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_sector_benchmarks (
    benchmark_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    metric_id                   UUID            REFERENCES pack028_sector_pathway.gl_sector_intensity_metrics(metric_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    intensity_metric            VARCHAR(60)     NOT NULL,
    -- Benchmark period
    benchmark_year              INTEGER         NOT NULL,
    benchmark_date              DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- Company performance
    company_intensity           DECIMAL(18,8)   NOT NULL,
    company_intensity_unit      VARCHAR(80)     NOT NULL,
    company_trend_3yr_pct       DECIMAL(8,4),
    company_trend_5yr_pct       DECIMAL(8,4),
    company_rank                INTEGER,
    total_peer_count            INTEGER,
    -- Percentile rankings
    percentile_overall          DECIMAL(5,2),
    percentile_by_region        DECIMAL(5,2),
    percentile_by_size          DECIMAL(5,2),
    percentile_sbti_peers       DECIMAL(5,2),
    -- Peer group benchmarks
    peer_group_name             VARCHAR(100),
    peer_group_size             INTEGER,
    peer_mean_intensity         DECIMAL(18,8),
    peer_median_intensity       DECIMAL(18,8),
    peer_p10_intensity          DECIMAL(18,8),
    peer_p25_intensity          DECIMAL(18,8),
    peer_p75_intensity          DECIMAL(18,8),
    peer_p90_intensity          DECIMAL(18,8),
    peer_min_intensity          DECIMAL(18,8),
    peer_max_intensity          DECIMAL(18,8),
    peer_std_deviation          DECIMAL(18,8),
    -- Sector leaders
    leader_intensity            DECIMAL(18,8),
    leader_name                 VARCHAR(255),
    leader_region               VARCHAR(30),
    gap_to_leader_pct           DECIMAL(8,4),
    gap_to_leader_absolute      DECIMAL(18,8),
    -- Top decile performers
    top_decile_avg_intensity    DECIMAL(18,8),
    gap_to_top_decile_pct       DECIMAL(8,4),
    -- SBTi-validated peers
    sbti_peer_count             INTEGER,
    sbti_peer_avg_intensity     DECIMAL(18,8),
    sbti_peer_median_intensity  DECIMAL(18,8),
    gap_to_sbti_avg_pct         DECIMAL(8,4),
    sbti_peer_trend_pct         DECIMAL(8,4),
    -- IEA pathway benchmark
    iea_pathway_intensity       DECIMAL(18,8),
    iea_scenario                VARCHAR(30),
    gap_to_iea_pct              DECIMAL(8,4),
    iea_alignment_score         DECIMAL(5,2),
    -- SBTi SDA pathway benchmark
    sda_pathway_intensity       DECIMAL(18,8),
    gap_to_sda_pct              DECIMAL(8,4),
    sda_alignment_score         DECIMAL(5,2),
    -- Regulatory benchmarks
    regulatory_benchmark        DECIMAL(18,8),
    regulatory_source           VARCHAR(100),
    gap_to_regulatory_pct       DECIMAL(8,4),
    -- Regional comparison
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    regional_avg_intensity      DECIMAL(18,8),
    gap_to_regional_avg_pct     DECIMAL(8,4),
    -- Size-based comparison
    size_category               VARCHAR(20),
    size_avg_intensity          DECIMAL(18,8),
    gap_to_size_avg_pct         DECIMAL(8,4),
    -- Year-over-year benchmark movement
    yoy_percentile_change       DECIMAL(6,2),
    yoy_rank_change             INTEGER,
    benchmark_trend             VARCHAR(20),
    -- Overall scores
    benchmark_composite_score   DECIMAL(5,2),
    performance_tier            VARCHAR(20),
    -- Data quality
    benchmark_data_quality      VARCHAR(20)     DEFAULT 'HIGH',
    peer_data_source            VARCHAR(100),
    data_vintage_year           INTEGER,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_bm_benchmark_year CHECK (
        benchmark_year >= 2015 AND benchmark_year <= 2100
    ),
    CONSTRAINT chk_p028_bm_percentile_overall CHECK (
        percentile_overall IS NULL OR (percentile_overall >= 0 AND percentile_overall <= 100)
    ),
    CONSTRAINT chk_p028_bm_percentile_region CHECK (
        percentile_by_region IS NULL OR (percentile_by_region >= 0 AND percentile_by_region <= 100)
    ),
    CONSTRAINT chk_p028_bm_percentile_size CHECK (
        percentile_by_size IS NULL OR (percentile_by_size >= 0 AND percentile_by_size <= 100)
    ),
    CONSTRAINT chk_p028_bm_percentile_sbti CHECK (
        percentile_sbti_peers IS NULL OR (percentile_sbti_peers >= 0 AND percentile_sbti_peers <= 100)
    ),
    CONSTRAINT chk_p028_bm_iea_scenario CHECK (
        iea_scenario IS NULL OR iea_scenario IN ('NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS')
    ),
    CONSTRAINT chk_p028_bm_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_bm_size_category CHECK (
        size_category IS NULL OR size_category IN (
            'MICRO', 'SMALL', 'MEDIUM', 'LARGE', 'VERY_LARGE', 'MEGA'
        )
    ),
    CONSTRAINT chk_p028_bm_benchmark_trend CHECK (
        benchmark_trend IS NULL OR benchmark_trend IN (
            'STRONGLY_IMPROVING', 'IMPROVING', 'STABLE', 'DECLINING', 'STRONGLY_DECLINING'
        )
    ),
    CONSTRAINT chk_p028_bm_performance_tier CHECK (
        performance_tier IS NULL OR performance_tier IN (
            'LEADER', 'FRONT_RUNNER', 'ALIGNED', 'LAGGING', 'SIGNIFICANTLY_BEHIND'
        )
    ),
    CONSTRAINT chk_p028_bm_data_quality CHECK (
        benchmark_data_quality IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p028_bm_composite_score CHECK (
        benchmark_composite_score IS NULL OR (benchmark_composite_score >= 0 AND benchmark_composite_score <= 100)
    ),
    CONSTRAINT chk_p028_bm_iea_alignment CHECK (
        iea_alignment_score IS NULL OR (iea_alignment_score >= 0 AND iea_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_bm_sda_alignment CHECK (
        sda_alignment_score IS NULL OR (sda_alignment_score >= 0 AND sda_alignment_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_bm_tenant             ON pack028_sector_pathway.gl_sector_benchmarks(tenant_id);
CREATE INDEX idx_p028_bm_company            ON pack028_sector_pathway.gl_sector_benchmarks(company_id);
CREATE INDEX idx_p028_bm_classification     ON pack028_sector_pathway.gl_sector_benchmarks(classification_id);
CREATE INDEX idx_p028_bm_metric             ON pack028_sector_pathway.gl_sector_benchmarks(metric_id);
CREATE INDEX idx_p028_bm_sector             ON pack028_sector_pathway.gl_sector_benchmarks(sector_code);
CREATE INDEX idx_p028_bm_year               ON pack028_sector_pathway.gl_sector_benchmarks(benchmark_year);
CREATE INDEX idx_p028_bm_company_year       ON pack028_sector_pathway.gl_sector_benchmarks(company_id, benchmark_year);
CREATE INDEX idx_p028_bm_company_sector     ON pack028_sector_pathway.gl_sector_benchmarks(company_id, sector_code, benchmark_year);
CREATE INDEX idx_p028_bm_percentile         ON pack028_sector_pathway.gl_sector_benchmarks(percentile_overall);
CREATE INDEX idx_p028_bm_perf_tier          ON pack028_sector_pathway.gl_sector_benchmarks(performance_tier);
CREATE INDEX idx_p028_bm_region             ON pack028_sector_pathway.gl_sector_benchmarks(region);
CREATE INDEX idx_p028_bm_size               ON pack028_sector_pathway.gl_sector_benchmarks(size_category);
CREATE INDEX idx_p028_bm_iea_scenario       ON pack028_sector_pathway.gl_sector_benchmarks(iea_scenario);
CREATE INDEX idx_p028_bm_composite_desc     ON pack028_sector_pathway.gl_sector_benchmarks(benchmark_composite_score DESC NULLS LAST);
CREATE INDEX idx_p028_bm_leader_gap         ON pack028_sector_pathway.gl_sector_benchmarks(gap_to_leader_pct);
CREATE INDEX idx_p028_bm_iea_alignment      ON pack028_sector_pathway.gl_sector_benchmarks(iea_alignment_score DESC NULLS LAST);
CREATE INDEX idx_p028_bm_sda_alignment      ON pack028_sector_pathway.gl_sector_benchmarks(sda_alignment_score DESC NULLS LAST);
CREATE INDEX idx_p028_bm_trend              ON pack028_sector_pathway.gl_sector_benchmarks(benchmark_trend);
CREATE INDEX idx_p028_bm_created            ON pack028_sector_pathway.gl_sector_benchmarks(created_at DESC);
CREATE INDEX idx_p028_bm_metadata           ON pack028_sector_pathway.gl_sector_benchmarks USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_sector_benchmarks_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_benchmarks
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_benchmarks ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_bm_tenant_isolation
    ON pack028_sector_pathway.gl_sector_benchmarks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_bm_service_bypass
    ON pack028_sector_pathway.gl_sector_benchmarks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_benchmarks TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_benchmarks IS
    'Multi-dimensional sector benchmarking with peer/leader/IEA/SBTi comparisons, percentile rankings, gap metrics, and trend analysis for sector intensity performance tracking.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.benchmark_id IS 'Unique benchmark record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.percentile_overall IS 'Company percentile rank vs. all sector peers (100 = best performer).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.gap_to_leader_pct IS 'Percentage gap between company and sector leader intensity (negative = better than leader).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.iea_alignment_score IS 'Alignment score (0-100) vs. IEA pathway intensity target for benchmark year.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.sda_alignment_score IS 'Alignment score (0-100) vs. SBTi SDA pathway intensity target for benchmark year.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.performance_tier IS 'Performance tier: LEADER, FRONT_RUNNER, ALIGNED, LAGGING, SIGNIFICANTLY_BEHIND.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.benchmark_composite_score IS 'Composite benchmark score (0-100) combining peer, IEA, and SDA alignment.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_benchmarks.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
