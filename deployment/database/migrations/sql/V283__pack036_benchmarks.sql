-- =============================================================================
-- V283: PACK-036 Utility Analysis Pack - Facility Benchmarks & Rankings
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Tables for facility-level energy performance benchmarking against
-- industry standards (ENERGY STAR, CIBSE TM46, ASHRAE 100), benchmark
-- target reference data by building type, and portfolio-level facility
-- rankings for performance comparison.
--
-- Tables (3):
--   1. pack036_utility_analysis.gl_facility_benchmarks
--   2. pack036_utility_analysis.gl_benchmark_targets
--   3. pack036_utility_analysis.gl_portfolio_rankings
--
-- Previous: V282__pack036_procurement.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_facility_benchmarks
-- =============================================================================
-- Annual facility energy performance benchmarks including site EUI,
-- source EUI, ENERGY STAR score, weather normalization, peer percentile,
-- and quartile ranking.

CREATE TABLE pack036_utility_analysis.gl_facility_benchmarks (
    benchmark_id            UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    period_year             INTEGER         NOT NULL,
    building_type           VARCHAR(50),
    floor_area_m2           NUMERIC(12,2),
    site_eui_kwh_m2         NUMERIC(12,4),
    source_eui_kwh_m2       NUMERIC(12,4),
    primary_eui_kwh_m2      NUMERIC(12,4),
    energy_star_score       INTEGER,
    weather_normalized      BOOLEAN         DEFAULT false,
    weather_normalized_eui  NUMERIC(12,4),
    benchmark_standard      VARCHAR(50)     NOT NULL,
    peer_percentile         NUMERIC(6,2),
    quartile                INTEGER,
    total_energy_kwh        NUMERIC(16,4),
    total_cost_eur          NUMERIC(14,2),
    cost_per_m2_eur         NUMERIC(10,4),
    co2_intensity_kg_m2     NUMERIC(10,4),
    data_completeness_pct   NUMERIC(6,2),
    yoy_eui_change_pct      NUMERIC(8,4),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_fb_year CHECK (
        period_year >= 2000 AND period_year <= 2100
    ),
    CONSTRAINT chk_p036_fb_site_eui CHECK (
        site_eui_kwh_m2 IS NULL OR site_eui_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p036_fb_source_eui CHECK (
        source_eui_kwh_m2 IS NULL OR source_eui_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p036_fb_primary_eui CHECK (
        primary_eui_kwh_m2 IS NULL OR primary_eui_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p036_fb_energy_star CHECK (
        energy_star_score IS NULL OR (energy_star_score >= 1 AND energy_star_score <= 100)
    ),
    CONSTRAINT chk_p036_fb_standard CHECK (
        benchmark_standard IN (
            'ENERGY_STAR', 'CIBSE_TM46', 'DIN_V_18599', 'ASHRAE_100',
            'NABERS', 'BPIE', 'EU_EED', 'CUSTOM', 'INTERNAL'
        )
    ),
    CONSTRAINT chk_p036_fb_percentile CHECK (
        peer_percentile IS NULL OR (peer_percentile >= 0 AND peer_percentile <= 100)
    ),
    CONSTRAINT chk_p036_fb_quartile CHECK (
        quartile IS NULL OR (quartile >= 1 AND quartile <= 4)
    ),
    CONSTRAINT chk_p036_fb_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p036_fb_area CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT uq_p036_fb_fac_year_std UNIQUE (facility_id, period_year, benchmark_standard)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_fb_tenant         ON pack036_utility_analysis.gl_facility_benchmarks(tenant_id);
CREATE INDEX idx_p036_fb_facility       ON pack036_utility_analysis.gl_facility_benchmarks(facility_id);
CREATE INDEX idx_p036_fb_year           ON pack036_utility_analysis.gl_facility_benchmarks(period_year DESC);
CREATE INDEX idx_p036_fb_building_type  ON pack036_utility_analysis.gl_facility_benchmarks(building_type);
CREATE INDEX idx_p036_fb_standard       ON pack036_utility_analysis.gl_facility_benchmarks(benchmark_standard);
CREATE INDEX idx_p036_fb_site_eui       ON pack036_utility_analysis.gl_facility_benchmarks(site_eui_kwh_m2);
CREATE INDEX idx_p036_fb_energy_star    ON pack036_utility_analysis.gl_facility_benchmarks(energy_star_score);
CREATE INDEX idx_p036_fb_quartile       ON pack036_utility_analysis.gl_facility_benchmarks(quartile);
CREATE INDEX idx_p036_fb_percentile     ON pack036_utility_analysis.gl_facility_benchmarks(peer_percentile);
CREATE INDEX idx_p036_fb_created        ON pack036_utility_analysis.gl_facility_benchmarks(created_at DESC);
CREATE INDEX idx_p036_fb_metadata       ON pack036_utility_analysis.gl_facility_benchmarks USING GIN(metadata);

-- Composite: facility + year for time-series benchmark lookup
CREATE INDEX idx_p036_fb_fac_year       ON pack036_utility_analysis.gl_facility_benchmarks(facility_id, period_year DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_fb_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_facility_benchmarks
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_benchmark_targets
-- =============================================================================
-- Reference data for energy benchmark targets by building type and
-- standard. Provides median, good practice, and top quartile EUI
-- targets for comparison.

CREATE TABLE pack036_utility_analysis.gl_benchmark_targets (
    target_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_type           VARCHAR(50)     NOT NULL,
    standard                VARCHAR(50)     NOT NULL,
    climate_zone            VARCHAR(20),
    country_code            CHAR(2),
    median_eui              NUMERIC(10,4)   NOT NULL,
    good_practice_eui       NUMERIC(10,4),
    top_quartile_eui        NUMERIC(10,4),
    best_in_class_eui       NUMERIC(10,4),
    unit                    VARCHAR(20)     NOT NULL DEFAULT 'kWh/m2/yr',
    source_year             INTEGER         NOT NULL,
    source_document         VARCHAR(255),
    notes                   TEXT,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_bt_standard CHECK (
        standard IN (
            'ENERGY_STAR', 'CIBSE_TM46', 'DIN_V_18599', 'ASHRAE_100',
            'NABERS', 'BPIE', 'EU_EED', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p036_bt_median CHECK (
        median_eui >= 0
    ),
    CONSTRAINT chk_p036_bt_good CHECK (
        good_practice_eui IS NULL OR good_practice_eui >= 0
    ),
    CONSTRAINT chk_p036_bt_top CHECK (
        top_quartile_eui IS NULL OR top_quartile_eui >= 0
    ),
    CONSTRAINT chk_p036_bt_best CHECK (
        best_in_class_eui IS NULL OR best_in_class_eui >= 0
    ),
    CONSTRAINT chk_p036_bt_year CHECK (
        source_year >= 2000 AND source_year <= 2100
    ),
    CONSTRAINT chk_p036_bt_country CHECK (
        country_code IS NULL OR LENGTH(country_code) = 2
    ),
    CONSTRAINT uq_p036_bt_type_std_clim UNIQUE (building_type, standard, climate_zone, country_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_bt_building_type  ON pack036_utility_analysis.gl_benchmark_targets(building_type);
CREATE INDEX idx_p036_bt_standard       ON pack036_utility_analysis.gl_benchmark_targets(standard);
CREATE INDEX idx_p036_bt_climate        ON pack036_utility_analysis.gl_benchmark_targets(climate_zone);
CREATE INDEX idx_p036_bt_country        ON pack036_utility_analysis.gl_benchmark_targets(country_code);
CREATE INDEX idx_p036_bt_year           ON pack036_utility_analysis.gl_benchmark_targets(source_year DESC);
CREATE INDEX idx_p036_bt_active         ON pack036_utility_analysis.gl_benchmark_targets(is_active);

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_portfolio_rankings
-- =============================================================================
-- Portfolio-level facility rankings for performance comparison.
-- Ranks facilities within a portfolio by EUI, cost, or other metrics
-- for a given period.

CREATE TABLE pack036_utility_analysis.gl_portfolio_rankings (
    ranking_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    portfolio_id            UUID            NOT NULL,
    portfolio_name          VARCHAR(255),
    period_year             INTEGER         NOT NULL,
    facility_id             UUID            NOT NULL,
    facility_name           VARCHAR(255),
    building_type           VARCHAR(50),
    rank_position           INTEGER         NOT NULL,
    total_facilities        INTEGER         NOT NULL,
    eui_value               NUMERIC(12,4)   NOT NULL,
    eui_unit                VARCHAR(20)     DEFAULT 'kWh/m2/yr',
    percentile              NUMERIC(6,2),
    quartile                INTEGER,
    ranking_metric          VARCHAR(50)     NOT NULL DEFAULT 'SITE_EUI',
    floor_area_m2           NUMERIC(12,2),
    total_cost_eur          NUMERIC(14,2),
    yoy_improvement_pct     NUMERIC(8,4),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_pr_year CHECK (
        period_year >= 2000 AND period_year <= 2100
    ),
    CONSTRAINT chk_p036_pr_rank CHECK (
        rank_position >= 1
    ),
    CONSTRAINT chk_p036_pr_total CHECK (
        total_facilities >= 1
    ),
    CONSTRAINT chk_p036_pr_rank_valid CHECK (
        rank_position <= total_facilities
    ),
    CONSTRAINT chk_p036_pr_eui CHECK (
        eui_value >= 0
    ),
    CONSTRAINT chk_p036_pr_percentile CHECK (
        percentile IS NULL OR (percentile >= 0 AND percentile <= 100)
    ),
    CONSTRAINT chk_p036_pr_quartile CHECK (
        quartile IS NULL OR (quartile >= 1 AND quartile <= 4)
    ),
    CONSTRAINT chk_p036_pr_metric CHECK (
        ranking_metric IN (
            'SITE_EUI', 'SOURCE_EUI', 'WEATHER_NORMALIZED_EUI',
            'COST_PER_M2', 'CO2_INTENSITY', 'ENERGY_STAR_SCORE'
        )
    ),
    CONSTRAINT uq_p036_pr_portfolio_fac UNIQUE (portfolio_id, period_year, facility_id, ranking_metric)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_pr_tenant         ON pack036_utility_analysis.gl_portfolio_rankings(tenant_id);
CREATE INDEX idx_p036_pr_portfolio      ON pack036_utility_analysis.gl_portfolio_rankings(portfolio_id);
CREATE INDEX idx_p036_pr_year           ON pack036_utility_analysis.gl_portfolio_rankings(period_year DESC);
CREATE INDEX idx_p036_pr_facility       ON pack036_utility_analysis.gl_portfolio_rankings(facility_id);
CREATE INDEX idx_p036_pr_rank           ON pack036_utility_analysis.gl_portfolio_rankings(rank_position);
CREATE INDEX idx_p036_pr_eui            ON pack036_utility_analysis.gl_portfolio_rankings(eui_value);
CREATE INDEX idx_p036_pr_metric         ON pack036_utility_analysis.gl_portfolio_rankings(ranking_metric);
CREATE INDEX idx_p036_pr_quartile       ON pack036_utility_analysis.gl_portfolio_rankings(quartile);
CREATE INDEX idx_p036_pr_created        ON pack036_utility_analysis.gl_portfolio_rankings(created_at DESC);

-- Composite: portfolio + year + metric for ranked listing
CREATE INDEX idx_p036_pr_port_yr_met    ON pack036_utility_analysis.gl_portfolio_rankings(portfolio_id, period_year DESC, ranking_metric, rank_position);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_facility_benchmarks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_benchmark_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_portfolio_rankings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_fb_tenant_isolation
    ON pack036_utility_analysis.gl_facility_benchmarks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_fb_service_bypass
    ON pack036_utility_analysis.gl_facility_benchmarks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Benchmark targets are shared reference data (no tenant isolation)
CREATE POLICY p036_bt_public_read
    ON pack036_utility_analysis.gl_benchmark_targets
    USING (TRUE);
CREATE POLICY p036_bt_service_bypass
    ON pack036_utility_analysis.gl_benchmark_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_pr_tenant_isolation
    ON pack036_utility_analysis.gl_portfolio_rankings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_pr_service_bypass
    ON pack036_utility_analysis.gl_portfolio_rankings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_facility_benchmarks TO PUBLIC;
GRANT SELECT, INSERT ON pack036_utility_analysis.gl_benchmark_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_portfolio_rankings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Seed Data: CIBSE TM46 Benchmark Targets
-- ---------------------------------------------------------------------------
INSERT INTO pack036_utility_analysis.gl_benchmark_targets
    (building_type, standard, country_code, median_eui, good_practice_eui, top_quartile_eui, unit, source_year, source_document) VALUES
    ('OFFICE', 'CIBSE_TM46', 'GB', 120.0, 95.0, 70.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('RETAIL', 'CIBSE_TM46', 'GB', 165.0, 130.0, 100.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('WAREHOUSE', 'CIBSE_TM46', 'GB', 55.0, 40.0, 30.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('HEALTHCARE', 'CIBSE_TM46', 'GB', 410.0, 340.0, 280.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('EDUCATION', 'CIBSE_TM46', 'GB', 130.0, 100.0, 75.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('HOTEL', 'CIBSE_TM46', 'GB', 240.0, 200.0, 160.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks'),
    ('RESTAURANT', 'CIBSE_TM46', 'GB', 370.0, 300.0, 240.0, 'kWh/m2/yr', 2008, 'CIBSE TM46: Energy Benchmarks');

-- ---------------------------------------------------------------------------
-- Seed Data: ENERGY STAR Benchmark Targets (US National Medians)
-- ---------------------------------------------------------------------------
INSERT INTO pack036_utility_analysis.gl_benchmark_targets
    (building_type, standard, country_code, median_eui, good_practice_eui, top_quartile_eui, unit, source_year, source_document) VALUES
    ('OFFICE', 'ENERGY_STAR', 'US', 148.1, 100.0, 74.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('RETAIL', 'ENERGY_STAR', 'US', 185.0, 140.0, 105.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('WAREHOUSE', 'ENERGY_STAR', 'US', 65.0, 45.0, 32.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('HEALTHCARE', 'ENERGY_STAR', 'US', 480.0, 380.0, 300.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('EDUCATION', 'ENERGY_STAR', 'US', 135.0, 105.0, 80.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('DATA_CENTER', 'ENERGY_STAR', 'US', 2000.0, 1400.0, 1000.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference'),
    ('HOTEL', 'ENERGY_STAR', 'US', 270.0, 210.0, 165.0, 'kWh/m2/yr', 2024, 'CBECS 2018 / ENERGY STAR Technical Reference');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_facility_benchmarks IS
    'Annual facility energy performance benchmarks with EUI, ENERGY STAR score, weather normalization, and peer percentile.';

COMMENT ON TABLE pack036_utility_analysis.gl_benchmark_targets IS
    'Reference data for energy benchmark targets by building type and standard with median, good practice, and top quartile EUI.';

COMMENT ON TABLE pack036_utility_analysis.gl_portfolio_rankings IS
    'Portfolio-level facility rankings for performance comparison by EUI, cost, or other metrics.';

COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.benchmark_id IS
    'Unique identifier for the facility benchmark record.';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.site_eui_kwh_m2 IS
    'Site Energy Use Intensity in kWh per square metre per year.';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.source_eui_kwh_m2 IS
    'Source Energy Use Intensity (accounts for primary energy conversion factors).';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.energy_star_score IS
    'ENERGY STAR score (1-100, 75+ qualifies for ENERGY STAR certification).';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.weather_normalized IS
    'Whether the EUI values are weather-normalized.';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.benchmark_standard IS
    'Benchmark standard used: ENERGY_STAR, CIBSE_TM46, DIN_V_18599, ASHRAE_100, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.peer_percentile IS
    'Percentile rank among peers (0 = worst, 100 = best).';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.quartile IS
    'Performance quartile (1 = top/best, 4 = bottom/worst).';
COMMENT ON COLUMN pack036_utility_analysis.gl_facility_benchmarks.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_benchmark_targets.median_eui IS
    'Median EUI value for this building type and standard (typical performance).';
COMMENT ON COLUMN pack036_utility_analysis.gl_benchmark_targets.good_practice_eui IS
    'Good practice EUI target (better than median, achievable with standard measures).';
COMMENT ON COLUMN pack036_utility_analysis.gl_benchmark_targets.top_quartile_eui IS
    'Top quartile EUI target (top 25% performance).';
COMMENT ON COLUMN pack036_utility_analysis.gl_portfolio_rankings.rank_position IS
    'Facility rank within the portfolio (1 = best performer).';
COMMENT ON COLUMN pack036_utility_analysis.gl_portfolio_rankings.ranking_metric IS
    'Metric used for ranking: SITE_EUI, SOURCE_EUI, WEATHER_NORMALIZED_EUI, COST_PER_M2, CO2_INTENSITY, ENERGY_STAR_SCORE.';
COMMENT ON COLUMN pack036_utility_analysis.gl_portfolio_rankings.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
