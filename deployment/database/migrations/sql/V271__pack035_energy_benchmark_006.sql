-- =============================================================================
-- V271: PACK-035 Energy Benchmark Pack - Portfolio Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Portfolio management for multi-facility energy benchmarking. Supports
-- grouping facilities by region, business unit, or entity level with
-- aggregated metrics (area-weighted EUI, total energy, total CO2) and
-- facility rankings within the portfolio.
--
-- Tables (4):
--   1. pack035_energy_benchmark.portfolios
--   2. pack035_energy_benchmark.portfolio_memberships
--   3. pack035_energy_benchmark.portfolio_metrics
--   4. pack035_energy_benchmark.facility_rankings
--
-- Previous: V270__pack035_energy_benchmark_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.portfolios
-- =============================================================================
-- Portfolio definitions for grouping facilities under a single
-- management or reporting boundary (e.g., corporate portfolio,
-- regional portfolio, fund portfolio).

CREATE TABLE pack035_energy_benchmark.portfolios (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    portfolio_name          VARCHAR(255)    NOT NULL,
    description             TEXT,
    portfolio_type          VARCHAR(50)     DEFAULT 'CORPORATE',
    aggregation_method      VARCHAR(30)     DEFAULT 'AREA_WEIGHTED',
    currency                VARCHAR(3)      DEFAULT 'EUR',
    reporting_period        VARCHAR(20)     DEFAULT 'ANNUAL',
    is_active               BOOLEAN         DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p035_pf_type CHECK (
        portfolio_type IN ('CORPORATE', 'REGIONAL', 'FUND', 'SECTOR', 'CUSTOM')
    ),
    CONSTRAINT chk_p035_pf_aggregation CHECK (
        aggregation_method IN ('AREA_WEIGHTED', 'SIMPLE_AVERAGE', 'MEDIAN', 'SUM')
    ),
    CONSTRAINT chk_p035_pf_period CHECK (
        reporting_period IN ('MONTHLY', 'QUARTERLY', 'ANNUAL', 'ROLLING_12M')
    )
);

-- Indexes
CREATE INDEX idx_p035_pf_tenant          ON pack035_energy_benchmark.portfolios(tenant_id);
CREATE INDEX idx_p035_pf_type            ON pack035_energy_benchmark.portfolios(portfolio_type);
CREATE INDEX idx_p035_pf_active          ON pack035_energy_benchmark.portfolios(is_active);
CREATE INDEX idx_p035_pf_created         ON pack035_energy_benchmark.portfolios(created_at DESC);

-- Trigger
CREATE TRIGGER trg_p035_pf_updated
    BEFORE UPDATE ON pack035_energy_benchmark.portfolios
    FOR EACH ROW EXECUTE FUNCTION pack035_energy_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack035_energy_benchmark.portfolio_memberships
-- =============================================================================
-- Facility membership within portfolios with optional organisational
-- segmentation by entity level, region, and business unit.

CREATE TABLE pack035_energy_benchmark.portfolio_memberships (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id            UUID            NOT NULL REFERENCES pack035_energy_benchmark.portfolios(id) ON DELETE CASCADE,
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    entity_level            VARCHAR(100),
    region                  VARCHAR(100),
    business_unit           VARCHAR(100),
    cost_centre             VARCHAR(100),
    ownership_pct           DECIMAL(5, 2)   DEFAULT 100.00,
    added_at                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    removed_at              TIMESTAMPTZ,
    is_active               BOOLEAN         DEFAULT true,
    metadata                JSONB           DEFAULT '{}',
    -- Constraints
    CONSTRAINT chk_p035_pm_ownership CHECK (
        ownership_pct >= 0 AND ownership_pct <= 100
    ),
    CONSTRAINT uq_p035_pm_portfolio_facility UNIQUE (portfolio_id, facility_id)
);

-- Indexes
CREATE INDEX idx_p035_pm_portfolio       ON pack035_energy_benchmark.portfolio_memberships(portfolio_id);
CREATE INDEX idx_p035_pm_facility        ON pack035_energy_benchmark.portfolio_memberships(facility_id);
CREATE INDEX idx_p035_pm_tenant          ON pack035_energy_benchmark.portfolio_memberships(tenant_id);
CREATE INDEX idx_p035_pm_entity          ON pack035_energy_benchmark.portfolio_memberships(entity_level);
CREATE INDEX idx_p035_pm_region          ON pack035_energy_benchmark.portfolio_memberships(region);
CREATE INDEX idx_p035_pm_bu              ON pack035_energy_benchmark.portfolio_memberships(business_unit);
CREATE INDEX idx_p035_pm_active          ON pack035_energy_benchmark.portfolio_memberships(is_active);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.portfolio_metrics
-- =============================================================================
-- Aggregated portfolio-level energy performance metrics calculated
-- periodically. Tracks total floor area, total energy, weighted EUI,
-- total cost, total CO2, and year-over-year change.

CREATE TABLE pack035_energy_benchmark.portfolio_metrics (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id            UUID            NOT NULL REFERENCES pack035_energy_benchmark.portfolios(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    calculation_date        DATE            NOT NULL,
    period_start            DATE,
    period_end              DATE,
    -- Facility counts
    total_facilities        INTEGER         NOT NULL DEFAULT 0,
    facilities_with_data    INTEGER         DEFAULT 0,
    data_coverage_pct       DECIMAL(5, 2),
    -- Aggregated metrics
    total_floor_area_m2     DECIMAL(14, 2),
    total_energy_kwh        DECIMAL(16, 2),
    area_weighted_eui       DECIMAL(10, 4),
    simple_average_eui      DECIMAL(10, 4),
    median_eui              DECIMAL(10, 4),
    eui_std_dev             DECIMAL(10, 4),
    -- Cost and emissions
    total_cost_eur          DECIMAL(16, 4),
    total_co2_tonnes        DECIMAL(14, 4),
    co2_intensity_kg_m2     DECIMAL(10, 4),
    cost_per_m2_eur         DECIMAL(10, 4),
    -- Performance distribution
    pct_top_quartile        DECIMAL(5, 2),
    pct_bottom_quartile     DECIMAL(5, 2),
    -- Best and worst
    best_performer_id       UUID            REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE SET NULL,
    worst_performer_id      UUID            REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE SET NULL,
    best_eui                DECIMAL(10, 4),
    worst_eui               DECIMAL(10, 4),
    -- Year-over-year
    yoy_eui_change_pct      DECIMAL(8, 4),
    yoy_energy_change_pct   DECIMAL(8, 4),
    yoy_co2_change_pct      DECIMAL(8, 4),
    -- Metadata
    provenance_hash         VARCHAR(64)     NOT NULL,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_pmet_facilities CHECK (
        total_facilities >= 0
    ),
    CONSTRAINT chk_p035_pmet_coverage CHECK (
        data_coverage_pct IS NULL OR (data_coverage_pct >= 0 AND data_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p035_pmet_area CHECK (
        total_floor_area_m2 IS NULL OR total_floor_area_m2 >= 0
    ),
    CONSTRAINT chk_p035_pmet_energy CHECK (
        total_energy_kwh IS NULL OR total_energy_kwh >= 0
    ),
    CONSTRAINT chk_p035_pmet_eui CHECK (
        area_weighted_eui IS NULL OR area_weighted_eui >= 0
    )
);

-- Indexes
CREATE INDEX idx_p035_pmet_portfolio     ON pack035_energy_benchmark.portfolio_metrics(portfolio_id);
CREATE INDEX idx_p035_pmet_tenant        ON pack035_energy_benchmark.portfolio_metrics(tenant_id);
CREATE INDEX idx_p035_pmet_date          ON pack035_energy_benchmark.portfolio_metrics(calculation_date DESC);
CREATE INDEX idx_p035_pmet_eui           ON pack035_energy_benchmark.portfolio_metrics(area_weighted_eui);
CREATE INDEX idx_p035_pmet_pf_date       ON pack035_energy_benchmark.portfolio_metrics(portfolio_id, calculation_date DESC);

-- =============================================================================
-- Table 4: pack035_energy_benchmark.facility_rankings
-- =============================================================================
-- Individual facility rankings within a portfolio metrics snapshot,
-- providing rank position, EUI value, tier classification, and
-- improvement rate for performance league tables.

CREATE TABLE pack035_energy_benchmark.facility_rankings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_metrics_id    UUID            NOT NULL REFERENCES pack035_energy_benchmark.portfolio_metrics(id) ON DELETE CASCADE,
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    rank_position           INTEGER         NOT NULL,
    eui_value               DECIMAL(10, 4)  NOT NULL,
    normalised_eui          DECIMAL(10, 4),
    percentile_in_portfolio DECIMAL(6, 3),
    tier                    VARCHAR(20),
    improvement_rate_pct    DECIMAL(8, 4),
    previous_rank           INTEGER,
    rank_change             INTEGER,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_fr_rank CHECK (
        rank_position >= 1
    ),
    CONSTRAINT chk_p035_fr_eui CHECK (
        eui_value >= 0
    ),
    CONSTRAINT chk_p035_fr_percentile CHECK (
        percentile_in_portfolio IS NULL OR (percentile_in_portfolio >= 0 AND percentile_in_portfolio <= 100)
    ),
    CONSTRAINT chk_p035_fr_tier CHECK (
        tier IS NULL OR tier IN ('LEADER', 'GOOD', 'AVERAGE', 'BELOW_AVERAGE', 'LAGGARD')
    )
);

-- Indexes
CREATE INDEX idx_p035_fr_metrics         ON pack035_energy_benchmark.facility_rankings(portfolio_metrics_id);
CREATE INDEX idx_p035_fr_facility        ON pack035_energy_benchmark.facility_rankings(facility_id);
CREATE INDEX idx_p035_fr_tenant          ON pack035_energy_benchmark.facility_rankings(tenant_id);
CREATE INDEX idx_p035_fr_rank            ON pack035_energy_benchmark.facility_rankings(rank_position);
CREATE INDEX idx_p035_fr_tier            ON pack035_energy_benchmark.facility_rankings(tier);
CREATE INDEX idx_p035_fr_eui             ON pack035_energy_benchmark.facility_rankings(eui_value);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.portfolio_memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.portfolio_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.facility_rankings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_pf_tenant_isolation ON pack035_energy_benchmark.portfolios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pf_service_bypass ON pack035_energy_benchmark.portfolios
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_pm_tenant_isolation ON pack035_energy_benchmark.portfolio_memberships
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pm_service_bypass ON pack035_energy_benchmark.portfolio_memberships
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_pmet_tenant_isolation ON pack035_energy_benchmark.portfolio_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pmet_service_bypass ON pack035_energy_benchmark.portfolio_metrics
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_fr_tenant_isolation ON pack035_energy_benchmark.facility_rankings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_fr_service_bypass ON pack035_energy_benchmark.facility_rankings
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.portfolios TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.portfolio_memberships TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.portfolio_metrics TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.facility_rankings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.portfolios IS
    'Portfolio definitions for multi-facility energy benchmarking with aggregation method and reporting period.';
COMMENT ON TABLE pack035_energy_benchmark.portfolio_memberships IS
    'Facility membership in portfolios with organisational segmentation (entity, region, business unit).';
COMMENT ON TABLE pack035_energy_benchmark.portfolio_metrics IS
    'Aggregated portfolio-level energy performance metrics: weighted EUI, total energy/cost/CO2, YoY change.';
COMMENT ON TABLE pack035_energy_benchmark.facility_rankings IS
    'Facility rankings within a portfolio snapshot for performance league tables and tier classification.';

COMMENT ON COLUMN pack035_energy_benchmark.portfolio_metrics.area_weighted_eui IS
    'Area-weighted EUI = SUM(facility_energy) / SUM(facility_area). Primary portfolio metric.';
COMMENT ON COLUMN pack035_energy_benchmark.portfolio_memberships.ownership_pct IS
    'Ownership percentage (0-100) for partial ownership accounting (e.g., 50% JV).';
COMMENT ON COLUMN pack035_energy_benchmark.facility_rankings.tier IS
    'Performance tier: LEADER (top 10%), GOOD (10-25%), AVERAGE (25-75%), BELOW_AVERAGE (75-90%), LAGGARD (bottom 10%).';
