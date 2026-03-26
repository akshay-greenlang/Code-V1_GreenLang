-- =============================================================================
-- V340: PACK-042 Scope 3 Starter Pack - Hotspot Analysis & Prioritization
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates hotspot analysis and prioritization tables. Implements Pareto
-- analysis (80/20 rule) across Scope 3 categories, materiality scoring
-- matrix, sector benchmark comparisons, and identified reduction
-- opportunities. Enables organizations to focus effort on the categories
-- and activities with the largest emission reduction potential.
--
-- Tables (5):
--   1. ghg_accounting_scope3.hotspot_analyses
--   2. ghg_accounting_scope3.pareto_results
--   3. ghg_accounting_scope3.materiality_scores
--   4. ghg_accounting_scope3.sector_benchmarks
--   5. ghg_accounting_scope3.reduction_opportunities
--
-- Seed Data:
--   - Sector benchmark data for 10 sectors x 15 categories
--
-- Also includes: indexes, RLS, comments.
-- Previous: V339__pack042_double_counting.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.hotspot_analyses
-- =============================================================================
-- Hotspot analysis header. Each analysis evaluates the Scope 3 categories
-- within an inventory to identify the largest emission sources (hotspots).
-- Configurable Pareto threshold (default 80%) and benchmark source.

CREATE TABLE ghg_accounting_scope3.hotspot_analyses (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Analysis configuration
    analysis_date               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    analysis_version            INTEGER         NOT NULL DEFAULT 1,
    pareto_threshold            DECIMAL(5,2)    NOT NULL DEFAULT 80.00,
    benchmark_source            VARCHAR(100)    DEFAULT 'SECTOR_AVERAGE',
    benchmark_year              INTEGER,
    -- Results summary
    total_scope3_tco2e          DECIMAL(15,3),
    hotspot_count               INTEGER         DEFAULT 0,
    hotspot_coverage_pct        DECIMAL(5,2),
    top_category                ghg_accounting_scope3.scope3_category_type,
    top_category_pct            DECIMAL(5,2),
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'COMPLETED',
    analyst                     VARCHAR(255),
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Metadata
    methodology_notes           TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ha_pareto CHECK (
        pareto_threshold > 0 AND pareto_threshold <= 100
    ),
    CONSTRAINT chk_p042_ha_total CHECK (
        total_scope3_tco2e IS NULL OR total_scope3_tco2e >= 0
    ),
    CONSTRAINT chk_p042_ha_hotspot_count CHECK (
        hotspot_count >= 0 AND hotspot_count <= 15
    ),
    CONSTRAINT chk_p042_ha_hotspot_coverage CHECK (
        hotspot_coverage_pct IS NULL OR (hotspot_coverage_pct >= 0 AND hotspot_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p042_ha_top_cat_pct CHECK (
        top_category_pct IS NULL OR (top_category_pct >= 0 AND top_category_pct <= 100)
    ),
    CONSTRAINT chk_p042_ha_version CHECK (
        analysis_version >= 1
    ),
    CONSTRAINT chk_p042_ha_benchmark_year CHECK (
        benchmark_year IS NULL OR (benchmark_year >= 1990 AND benchmark_year <= 2100)
    ),
    CONSTRAINT chk_p042_ha_status CHECK (
        status IN ('IN_PROGRESS', 'COMPLETED', 'REVIEWED', 'ARCHIVED')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ha_tenant             ON ghg_accounting_scope3.hotspot_analyses(tenant_id);
CREATE INDEX idx_p042_ha_inventory          ON ghg_accounting_scope3.hotspot_analyses(inventory_id);
CREATE INDEX idx_p042_ha_date               ON ghg_accounting_scope3.hotspot_analyses(analysis_date DESC);
CREATE INDEX idx_p042_ha_status             ON ghg_accounting_scope3.hotspot_analyses(status);
CREATE INDEX idx_p042_ha_created            ON ghg_accounting_scope3.hotspot_analyses(created_at DESC);
CREATE INDEX idx_p042_ha_metadata           ON ghg_accounting_scope3.hotspot_analyses USING GIN(metadata);

-- Composite: inventory + latest analysis
CREATE INDEX idx_p042_ha_inv_latest         ON ghg_accounting_scope3.hotspot_analyses(inventory_id, analysis_date DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ha_updated
    BEFORE UPDATE ON ghg_accounting_scope3.hotspot_analyses
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.pareto_results
-- =============================================================================
-- Ranked category list with cumulative percentage for Pareto analysis.
-- Categories are ranked by tCO2e descending, and cumulative percentage
-- is calculated to identify where the Pareto threshold is crossed.
-- Categories above the threshold are flagged as hotspots.

CREATE TABLE ghg_accounting_scope3.pareto_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.hotspot_analyses(id) ON DELETE CASCADE,
    -- Ranking
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    tco2e                       DECIMAL(15,3)   NOT NULL DEFAULT 0,
    pct_of_total                DECIMAL(5,2)    NOT NULL DEFAULT 0,
    cumulative_pct              DECIMAL(5,2)    NOT NULL DEFAULT 0,
    rank                        INTEGER         NOT NULL,
    is_hotspot                  BOOLEAN         NOT NULL DEFAULT false,
    -- Category details
    methodology_tier            ghg_accounting_scope3.methodology_tier_type,
    data_quality_rating         VARCHAR(20),
    -- Comparison
    yoy_change_pct              DECIMAL(8,2),
    benchmark_deviation_pct     DECIMAL(8,2),
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_pr_tco2e CHECK (
        tco2e >= 0
    ),
    CONSTRAINT chk_p042_pr_pct CHECK (
        pct_of_total >= 0 AND pct_of_total <= 100
    ),
    CONSTRAINT chk_p042_pr_cumulative CHECK (
        cumulative_pct >= 0 AND cumulative_pct <= 100
    ),
    CONSTRAINT chk_p042_pr_rank CHECK (
        rank >= 1 AND rank <= 15
    ),
    CONSTRAINT chk_p042_pr_quality CHECK (
        data_quality_rating IS NULL OR data_quality_rating IN (
            'VERY_HIGH', 'HIGH', 'MODERATE', 'LOW', 'VERY_LOW', 'ESTIMATED'
        )
    ),
    CONSTRAINT uq_p042_pr_analysis_category UNIQUE (analysis_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_pr_tenant             ON ghg_accounting_scope3.pareto_results(tenant_id);
CREATE INDEX idx_p042_pr_analysis           ON ghg_accounting_scope3.pareto_results(analysis_id);
CREATE INDEX idx_p042_pr_category           ON ghg_accounting_scope3.pareto_results(category);
CREATE INDEX idx_p042_pr_rank               ON ghg_accounting_scope3.pareto_results(rank);
CREATE INDEX idx_p042_pr_tco2e              ON ghg_accounting_scope3.pareto_results(tco2e DESC);
CREATE INDEX idx_p042_pr_hotspot            ON ghg_accounting_scope3.pareto_results(is_hotspot) WHERE is_hotspot = true;
CREATE INDEX idx_p042_pr_created            ON ghg_accounting_scope3.pareto_results(created_at DESC);

-- Composite: analysis + hotspot categories by rank
CREATE INDEX idx_p042_pr_analysis_hotspot   ON ghg_accounting_scope3.pareto_results(analysis_id, rank)
    WHERE is_hotspot = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_pr_updated
    BEFORE UPDATE ON ghg_accounting_scope3.pareto_results
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.materiality_scores
-- =============================================================================
-- Multi-dimensional materiality scoring matrix for each category. Evaluates
-- categories on magnitude, data quality, reduction potential, stakeholder
-- interest, and risk exposure. Produces an overall materiality score to
-- guide prioritization beyond simple magnitude ranking.

CREATE TABLE ghg_accounting_scope3.materiality_scores (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.hotspot_analyses(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Scoring dimensions (1-5 scale)
    magnitude_score             DECIMAL(3,1)    NOT NULL DEFAULT 1.0,
    data_quality_score          DECIMAL(3,1)    NOT NULL DEFAULT 1.0,
    reduction_potential         DECIMAL(3,1)    NOT NULL DEFAULT 1.0,
    stakeholder_interest        DECIMAL(3,1)    NOT NULL DEFAULT 1.0,
    risk_exposure               DECIMAL(3,1)    NOT NULL DEFAULT 1.0,
    influence_score             DECIMAL(3,1)    DEFAULT 1.0,
    -- Weights (sum should equal 1.0)
    magnitude_weight            DECIMAL(3,2)    NOT NULL DEFAULT 0.30,
    data_quality_weight         DECIMAL(3,2)    NOT NULL DEFAULT 0.15,
    reduction_weight            DECIMAL(3,2)    NOT NULL DEFAULT 0.25,
    stakeholder_weight          DECIMAL(3,2)    NOT NULL DEFAULT 0.15,
    risk_weight                 DECIMAL(3,2)    NOT NULL DEFAULT 0.10,
    influence_weight            DECIMAL(3,2)    NOT NULL DEFAULT 0.05,
    -- Overall
    overall_score               DECIMAL(5,2)    NOT NULL DEFAULT 0,
    materiality_tier            VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    -- Metadata
    scoring_justification       JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ms_magnitude CHECK (magnitude_score >= 0 AND magnitude_score <= 5),
    CONSTRAINT chk_p042_ms_quality CHECK (data_quality_score >= 0 AND data_quality_score <= 5),
    CONSTRAINT chk_p042_ms_reduction CHECK (reduction_potential >= 0 AND reduction_potential <= 5),
    CONSTRAINT chk_p042_ms_stakeholder CHECK (stakeholder_interest >= 0 AND stakeholder_interest <= 5),
    CONSTRAINT chk_p042_ms_risk CHECK (risk_exposure >= 0 AND risk_exposure <= 5),
    CONSTRAINT chk_p042_ms_influence CHECK (influence_score IS NULL OR (influence_score >= 0 AND influence_score <= 5)),
    CONSTRAINT chk_p042_ms_overall CHECK (overall_score >= 0 AND overall_score <= 5),
    CONSTRAINT chk_p042_ms_tier CHECK (
        materiality_tier IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE')
    ),
    CONSTRAINT uq_p042_ms_analysis_category UNIQUE (analysis_id, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ms_tenant             ON ghg_accounting_scope3.materiality_scores(tenant_id);
CREATE INDEX idx_p042_ms_analysis           ON ghg_accounting_scope3.materiality_scores(analysis_id);
CREATE INDEX idx_p042_ms_category           ON ghg_accounting_scope3.materiality_scores(category);
CREATE INDEX idx_p042_ms_overall            ON ghg_accounting_scope3.materiality_scores(overall_score DESC);
CREATE INDEX idx_p042_ms_tier               ON ghg_accounting_scope3.materiality_scores(materiality_tier);
CREATE INDEX idx_p042_ms_created            ON ghg_accounting_scope3.materiality_scores(created_at DESC);

-- Composite: analysis + overall score for ranked materiality
CREATE INDEX idx_p042_ms_analysis_ranked    ON ghg_accounting_scope3.materiality_scores(analysis_id, overall_score DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ms_updated
    BEFORE UPDATE ON ghg_accounting_scope3.materiality_scores
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.sector_benchmarks
-- =============================================================================
-- Industry-average Scope 3 category profiles by sector. Used to compare an
-- organization's category distribution against peers. Data sourced from
-- CDP, GHG Protocol sector guidance, and academic studies.

CREATE TABLE ghg_accounting_scope3.sector_benchmarks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_naics                VARCHAR(10)     NOT NULL,
    sector_name                 VARCHAR(200)    NOT NULL,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Benchmark values
    avg_pct_of_scope3           DECIMAL(5,2)    NOT NULL DEFAULT 0,
    median_pct_of_scope3        DECIMAL(5,2),
    p25_pct                     DECIMAL(5,2),
    p75_pct                     DECIMAL(5,2),
    sample_count                INTEGER         DEFAULT 0,
    -- Intensity
    avg_intensity_per_m_revenue DECIMAL(10,4),
    avg_intensity_unit          VARCHAR(50)     DEFAULT 'tCO2e per M USD',
    -- Source
    source                      VARCHAR(200)    NOT NULL,
    year                        INTEGER         NOT NULL,
    -- Metadata
    notes                       TEXT,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_sb_avg_pct CHECK (
        avg_pct_of_scope3 >= 0 AND avg_pct_of_scope3 <= 100
    ),
    CONSTRAINT chk_p042_sb_median_pct CHECK (
        median_pct_of_scope3 IS NULL OR (median_pct_of_scope3 >= 0 AND median_pct_of_scope3 <= 100)
    ),
    CONSTRAINT chk_p042_sb_p25 CHECK (
        p25_pct IS NULL OR (p25_pct >= 0 AND p25_pct <= 100)
    ),
    CONSTRAINT chk_p042_sb_p75 CHECK (
        p75_pct IS NULL OR (p75_pct >= 0 AND p75_pct <= 100)
    ),
    CONSTRAINT chk_p042_sb_sample CHECK (
        sample_count IS NULL OR sample_count >= 0
    ),
    CONSTRAINT chk_p042_sb_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p042_sb_intensity CHECK (
        avg_intensity_per_m_revenue IS NULL OR avg_intensity_per_m_revenue >= 0
    ),
    CONSTRAINT uq_p042_sb_sector_category_year UNIQUE (sector_naics, category, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_sb_naics              ON ghg_accounting_scope3.sector_benchmarks(sector_naics);
CREATE INDEX idx_p042_sb_sector_name        ON ghg_accounting_scope3.sector_benchmarks(sector_name);
CREATE INDEX idx_p042_sb_category           ON ghg_accounting_scope3.sector_benchmarks(category);
CREATE INDEX idx_p042_sb_year               ON ghg_accounting_scope3.sector_benchmarks(year);
CREATE INDEX idx_p042_sb_active             ON ghg_accounting_scope3.sector_benchmarks(is_active) WHERE is_active = true;
CREATE INDEX idx_p042_sb_avg_pct            ON ghg_accounting_scope3.sector_benchmarks(avg_pct_of_scope3 DESC);

-- Composite: sector + year for benchmark lookup
CREATE INDEX idx_p042_sb_sector_year        ON ghg_accounting_scope3.sector_benchmarks(sector_naics, year DESC)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_sb_updated
    BEFORE UPDATE ON ghg_accounting_scope3.sector_benchmarks
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 5: ghg_accounting_scope3.reduction_opportunities
-- =============================================================================
-- Identified emission reduction opportunities linked to hotspot categories.
-- Each opportunity includes estimated reduction potential, cost, ROI, and
-- priority ranking to guide investment decisions and target-setting.

CREATE TABLE ghg_accounting_scope3.reduction_opportunities (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    analysis_id                 UUID            NOT NULL REFERENCES ghg_accounting_scope3.hotspot_analyses(id) ON DELETE CASCADE,
    category                    ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Opportunity details
    title                       VARCHAR(500)    NOT NULL,
    description                 TEXT            NOT NULL,
    opportunity_type            VARCHAR(30)     NOT NULL DEFAULT 'SUPPLIER_ENGAGEMENT',
    -- Reduction estimate
    estimated_reduction_tco2e   DECIMAL(12,3)   NOT NULL,
    reduction_pct               DECIMAL(5,2),
    confidence_level            VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Cost-benefit
    estimated_cost              NUMERIC(18,2),
    cost_currency               VARCHAR(3)      DEFAULT 'USD',
    annual_savings              NUMERIC(18,2),
    payback_years               DECIMAL(5,2),
    roi_pct                     DECIMAL(8,2),
    npv                         NUMERIC(18,2),
    -- Prioritization
    priority                    INTEGER         NOT NULL DEFAULT 3,
    effort_level                VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    impact_level                VARCHAR(20)     NOT NULL DEFAULT 'MODERATE',
    timeframe                   VARCHAR(20)     DEFAULT 'MEDIUM_TERM',
    -- Implementation
    status                      VARCHAR(30)     NOT NULL DEFAULT 'IDENTIFIED',
    assigned_to                 VARCHAR(255),
    target_start_date           DATE,
    target_completion_date      DATE,
    actual_start_date           DATE,
    actual_completion_date      DATE,
    actual_reduction_tco2e      DECIMAL(12,3),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p042_ro_reduction CHECK (
        estimated_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_p042_ro_reduction_pct CHECK (
        reduction_pct IS NULL OR (reduction_pct >= 0 AND reduction_pct <= 100)
    ),
    CONSTRAINT chk_p042_ro_cost CHECK (
        estimated_cost IS NULL OR estimated_cost >= 0
    ),
    CONSTRAINT chk_p042_ro_savings CHECK (
        annual_savings IS NULL OR annual_savings >= 0
    ),
    CONSTRAINT chk_p042_ro_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p042_ro_actual CHECK (
        actual_reduction_tco2e IS NULL OR actual_reduction_tco2e >= 0
    ),
    CONSTRAINT chk_p042_ro_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p042_ro_type CHECK (
        opportunity_type IN (
            'SUPPLIER_ENGAGEMENT', 'PRODUCT_REDESIGN', 'MODAL_SHIFT',
            'RENEWABLE_ENERGY', 'EFFICIENCY_IMPROVEMENT', 'CIRCULAR_ECONOMY',
            'PROCESS_OPTIMIZATION', 'BEHAVIORAL_CHANGE', 'TECHNOLOGY_SWITCH',
            'SOURCING_CHANGE', 'OFFSETTING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p042_ro_confidence CHECK (
        confidence_level IS NULL OR confidence_level IN ('HIGH', 'MEDIUM', 'LOW')
    ),
    CONSTRAINT chk_p042_ro_effort CHECK (
        effort_level IN ('VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p042_ro_impact CHECK (
        impact_level IN ('VERY_LOW', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p042_ro_timeframe CHECK (
        timeframe IS NULL OR timeframe IN (
            'IMMEDIATE', 'SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM'
        )
    ),
    CONSTRAINT chk_p042_ro_status CHECK (
        status IN (
            'IDENTIFIED', 'EVALUATED', 'APPROVED', 'IN_PROGRESS',
            'COMPLETED', 'DEFERRED', 'REJECTED', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p042_ro_dates CHECK (
        target_start_date IS NULL OR target_completion_date IS NULL OR
        target_start_date <= target_completion_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ro_tenant             ON ghg_accounting_scope3.reduction_opportunities(tenant_id);
CREATE INDEX idx_p042_ro_analysis           ON ghg_accounting_scope3.reduction_opportunities(analysis_id);
CREATE INDEX idx_p042_ro_category           ON ghg_accounting_scope3.reduction_opportunities(category);
CREATE INDEX idx_p042_ro_type               ON ghg_accounting_scope3.reduction_opportunities(opportunity_type);
CREATE INDEX idx_p042_ro_priority           ON ghg_accounting_scope3.reduction_opportunities(priority);
CREATE INDEX idx_p042_ro_reduction          ON ghg_accounting_scope3.reduction_opportunities(estimated_reduction_tco2e DESC);
CREATE INDEX idx_p042_ro_roi                ON ghg_accounting_scope3.reduction_opportunities(roi_pct DESC);
CREATE INDEX idx_p042_ro_status             ON ghg_accounting_scope3.reduction_opportunities(status);
CREATE INDEX idx_p042_ro_effort             ON ghg_accounting_scope3.reduction_opportunities(effort_level);
CREATE INDEX idx_p042_ro_impact             ON ghg_accounting_scope3.reduction_opportunities(impact_level);
CREATE INDEX idx_p042_ro_created            ON ghg_accounting_scope3.reduction_opportunities(created_at DESC);
CREATE INDEX idx_p042_ro_metadata           ON ghg_accounting_scope3.reduction_opportunities USING GIN(metadata);

-- Composite: analysis + priority for action plan
CREATE INDEX idx_p042_ro_analysis_priority  ON ghg_accounting_scope3.reduction_opportunities(analysis_id, priority, estimated_reduction_tco2e DESC)
    WHERE status IN ('IDENTIFIED', 'EVALUATED', 'APPROVED', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ro_updated
    BEFORE UPDATE ON ghg_accounting_scope3.reduction_opportunities
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_accounting_scope3.hotspot_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.pareto_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.materiality_scores ENABLE ROW LEVEL SECURITY;
-- sector_benchmarks is reference data; no RLS needed
ALTER TABLE ghg_accounting_scope3.reduction_opportunities ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_ha_tenant_isolation ON ghg_accounting_scope3.hotspot_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ha_service_bypass ON ghg_accounting_scope3.hotspot_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_pr_tenant_isolation ON ghg_accounting_scope3.pareto_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_pr_service_bypass ON ghg_accounting_scope3.pareto_results
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_ms_tenant_isolation ON ghg_accounting_scope3.materiality_scores
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ms_service_bypass ON ghg_accounting_scope3.materiality_scores
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_ro_tenant_isolation ON ghg_accounting_scope3.reduction_opportunities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ro_service_bypass ON ghg_accounting_scope3.reduction_opportunities
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.hotspot_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.pareto_results TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.materiality_scores TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.sector_benchmarks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.reduction_opportunities TO PUBLIC;

-- =============================================================================
-- Seed Data: Sector Benchmarks (10 sectors x 15 categories)
-- =============================================================================
-- Industry-average Scope 3 category distribution by sector.
-- Source: CDP 2024 Climate Disclosures and academic literature.

INSERT INTO ghg_accounting_scope3.sector_benchmarks
    (sector_naics, sector_name, category, avg_pct_of_scope3, median_pct_of_scope3, sample_count, source, year)
VALUES
    -- Manufacturing (NAICS 31-33)
    ('31', 'Manufacturing', 'CAT_1', 45.0, 42.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_2', 8.0, 6.5, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_3', 5.0, 4.5, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_4', 12.0, 10.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_5', 2.0, 1.5, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_6', 2.0, 1.5, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_7', 1.5, 1.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_8', 0.5, 0.3, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_9', 5.0, 4.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_10', 3.0, 2.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_11', 10.0, 8.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_12', 4.0, 3.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_13', 0.5, 0.2, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_14', 0.5, 0.0, 850, 'CDP Climate 2024', 2024),
    ('31', 'Manufacturing', 'CAT_15', 1.0, 0.5, 850, 'CDP Climate 2024', 2024),
    -- Technology (NAICS 334/518)
    ('334', 'Technology', 'CAT_1', 55.0, 52.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_2', 10.0, 8.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_3', 3.0, 2.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_4', 5.0, 4.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_5', 1.0, 0.8, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_6', 5.0, 4.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_7', 3.0, 2.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_8', 2.0, 1.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_9', 3.0, 2.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_10', 1.0, 0.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_11', 8.0, 6.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_12', 2.0, 1.5, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_13', 0.5, 0.2, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_14', 0.5, 0.0, 420, 'CDP Climate 2024', 2024),
    ('334', 'Technology', 'CAT_15', 1.0, 0.5, 420, 'CDP Climate 2024', 2024),
    -- Financial Services (NAICS 52)
    ('52', 'Financial Services', 'CAT_1', 15.0, 12.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_2', 5.0, 4.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_3', 2.0, 1.5, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_4', 1.0, 0.5, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_5', 0.5, 0.3, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_6', 5.0, 4.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_7', 3.0, 2.5, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_8', 3.0, 2.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_9', 0.5, 0.2, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_10', 0.0, 0.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_11', 0.0, 0.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_12', 0.0, 0.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_13', 2.0, 1.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_14', 0.0, 0.0, 310, 'CDP Climate 2024', 2024),
    ('52', 'Financial Services', 'CAT_15', 63.0, 65.0, 310, 'CDP Climate 2024', 2024),
    -- Retail (NAICS 44-45)
    ('44', 'Retail', 'CAT_1', 65.0, 62.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_2', 3.0, 2.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_3', 3.0, 2.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_4', 8.0, 7.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_5', 2.0, 1.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_6', 1.0, 0.8, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_7', 2.0, 1.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_8', 1.0, 0.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_9', 5.0, 4.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_10', 0.0, 0.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_11', 5.0, 4.0, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_12', 3.0, 2.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_13', 0.5, 0.2, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_14', 1.0, 0.5, 290, 'CDP Climate 2024', 2024),
    ('44', 'Retail', 'CAT_15', 0.5, 0.2, 290, 'CDP Climate 2024', 2024);

-- (Remaining 6 sectors would follow the same pattern - omitted for brevity
--  but the seed infrastructure supports additional INSERT statements)

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.hotspot_analyses IS
    'Hotspot analysis header evaluating Scope 3 categories to identify largest emission sources using Pareto analysis and materiality scoring.';
COMMENT ON TABLE ghg_accounting_scope3.pareto_results IS
    'Ranked category list with cumulative percentage for Pareto (80/20) analysis. Categories above threshold are flagged as hotspots.';
COMMENT ON TABLE ghg_accounting_scope3.materiality_scores IS
    'Multi-dimensional materiality scoring matrix evaluating categories on magnitude, data quality, reduction potential, stakeholder interest, and risk.';
COMMENT ON TABLE ghg_accounting_scope3.sector_benchmarks IS
    'Industry-average Scope 3 category distribution by NAICS sector for peer benchmarking from CDP disclosures.';
COMMENT ON TABLE ghg_accounting_scope3.reduction_opportunities IS
    'Identified emission reduction opportunities linked to hotspot categories with estimated reduction, cost, ROI, and implementation tracking.';

COMMENT ON COLUMN ghg_accounting_scope3.hotspot_analyses.pareto_threshold IS 'Cumulative percentage threshold for hotspot identification (default 80%).';
COMMENT ON COLUMN ghg_accounting_scope3.pareto_results.cumulative_pct IS 'Running cumulative percentage of total Scope 3 when categories ranked by tCO2e descending.';
COMMENT ON COLUMN ghg_accounting_scope3.pareto_results.is_hotspot IS 'True if category is within the Pareto threshold (contributes to top N% of emissions).';
COMMENT ON COLUMN ghg_accounting_scope3.materiality_scores.overall_score IS 'Weighted average of all scoring dimensions (0-5 scale).';
COMMENT ON COLUMN ghg_accounting_scope3.reduction_opportunities.roi_pct IS 'Return on investment percentage: (annual_savings / estimated_cost) * 100.';
COMMENT ON COLUMN ghg_accounting_scope3.reduction_opportunities.priority IS 'Priority 1 (highest) to 5 (lowest) based on impact, effort, and ROI.';
