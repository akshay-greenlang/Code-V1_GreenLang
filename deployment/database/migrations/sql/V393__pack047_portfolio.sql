-- =============================================================================
-- V393: PACK-047 GHG Emissions Benchmark Pack - Portfolio Carbon Benchmarking
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for portfolio-level carbon benchmarking per PCAF, TCFD,
-- and PAB/CTB regulations. Portfolios hold investment holdings across
-- asset classes (listed equity, corporate bonds, project finance, commercial
-- real estate, mortgages, sovereign debt). Portfolio results calculate WACI,
-- carbon footprint, carbon intensity, total financed emissions, tracking
-- error, sector attribution, and top contributor analysis.
--
-- Tables (3):
--   1. ghg_benchmark.gl_bm_portfolios
--   2. ghg_benchmark.gl_bm_holdings
--   3. ghg_benchmark.gl_bm_portfolio_results
--
-- Also includes: indexes, RLS, comments.
-- Previous: V392__pack047_trajectory.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_portfolios
-- =============================================================================
-- Portfolio definitions for carbon benchmarking. Each portfolio has assets
-- under management (AUM), currency, holding count, PCAF coverage percentage,
-- and aggregate data quality score. Portfolios are linked to a benchmark
-- configuration and may represent investment funds, lending books, insurance
-- underwriting portfolios, or custom groupings.

CREATE TABLE ghg_benchmark.gl_bm_portfolios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    portfolio_name              VARCHAR(255)    NOT NULL,
    portfolio_type              VARCHAR(50)     NOT NULL DEFAULT 'INVESTMENT',
    aum                         NUMERIC(20,2),
    currency                    VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    holding_count               INTEGER         NOT NULL DEFAULT 0,
    coverage_pct                NUMERIC(5,2)    NOT NULL DEFAULT 0,
    pcaf_quality                NUMERIC(3,1),
    benchmark_index             VARCHAR(100),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p047_pf_type CHECK (
        portfolio_type IN (
            'INVESTMENT', 'LENDING', 'UNDERWRITING', 'REAL_ESTATE',
            'PROJECT_FINANCE', 'SOVEREIGN', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p047_pf_aum CHECK (
        aum IS NULL OR aum >= 0
    ),
    CONSTRAINT chk_p047_pf_currency CHECK (
        LENGTH(currency) = 3
    ),
    CONSTRAINT chk_p047_pf_holdings CHECK (
        holding_count >= 0
    ),
    CONSTRAINT chk_p047_pf_coverage CHECK (
        coverage_pct >= 0 AND coverage_pct <= 100
    ),
    CONSTRAINT chk_p047_pf_pcaf CHECK (
        pcaf_quality IS NULL OR (pcaf_quality >= 1 AND pcaf_quality <= 5)
    ),
    CONSTRAINT uq_p047_pf_config_name UNIQUE (config_id, portfolio_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pf_tenant            ON ghg_benchmark.gl_bm_portfolios(tenant_id);
CREATE INDEX idx_p047_pf_config            ON ghg_benchmark.gl_bm_portfolios(config_id);
CREATE INDEX idx_p047_pf_name              ON ghg_benchmark.gl_bm_portfolios(portfolio_name);
CREATE INDEX idx_p047_pf_type              ON ghg_benchmark.gl_bm_portfolios(portfolio_type);
CREATE INDEX idx_p047_pf_active            ON ghg_benchmark.gl_bm_portfolios(is_active) WHERE is_active = true;
CREATE INDEX idx_p047_pf_benchmark         ON ghg_benchmark.gl_bm_portfolios(benchmark_index);
CREATE INDEX idx_p047_pf_created           ON ghg_benchmark.gl_bm_portfolios(created_at DESC);
CREATE INDEX idx_p047_pf_metadata          ON ghg_benchmark.gl_bm_portfolios USING GIN(metadata);

-- Composite: tenant + config for listing
CREATE INDEX idx_p047_pf_tenant_config     ON ghg_benchmark.gl_bm_portfolios(tenant_id, config_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_pf_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_portfolios
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_holdings
-- =============================================================================
-- Individual holdings within a portfolio. Each holding represents an
-- investment position with asset class classification per PCAF standard
-- (listed equity, corporate bonds, project finance, commercial real estate,
-- mortgages, sovereign debt). Holdings carry investment value, EVIC
-- (Enterprise Value Including Cash), ownership share, sector and country
-- classification, emissions by scope, revenue for intensity calculation,
-- PCAF quality score, and data source reference.

CREATE TABLE ghg_benchmark.gl_bm_holdings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    portfolio_id                UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_portfolios(id) ON DELETE CASCADE,
    holding_name                VARCHAR(255)    NOT NULL,
    holding_identifier          VARCHAR(100),
    asset_class                 VARCHAR(30)     NOT NULL,
    investment_value            NUMERIC(20,2),
    evic                        NUMERIC(20,2),
    ownership_share             NUMERIC(8,6),
    sector_code                 VARCHAR(50),
    country_code                VARCHAR(3),
    scope1                      NUMERIC(20,6),
    scope2                      NUMERIC(20,6),
    scope3                      NUMERIC(20,6),
    revenue                     NUMERIC(20,2),
    pcaf_score                  INTEGER,
    data_source                 VARCHAR(100),
    is_excluded                 BOOLEAN         NOT NULL DEFAULT false,
    exclusion_reason            TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_hd_asset_class CHECK (
        asset_class IN (
            'LISTED_EQUITY', 'CORPORATE_BONDS', 'PROJECT_FINANCE',
            'COMMERCIAL_RE', 'MORTGAGES', 'SOVEREIGN_DEBT'
        )
    ),
    CONSTRAINT chk_p047_hd_investment CHECK (
        investment_value IS NULL OR investment_value >= 0
    ),
    CONSTRAINT chk_p047_hd_evic CHECK (
        evic IS NULL OR evic >= 0
    ),
    CONSTRAINT chk_p047_hd_ownership CHECK (
        ownership_share IS NULL OR (ownership_share >= 0 AND ownership_share <= 1)
    ),
    CONSTRAINT chk_p047_hd_country CHECK (
        country_code IS NULL OR LENGTH(country_code) BETWEEN 2 AND 3
    ),
    CONSTRAINT chk_p047_hd_scope1 CHECK (
        scope1 IS NULL OR scope1 >= 0
    ),
    CONSTRAINT chk_p047_hd_scope2 CHECK (
        scope2 IS NULL OR scope2 >= 0
    ),
    CONSTRAINT chk_p047_hd_scope3 CHECK (
        scope3 IS NULL OR scope3 >= 0
    ),
    CONSTRAINT chk_p047_hd_revenue CHECK (
        revenue IS NULL OR revenue >= 0
    ),
    CONSTRAINT chk_p047_hd_pcaf CHECK (
        pcaf_score IS NULL OR (pcaf_score >= 1 AND pcaf_score <= 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_hd_tenant            ON ghg_benchmark.gl_bm_holdings(tenant_id);
CREATE INDEX idx_p047_hd_portfolio         ON ghg_benchmark.gl_bm_holdings(portfolio_id);
CREATE INDEX idx_p047_hd_name              ON ghg_benchmark.gl_bm_holdings(holding_name);
CREATE INDEX idx_p047_hd_identifier        ON ghg_benchmark.gl_bm_holdings(holding_identifier);
CREATE INDEX idx_p047_hd_asset_class       ON ghg_benchmark.gl_bm_holdings(asset_class);
CREATE INDEX idx_p047_hd_sector            ON ghg_benchmark.gl_bm_holdings(sector_code);
CREATE INDEX idx_p047_hd_country           ON ghg_benchmark.gl_bm_holdings(country_code);
CREATE INDEX idx_p047_hd_pcaf              ON ghg_benchmark.gl_bm_holdings(pcaf_score);
CREATE INDEX idx_p047_hd_excluded          ON ghg_benchmark.gl_bm_holdings(is_excluded) WHERE is_excluded = true;
CREATE INDEX idx_p047_hd_created           ON ghg_benchmark.gl_bm_holdings(created_at DESC);
CREATE INDEX idx_p047_hd_metadata          ON ghg_benchmark.gl_bm_holdings USING GIN(metadata);

-- Composite: portfolio + asset class for class-level analysis
CREATE INDEX idx_p047_hd_port_class        ON ghg_benchmark.gl_bm_holdings(portfolio_id, asset_class);

-- Composite: portfolio + sector for sector attribution
CREATE INDEX idx_p047_hd_port_sector       ON ghg_benchmark.gl_bm_holdings(portfolio_id, sector_code);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p047_hd_updated
    BEFORE UPDATE ON ghg_benchmark.gl_bm_holdings
    FOR EACH ROW EXECUTE FUNCTION ghg_benchmark.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_benchmark.gl_bm_portfolio_results
-- =============================================================================
-- Portfolio carbon benchmark results per PCAF/TCFD methodology. Calculates
-- WACI (Weighted Average Carbon Intensity), carbon footprint, carbon
-- intensity, total financed emissions, tracking error (vs benchmark index),
-- sector attribution (JSONB decomposition of emissions by sector), and top
-- contributors (JSONB list of largest emitters by financed emissions).

CREATE TABLE ghg_benchmark.gl_bm_portfolio_results (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    portfolio_id                UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_portfolios(id) ON DELETE CASCADE,
    benchmark_index             VARCHAR(100),
    waci                        NUMERIC(20,10),
    carbon_footprint            NUMERIC(20,10),
    carbon_intensity            NUMERIC(20,10),
    total_financed_emissions    NUMERIC(20,6),
    tracking_error              NUMERIC(20,10),
    coverage_pct                NUMERIC(5,2),
    pcaf_quality                NUMERIC(3,1),
    sector_attribution          JSONB           DEFAULT '{}',
    top_contributors            JSONB           DEFAULT '[]',
    yoy_change_pct              NUMERIC(10,6),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'CALCULATED',
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_pr_waci CHECK (
        waci IS NULL OR waci >= 0
    ),
    CONSTRAINT chk_p047_pr_footprint CHECK (
        carbon_footprint IS NULL OR carbon_footprint >= 0
    ),
    CONSTRAINT chk_p047_pr_intensity CHECK (
        carbon_intensity IS NULL OR carbon_intensity >= 0
    ),
    CONSTRAINT chk_p047_pr_financed CHECK (
        total_financed_emissions IS NULL OR total_financed_emissions >= 0
    ),
    CONSTRAINT chk_p047_pr_coverage CHECK (
        coverage_pct IS NULL OR (coverage_pct >= 0 AND coverage_pct <= 100)
    ),
    CONSTRAINT chk_p047_pr_pcaf CHECK (
        pcaf_quality IS NULL OR (pcaf_quality >= 1 AND pcaf_quality <= 5)
    ),
    CONSTRAINT chk_p047_pr_status CHECK (
        status IN ('CALCULATED', 'VERIFIED', 'PUBLISHED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p047_pr_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_pr_tenant            ON ghg_benchmark.gl_bm_portfolio_results(tenant_id);
CREATE INDEX idx_p047_pr_portfolio         ON ghg_benchmark.gl_bm_portfolio_results(portfolio_id);
CREATE INDEX idx_p047_pr_benchmark         ON ghg_benchmark.gl_bm_portfolio_results(benchmark_index);
CREATE INDEX idx_p047_pr_status            ON ghg_benchmark.gl_bm_portfolio_results(status);
CREATE INDEX idx_p047_pr_calculated        ON ghg_benchmark.gl_bm_portfolio_results(calculated_at DESC);
CREATE INDEX idx_p047_pr_created           ON ghg_benchmark.gl_bm_portfolio_results(created_at DESC);
CREATE INDEX idx_p047_pr_provenance        ON ghg_benchmark.gl_bm_portfolio_results(provenance_hash);
CREATE INDEX idx_p047_pr_sector_attr       ON ghg_benchmark.gl_bm_portfolio_results USING GIN(sector_attribution);
CREATE INDEX idx_p047_pr_top_contrib       ON ghg_benchmark.gl_bm_portfolio_results USING GIN(top_contributors);

-- Composite: portfolio + status for lifecycle queries
CREATE INDEX idx_p047_pr_port_status       ON ghg_benchmark.gl_bm_portfolio_results(portfolio_id, status);

-- Composite: tenant + portfolio for tenant-scoped queries
CREATE INDEX idx_p047_pr_tenant_port       ON ghg_benchmark.gl_bm_portfolio_results(tenant_id, portfolio_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_holdings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_portfolio_results ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_pf_tenant_isolation
    ON ghg_benchmark.gl_bm_portfolios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pf_service_bypass
    ON ghg_benchmark.gl_bm_portfolios
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_hd_tenant_isolation
    ON ghg_benchmark.gl_bm_holdings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_hd_service_bypass
    ON ghg_benchmark.gl_bm_holdings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_pr_tenant_isolation
    ON ghg_benchmark.gl_bm_portfolio_results
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_pr_service_bypass
    ON ghg_benchmark.gl_bm_portfolio_results
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_portfolios TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_holdings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_portfolio_results TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_portfolios IS
    'Portfolio definitions for carbon benchmarking with AUM, PCAF coverage, and benchmark index association.';
COMMENT ON TABLE ghg_benchmark.gl_bm_holdings IS
    'Individual holdings per portfolio with PCAF asset class, emissions by scope, ownership share, and data quality.';
COMMENT ON TABLE ghg_benchmark.gl_bm_portfolio_results IS
    'Portfolio benchmark results: WACI, carbon footprint, intensity, financed emissions, sector attribution, and top contributors.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolios.portfolio_type IS 'Portfolio classification: INVESTMENT, LENDING, UNDERWRITING, REAL_ESTATE, PROJECT_FINANCE, SOVEREIGN, CUSTOM.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolios.pcaf_quality IS 'Aggregate PCAF data quality score (1-5 scale) for the portfolio, weighted by investment value.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolios.benchmark_index IS 'Reference benchmark index for tracking error (e.g., MSCI World, S&P 500, Paris-Aligned Benchmark).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_holdings.asset_class IS 'PCAF asset class: LISTED_EQUITY, CORPORATE_BONDS, PROJECT_FINANCE, COMMERCIAL_RE, MORTGAGES, SOVEREIGN_DEBT.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_holdings.evic IS 'Enterprise Value Including Cash for attribution calculation (PCAF standard).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_holdings.ownership_share IS 'Attribution factor: investment_value / evic (listed equity) or outstanding_amount / total_debt (bonds).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.waci IS 'Weighted Average Carbon Intensity: sum of (portfolio weight * company intensity) in tCO2e per MEUR revenue.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.carbon_footprint IS 'Carbon Footprint: sum of (ownership_share * company_emissions) / AUM in tCO2e per MEUR invested.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.total_financed_emissions IS 'Total financed emissions: sum of (ownership_share * company_emissions) in tCO2e.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.tracking_error IS 'Carbon intensity tracking error vs benchmark index. Positive = higher intensity than benchmark.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.sector_attribution IS 'JSON sector decomposition: {"energy": {"weight": 0.12, "waci": 450, "contribution": 54.0}, ...}.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_portfolio_results.top_contributors IS 'JSON array of top emitters: [{"name": "Company A", "financed_emissions": 12500, "pct_of_total": 15.3}].';
