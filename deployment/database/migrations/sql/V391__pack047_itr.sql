-- =============================================================================
-- V391: PACK-047 GHG Emissions Benchmark Pack - Implied Temperature Rating
-- =============================================================================
-- Pack:         PACK-047 (GHG Emissions Benchmark Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for Implied Temperature Rating (ITR) calculations and
-- portfolio-level temperature scoring. ITR translates an entity's
-- emissions trajectory into an implied warming outcome in degrees Celsius.
-- Three methods are supported: budget-based (cumulative emissions vs carbon
-- budget), sector-relative (intensity vs sector pathway), and rate-of-
-- reduction (CARR extrapolation). Portfolio ITR aggregates holdings-level
-- ITR into a portfolio temperature score with coverage and quality metrics.
--
-- Tables (2):
--   1. ghg_benchmark.gl_bm_itr_calculations
--   2. ghg_benchmark.gl_bm_itr_portfolio
--
-- Also includes: indexes, RLS, comments.
-- Previous: V390__pack047_pathway_alignment.sql
-- =============================================================================

SET search_path TO ghg_benchmark, public;

-- =============================================================================
-- Table 1: ghg_benchmark.gl_bm_itr_calculations
-- =============================================================================
-- Entity-level Implied Temperature Rating calculations. Each record holds
-- the ITR for a specific entity using one of three methods: BUDGET_BASED
-- (cumulative emissions vs 1.5/2.0C carbon budget), SECTOR_RELATIVE
-- (sector pathway comparison), or RATE_OF_REDUCTION (CARR extrapolation).
-- Includes confidence interval, cumulative emissions, allocated budget,
-- overshoot probability, and data quality scoring.

CREATE TABLE ghg_benchmark.gl_bm_itr_calculations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_benchmark.gl_bm_configurations(id) ON DELETE CASCADE,
    entity_name                 VARCHAR(255)    NOT NULL,
    entity_identifier           VARCHAR(100),
    method                      VARCHAR(30)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL DEFAULT 'SCOPE_1_2_LOCATION',
    implied_temperature         NUMERIC(4,2)    NOT NULL,
    confidence_lower            NUMERIC(4,2),
    confidence_upper            NUMERIC(4,2),
    cumulative_emissions        NUMERIC(20,6),
    allocated_budget            NUMERIC(20,6),
    overshoot_probability       NUMERIC(5,4),
    data_quality_score          NUMERIC(3,1),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_itr_method CHECK (
        method IN ('BUDGET_BASED', 'SECTOR_RELATIVE', 'RATE_OF_REDUCTION')
    ),
    CONSTRAINT chk_p047_itr_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p047_itr_temp CHECK (
        implied_temperature >= 0.5 AND implied_temperature <= 10.0
    ),
    CONSTRAINT chk_p047_itr_conf_lower CHECK (
        confidence_lower IS NULL OR (confidence_lower >= 0.5 AND confidence_lower <= implied_temperature)
    ),
    CONSTRAINT chk_p047_itr_conf_upper CHECK (
        confidence_upper IS NULL OR (confidence_upper >= implied_temperature AND confidence_upper <= 10.0)
    ),
    CONSTRAINT chk_p047_itr_cumulative CHECK (
        cumulative_emissions IS NULL OR cumulative_emissions >= 0
    ),
    CONSTRAINT chk_p047_itr_budget CHECK (
        allocated_budget IS NULL OR allocated_budget >= 0
    ),
    CONSTRAINT chk_p047_itr_overshoot CHECK (
        overshoot_probability IS NULL OR (overshoot_probability >= 0 AND overshoot_probability <= 1)
    ),
    CONSTRAINT chk_p047_itr_dq CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 1 AND data_quality_score <= 5)
    ),
    CONSTRAINT chk_p047_itr_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_itr_tenant           ON ghg_benchmark.gl_bm_itr_calculations(tenant_id);
CREATE INDEX idx_p047_itr_config           ON ghg_benchmark.gl_bm_itr_calculations(config_id);
CREATE INDEX idx_p047_itr_entity           ON ghg_benchmark.gl_bm_itr_calculations(entity_name);
CREATE INDEX idx_p047_itr_entity_id        ON ghg_benchmark.gl_bm_itr_calculations(entity_identifier);
CREATE INDEX idx_p047_itr_method           ON ghg_benchmark.gl_bm_itr_calculations(method);
CREATE INDEX idx_p047_itr_scope            ON ghg_benchmark.gl_bm_itr_calculations(scope_inclusion);
CREATE INDEX idx_p047_itr_temp             ON ghg_benchmark.gl_bm_itr_calculations(implied_temperature);
CREATE INDEX idx_p047_itr_dq               ON ghg_benchmark.gl_bm_itr_calculations(data_quality_score);
CREATE INDEX idx_p047_itr_calculated       ON ghg_benchmark.gl_bm_itr_calculations(calculated_at DESC);
CREATE INDEX idx_p047_itr_created          ON ghg_benchmark.gl_bm_itr_calculations(created_at DESC);
CREATE INDEX idx_p047_itr_provenance       ON ghg_benchmark.gl_bm_itr_calculations(provenance_hash);

-- Composite: config + method for method-filtered queries
CREATE INDEX idx_p047_itr_config_method    ON ghg_benchmark.gl_bm_itr_calculations(config_id, method);

-- Composite: tenant + entity for entity-level history
CREATE INDEX idx_p047_itr_tenant_entity    ON ghg_benchmark.gl_bm_itr_calculations(tenant_id, entity_name);

-- =============================================================================
-- Table 2: ghg_benchmark.gl_bm_itr_portfolio
-- =============================================================================
-- Portfolio-level Implied Temperature Rating aggregation. Aggregates
-- entity-level ITR into a portfolio temperature score using AUM/investment-
-- weighted or equal-weighted methods. Tracks the number of holdings, coverage
-- percentage, weighted data quality, and methodology used. Portfolio ITR
-- may be calculated without an explicit portfolio reference (standalone).

CREATE TABLE ghg_benchmark.gl_bm_itr_portfolio (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    portfolio_id                UUID,
    itr_calculation_id          UUID,
    portfolio_name              VARCHAR(255)    NOT NULL,
    portfolio_itr               NUMERIC(4,2)    NOT NULL,
    holding_count               INTEGER         NOT NULL DEFAULT 0,
    coverage_pct                NUMERIC(5,2)    NOT NULL DEFAULT 0,
    weighted_quality            NUMERIC(3,1),
    method                      VARCHAR(30)     NOT NULL DEFAULT 'AUM_WEIGHTED',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'CALCULATED',
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p047_itrp_itr CHECK (
        portfolio_itr >= 0.5 AND portfolio_itr <= 10.0
    ),
    CONSTRAINT chk_p047_itrp_holdings CHECK (
        holding_count >= 0
    ),
    CONSTRAINT chk_p047_itrp_coverage CHECK (
        coverage_pct >= 0 AND coverage_pct <= 100
    ),
    CONSTRAINT chk_p047_itrp_quality CHECK (
        weighted_quality IS NULL OR (weighted_quality >= 1 AND weighted_quality <= 5)
    ),
    CONSTRAINT chk_p047_itrp_method CHECK (
        method IN (
            'AUM_WEIGHTED', 'EQUAL_WEIGHTED', 'OWNERSHIP_WEIGHTED',
            'EVIC_WEIGHTED', 'REVENUE_WEIGHTED'
        )
    ),
    CONSTRAINT chk_p047_itrp_status CHECK (
        status IN ('CALCULATED', 'VERIFIED', 'PUBLISHED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p047_itrp_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p047_itrp_tenant          ON ghg_benchmark.gl_bm_itr_portfolio(tenant_id);
CREATE INDEX idx_p047_itrp_portfolio       ON ghg_benchmark.gl_bm_itr_portfolio(portfolio_id);
CREATE INDEX idx_p047_itrp_itr_calc        ON ghg_benchmark.gl_bm_itr_portfolio(itr_calculation_id);
CREATE INDEX idx_p047_itrp_name            ON ghg_benchmark.gl_bm_itr_portfolio(portfolio_name);
CREATE INDEX idx_p047_itrp_itr             ON ghg_benchmark.gl_bm_itr_portfolio(portfolio_itr);
CREATE INDEX idx_p047_itrp_method          ON ghg_benchmark.gl_bm_itr_portfolio(method);
CREATE INDEX idx_p047_itrp_status          ON ghg_benchmark.gl_bm_itr_portfolio(status);
CREATE INDEX idx_p047_itrp_calculated      ON ghg_benchmark.gl_bm_itr_portfolio(calculated_at DESC);
CREATE INDEX idx_p047_itrp_created         ON ghg_benchmark.gl_bm_itr_portfolio(created_at DESC);
CREATE INDEX idx_p047_itrp_provenance      ON ghg_benchmark.gl_bm_itr_portfolio(provenance_hash);

-- Composite: tenant + portfolio for portfolio-level queries
CREATE INDEX idx_p047_itrp_tenant_port     ON ghg_benchmark.gl_bm_itr_portfolio(tenant_id, portfolio_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_benchmark.gl_bm_itr_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_benchmark.gl_bm_itr_portfolio ENABLE ROW LEVEL SECURITY;

CREATE POLICY p047_itr_tenant_isolation
    ON ghg_benchmark.gl_bm_itr_calculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_itr_service_bypass
    ON ghg_benchmark.gl_bm_itr_calculations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p047_itrp_tenant_isolation
    ON ghg_benchmark.gl_bm_itr_portfolio
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p047_itrp_service_bypass
    ON ghg_benchmark.gl_bm_itr_portfolio
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_itr_calculations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_benchmark.gl_bm_itr_portfolio TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_benchmark.gl_bm_itr_calculations IS
    'Entity-level Implied Temperature Rating calculations using budget-based, sector-relative, or rate-of-reduction methods with confidence intervals.';
COMMENT ON TABLE ghg_benchmark.gl_bm_itr_portfolio IS
    'Portfolio-level ITR aggregation using AUM, equal, ownership, EVIC, or revenue weighting with coverage and quality tracking.';

COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.method IS 'ITR calculation method: BUDGET_BASED (cumulative vs budget), SECTOR_RELATIVE (pathway comparison), RATE_OF_REDUCTION (CARR extrapolation).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.implied_temperature IS 'Implied warming outcome in degrees Celsius (0.5 to 10.0). Below 1.5 = Paris-aligned, 1.5-2.0 = well-below 2C, >3.0 = high risk.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.confidence_lower IS 'Lower bound of ITR confidence interval at 90% confidence level.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.confidence_upper IS 'Upper bound of ITR confidence interval at 90% confidence level.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.cumulative_emissions IS 'Projected cumulative emissions from base year to 2050 in tCO2e (budget-based method).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.allocated_budget IS 'Carbon budget allocated to entity based on fair-share principle in tCO2e.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_calculations.overshoot_probability IS 'Probability (0-1) that cumulative emissions will exceed the allocated carbon budget.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_portfolio.portfolio_itr IS 'Weighted Implied Temperature Rating for the portfolio in degrees Celsius.';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_portfolio.coverage_pct IS 'Percentage of portfolio AUM/holdings covered by ITR calculations (0-100).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_portfolio.weighted_quality IS 'AUM-weighted average data quality score for the portfolio (1-5 scale).';
COMMENT ON COLUMN ghg_benchmark.gl_bm_itr_portfolio.method IS 'Weighting method: AUM_WEIGHTED, EQUAL_WEIGHTED, OWNERSHIP_WEIGHTED, EVIC_WEIGHTED, REVENUE_WEIGHTED.';
