-- =============================================================================
-- V134: PACK-023-sbti-alignment-006: Temperature Rating Results and Portfolio Scores
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for SBTi Temperature Rating v2.0 implementation.
-- Covers company-level temperature scoring (1.0-6.0C range), portfolio
-- aggregation methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS), contribution analysis,
-- and scenario comparison with Paris Agreement and carbon budget alignment.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (temperature baseline)
--   PACK-022: Net Zero Acceleration Pack (portfolio scenarios)
--   V129: PACK-023 Target Definitions
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the temperature rating assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_temperature_company_scores   - Company-level temperature scores
--   2. pack023_temperature_portfolio_scores - Portfolio-level aggregated scores
--   3. pack023_temperature_portfolio_holdings - Portfolio entity contribution
--   4. pack023_temperature_scenario_comparison - Scenario comparison analysis
--
-- Hypertables (2):
--   pack023_temperature_company_scores   on score_date (chunk: 3 months)
--   pack023_temperature_portfolio_scores on score_date (chunk: 3 months)
--
-- Also includes: 45+ indexes, update triggers, security grants, and comments.
-- Previous: V133__pack023_flag_assessment_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_temperature_company_scores
-- =============================================================================
-- Company-level temperature scores for single organizations based on
-- near-term, long-term, and net-zero targets with methodology details.

CREATE TABLE pack023_sbti_alignment.pack023_temperature_company_scores (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    score_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    scope                   VARCHAR(50),
    assessment_year         INTEGER,
    temperature_score       DECIMAL(4,2),
    temperature_unit        VARCHAR(10)     DEFAULT 'C',
    temperature_category    VARCHAR(30),
    temperature_alignment   VARCHAR(100),
    annual_reduction_rate   DECIMAL(6,4),
    implied_warming_rate    DECIMAL(6,4),
    target_ambition         VARCHAR(30),
    near_term_contribution  DECIMAL(6,4),
    long_term_contribution  DECIMAL(6,4),
    net_zero_contribution   DECIMAL(6,4),
    scope1_weight           DECIMAL(6,2),
    scope2_weight           DECIMAL(6,2),
    scope3_weight           DECIMAL(6,2),
    methodology_version     VARCHAR(50),
    calculation_date        TIMESTAMPTZ,
    calculation_method      VARCHAR(200),
    data_quality_assessment VARCHAR(30),
    assumptions_applied     JSONB           DEFAULT '{}',
    sensitivities           JSONB           DEFAULT '{}',
    confidence_range_low    DECIMAL(4,2),
    confidence_range_high   DECIMAL(4,2),
    comparability_factors   TEXT[],
    validation_status       VARCHAR(30),
    validated_by            VARCHAR(255),
    validated_at            TIMESTAMPTZ,
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_temp_score CHECK (
        temperature_score IS NULL OR (temperature_score >= 1.0 AND temperature_score <= 6.0)
    ),
    CONSTRAINT chk_pk_temp_category CHECK (
        temperature_category IN ('EXCEEDS_TARGETS', 'ALIGNED_1.5C', 'ALIGNED_2C',
                               'UNALIGNED', 'INSUFFICIENT_DATA')
    ),
    CONSTRAINT chk_pk_temp_arr CHECK (
        annual_reduction_rate IS NULL OR (annual_reduction_rate >= -10 AND annual_reduction_rate <= 10)
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_temperature_company_scores',
    'score_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_temp_tenant ON pack023_sbti_alignment.pack023_temperature_company_scores(tenant_id);
CREATE INDEX idx_pk_temp_org ON pack023_sbti_alignment.pack023_temperature_company_scores(org_id);
CREATE INDEX idx_pk_temp_date ON pack023_sbti_alignment.pack023_temperature_company_scores(score_date DESC);
CREATE INDEX idx_pk_temp_year ON pack023_sbti_alignment.pack023_temperature_company_scores(assessment_year);
CREATE INDEX idx_pk_temp_score ON pack023_sbti_alignment.pack023_temperature_company_scores(temperature_score);
CREATE INDEX idx_pk_temp_category ON pack023_sbti_alignment.pack023_temperature_company_scores(temperature_category);
CREATE INDEX idx_pk_temp_alignment ON pack023_sbti_alignment.pack023_temperature_company_scores(temperature_alignment);
CREATE INDEX idx_pk_temp_validation ON pack023_sbti_alignment.pack023_temperature_company_scores(validation_status);
CREATE INDEX idx_pk_temp_org_date ON pack023_sbti_alignment.pack023_temperature_company_scores(org_id, score_date DESC);
CREATE INDEX idx_pk_temp_assumptions ON pack023_sbti_alignment.pack023_temperature_company_scores USING GIN(assumptions_applied);
CREATE INDEX idx_pk_temp_sensitivities ON pack023_sbti_alignment.pack023_temperature_company_scores USING GIN(sensitivities);

-- Updated_at trigger
CREATE TRIGGER trg_pk_temp_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_temperature_company_scores
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_temperature_portfolio_scores
-- =============================================================================
-- Portfolio-level aggregated temperature scores using 6 different aggregation
-- methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS) with weighting and coverage metrics.

CREATE TABLE pack023_sbti_alignment.pack023_temperature_portfolio_scores (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    portfolio_id            UUID            NOT NULL,
    portfolio_name          VARCHAR(500),
    score_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    portfolio_value_millions DECIMAL(18,2),
    portfolio_emissions_mt  DECIMAL(18,6),
    holding_count           INTEGER,
    holdings_with_targets   INTEGER,
    target_coverage_pct     DECIMAL(6,2),
    wats_score              DECIMAL(4,2),
    wats_weight_metric      VARCHAR(50),
    tets_score              DECIMAL(4,2),
    tets_weight_metric      VARCHAR(50),
    mots_score              DECIMAL(4,2),
    mots_weight_metric      VARCHAR(50),
    eots_score              DECIMAL(4,2),
    eots_weight_metric      VARCHAR(50),
    ecots_score             DECIMAL(4,2),
    ecots_weight_metric     VARCHAR(50),
    aots_score              DECIMAL(4,2),
    aots_weight_metric      VARCHAR(50),
    primary_methodology     VARCHAR(50),
    primary_score           DECIMAL(4,2),
    methodology_version     VARCHAR(50),
    carbon_budget_remaining DECIMAL(18,6),
    carbon_budget_unit      VARCHAR(20),
    paris_alignment         VARCHAR(100),
    extreme_values_present  BOOLEAN         DEFAULT FALSE,
    extreme_value_handling  VARCHAR(100),
    data_quality_score      DECIMAL(5,2),
    missing_target_count    INTEGER,
    proxy_target_count      INTEGER,
    validation_status       VARCHAR(30),
    validated_by            VARCHAR(255),
    validated_at            TIMESTAMPTZ,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_port_scores CHECK (
        (wats_score IS NULL OR (wats_score >= 1.0 AND wats_score <= 6.0)) AND
        (tets_score IS NULL OR (tets_score >= 1.0 AND tets_score <= 6.0)) AND
        (primary_score IS NULL OR (primary_score >= 1.0 AND primary_score <= 6.0))
    ),
    CONSTRAINT chk_pk_port_coverage CHECK (
        target_coverage_pct >= 0 AND target_coverage_pct <= 100
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_temperature_portfolio_scores',
    'score_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_port_tenant ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(tenant_id);
CREATE INDEX idx_pk_port_portfolio ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(portfolio_id);
CREATE INDEX idx_pk_port_date ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(score_date DESC);
CREATE INDEX idx_pk_port_primary_score ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(primary_score);
CREATE INDEX idx_pk_port_wats ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(wats_score);
CREATE INDEX idx_pk_port_tets ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(tets_score);
CREATE INDEX idx_pk_port_methodology ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(primary_methodology);
CREATE INDEX idx_pk_port_alignment ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(paris_alignment);
CREATE INDEX idx_pk_port_validation ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(validation_status);
CREATE INDEX idx_pk_port_coverage ON pack023_sbti_alignment.pack023_temperature_portfolio_scores(target_coverage_pct);
CREATE INDEX idx_pk_port_metadata ON pack023_sbti_alignment.pack023_temperature_portfolio_scores USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_port_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_temperature_portfolio_scores
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_temperature_portfolio_holdings
-- =============================================================================
-- Individual portfolio holding contribution to portfolio temperature score
-- with entity-level temperature score and weighting in each aggregation method.

CREATE TABLE pack023_sbti_alignment.pack023_temperature_portfolio_holdings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_score_id      UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_temperature_portfolio_scores(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    portfolio_id            UUID            NOT NULL,
    holding_entity_id       UUID            NOT NULL,
    holding_name            VARCHAR(500),
    isin_code               VARCHAR(50),
    sedol_code              VARCHAR(50),
    ticker                  VARCHAR(20),
    company_temperature     DECIMAL(4,2),
    company_score_date      TIMESTAMPTZ,
    has_validated_targets   BOOLEAN         DEFAULT FALSE,
    target_year             INTEGER,
    market_value_millions   DECIMAL(18,2),
    market_cap_millions     DECIMAL(18,2),
    enterprise_value_millions DECIMAL(18,2),
    total_invested_capital  DECIMAL(18,2),
    revenue_millions        DECIMAL(18,2),
    emissions_mt_co2e       DECIMAL(18,6),
    wats_weight_pct         DECIMAL(6,2),
    tets_weight_pct         DECIMAL(6,2),
    mots_weight_pct         DECIMAL(6,2),
    eots_weight_pct         DECIMAL(6,2),
    ecots_weight_pct        DECIMAL(6,2),
    aots_weight_pct         DECIMAL(6,2),
    weighted_contribution   DECIMAL(6,4),
    sector_classification   VARCHAR(100),
    region                  VARCHAR(100),
    data_quality_tier       VARCHAR(30),
    temperature_source      VARCHAR(100),
    last_updated            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_hold_temp CHECK (
        company_temperature IS NULL OR (company_temperature >= 1.0 AND company_temperature <= 6.0)
    ),
    CONSTRAINT chk_pk_hold_weights CHECK (
        (wats_weight_pct IS NULL OR (wats_weight_pct >= 0 AND wats_weight_pct <= 100)) AND
        (tets_weight_pct IS NULL OR (tets_weight_pct >= 0 AND tets_weight_pct <= 100))
    )
);

-- Indexes
CREATE INDEX idx_pk_hold_portfolio_id ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(portfolio_score_id);
CREATE INDEX idx_pk_hold_tenant ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(tenant_id);
CREATE INDEX idx_pk_hold_portfolio ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(portfolio_id);
CREATE INDEX idx_pk_hold_entity ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(holding_entity_id);
CREATE INDEX idx_pk_hold_isin ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(isin_code);
CREATE INDEX idx_pk_hold_ticker ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(ticker);
CREATE INDEX idx_pk_hold_temp ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(company_temperature);
CREATE INDEX idx_pk_hold_sector ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(sector_classification);
CREATE INDEX idx_pk_hold_region ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(region);
CREATE INDEX idx_pk_hold_data_quality ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(data_quality_tier);
CREATE INDEX idx_pk_hold_created_at ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_pk_hold_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_temperature_portfolio_holdings
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_temperature_scenario_comparison
-- =============================================================================
-- Scenario comparison analysis showing temperature trajectory under different
-- warming scenarios and policy/climate pathways (RCP 2.6, 4.5, 6.0, 8.5, etc.).

CREATE TABLE pack023_sbti_alignment.pack023_temperature_scenario_comparison (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_score_id        UUID            REFERENCES pack023_sbti_alignment.pack023_temperature_company_scores(id) ON DELETE CASCADE,
    portfolio_score_id      UUID            REFERENCES pack023_sbti_alignment.pack023_temperature_portfolio_scores(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    comparison_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    scenario_name           VARCHAR(200)    NOT NULL,
    scenario_code           VARCHAR(50),
    scenario_type           VARCHAR(100),
    implied_warming_2030    DECIMAL(4,2),
    implied_warming_2050    DECIMAL(4,2),
    implied_warming_2100    DECIMAL(4,2),
    current_trajectory      DECIMAL(4,2),
    policy_pathway          VARCHAR(100),
    carbon_budget_2030      DECIMAL(18,6),
    carbon_budget_2050      DECIMAL(18,6),
    carbon_budget_unit      VARCHAR(20),
    cumulative_emissions_2030 DECIMAL(18,6),
    cumulative_emissions_2050 DECIMAL(18,6),
    net_zero_year           INTEGER,
    nzc_achievement_probability DECIMAL(6,2),
    likelihood              VARCHAR(30),
    policy_assumptions      TEXT[],
    technology_assumptions  JSONB           DEFAULT '{}',
    mitigation_potential    DECIMAL(6,2),
    adaptation_required     DECIMAL(6,2),
    risk_assessment         TEXT,
    opportunity_assessment  TEXT,
    financial_impact_low    DECIMAL(18,2),
    financial_impact_mid    DECIMAL(18,2),
    financial_impact_high   DECIMAL(18,2),
    data_source             VARCHAR(255),
    publication_year        INTEGER,
    confidence_level        VARCHAR(30),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_scen_warming CHECK (
        (implied_warming_2030 IS NULL OR implied_warming_2030 >= 1.0) AND
        (implied_warming_2050 IS NULL OR implied_warming_2050 >= 1.0)
    ),
    CONSTRAINT chk_pk_scen_type CHECK (
        scenario_type IN ('PARIS_ALIGNMENT', 'IEA_PATHWAY', 'RCP_SCENARIO', 'IPCC_AR6', 'CUSTOM')
    )
);

-- Indexes
CREATE INDEX idx_pk_scen_company_id ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(company_score_id);
CREATE INDEX idx_pk_scen_portfolio_id ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(portfolio_score_id);
CREATE INDEX idx_pk_scen_tenant ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(tenant_id);
CREATE INDEX idx_pk_scen_date ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(comparison_date DESC);
CREATE INDEX idx_pk_scen_name ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(scenario_name);
CREATE INDEX idx_pk_scen_type ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(scenario_type);
CREATE INDEX idx_pk_scen_policy ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(policy_pathway);
CREATE INDEX idx_pk_scen_warming_2100 ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(implied_warming_2100);
CREATE INDEX idx_pk_scen_nzc_year ON pack023_sbti_alignment.pack023_temperature_scenario_comparison(net_zero_year);
CREATE INDEX idx_pk_scen_assumptions ON pack023_sbti_alignment.pack023_temperature_scenario_comparison USING GIN(policy_assumptions);
CREATE INDEX idx_pk_scen_tech ON pack023_sbti_alignment.pack023_temperature_scenario_comparison USING GIN(technology_assumptions);

-- Updated_at trigger
CREATE TRIGGER trg_pk_scen_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_temperature_scenario_comparison
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_temperature_company_scores IS
'Company-level SBTi Temperature Rating v2.0 scores (1.0-6.0C range) with methodology details and validation status.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_temperature_portfolio_scores IS
'Portfolio-level aggregated temperature scores using 6 methods (WATS/TETS/MOTS/EOTS/ECOTS/AOTS) with Paris alignment assessment.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_temperature_portfolio_holdings IS
'Individual portfolio holding contributions to portfolio temperature score with entity-level temperatures and weighting per aggregation method.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_temperature_scenario_comparison IS
'Scenario comparison analysis showing temperature trajectories under different warming scenarios (RCP, IEA, Paris pathways) with financial impact.';
