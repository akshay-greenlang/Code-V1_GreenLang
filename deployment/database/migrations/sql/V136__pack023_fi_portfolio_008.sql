-- =============================================================================
-- V136: PACK-023-sbti-alignment-008: FI Portfolio Targets and PCAF Scores
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for Financial Institution (FI) portfolio targets per
-- SBTi FINZ V1.0 standard. Covers 8 asset classes with portfolio-level target
-- definitions, PCAF financed emissions data quality scoring (1-5 scale),
-- coverage calculations, and engagement target tracking.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (FI baseline)
--   V134: Temperature Rating (portfolio methodology)
--   V129: PACK-023 Target Definitions
--
-- 8 Asset Classes: listed_equity, corporate_bonds, business_loans,
--   mortgages, commercial_real_estate, project_finance,
--   sovereign_bonds, securitized
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the FI portfolio targets assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_fi_portfolio_targets         - FI portfolio targets by asset class
--   2. pack023_fi_pcaf_data_quality         - PCAF data quality scores
--   3. pack023_fi_asset_class_coverage      - Asset class coverage analysis
--   4. pack023_fi_engagement_targets        - Investee engagement goals
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V135__pack023_progress_tracking_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_fi_portfolio_targets
-- =============================================================================
-- Financial institution portfolio targets by asset class per FINZ V1.0
-- with target year, reduction rate, and coverage percentage.

CREATE TABLE pack023_sbti_alignment.pack023_fi_portfolio_targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    portfolio_id            UUID            NOT NULL,
    portfolio_name          VARCHAR(500),
    target_type             VARCHAR(50),
    asset_class_code        VARCHAR(50)     NOT NULL,
    asset_class_name        VARCHAR(200)    NOT NULL,
    target_year             INTEGER         NOT NULL,
    baseline_year           INTEGER         NOT NULL,
    portfolio_value_millions DECIMAL(18,2),
    portfolio_emissions_mt  DECIMAL(18,6),
    target_emissions_mt     DECIMAL(18,6),
    reduction_percentage    DECIMAL(8,4),
    annual_reduction_rate   DECIMAL(6,4),
    coverage_percentage     DECIMAL(6,2),
    financed_emissions_baseline DECIMAL(18,6),
    financed_emissions_target DECIMAL(18,6),
    financed_emissions_coverage DECIMAL(6,2),
    intensity_metric        VARCHAR(100),
    intensity_unit          VARCHAR(100),
    baseline_intensity      DECIMAL(16,8),
    target_intensity        DECIMAL(16,8),
    portfolio_temperature   DECIMAL(4,2),
    data_quality_score      DECIMAL(5,2),
    engagement_target_count INTEGER,
    engagement_coverage_pct DECIMAL(6,2),
    sector_weighting        JSONB           DEFAULT '{}',
    geographic_weighting    JSONB           DEFAULT '{}',
    exclusions_applied      TEXT[],
    exclusion_methodology   VARCHAR(200),
    scope_coverage          VARCHAR(100),
    financed_vs_attributed  VARCHAR(50),
    attribution_method      VARCHAR(100),
    market_based_approach   BOOLEAN         DEFAULT TRUE,
    location_based_approach BOOLEAN         DEFAULT FALSE,
    third_party_data        BOOLEAN         DEFAULT FALSE,
    proxy_data_percentage   DECIMAL(6,2),
    validation_status       VARCHAR(30),
    validated_by            VARCHAR(255),
    validated_date          DATE,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_fi_asset_class CHECK (
        asset_class_code IN ('LISTED_EQUITY', 'CORPORATE_BONDS', 'BUSINESS_LOANS',
                            'MORTGAGES', 'COMMERCIAL_RE', 'PROJECT_FINANCE',
                            'SOVEREIGN_BONDS', 'SECURITIZED')
    ),
    CONSTRAINT chk_pk_fi_coverage CHECK (
        coverage_percentage >= 0 AND coverage_percentage <= 100
    ),
    CONSTRAINT chk_pk_fi_reduction CHECK (
        reduction_percentage >= 0 AND reduction_percentage <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_fi_org ON pack023_sbti_alignment.pack023_fi_portfolio_targets(org_id);
CREATE INDEX idx_pk_fi_tenant ON pack023_sbti_alignment.pack023_fi_portfolio_targets(tenant_id);
CREATE INDEX idx_pk_fi_portfolio ON pack023_sbti_alignment.pack023_fi_portfolio_targets(portfolio_id);
CREATE INDEX idx_pk_fi_asset_class ON pack023_sbti_alignment.pack023_fi_portfolio_targets(asset_class_code);
CREATE INDEX idx_pk_fi_target_year ON pack023_sbti_alignment.pack023_fi_portfolio_targets(target_year);
CREATE INDEX idx_pk_fi_type ON pack023_sbti_alignment.pack023_fi_portfolio_targets(target_type);
CREATE INDEX idx_pk_fi_temp ON pack023_sbti_alignment.pack023_fi_portfolio_targets(portfolio_temperature);
CREATE INDEX idx_pk_fi_data_quality ON pack023_sbti_alignment.pack023_fi_portfolio_targets(data_quality_score);
CREATE INDEX idx_pk_fi_validation ON pack023_sbti_alignment.pack023_fi_portfolio_targets(validation_status);
CREATE INDEX idx_pk_fi_created_at ON pack023_sbti_alignment.pack023_fi_portfolio_targets(created_at DESC);
CREATE INDEX idx_pk_fi_sector_weighting ON pack023_sbti_alignment.pack023_fi_portfolio_targets USING GIN(sector_weighting);
CREATE INDEX idx_pk_fi_geographic ON pack023_sbti_alignment.pack023_fi_portfolio_targets USING GIN(geographic_weighting);

-- Updated_at trigger
CREATE TRIGGER trg_pk_fi_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_fi_portfolio_targets
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_fi_pcaf_data_quality
-- =============================================================================
-- PCAF data quality assessment for each asset class showing data source,
-- quality tier, and scoring (1-5 scale per PCAF Global Standard v2.0).

CREATE TABLE pack023_sbti_alignment.pack023_fi_pcaf_data_quality (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_target_id     UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_fi_portfolio_targets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    asset_class_code        VARCHAR(50)     NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    assessment_period_start DATE,
    assessment_period_end   DATE,
    holdings_assessed       INTEGER,
    holdings_with_company_data INTEGER,
    holdings_with_proxy_data INTEGER,
    holdings_excluded       INTEGER,
    company_data_percentage DECIMAL(6,2),
    proxy_data_percentage   DECIMAL(6,2),
    excluded_percentage     DECIMAL(6,2),
    emissions_data_score    DECIMAL(3,1),
    activity_data_score     DECIMAL(3,1),
    ef_score                DECIMAL(3,1),
    waci_score              DECIMAL(3,1),
    completeness_score      DECIMAL(3,1),
    temporal_score          DECIMAL(3,1),
    geographical_score      DECIMAL(3,1),
    technological_score     DECIMAL(3,1),
    overall_score           DECIMAL(3,1),
    data_quality_tier       VARCHAR(30),
    data_quality_category   VARCHAR(50),
    primary_data_sources    VARCHAR(200)[],
    proxy_data_sources      VARCHAR(200)[],
    calculation_method      VARCHAR(200),
    equity_weighting        BOOLEAN         DEFAULT FALSE,
    market_cap_weighting    BOOLEAN         DEFAULT FALSE,
    enterprise_value_weighting BOOLEAN     DEFAULT FALSE,
    data_challenges         TEXT[],
    improvement_actions     TEXT[],
    improvement_timeline    VARCHAR(200),
    assessor                VARCHAR(255),
    assessment_notes        TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_pcaf_score CHECK (
        overall_score >= 1 AND overall_score <= 5
    ),
    CONSTRAINT chk_pk_pcaf_percentages CHECK (
        (company_data_percentage + proxy_data_percentage + excluded_percentage) = 100
    ),
    CONSTRAINT chk_pk_pcaf_tier CHECK (
        data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3', 'TIER_4', 'TIER_5')
    )
);

-- Indexes
CREATE INDEX idx_pk_pcaf_target_id ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(portfolio_target_id);
CREATE INDEX idx_pk_pcaf_tenant ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(tenant_id);
CREATE INDEX idx_pk_pcaf_org ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(org_id);
CREATE INDEX idx_pk_pcaf_asset_class ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(asset_class_code);
CREATE INDEX idx_pk_pcaf_date ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(assessment_date DESC);
CREATE INDEX idx_pk_pcaf_overall ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(overall_score);
CREATE INDEX idx_pk_pcaf_tier ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality(data_quality_tier);
CREATE INDEX idx_pk_pcaf_sources ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality USING GIN(primary_data_sources);
CREATE INDEX idx_pk_pcaf_challenges ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality USING GIN(data_challenges);

-- Updated_at trigger
CREATE TRIGGER trg_pk_pcaf_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_fi_pcaf_data_quality
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_fi_asset_class_coverage
-- =============================================================================
-- Asset class coverage analysis showing % of portfolio value, emissions,
-- and number of investees with targets by asset class.

CREATE TABLE pack023_sbti_alignment.pack023_fi_asset_class_coverage (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_target_id     UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_fi_portfolio_targets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    asset_class_code        VARCHAR(50)     NOT NULL,
    asset_class_name        VARCHAR(200),
    analysis_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_holding_count     INTEGER,
    holdings_with_targets   INTEGER,
    target_coverage_count_pct DECIMAL(6,2),
    total_value_millions    DECIMAL(18,2),
    value_with_targets_millions DECIMAL(18,2),
    target_coverage_value_pct DECIMAL(6,2),
    total_emissions_mt      DECIMAL(18,6),
    emissions_from_targeted DECIMAL(18,6),
    target_coverage_emissions_pct DECIMAL(6,2),
    weighted_avg_target_ambition VARCHAR(50),
    number_sbti_validated   INTEGER,
    number_sbti_committed   INTEGER,
    number_interim_targets  INTEGER,
    number_with_nz_goals    INTEGER,
    companies_exceeding_1_5c DECIMAL(6,2),
    companies_aligned_1_5c  DECIMAL(6,2),
    companies_unaligned     DECIMAL(6,2),
    most_common_sectors     TEXT[],
    most_common_regions     VARCHAR(100)[],
    emerging_target_trends  TEXT,
    coverage_gaps           TEXT[],
    required_coverage_pct   DECIMAL(6,2),
    coverage_action_plan    TEXT,
    assessed_by             VARCHAR(255),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_asset_coverage CHECK (
        target_coverage_count_pct >= 0 AND target_coverage_count_pct <= 100
    ),
    CONSTRAINT chk_pk_asset_value_coverage CHECK (
        target_coverage_value_pct >= 0 AND target_coverage_value_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_asset_target_id ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(portfolio_target_id);
CREATE INDEX idx_pk_asset_tenant ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(tenant_id);
CREATE INDEX idx_pk_asset_org ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(org_id);
CREATE INDEX idx_pk_asset_class ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(asset_class_code);
CREATE INDEX idx_pk_asset_date ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(analysis_date DESC);
CREATE INDEX idx_pk_asset_coverage_pct ON pack023_sbti_alignment.pack023_fi_asset_class_coverage(target_coverage_value_pct);
CREATE INDEX idx_pk_asset_sectors ON pack023_sbti_alignment.pack023_fi_asset_class_coverage USING GIN(most_common_sectors);
CREATE INDEX idx_pk_asset_regions ON pack023_sbti_alignment.pack023_fi_asset_class_coverage USING GIN(most_common_regions);
CREATE INDEX idx_pk_asset_gaps ON pack023_sbti_alignment.pack023_fi_asset_class_coverage USING GIN(coverage_gaps);

-- Updated_at trigger
CREATE TRIGGER trg_pk_asset_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_fi_asset_class_coverage
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_fi_engagement_targets
-- =============================================================================
-- Investee engagement target details per asset class and sector
-- with progress tracking and engagement effectiveness metrics.

CREATE TABLE pack023_sbti_alignment.pack023_fi_engagement_targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_target_id     UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_fi_portfolio_targets(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    asset_class_code        VARCHAR(50)     NOT NULL,
    engagement_year         INTEGER,
    engagement_type         VARCHAR(100),
    target_sector           VARCHAR(200),
    target_geography        VARCHAR(200),
    target_company_count    INTEGER,
    target_coverage_pct     DECIMAL(6,2),
    companies_engaged       INTEGER,
    engagement_status       VARCHAR(30),
    engagement_methodology  VARCHAR(200)[],
    collaborative_initiative BOOLEAN        DEFAULT FALSE,
    initiative_name         VARCHAR(200),
    num_collaborative_partners INTEGER,
    escalation_pathway      BOOLEAN         DEFAULT FALSE,
    escalation_triggers     TEXT[],
    divestment_criteria     TEXT[],
    engagement_objectives   TEXT,
    engagement_progress     DECIMAL(6,2),
    target_setting_pressure BOOLEAN         DEFAULT TRUE,
    science_based_pressure  BOOLEAN         DEFAULT TRUE,
    net_zero_pressure       BOOLEAN         DEFAULT FALSE,
    timeline_pressure       BOOLEAN         DEFAULT TRUE,
    companies_committed_post_engagement INTEGER,
    companies_validated_post_engagement INTEGER,
    effectiveness_score     DECIMAL(5,2),
    financial_impact        DECIMAL(18,2),
    carbon_impact_potential DECIMAL(18,6),
    expected_emissions_reduction DECIMAL(8,4),
    assessed_by             VARCHAR(255),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_engage_type CHECK (
        engagement_type IN ('ACTIVE_DIALOGUE', 'PUBLIC_CAMPAIGN', 'VOTING', 'COLLABORATIVE',
                           'ESCALATION', 'DIVESTMENT')
    ),
    CONSTRAINT chk_pk_engage_status CHECK (
        engagement_status IN ('PLANNED', 'IN_PROGRESS', 'ON_TRACK', 'OFF_TRACK', 'COMPLETE')
    ),
    CONSTRAINT chk_pk_engage_progress CHECK (
        engagement_progress >= 0 AND engagement_progress <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_engage_target_id ON pack023_sbti_alignment.pack023_fi_engagement_targets(portfolio_target_id);
CREATE INDEX idx_pk_engage_tenant ON pack023_sbti_alignment.pack023_fi_engagement_targets(tenant_id);
CREATE INDEX idx_pk_engage_org ON pack023_sbti_alignment.pack023_fi_engagement_targets(org_id);
CREATE INDEX idx_pk_engage_asset_class ON pack023_sbti_alignment.pack023_fi_engagement_targets(asset_class_code);
CREATE INDEX idx_pk_engage_year ON pack023_sbti_alignment.pack023_fi_engagement_targets(engagement_year);
CREATE INDEX idx_pk_engage_type ON pack023_sbti_alignment.pack023_fi_engagement_targets(engagement_type);
CREATE INDEX idx_pk_engage_status ON pack023_sbti_alignment.pack023_fi_engagement_targets(engagement_status);
CREATE INDEX idx_pk_engage_sector ON pack023_sbti_alignment.pack023_fi_engagement_targets(target_sector);
CREATE INDEX idx_pk_engage_effectiveness ON pack023_sbti_alignment.pack023_fi_engagement_targets(effectiveness_score);
CREATE INDEX idx_pk_engage_created_at ON pack023_sbti_alignment.pack023_fi_engagement_targets(created_at DESC);
CREATE INDEX idx_pk_engage_methods ON pack023_sbti_alignment.pack023_fi_engagement_targets USING GIN(engagement_methodology);
CREATE INDEX idx_pk_engage_triggers ON pack023_sbti_alignment.pack023_fi_engagement_targets USING GIN(escalation_triggers);

-- Updated_at trigger
CREATE TRIGGER trg_pk_engage_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_fi_engagement_targets
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

COMMENT ON TABLE pack023_sbti_alignment.pack023_fi_portfolio_targets IS
'FI portfolio targets by asset class per SBTi FINZ V1.0 with target year, reduction rate, and financed emissions coverage.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_fi_pcaf_data_quality IS
'PCAF data quality assessment for each asset class showing overall score (1-5 per PCAF v2.0) and component tier scoring.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_fi_asset_class_coverage IS
'Asset class coverage analysis showing % of portfolio value, emissions, and investees with SBTi targets by asset class.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_fi_engagement_targets IS
'Investee engagement target details per asset class and sector with methodology, progress tracking, and effectiveness scoring.';
