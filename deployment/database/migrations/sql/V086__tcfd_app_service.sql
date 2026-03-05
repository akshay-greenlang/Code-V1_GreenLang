-- =============================================================================
-- V086: GL-TCFD-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-TCFD-APP (TCFD Disclosure & Scenario Analysis Platform)
-- Date:        March 2026
--
-- Application-level tables for the Task Force on Climate-related Financial
-- Disclosures (TCFD) four-pillar framework.  Covers governance assessments,
-- climate risk/opportunity identification, scenario analysis (IEA/NGFS),
-- physical & transition risk assessment, financial impact quantification,
-- risk management integration, metrics & targets tracking, disclosure
-- generation, ISSB/IFRS S2 cross-walk, and gap analysis.
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--   V081: Audit Trail & Lineage Service
--   V083: GL-GHG-APP v1.0
--   V084: GL-ISO14064-APP v1.0
--   V085: GL-CDP-APP v1.0
--
-- These tables sit in the tcfd_app schema and integrate with the underlying
-- MRV agent data for auto-population of GHG emissions into scenario analysis
-- and metrics/targets disclosures.
-- =============================================================================
-- Tables (22):
--   1.  gl_tcfd_organizations               - Organization profiles
--   2.  gl_tcfd_governance_assessments       - Governance pillar assessments
--   3.  gl_tcfd_governance_roles             - Climate governance roles
--   4.  gl_tcfd_climate_risks                - Climate-related risks
--   5.  gl_tcfd_climate_opportunities        - Climate-related opportunities
--   6.  gl_tcfd_scenarios                    - Scenario definitions
--   7.  gl_tcfd_scenario_results             - Scenario analysis results (HT)
--   8.  gl_tcfd_scenario_parameters          - Scenario parameter values
--   9.  gl_tcfd_physical_risk_assessments    - Physical risk assessments (HT)
--  10.  gl_tcfd_asset_locations              - Physical asset locations
--  11.  gl_tcfd_transition_risk_assessments  - Transition risk assessments
--  12.  gl_tcfd_financial_impacts            - Financial impacts (HT)
--  13.  gl_tcfd_risk_management_records      - Risk management records
--  14.  gl_tcfd_risk_responses               - Risk response actions
--  15.  gl_tcfd_metrics                      - Climate metric definitions
--  16.  gl_tcfd_metric_values                - Metric period values
--  17.  gl_tcfd_targets                      - Climate targets
--  18.  gl_tcfd_target_progress              - Target progress tracking
--  19.  gl_tcfd_disclosures                  - Disclosure documents
--  20.  gl_tcfd_disclosure_sections          - Disclosure sections (11)
--  21.  gl_tcfd_gap_assessments              - Gap analysis results
--  22.  gl_tcfd_issb_mappings                - TCFD-to-ISSB cross-walk
--
-- Hypertables (3):
--   gl_tcfd_scenario_results        on analysis_date  (chunk: 3 months)
--   gl_tcfd_physical_risk_assessments on assessment_date (chunk: 3 months)
--   gl_tcfd_financial_impacts       on impact_date    (chunk: 3 months)
--
-- Continuous Aggregates (2):
--   tcfd_app.quarterly_risk_scores      - Quarterly risk score aggregation
--   tcfd_app.annual_scenario_summary    - Annual scenario result averages
--
-- Also includes: 80+ indexes (B-tree, GIN), update triggers, security
-- grants, retention policies, compression policies, permissions, and comments.
-- Previous: V085__cdp_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS tcfd_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION tcfd_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Enum Types
-- =============================================================================

CREATE TYPE tcfd_app.gl_tcfd_risk_type_enum AS ENUM (
    'physical_acute', 'physical_chronic',
    'transition_policy', 'transition_technology',
    'transition_market', 'transition_reputation'
);

CREATE TYPE tcfd_app.gl_tcfd_opportunity_category_enum AS ENUM (
    'resource_efficiency', 'energy_source',
    'products_services', 'markets', 'resilience'
);

CREATE TYPE tcfd_app.gl_tcfd_scenario_type_enum AS ENUM (
    'iea_nze', 'iea_aps', 'iea_steps',
    'ngfs_current_policies', 'ngfs_delayed_transition',
    'ngfs_below_2c', 'ngfs_divergent_nz', 'custom'
);

CREATE TYPE tcfd_app.gl_tcfd_time_horizon_enum AS ENUM (
    'short_term', 'medium_term', 'long_term'
);

CREATE TYPE tcfd_app.gl_tcfd_financial_statement_enum AS ENUM (
    'income_statement', 'balance_sheet', 'cash_flow'
);

CREATE TYPE tcfd_app.gl_tcfd_impact_category_enum AS ENUM (
    'revenue', 'cost', 'asset_impairment', 'capex',
    'carbon_cost', 'insurance', 'litigation', 'other'
);

CREATE TYPE tcfd_app.gl_tcfd_asset_type_enum AS ENUM (
    'building', 'infrastructure', 'equipment',
    'vehicle', 'land', 'financial_asset'
);

CREATE TYPE tcfd_app.gl_tcfd_risk_response_enum AS ENUM (
    'accept', 'mitigate', 'transfer', 'avoid'
);

CREATE TYPE tcfd_app.gl_tcfd_target_type_enum AS ENUM (
    'absolute', 'intensity', 'net_zero',
    'renewable_energy', 'custom'
);

CREATE TYPE tcfd_app.gl_tcfd_disclosure_status_enum AS ENUM (
    'draft', 'review', 'approved', 'published'
);

CREATE TYPE tcfd_app.gl_tcfd_disclosure_code_enum AS ENUM (
    'gov_a', 'gov_b',
    'str_a', 'str_b', 'str_c',
    'rm_a', 'rm_b', 'rm_c',
    'mt_a', 'mt_b', 'mt_c'
);

CREATE TYPE tcfd_app.gl_tcfd_pillar_enum AS ENUM (
    'governance', 'strategy', 'risk_management', 'metrics_targets'
);

CREATE TYPE tcfd_app.gl_tcfd_metric_category_enum AS ENUM (
    'ghg_emissions', 'transition_risk_assets', 'physical_risk_assets',
    'opportunity_revenue', 'capital_deployment',
    'internal_carbon_price', 'remuneration_linked',
    'cross_industry', 'industry_specific'
);

CREATE TYPE tcfd_app.gl_tcfd_hazard_type_enum AS ENUM (
    'cyclone', 'flood', 'wildfire', 'heatwave',
    'drought', 'sea_level_rise', 'water_stress',
    'temperature_rise', 'precipitation_change'
);

-- =============================================================================
-- Table 1: tcfd_app.gl_tcfd_organizations
-- =============================================================================
-- Organization profiles for TCFD disclosure.  Each organization represents
-- the top-level entity that produces TCFD-aligned climate disclosures,
-- including scenario analysis and risk assessments.

CREATE TABLE tcfd_app.gl_tcfd_organizations (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           UUID,
    name                VARCHAR(500)    NOT NULL,
    sector              VARCHAR(200),
    industry            VARCHAR(200),
    jurisdiction        VARCHAR(100),
    reporting_currency  VARCHAR(3)      DEFAULT 'USD',
    fiscal_year_end     VARCHAR(10),
    metadata            JSONB           DEFAULT '{}',
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_org_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tcfd_org_currency_length CHECK (
        reporting_currency IS NULL OR LENGTH(TRIM(reporting_currency)) = 3
    )
);

-- Indexes
CREATE INDEX idx_tcfd_org_tenant ON tcfd_app.gl_tcfd_organizations(tenant_id);
CREATE INDEX idx_tcfd_org_name ON tcfd_app.gl_tcfd_organizations(name);
CREATE INDEX idx_tcfd_org_sector ON tcfd_app.gl_tcfd_organizations(sector);
CREATE INDEX idx_tcfd_org_industry ON tcfd_app.gl_tcfd_organizations(industry);
CREATE INDEX idx_tcfd_org_jurisdiction ON tcfd_app.gl_tcfd_organizations(jurisdiction);
CREATE INDEX idx_tcfd_org_created_at ON tcfd_app.gl_tcfd_organizations(created_at DESC);
CREATE INDEX idx_tcfd_org_metadata ON tcfd_app.gl_tcfd_organizations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_org_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_organizations
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 2: tcfd_app.gl_tcfd_governance_assessments
-- =============================================================================
-- Governance pillar assessments per TCFD Governance (a) and (b) disclosures.
-- Evaluates board oversight, management roles, competency, reporting
-- structures, and incentive linkages across 8 maturity dimensions.

CREATE TABLE tcfd_app.gl_tcfd_governance_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    reporting_period_start  DATE            NOT NULL,
    reporting_period_end    DATE            NOT NULL,
    board_oversight         JSONB           DEFAULT '{}',
    management_roles        JSONB           DEFAULT '{}',
    maturity_scores         JSONB           DEFAULT '{}',
    competency_matrix       JSONB           DEFAULT '{}',
    incentive_linkages      JSONB           DEFAULT '{}',
    overall_maturity_score  DECIMAL(5,2)    DEFAULT 0,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_ga_period CHECK (
        reporting_period_end >= reporting_period_start
    ),
    CONSTRAINT chk_tcfd_ga_maturity_range CHECK (
        overall_maturity_score >= 0 AND overall_maturity_score <= 5
    ),
    CONSTRAINT chk_tcfd_ga_status CHECK (
        status IN ('draft', 'review', 'approved', 'published')
    )
);

-- Indexes
CREATE INDEX idx_tcfd_ga_org ON tcfd_app.gl_tcfd_governance_assessments(org_id);
CREATE INDEX idx_tcfd_ga_status ON tcfd_app.gl_tcfd_governance_assessments(status);
CREATE INDEX idx_tcfd_ga_period ON tcfd_app.gl_tcfd_governance_assessments(org_id, reporting_period_start);
CREATE INDEX idx_tcfd_ga_created_at ON tcfd_app.gl_tcfd_governance_assessments(created_at DESC);
CREATE INDEX idx_tcfd_ga_board ON tcfd_app.gl_tcfd_governance_assessments USING GIN(board_oversight);
CREATE INDEX idx_tcfd_ga_maturity ON tcfd_app.gl_tcfd_governance_assessments USING GIN(maturity_scores);
CREATE INDEX idx_tcfd_ga_competency ON tcfd_app.gl_tcfd_governance_assessments USING GIN(competency_matrix);
CREATE INDEX idx_tcfd_ga_incentives ON tcfd_app.gl_tcfd_governance_assessments USING GIN(incentive_linkages);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_ga_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_governance_assessments
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 3: tcfd_app.gl_tcfd_governance_roles
-- =============================================================================
-- Climate governance roles tracking board and management positions with
-- climate accountability, responsibilities, and reporting lines.

CREATE TABLE tcfd_app.gl_tcfd_governance_roles (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_governance_assessments(id) ON DELETE CASCADE,
    role_title              VARCHAR(255)    NOT NULL,
    role_holder             VARCHAR(255),
    responsibilities        TEXT[]          DEFAULT '{}',
    climate_accountability  BOOLEAN         NOT NULL DEFAULT FALSE,
    reporting_line          VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_gr_title_not_empty CHECK (
        LENGTH(TRIM(role_title)) > 0
    )
);

-- Indexes
CREATE INDEX idx_tcfd_gr_assessment ON tcfd_app.gl_tcfd_governance_roles(assessment_id);
CREATE INDEX idx_tcfd_gr_accountability ON tcfd_app.gl_tcfd_governance_roles(climate_accountability);
CREATE INDEX idx_tcfd_gr_created_at ON tcfd_app.gl_tcfd_governance_roles(created_at DESC);

-- =============================================================================
-- Table 4: tcfd_app.gl_tcfd_climate_risks
-- =============================================================================
-- Climate-related risks per TCFD Strategy (a).  Captures physical and
-- transition risk types, time horizons, likelihood/impact scoring,
-- financial impact estimates, and strategic responses.

CREATE TABLE tcfd_app.gl_tcfd_climate_risks (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    risk_type                   tcfd_app.gl_tcfd_risk_type_enum NOT NULL,
    sub_type                    VARCHAR(100),
    name                        VARCHAR(255)    NOT NULL,
    description                 TEXT,
    time_horizon                tcfd_app.gl_tcfd_time_horizon_enum NOT NULL DEFAULT 'medium_term',
    likelihood                  INTEGER         NOT NULL DEFAULT 3,
    impact                      INTEGER         NOT NULL DEFAULT 3,
    risk_score                  DECIMAL(6,2)    DEFAULT 0,
    financial_impact_estimate   DECIMAL(18,2)   DEFAULT 0,
    currency                    VARCHAR(3)      DEFAULT 'USD',
    affected_assets             TEXT[]          DEFAULT '{}',
    affected_value_chain        TEXT[]          DEFAULT '{}',
    strategic_response          JSONB           DEFAULT '{}',
    status                      VARCHAR(30)     NOT NULL DEFAULT 'active',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_cr_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tcfd_cr_likelihood_range CHECK (
        likelihood >= 1 AND likelihood <= 5
    ),
    CONSTRAINT chk_tcfd_cr_impact_range CHECK (
        impact >= 1 AND impact <= 5
    ),
    CONSTRAINT chk_tcfd_cr_score_non_neg CHECK (
        risk_score >= 0
    ),
    CONSTRAINT chk_tcfd_cr_financial_non_neg CHECK (
        financial_impact_estimate >= 0
    ),
    CONSTRAINT chk_tcfd_cr_status CHECK (
        status IN ('active', 'mitigated', 'closed', 'monitoring')
    )
);

-- Indexes
CREATE INDEX idx_tcfd_cr_org ON tcfd_app.gl_tcfd_climate_risks(org_id);
CREATE INDEX idx_tcfd_cr_risk_type ON tcfd_app.gl_tcfd_climate_risks(risk_type);
CREATE INDEX idx_tcfd_cr_time_horizon ON tcfd_app.gl_tcfd_climate_risks(time_horizon);
CREATE INDEX idx_tcfd_cr_status ON tcfd_app.gl_tcfd_climate_risks(status);
CREATE INDEX idx_tcfd_cr_score ON tcfd_app.gl_tcfd_climate_risks(risk_score DESC);
CREATE INDEX idx_tcfd_cr_created_at ON tcfd_app.gl_tcfd_climate_risks(created_at DESC);
CREATE INDEX idx_tcfd_cr_response ON tcfd_app.gl_tcfd_climate_risks USING GIN(strategic_response);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_cr_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_climate_risks
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 5: tcfd_app.gl_tcfd_climate_opportunities
-- =============================================================================
-- Climate-related opportunities per TCFD Strategy (a).  Covers resource
-- efficiency, energy source, products/services, markets, and resilience
-- categories with financial sizing and pipeline tracking.

CREATE TABLE tcfd_app.gl_tcfd_climate_opportunities (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    category                tcfd_app.gl_tcfd_opportunity_category_enum NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    description             TEXT,
    time_horizon            tcfd_app.gl_tcfd_time_horizon_enum NOT NULL DEFAULT 'medium_term',
    revenue_potential       DECIMAL(18,2)   DEFAULT 0,
    cost_savings_potential  DECIMAL(18,2)   DEFAULT 0,
    investment_required     DECIMAL(18,2)   DEFAULT 0,
    currency                VARCHAR(3)      DEFAULT 'USD',
    pipeline_stage          VARCHAR(50)     DEFAULT 'identified',
    strategic_response      JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_co_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tcfd_co_revenue_non_neg CHECK (
        revenue_potential >= 0
    ),
    CONSTRAINT chk_tcfd_co_savings_non_neg CHECK (
        cost_savings_potential >= 0
    ),
    CONSTRAINT chk_tcfd_co_investment_non_neg CHECK (
        investment_required >= 0
    ),
    CONSTRAINT chk_tcfd_co_pipeline CHECK (
        pipeline_stage IN ('identified', 'evaluated', 'planned', 'in_progress', 'realized', 'deferred')
    )
);

-- Indexes
CREATE INDEX idx_tcfd_co_org ON tcfd_app.gl_tcfd_climate_opportunities(org_id);
CREATE INDEX idx_tcfd_co_category ON tcfd_app.gl_tcfd_climate_opportunities(category);
CREATE INDEX idx_tcfd_co_time_horizon ON tcfd_app.gl_tcfd_climate_opportunities(time_horizon);
CREATE INDEX idx_tcfd_co_pipeline ON tcfd_app.gl_tcfd_climate_opportunities(pipeline_stage);
CREATE INDEX idx_tcfd_co_created_at ON tcfd_app.gl_tcfd_climate_opportunities(created_at DESC);
CREATE INDEX idx_tcfd_co_response ON tcfd_app.gl_tcfd_climate_opportunities USING GIN(strategic_response);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_co_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_climate_opportunities
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 6: tcfd_app.gl_tcfd_scenarios
-- =============================================================================
-- Scenario definitions for TCFD Strategy (c) analysis.  Supports 7 pre-built
-- scenarios (IEA NZE/APS/STEPS, NGFS Current/Delayed/Below2C/Divergent) plus
-- custom user-defined scenarios with full parameter customization.

CREATE TABLE tcfd_app.gl_tcfd_scenarios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    scenario_type               tcfd_app.gl_tcfd_scenario_type_enum NOT NULL,
    name                        VARCHAR(255)    NOT NULL,
    description                 TEXT,
    temperature_outcome         DECIMAL(4,2)    DEFAULT 0,
    time_horizons               INTEGER[]       DEFAULT '{2030, 2040, 2050}',
    parameters                  JSONB           DEFAULT '{}',
    carbon_price_trajectory     JSONB           DEFAULT '{}',
    energy_mix_trajectory       JSONB           DEFAULT '{}',
    is_custom                   BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_sc_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tcfd_sc_temp_range CHECK (
        temperature_outcome >= 0 AND temperature_outcome <= 10
    )
);

-- Indexes
CREATE INDEX idx_tcfd_sc_org ON tcfd_app.gl_tcfd_scenarios(org_id);
CREATE INDEX idx_tcfd_sc_type ON tcfd_app.gl_tcfd_scenarios(scenario_type);
CREATE INDEX idx_tcfd_sc_custom ON tcfd_app.gl_tcfd_scenarios(is_custom);
CREATE INDEX idx_tcfd_sc_created_at ON tcfd_app.gl_tcfd_scenarios(created_at DESC);
CREATE INDEX idx_tcfd_sc_parameters ON tcfd_app.gl_tcfd_scenarios USING GIN(parameters);
CREATE INDEX idx_tcfd_sc_carbon ON tcfd_app.gl_tcfd_scenarios USING GIN(carbon_price_trajectory);
CREATE INDEX idx_tcfd_sc_energy ON tcfd_app.gl_tcfd_scenarios USING GIN(energy_mix_trajectory);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_sc_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_scenarios
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 7: tcfd_app.gl_tcfd_scenario_results (HYPERTABLE)
-- =============================================================================
-- Scenario analysis results partitioned by analysis_date for time-series
-- querying.  Stores revenue/cost/asset impacts, carbon cost, capex
-- requirements, resilience scoring, and narrative summaries per scenario run.

CREATE TABLE tcfd_app.gl_tcfd_scenario_results (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    scenario_id             UUID            NOT NULL,
    analysis_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    target_year             INTEGER         NOT NULL,
    revenue_impact_pct      DECIMAL(8,4)    DEFAULT 0,
    cost_impact_abs         DECIMAL(18,2)   DEFAULT 0,
    asset_impairment_pct    DECIMAL(6,4)    DEFAULT 0,
    capex_requirement       DECIMAL(18,2)   DEFAULT 0,
    carbon_cost             DECIMAL(18,2)   DEFAULT 0,
    currency                VARCHAR(3)      DEFAULT 'USD',
    resilience_score        DECIMAL(5,2)    DEFAULT 0,
    narrative               TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_sr_year_range CHECK (
        target_year >= 2020 AND target_year <= 2100
    ),
    CONSTRAINT chk_tcfd_sr_resilience_range CHECK (
        resilience_score >= 0 AND resilience_score <= 100
    ),
    CONSTRAINT chk_tcfd_sr_capex_non_neg CHECK (
        capex_requirement >= 0
    ),
    CONSTRAINT chk_tcfd_sr_carbon_non_neg CHECK (
        carbon_cost >= 0
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('tcfd_app.gl_tcfd_scenario_results', 'analysis_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tcfd_sr_org ON tcfd_app.gl_tcfd_scenario_results(org_id, analysis_date DESC);
CREATE INDEX idx_tcfd_sr_scenario ON tcfd_app.gl_tcfd_scenario_results(scenario_id, analysis_date DESC);
CREATE INDEX idx_tcfd_sr_year ON tcfd_app.gl_tcfd_scenario_results(target_year, analysis_date DESC);
CREATE INDEX idx_tcfd_sr_org_scenario ON tcfd_app.gl_tcfd_scenario_results(org_id, scenario_id, analysis_date DESC);
CREATE INDEX idx_tcfd_sr_scenario_year ON tcfd_app.gl_tcfd_scenario_results(scenario_id, target_year);

-- =============================================================================
-- Table 8: tcfd_app.gl_tcfd_scenario_parameters
-- =============================================================================
-- Individual parameter values within a scenario definition.  Stores yearly
-- parameter projections for carbon prices, energy mix, technology adoption,
-- and regulatory milestones.

CREATE TABLE tcfd_app.gl_tcfd_scenario_parameters (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_id             UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_scenarios(id) ON DELETE CASCADE,
    parameter_name          VARCHAR(200)    NOT NULL,
    parameter_category      VARCHAR(100)    DEFAULT '',
    year                    INTEGER         NOT NULL,
    value                   DECIMAL(18,6)   NOT NULL,
    unit                    VARCHAR(50)     DEFAULT '',
    source                  VARCHAR(500)    DEFAULT '',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_sp_name_not_empty CHECK (
        LENGTH(TRIM(parameter_name)) > 0
    ),
    CONSTRAINT chk_tcfd_sp_year_range CHECK (
        year >= 2020 AND year <= 2100
    )
);

-- Indexes
CREATE INDEX idx_tcfd_sp_scenario ON tcfd_app.gl_tcfd_scenario_parameters(scenario_id);
CREATE INDEX idx_tcfd_sp_name ON tcfd_app.gl_tcfd_scenario_parameters(parameter_name);
CREATE INDEX idx_tcfd_sp_category ON tcfd_app.gl_tcfd_scenario_parameters(parameter_category);
CREATE INDEX idx_tcfd_sp_year ON tcfd_app.gl_tcfd_scenario_parameters(year);
CREATE INDEX idx_tcfd_sp_scenario_year ON tcfd_app.gl_tcfd_scenario_parameters(scenario_id, year);

-- =============================================================================
-- Table 9: tcfd_app.gl_tcfd_physical_risk_assessments (HYPERTABLE)
-- =============================================================================
-- Physical climate risk assessments per asset partitioned by assessment_date.
-- Computes exposure, vulnerability, and adaptive capacity scores to derive
-- composite risk scores and financial damage estimates per hazard type.

CREATE TABLE tcfd_app.gl_tcfd_physical_risk_assessments (
    id                          UUID            NOT NULL DEFAULT gen_random_uuid(),
    asset_id                    UUID            NOT NULL,
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    hazard_type                 tcfd_app.gl_tcfd_hazard_type_enum NOT NULL,
    exposure_score              DECIMAL(5,2)    DEFAULT 0,
    vulnerability_score         DECIMAL(5,2)    DEFAULT 0,
    adaptive_capacity_score     DECIMAL(5,2)    DEFAULT 0,
    composite_risk_score        DECIMAL(5,2)    DEFAULT 0,
    damage_estimate_pct         DECIMAL(6,4)    DEFAULT 0,
    insurance_premium_change_pct DECIMAL(6,4)   DEFAULT 0,
    rcp_ssp_scenario            VARCHAR(20)     DEFAULT 'ssp2_45',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_pra_exposure_range CHECK (
        exposure_score >= 0 AND exposure_score <= 5
    ),
    CONSTRAINT chk_tcfd_pra_vulnerability_range CHECK (
        vulnerability_score >= 0 AND vulnerability_score <= 5
    ),
    CONSTRAINT chk_tcfd_pra_adaptive_range CHECK (
        adaptive_capacity_score >= 0 AND adaptive_capacity_score <= 5
    ),
    CONSTRAINT chk_tcfd_pra_composite_range CHECK (
        composite_risk_score >= 0 AND composite_risk_score <= 100
    ),
    CONSTRAINT chk_tcfd_pra_damage_range CHECK (
        damage_estimate_pct >= 0 AND damage_estimate_pct <= 100
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('tcfd_app.gl_tcfd_physical_risk_assessments', 'assessment_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tcfd_pra_asset ON tcfd_app.gl_tcfd_physical_risk_assessments(asset_id, assessment_date DESC);
CREATE INDEX idx_tcfd_pra_hazard ON tcfd_app.gl_tcfd_physical_risk_assessments(hazard_type, assessment_date DESC);
CREATE INDEX idx_tcfd_pra_rcp ON tcfd_app.gl_tcfd_physical_risk_assessments(rcp_ssp_scenario, assessment_date DESC);
CREATE INDEX idx_tcfd_pra_composite ON tcfd_app.gl_tcfd_physical_risk_assessments(composite_risk_score DESC, assessment_date DESC);

-- =============================================================================
-- Table 10: tcfd_app.gl_tcfd_asset_locations
-- =============================================================================
-- Physical asset locations subject to climate risk assessment.  Stores
-- geographic coordinates, asset type, replacement value, building
-- characteristics, and insurance coverage for physical risk scoring.

CREATE TABLE tcfd_app.gl_tcfd_asset_locations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    asset_name              VARCHAR(255)    NOT NULL,
    asset_type              tcfd_app.gl_tcfd_asset_type_enum NOT NULL DEFAULT 'building',
    latitude                DECIMAL(10,7)   NOT NULL,
    longitude               DECIMAL(10,7)   NOT NULL,
    country                 VARCHAR(3)      NOT NULL,
    region                  VARCHAR(100),
    city                    VARCHAR(100),
    value                   DECIMAL(18,2)   DEFAULT 0,
    currency                VARCHAR(3)      DEFAULT 'USD',
    building_type           VARCHAR(100),
    elevation_m             DECIMAL(8,2)    DEFAULT 0,
    infrastructure_quality  VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_al_name_not_empty CHECK (
        LENGTH(TRIM(asset_name)) > 0
    ),
    CONSTRAINT chk_tcfd_al_lat_range CHECK (
        latitude >= -90 AND latitude <= 90
    ),
    CONSTRAINT chk_tcfd_al_lon_range CHECK (
        longitude >= -180 AND longitude <= 180
    ),
    CONSTRAINT chk_tcfd_al_value_non_neg CHECK (
        value >= 0
    ),
    CONSTRAINT chk_tcfd_al_country_length CHECK (
        LENGTH(TRIM(country)) >= 2 AND LENGTH(TRIM(country)) <= 3
    )
);

-- Indexes
CREATE INDEX idx_tcfd_al_org ON tcfd_app.gl_tcfd_asset_locations(org_id);
CREATE INDEX idx_tcfd_al_type ON tcfd_app.gl_tcfd_asset_locations(asset_type);
CREATE INDEX idx_tcfd_al_country ON tcfd_app.gl_tcfd_asset_locations(country);
CREATE INDEX idx_tcfd_al_region ON tcfd_app.gl_tcfd_asset_locations(region);
CREATE INDEX idx_tcfd_al_created_at ON tcfd_app.gl_tcfd_asset_locations(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_al_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_asset_locations
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 11: tcfd_app.gl_tcfd_transition_risk_assessments
-- =============================================================================
-- Transition risk assessments covering policy, technology, market, and
-- reputation risk sub-categories with composite scoring, carbon price
-- impact modeling, and stranding probability estimation.

CREATE TABLE tcfd_app.gl_tcfd_transition_risk_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    policy_risk_score           DECIMAL(5,2)    DEFAULT 0,
    technology_risk_score       DECIMAL(5,2)    DEFAULT 0,
    market_risk_score           DECIMAL(5,2)    DEFAULT 0,
    reputation_risk_score       DECIMAL(5,2)    DEFAULT 0,
    composite_score             DECIMAL(5,2)    DEFAULT 0,
    carbon_price_impact         DECIMAL(18,2)   DEFAULT 0,
    stranding_probability       DECIMAL(5,4)    DEFAULT 0,
    sector_profile              JSONB           DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_tra_policy_range CHECK (
        policy_risk_score >= 0 AND policy_risk_score <= 100
    ),
    CONSTRAINT chk_tcfd_tra_tech_range CHECK (
        technology_risk_score >= 0 AND technology_risk_score <= 100
    ),
    CONSTRAINT chk_tcfd_tra_market_range CHECK (
        market_risk_score >= 0 AND market_risk_score <= 100
    ),
    CONSTRAINT chk_tcfd_tra_rep_range CHECK (
        reputation_risk_score >= 0 AND reputation_risk_score <= 100
    ),
    CONSTRAINT chk_tcfd_tra_composite_range CHECK (
        composite_score >= 0 AND composite_score <= 100
    ),
    CONSTRAINT chk_tcfd_tra_carbon_non_neg CHECK (
        carbon_price_impact >= 0
    ),
    CONSTRAINT chk_tcfd_tra_stranding_range CHECK (
        stranding_probability >= 0 AND stranding_probability <= 1
    )
);

-- Indexes
CREATE INDEX idx_tcfd_tra_org ON tcfd_app.gl_tcfd_transition_risk_assessments(org_id);
CREATE INDEX idx_tcfd_tra_date ON tcfd_app.gl_tcfd_transition_risk_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_tcfd_tra_composite ON tcfd_app.gl_tcfd_transition_risk_assessments(composite_score DESC);
CREATE INDEX idx_tcfd_tra_created_at ON tcfd_app.gl_tcfd_transition_risk_assessments(created_at DESC);
CREATE INDEX idx_tcfd_tra_sector ON tcfd_app.gl_tcfd_transition_risk_assessments USING GIN(sector_profile);

-- =============================================================================
-- Table 12: tcfd_app.gl_tcfd_financial_impacts (HYPERTABLE)
-- =============================================================================
-- Climate financial impacts on income statement, balance sheet, and cash
-- flow per TCFD/IFRS S2 requirements.  Partitioned by impact_date for
-- time-series analysis of projected financial impacts under scenarios.

CREATE TABLE tcfd_app.gl_tcfd_financial_impacts (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    scenario_id             UUID,
    impact_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    statement_type          tcfd_app.gl_tcfd_financial_statement_enum NOT NULL,
    impact_category         tcfd_app.gl_tcfd_impact_category_enum NOT NULL,
    amount                  DECIMAL(18,2)   NOT NULL DEFAULT 0,
    currency                VARCHAR(3)      DEFAULT 'USD',
    description             TEXT,
    assumptions             JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('tcfd_app.gl_tcfd_financial_impacts', 'impact_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tcfd_fi_org ON tcfd_app.gl_tcfd_financial_impacts(org_id, impact_date DESC);
CREATE INDEX idx_tcfd_fi_scenario ON tcfd_app.gl_tcfd_financial_impacts(scenario_id, impact_date DESC);
CREATE INDEX idx_tcfd_fi_statement ON tcfd_app.gl_tcfd_financial_impacts(statement_type, impact_date DESC);
CREATE INDEX idx_tcfd_fi_category ON tcfd_app.gl_tcfd_financial_impacts(impact_category, impact_date DESC);
CREATE INDEX idx_tcfd_fi_org_scenario ON tcfd_app.gl_tcfd_financial_impacts(org_id, scenario_id, impact_date DESC);
CREATE INDEX idx_tcfd_fi_assumptions ON tcfd_app.gl_tcfd_financial_impacts USING GIN(assumptions);

-- =============================================================================
-- Table 13: tcfd_app.gl_tcfd_risk_management_records
-- =============================================================================
-- Risk management records per TCFD Pillar 3 (RM a/b/c).  Tracks risk
-- register configuration, ERM framework details, risk appetite, review
-- schedules, and entry counts.

CREATE TABLE tcfd_app.gl_tcfd_risk_management_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    register_name               VARCHAR(255)    NOT NULL,
    erm_framework               VARCHAR(200),
    risk_appetite_statement     TEXT,
    review_frequency            VARCHAR(50)     DEFAULT 'quarterly',
    last_review_date            DATE,
    entries_count               INTEGER         NOT NULL DEFAULT 0,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_rmr_name_not_empty CHECK (
        LENGTH(TRIM(register_name)) > 0
    ),
    CONSTRAINT chk_tcfd_rmr_entries_non_neg CHECK (
        entries_count >= 0
    ),
    CONSTRAINT chk_tcfd_rmr_review_freq CHECK (
        review_frequency IN ('monthly', 'quarterly', 'semi_annually', 'annually')
    )
);

-- Indexes
CREATE INDEX idx_tcfd_rmr_org ON tcfd_app.gl_tcfd_risk_management_records(org_id);
CREATE INDEX idx_tcfd_rmr_framework ON tcfd_app.gl_tcfd_risk_management_records(erm_framework);
CREATE INDEX idx_tcfd_rmr_review ON tcfd_app.gl_tcfd_risk_management_records(review_frequency);
CREATE INDEX idx_tcfd_rmr_created_at ON tcfd_app.gl_tcfd_risk_management_records(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_rmr_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_risk_management_records
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 14: tcfd_app.gl_tcfd_risk_responses
-- =============================================================================
-- Risk response actions linked to risk entries.  Tracks response type
-- (accept/mitigate/transfer/avoid), ownership, progress, and cost estimates.

CREATE TABLE tcfd_app.gl_tcfd_risk_responses (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_entry_id           UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_climate_risks(id) ON DELETE CASCADE,
    response_type           tcfd_app.gl_tcfd_risk_response_enum NOT NULL DEFAULT 'accept',
    description             TEXT,
    owner                   VARCHAR(255),
    target_date             DATE,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'planned',
    progress_pct            DECIMAL(5,2)    DEFAULT 0,
    cost_estimate           DECIMAL(18,2)   DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_rr_status CHECK (
        status IN ('planned', 'in_progress', 'completed', 'cancelled')
    ),
    CONSTRAINT chk_tcfd_rr_progress_range CHECK (
        progress_pct >= 0 AND progress_pct <= 100
    ),
    CONSTRAINT chk_tcfd_rr_cost_non_neg CHECK (
        cost_estimate >= 0
    )
);

-- Indexes
CREATE INDEX idx_tcfd_rr_risk ON tcfd_app.gl_tcfd_risk_responses(risk_entry_id);
CREATE INDEX idx_tcfd_rr_type ON tcfd_app.gl_tcfd_risk_responses(response_type);
CREATE INDEX idx_tcfd_rr_status ON tcfd_app.gl_tcfd_risk_responses(status);
CREATE INDEX idx_tcfd_rr_target ON tcfd_app.gl_tcfd_risk_responses(target_date);
CREATE INDEX idx_tcfd_rr_created_at ON tcfd_app.gl_tcfd_risk_responses(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_rr_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_risk_responses
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 15: tcfd_app.gl_tcfd_metrics
-- =============================================================================
-- Climate metric definitions per TCFD Metrics & Targets (a) and ISSB para 29.
-- Tracks metric type, unit, SASB code, cross-industry flag, and description.

CREATE TABLE tcfd_app.gl_tcfd_metrics (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    metric_name             VARCHAR(255)    NOT NULL,
    metric_category         tcfd_app.gl_tcfd_metric_category_enum NOT NULL,
    unit                    VARCHAR(50)     DEFAULT '',
    description             TEXT,
    sasb_code               VARCHAR(50),
    is_cross_industry       BOOLEAN         NOT NULL DEFAULT TRUE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_mt_name_not_empty CHECK (
        LENGTH(TRIM(metric_name)) > 0
    )
);

-- Indexes
CREATE INDEX idx_tcfd_mt_org ON tcfd_app.gl_tcfd_metrics(org_id);
CREATE INDEX idx_tcfd_mt_category ON tcfd_app.gl_tcfd_metrics(metric_category);
CREATE INDEX idx_tcfd_mt_cross ON tcfd_app.gl_tcfd_metrics(is_cross_industry);
CREATE INDEX idx_tcfd_mt_sasb ON tcfd_app.gl_tcfd_metrics(sasb_code);
CREATE INDEX idx_tcfd_mt_created_at ON tcfd_app.gl_tcfd_metrics(created_at DESC);

-- =============================================================================
-- Table 16: tcfd_app.gl_tcfd_metric_values
-- =============================================================================
-- Metric period values linking to metric definitions.  Tracks reporting
-- period values, data quality tiers, sources, and verification status.

CREATE TABLE tcfd_app.gl_tcfd_metric_values (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_id                   UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_metrics(id) ON DELETE CASCADE,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    value                       DECIMAL(18,6)   NOT NULL,
    data_quality_tier           INTEGER         DEFAULT 3,
    source                      VARCHAR(500)    DEFAULT '',
    verified                    BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_mv_period CHECK (
        reporting_period_end >= reporting_period_start
    ),
    CONSTRAINT chk_tcfd_mv_quality_range CHECK (
        data_quality_tier >= 1 AND data_quality_tier <= 5
    )
);

-- Indexes
CREATE INDEX idx_tcfd_mv_metric ON tcfd_app.gl_tcfd_metric_values(metric_id);
CREATE INDEX idx_tcfd_mv_period ON tcfd_app.gl_tcfd_metric_values(reporting_period_start, reporting_period_end);
CREATE INDEX idx_tcfd_mv_verified ON tcfd_app.gl_tcfd_metric_values(verified);
CREATE INDEX idx_tcfd_mv_quality ON tcfd_app.gl_tcfd_metric_values(data_quality_tier);
CREATE INDEX idx_tcfd_mv_created_at ON tcfd_app.gl_tcfd_metric_values(created_at DESC);

-- =============================================================================
-- Table 17: tcfd_app.gl_tcfd_targets
-- =============================================================================
-- Climate targets per TCFD Metrics & Targets (c).  Tracks absolute, intensity,
-- net-zero, and renewable energy targets with base/target year values,
-- SBTi alignment, and pathway specification.

CREATE TABLE tcfd_app.gl_tcfd_targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    target_name             VARCHAR(255)    NOT NULL,
    target_type             tcfd_app.gl_tcfd_target_type_enum NOT NULL,
    base_year               INTEGER         NOT NULL,
    base_value              DECIMAL(18,6)   NOT NULL,
    target_year             INTEGER         NOT NULL,
    target_value            DECIMAL(18,6)   NOT NULL,
    reduction_pct           DECIMAL(6,2)    DEFAULT 0,
    sbti_aligned            BOOLEAN         NOT NULL DEFAULT FALSE,
    sbti_pathway            VARCHAR(100),
    unit                    VARCHAR(50)     DEFAULT 'tCO2e',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_tgt_name_not_empty CHECK (
        LENGTH(TRIM(target_name)) > 0
    ),
    CONSTRAINT chk_tcfd_tgt_year_range CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_tcfd_tgt_target_year CHECK (
        target_year >= 1990 AND target_year <= 2100
    ),
    CONSTRAINT chk_tcfd_tgt_target_after_base CHECK (
        target_year > base_year
    ),
    CONSTRAINT chk_tcfd_tgt_reduction_range CHECK (
        reduction_pct >= 0 AND reduction_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_tcfd_tgt_org ON tcfd_app.gl_tcfd_targets(org_id);
CREATE INDEX idx_tcfd_tgt_type ON tcfd_app.gl_tcfd_targets(target_type);
CREATE INDEX idx_tcfd_tgt_sbti ON tcfd_app.gl_tcfd_targets(sbti_aligned);
CREATE INDEX idx_tcfd_tgt_base_year ON tcfd_app.gl_tcfd_targets(base_year);
CREATE INDEX idx_tcfd_tgt_target_year ON tcfd_app.gl_tcfd_targets(target_year);
CREATE INDEX idx_tcfd_tgt_created_at ON tcfd_app.gl_tcfd_targets(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_tgt_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_targets
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 18: tcfd_app.gl_tcfd_target_progress
-- =============================================================================
-- Target progress tracking records.  Stores current values, gap to target,
-- on-track assessment, and notes per reporting period.

CREATE TABLE tcfd_app.gl_tcfd_target_progress (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id               UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_targets(id) ON DELETE CASCADE,
    progress_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    current_value           DECIMAL(18,6)   NOT NULL,
    gap_to_target           DECIMAL(18,6)   DEFAULT 0,
    on_track                BOOLEAN         NOT NULL DEFAULT FALSE,
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tcfd_tp_target ON tcfd_app.gl_tcfd_target_progress(target_id);
CREATE INDEX idx_tcfd_tp_date ON tcfd_app.gl_tcfd_target_progress(progress_date DESC);
CREATE INDEX idx_tcfd_tp_on_track ON tcfd_app.gl_tcfd_target_progress(on_track);
CREATE INDEX idx_tcfd_tp_created_at ON tcfd_app.gl_tcfd_target_progress(created_at DESC);

-- =============================================================================
-- Table 19: tcfd_app.gl_tcfd_disclosures
-- =============================================================================
-- TCFD disclosure documents per organization-year.  Tracks lifecycle status,
-- compliance scoring, approval workflow, and publication timestamps.

CREATE TABLE tcfd_app.gl_tcfd_disclosures (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    title                       VARCHAR(500)    NOT NULL,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    status                      tcfd_app.gl_tcfd_disclosure_status_enum NOT NULL DEFAULT 'draft',
    compliance_score            DECIMAL(5,2)    DEFAULT 0,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    published_at                TIMESTAMPTZ,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_disc_title_not_empty CHECK (
        LENGTH(TRIM(title)) > 0
    ),
    CONSTRAINT chk_tcfd_disc_period CHECK (
        reporting_period_end >= reporting_period_start
    ),
    CONSTRAINT chk_tcfd_disc_compliance_range CHECK (
        compliance_score >= 0 AND compliance_score <= 100
    )
);

-- Indexes
CREATE INDEX idx_tcfd_disc_org ON tcfd_app.gl_tcfd_disclosures(org_id);
CREATE INDEX idx_tcfd_disc_status ON tcfd_app.gl_tcfd_disclosures(status);
CREATE INDEX idx_tcfd_disc_period ON tcfd_app.gl_tcfd_disclosures(org_id, reporting_period_start);
CREATE INDEX idx_tcfd_disc_compliance ON tcfd_app.gl_tcfd_disclosures(compliance_score DESC);
CREATE INDEX idx_tcfd_disc_created_at ON tcfd_app.gl_tcfd_disclosures(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_disc_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_disclosures
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 20: tcfd_app.gl_tcfd_disclosure_sections
-- =============================================================================
-- Individual sections within a TCFD disclosure document.  Maps to one of
-- the 11 recommended disclosures (gov_a/b, str_a/b/c, rm_a/b/c, mt_a/b/c)
-- and tracks content, evidence references, and completeness scoring.

CREATE TABLE tcfd_app.gl_tcfd_disclosure_sections (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    disclosure_id           UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_disclosures(id) ON DELETE CASCADE,
    disclosure_code         tcfd_app.gl_tcfd_disclosure_code_enum NOT NULL,
    pillar                  tcfd_app.gl_tcfd_pillar_enum NOT NULL,
    content                 TEXT            DEFAULT '',
    evidence_refs           JSONB           DEFAULT '[]',
    completeness_pct        DECIMAL(5,2)    DEFAULT 0,
    last_edited_by          UUID,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_ds_completeness_range CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    UNIQUE(disclosure_id, disclosure_code)
);

-- Indexes
CREATE INDEX idx_tcfd_ds_disclosure ON tcfd_app.gl_tcfd_disclosure_sections(disclosure_id);
CREATE INDEX idx_tcfd_ds_code ON tcfd_app.gl_tcfd_disclosure_sections(disclosure_code);
CREATE INDEX idx_tcfd_ds_pillar ON tcfd_app.gl_tcfd_disclosure_sections(pillar);
CREATE INDEX idx_tcfd_ds_completeness ON tcfd_app.gl_tcfd_disclosure_sections(completeness_pct DESC);
CREATE INDEX idx_tcfd_ds_created_at ON tcfd_app.gl_tcfd_disclosure_sections(created_at DESC);
CREATE INDEX idx_tcfd_ds_evidence ON tcfd_app.gl_tcfd_disclosure_sections USING GIN(evidence_refs);

-- Updated_at trigger
CREATE TRIGGER trg_tcfd_ds_updated_at
    BEFORE UPDATE ON tcfd_app.gl_tcfd_disclosure_sections
    FOR EACH ROW
    EXECUTE FUNCTION tcfd_app.set_updated_at();

-- =============================================================================
-- Table 21: tcfd_app.gl_tcfd_gap_assessments
-- =============================================================================
-- Gap analysis results evaluating organizational readiness across all four
-- TCFD pillars.  Identifies maturity gaps and generates action plans.

CREATE TABLE tcfd_app.gl_tcfd_gap_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    governance_score        DECIMAL(5,2)    DEFAULT 0,
    strategy_score          DECIMAL(5,2)    DEFAULT 0,
    risk_mgmt_score         DECIMAL(5,2)    DEFAULT 0,
    metrics_score           DECIMAL(5,2)    DEFAULT 0,
    overall_score           DECIMAL(5,2)    DEFAULT 0,
    gaps                    JSONB           DEFAULT '[]',
    action_plan             JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_gap_governance_range CHECK (
        governance_score >= 0 AND governance_score <= 5
    ),
    CONSTRAINT chk_tcfd_gap_strategy_range CHECK (
        strategy_score >= 0 AND strategy_score <= 5
    ),
    CONSTRAINT chk_tcfd_gap_rm_range CHECK (
        risk_mgmt_score >= 0 AND risk_mgmt_score <= 5
    ),
    CONSTRAINT chk_tcfd_gap_metrics_range CHECK (
        metrics_score >= 0 AND metrics_score <= 5
    ),
    CONSTRAINT chk_tcfd_gap_overall_range CHECK (
        overall_score >= 0 AND overall_score <= 5
    )
);

-- Indexes
CREATE INDEX idx_tcfd_gap_org ON tcfd_app.gl_tcfd_gap_assessments(org_id);
CREATE INDEX idx_tcfd_gap_date ON tcfd_app.gl_tcfd_gap_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_tcfd_gap_overall ON tcfd_app.gl_tcfd_gap_assessments(overall_score DESC);
CREATE INDEX idx_tcfd_gap_created_at ON tcfd_app.gl_tcfd_gap_assessments(created_at DESC);
CREATE INDEX idx_tcfd_gap_gaps ON tcfd_app.gl_tcfd_gap_assessments USING GIN(gaps);
CREATE INDEX idx_tcfd_gap_actions ON tcfd_app.gl_tcfd_gap_assessments USING GIN(action_plan);

-- =============================================================================
-- Table 22: tcfd_app.gl_tcfd_issb_mappings
-- =============================================================================
-- TCFD-to-ISSB/IFRS S2 cross-walk mapping.  Tracks disclosure code alignment,
-- compliance status, gap descriptions, and migration actions to transition
-- from TCFD to ISSB reporting requirements.

CREATE TABLE tcfd_app.gl_tcfd_issb_mappings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES tcfd_app.gl_tcfd_organizations(id) ON DELETE CASCADE,
    tcfd_disclosure_code    VARCHAR(10)     NOT NULL,
    issb_paragraph          VARCHAR(20)     NOT NULL,
    compliance_status       VARCHAR(30)     NOT NULL DEFAULT 'fully_mapped',
    gap_description         TEXT,
    migration_action        TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tcfd_issb_code_not_empty CHECK (
        LENGTH(TRIM(tcfd_disclosure_code)) > 0
    ),
    CONSTRAINT chk_tcfd_issb_paragraph_not_empty CHECK (
        LENGTH(TRIM(issb_paragraph)) > 0
    ),
    CONSTRAINT chk_tcfd_issb_compliance CHECK (
        compliance_status IN ('fully_mapped', 'enhanced', 'partial', 'gap')
    )
);

-- Indexes
CREATE INDEX idx_tcfd_issb_org ON tcfd_app.gl_tcfd_issb_mappings(org_id);
CREATE INDEX idx_tcfd_issb_code ON tcfd_app.gl_tcfd_issb_mappings(tcfd_disclosure_code);
CREATE INDEX idx_tcfd_issb_paragraph ON tcfd_app.gl_tcfd_issb_mappings(issb_paragraph);
CREATE INDEX idx_tcfd_issb_compliance ON tcfd_app.gl_tcfd_issb_mappings(compliance_status);
CREATE INDEX idx_tcfd_issb_created_at ON tcfd_app.gl_tcfd_issb_mappings(created_at DESC);

-- =============================================================================
-- Continuous Aggregate: tcfd_app.quarterly_risk_scores
-- =============================================================================
-- Precomputed quarterly aggregated physical risk scores by organization
-- derived from the physical_risk_assessments hypertable for dashboard
-- and trend analysis.

CREATE MATERIALIZED VIEW tcfd_app.quarterly_risk_scores
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('3 months', assessment_date)    AS bucket,
    asset_id,
    hazard_type,
    AVG(composite_risk_score)                   AS avg_risk_score,
    MAX(composite_risk_score)                   AS max_risk_score,
    MIN(composite_risk_score)                   AS min_risk_score,
    AVG(damage_estimate_pct)                    AS avg_damage_pct,
    COUNT(*)                                    AS assessment_count
FROM tcfd_app.gl_tcfd_physical_risk_assessments
GROUP BY bucket, asset_id, hazard_type
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('tcfd_app.quarterly_risk_scores',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate: tcfd_app.annual_scenario_summary
-- =============================================================================
-- Precomputed annual scenario result averages by organization and scenario
-- for strategic planning and year-over-year comparison dashboards.

CREATE MATERIALIZED VIEW tcfd_app.annual_scenario_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 year', analysis_date)        AS bucket,
    org_id,
    scenario_id,
    AVG(revenue_impact_pct)                     AS avg_revenue_impact,
    AVG(cost_impact_abs)                        AS avg_cost_impact,
    AVG(asset_impairment_pct)                   AS avg_impairment,
    AVG(capex_requirement)                      AS avg_capex,
    AVG(carbon_cost)                            AS avg_carbon_cost,
    AVG(resilience_score)                       AS avg_resilience,
    COUNT(*)                                    AS analysis_count
FROM tcfd_app.gl_tcfd_scenario_results
GROUP BY bucket, org_id, scenario_id
WITH NO DATA;

-- Refresh policy: every hour, covering last 3 days
SELECT add_continuous_aggregate_policy('tcfd_app.annual_scenario_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep scenario results for 3650 days (10 years, regulatory retention)
SELECT add_retention_policy('tcfd_app.gl_tcfd_scenario_results', INTERVAL '3650 days');

-- Keep physical risk assessments for 3650 days (10 years)
SELECT add_retention_policy('tcfd_app.gl_tcfd_physical_risk_assessments', INTERVAL '3650 days');

-- Keep financial impacts for 3650 days (10 years)
SELECT add_retention_policy('tcfd_app.gl_tcfd_financial_impacts', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on scenario_results after 90 days
ALTER TABLE tcfd_app.gl_tcfd_scenario_results SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'analysis_date DESC'
);

SELECT add_compression_policy('tcfd_app.gl_tcfd_scenario_results', INTERVAL '90 days');

-- Enable compression on physical_risk_assessments after 90 days
ALTER TABLE tcfd_app.gl_tcfd_physical_risk_assessments SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'assessment_date DESC'
);

SELECT add_compression_policy('tcfd_app.gl_tcfd_physical_risk_assessments', INTERVAL '90 days');

-- Enable compression on financial_impacts after 90 days
ALTER TABLE tcfd_app.gl_tcfd_financial_impacts SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'impact_date DESC'
);

SELECT add_compression_policy('tcfd_app.gl_tcfd_financial_impacts', INTERVAL '90 days');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA tcfd_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA tcfd_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA tcfd_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON tcfd_app.quarterly_risk_scores TO greenlang_app;
GRANT SELECT ON tcfd_app.annual_scenario_summary TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA tcfd_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA tcfd_app TO greenlang_readonly;
        GRANT SELECT ON tcfd_app.quarterly_risk_scores TO greenlang_readonly;
        GRANT SELECT ON tcfd_app.annual_scenario_summary TO greenlang_readonly;
    END IF;
END
$$;

-- Add TCFD app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'tcfd_app:organizations:read', 'tcfd_app', 'organizations_read', 'View TCFD organization profiles'),
    (gen_random_uuid(), 'tcfd_app:organizations:write', 'tcfd_app', 'organizations_write', 'Create and manage TCFD organization profiles'),
    (gen_random_uuid(), 'tcfd_app:governance:read', 'tcfd_app', 'governance_read', 'View TCFD governance assessments'),
    (gen_random_uuid(), 'tcfd_app:governance:write', 'tcfd_app', 'governance_write', 'Create and manage TCFD governance assessments'),
    (gen_random_uuid(), 'tcfd_app:risks:read', 'tcfd_app', 'risks_read', 'View TCFD climate risks'),
    (gen_random_uuid(), 'tcfd_app:risks:write', 'tcfd_app', 'risks_write', 'Create and manage TCFD climate risks'),
    (gen_random_uuid(), 'tcfd_app:opportunities:read', 'tcfd_app', 'opportunities_read', 'View TCFD climate opportunities'),
    (gen_random_uuid(), 'tcfd_app:opportunities:write', 'tcfd_app', 'opportunities_write', 'Create and manage TCFD climate opportunities'),
    (gen_random_uuid(), 'tcfd_app:scenarios:read', 'tcfd_app', 'scenarios_read', 'View TCFD scenario definitions and results'),
    (gen_random_uuid(), 'tcfd_app:scenarios:write', 'tcfd_app', 'scenarios_write', 'Create and manage TCFD scenarios'),
    (gen_random_uuid(), 'tcfd_app:scenarios:run', 'tcfd_app', 'scenarios_run', 'Execute TCFD scenario analyses'),
    (gen_random_uuid(), 'tcfd_app:physical_risk:read', 'tcfd_app', 'physical_risk_read', 'View TCFD physical risk assessments'),
    (gen_random_uuid(), 'tcfd_app:physical_risk:write', 'tcfd_app', 'physical_risk_write', 'Create and manage TCFD physical risk assessments'),
    (gen_random_uuid(), 'tcfd_app:transition_risk:read', 'tcfd_app', 'transition_risk_read', 'View TCFD transition risk assessments'),
    (gen_random_uuid(), 'tcfd_app:transition_risk:write', 'tcfd_app', 'transition_risk_write', 'Create and manage TCFD transition risk assessments'),
    (gen_random_uuid(), 'tcfd_app:financial_impact:read', 'tcfd_app', 'financial_impact_read', 'View TCFD financial impacts'),
    (gen_random_uuid(), 'tcfd_app:financial_impact:write', 'tcfd_app', 'financial_impact_write', 'Create and manage TCFD financial impacts'),
    (gen_random_uuid(), 'tcfd_app:risk_mgmt:read', 'tcfd_app', 'risk_mgmt_read', 'View TCFD risk management records'),
    (gen_random_uuid(), 'tcfd_app:risk_mgmt:write', 'tcfd_app', 'risk_mgmt_write', 'Create and manage TCFD risk management records'),
    (gen_random_uuid(), 'tcfd_app:metrics:read', 'tcfd_app', 'metrics_read', 'View TCFD climate metrics'),
    (gen_random_uuid(), 'tcfd_app:metrics:write', 'tcfd_app', 'metrics_write', 'Create and manage TCFD climate metrics'),
    (gen_random_uuid(), 'tcfd_app:targets:read', 'tcfd_app', 'targets_read', 'View TCFD climate targets and progress'),
    (gen_random_uuid(), 'tcfd_app:targets:write', 'tcfd_app', 'targets_write', 'Create and manage TCFD climate targets'),
    (gen_random_uuid(), 'tcfd_app:disclosures:read', 'tcfd_app', 'disclosures_read', 'View TCFD disclosure documents'),
    (gen_random_uuid(), 'tcfd_app:disclosures:write', 'tcfd_app', 'disclosures_write', 'Create and manage TCFD disclosure documents'),
    (gen_random_uuid(), 'tcfd_app:disclosures:approve', 'tcfd_app', 'disclosures_approve', 'Approve TCFD disclosure documents'),
    (gen_random_uuid(), 'tcfd_app:disclosures:publish', 'tcfd_app', 'disclosures_publish', 'Publish TCFD disclosure documents'),
    (gen_random_uuid(), 'tcfd_app:gaps:read', 'tcfd_app', 'gaps_read', 'View TCFD gap assessments'),
    (gen_random_uuid(), 'tcfd_app:gaps:write', 'tcfd_app', 'gaps_write', 'Run and manage TCFD gap assessments'),
    (gen_random_uuid(), 'tcfd_app:issb:read', 'tcfd_app', 'issb_read', 'View TCFD-to-ISSB cross-walk mappings'),
    (gen_random_uuid(), 'tcfd_app:issb:write', 'tcfd_app', 'issb_write', 'Manage TCFD-to-ISSB cross-walk mappings'),
    (gen_random_uuid(), 'tcfd_app:reports:read', 'tcfd_app', 'reports_read', 'View TCFD generated reports'),
    (gen_random_uuid(), 'tcfd_app:reports:generate', 'tcfd_app', 'reports_generate', 'Generate TCFD reports and exports'),
    (gen_random_uuid(), 'tcfd_app:dashboard:read', 'tcfd_app', 'dashboard_read', 'View TCFD dashboards and analytics'),
    (gen_random_uuid(), 'tcfd_app:admin', 'tcfd_app', 'admin', 'TCFD application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA tcfd_app IS 'GL-TCFD-APP v1.0 Application Schema - TCFD Disclosure & Scenario Analysis Platform with governance assessments, climate risk/opportunity analysis, scenario modeling (IEA/NGFS), physical/transition risk assessment, financial impact quantification, risk management integration, metrics & targets tracking, disclosure generation, ISSB/IFRS S2 cross-walk, and gap analysis';

COMMENT ON TABLE tcfd_app.gl_tcfd_organizations IS 'Organization profiles for TCFD disclosure with sector, industry, jurisdiction, and reporting currency';
COMMENT ON TABLE tcfd_app.gl_tcfd_governance_assessments IS 'TCFD Pillar 1 governance assessments with board oversight, management roles, maturity scoring, competency matrix, and incentive linkages';
COMMENT ON TABLE tcfd_app.gl_tcfd_governance_roles IS 'Climate governance role assignments with title, holder, responsibilities, accountability flag, and reporting line';
COMMENT ON TABLE tcfd_app.gl_tcfd_climate_risks IS 'Climate-related risks (physical/transition) with likelihood/impact scoring, financial impact estimates, and strategic responses per TCFD Strategy (a)';
COMMENT ON TABLE tcfd_app.gl_tcfd_climate_opportunities IS 'Climate-related opportunities across 5 TCFD categories with revenue/savings potential, investment requirements, and pipeline tracking';
COMMENT ON TABLE tcfd_app.gl_tcfd_scenarios IS 'Scenario definitions supporting 7 pre-built (IEA/NGFS) plus custom scenarios with carbon price and energy mix trajectories';
COMMENT ON TABLE tcfd_app.gl_tcfd_scenario_results IS 'TimescaleDB hypertable: Scenario analysis results with revenue/cost/asset impacts, carbon cost, capex requirements, and resilience scores';
COMMENT ON TABLE tcfd_app.gl_tcfd_scenario_parameters IS 'Scenario parameter values with yearly projections for carbon prices, energy mix, technology adoption, and regulatory milestones';
COMMENT ON TABLE tcfd_app.gl_tcfd_physical_risk_assessments IS 'TimescaleDB hypertable: Physical risk assessments per asset with exposure/vulnerability/adaptive capacity scoring and damage estimates by hazard type';
COMMENT ON TABLE tcfd_app.gl_tcfd_asset_locations IS 'Physical asset locations with geographic coordinates, type, replacement value, elevation, and building characteristics for climate risk mapping';
COMMENT ON TABLE tcfd_app.gl_tcfd_transition_risk_assessments IS 'Transition risk assessments covering policy/technology/market/reputation sub-categories with composite scoring and stranding probability';
COMMENT ON TABLE tcfd_app.gl_tcfd_financial_impacts IS 'TimescaleDB hypertable: Climate financial impacts on income statement, balance sheet, and cash flow per TCFD/IFRS S2 requirements';
COMMENT ON TABLE tcfd_app.gl_tcfd_risk_management_records IS 'Risk management records per TCFD Pillar 3 with ERM framework, risk appetite, review frequency, and register configuration';
COMMENT ON TABLE tcfd_app.gl_tcfd_risk_responses IS 'Risk response actions (accept/mitigate/transfer/avoid) with ownership, progress tracking, and cost estimates';
COMMENT ON TABLE tcfd_app.gl_tcfd_metrics IS 'Climate metric definitions per TCFD/ISSB para 29 with SASB codes, cross-industry flags, and metric categories';
COMMENT ON TABLE tcfd_app.gl_tcfd_metric_values IS 'Metric period values with data quality tiers, source tracking, and verification status';
COMMENT ON TABLE tcfd_app.gl_tcfd_targets IS 'Climate targets (absolute/intensity/net-zero/renewable) with base/target year values, SBTi alignment, and pathway specification';
COMMENT ON TABLE tcfd_app.gl_tcfd_target_progress IS 'Target progress tracking with current values, gap to target, on-track assessment per reporting period';
COMMENT ON TABLE tcfd_app.gl_tcfd_disclosures IS 'TCFD disclosure documents with lifecycle status (draft/review/approved/published), compliance scoring, and approval workflow';
COMMENT ON TABLE tcfd_app.gl_tcfd_disclosure_sections IS 'Disclosure sections mapping to 11 TCFD recommended disclosures (gov_a/b, str_a/b/c, rm_a/b/c, mt_a/b/c) with evidence and completeness scoring';
COMMENT ON TABLE tcfd_app.gl_tcfd_gap_assessments IS 'Gap analysis results evaluating organizational readiness across all four TCFD pillars with gaps identification and action plans';
COMMENT ON TABLE tcfd_app.gl_tcfd_issb_mappings IS 'TCFD-to-ISSB/IFRS S2 cross-walk mapping with compliance status, gap descriptions, and migration actions';

COMMENT ON MATERIALIZED VIEW tcfd_app.quarterly_risk_scores IS 'Continuous aggregate: quarterly physical risk score averages by asset and hazard type for trend analysis';
COMMENT ON MATERIALIZED VIEW tcfd_app.annual_scenario_summary IS 'Continuous aggregate: annual scenario result averages by organization and scenario for strategic planning';

COMMENT ON COLUMN tcfd_app.gl_tcfd_climate_risks.risk_type IS 'TCFD risk type: physical_acute, physical_chronic, transition_policy, transition_technology, transition_market, transition_reputation';
COMMENT ON COLUMN tcfd_app.gl_tcfd_climate_risks.time_horizon IS 'TCFD time horizon: short_term (0-3y), medium_term (3-10y), long_term (10-30+y)';
COMMENT ON COLUMN tcfd_app.gl_tcfd_climate_risks.likelihood IS 'Likelihood score 1-5: rare, unlikely, possible, likely, almost_certain';
COMMENT ON COLUMN tcfd_app.gl_tcfd_climate_risks.impact IS 'Impact score 1-5: insignificant, minor, moderate, major, catastrophic';
COMMENT ON COLUMN tcfd_app.gl_tcfd_scenarios.scenario_type IS 'Scenario archetype: iea_nze, iea_aps, iea_steps, ngfs_current_policies, ngfs_delayed_transition, ngfs_below_2c, ngfs_divergent_nz, custom';
COMMENT ON COLUMN tcfd_app.gl_tcfd_disclosures.status IS 'Disclosure lifecycle: draft, review, approved, published';
COMMENT ON COLUMN tcfd_app.gl_tcfd_disclosure_sections.disclosure_code IS 'One of 11 TCFD disclosure codes: gov_a, gov_b, str_a, str_b, str_c, rm_a, rm_b, rm_c, mt_a, mt_b, mt_c';
COMMENT ON COLUMN tcfd_app.gl_tcfd_disclosure_sections.pillar IS 'TCFD pillar: governance, strategy, risk_management, metrics_targets';
COMMENT ON COLUMN tcfd_app.gl_tcfd_targets.target_type IS 'Target type: absolute, intensity, net_zero, renewable_energy, custom';
COMMENT ON COLUMN tcfd_app.gl_tcfd_issb_mappings.compliance_status IS 'Mapping status: fully_mapped, enhanced, partial, gap';
