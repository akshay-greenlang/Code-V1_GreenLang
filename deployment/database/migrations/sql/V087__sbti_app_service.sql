-- =============================================================================
-- V087: GL-SBTi-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-SBTi-APP (SBTi Target Setting & Validation Platform)
-- Date:        March 2026
--
-- Application-level tables for the Science Based Targets initiative (SBTi)
-- target setting, validation, and progress tracking platform.  Covers
-- organization profiling, emissions inventories, near-term/long-term/net-zero
-- target definitions, ACA/SDA pathway calculation, criteria validation (C1-C28,
-- NZ-C1-NZ-C14), Scope 3 screening, FLAG assessments, sector pathways,
-- progress tracking, temperature scoring, base year recalculations,
-- five-year reviews, financial institution portfolios, framework mappings,
-- gap analysis, and reporting.
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--   V081: Audit Trail & Lineage Service
--   V083: GL-GHG-APP v1.0
--   V084: GL-ISO14064-APP v1.0
--   V085: GL-CDP-APP v1.0
--   V086: GL-TCFD-APP v1.0
--
-- These tables sit in the sbti_app schema and integrate with the underlying
-- MRV agent data for auto-population of Scope 1/2/3 emissions into SBTi
-- target setting, pathway calculation, and progress tracking.
-- =============================================================================
-- Tables (25):
--   1.  gl_sbti_organizations            - Organization profiles
--   2.  gl_sbti_emissions_inventories    - Base year and annual emissions
--   3.  gl_sbti_targets                  - Target definitions
--   4.  gl_sbti_target_scopes            - Per-scope target details
--   5.  gl_sbti_pathways                 - Calculated reduction pathways
--   6.  gl_sbti_pathway_milestones       - Annual milestone points
--   7.  gl_sbti_validation_results       - Criteria validation results (HT)
--   8.  gl_sbti_criteria_checks          - Individual criterion results
--   9.  gl_sbti_scope3_screenings        - Scope 3 screening results
--  10.  gl_sbti_scope3_categories        - Per-category breakdown
--  11.  gl_sbti_flag_assessments         - FLAG assessments
--  12.  gl_sbti_flag_commodities         - Commodity-level FLAG data
--  13.  gl_sbti_sector_pathways          - Sector intensity pathways
--  14.  gl_sbti_sector_benchmarks        - Sector benchmark data
--  15.  gl_sbti_progress_records         - Annual progress tracking (HT)
--  16.  gl_sbti_temperature_scores       - Temperature alignment (HT)
--  17.  gl_sbti_recalculations           - Base year recalculations
--  18.  gl_sbti_five_year_reviews        - Review records
--  19.  gl_sbti_fi_portfolios            - FI portfolios
--  20.  gl_sbti_fi_portfolio_holdings    - Portfolio holdings
--  21.  gl_sbti_fi_engagement            - Investee engagement
--  22.  gl_sbti_framework_mappings       - Cross-framework alignment
--  23.  gl_sbti_reports                  - Generated reports
--  24.  gl_sbti_gap_assessments          - Gap analysis results
--  25.  gl_sbti_gap_items                - Individual gap items
--
-- Hypertables (3):
--   gl_sbti_validation_results   on validation_date  (chunk: 3 months)
--   gl_sbti_progress_records     on tracking_date    (chunk: 3 months)
--   gl_sbti_temperature_scores   on score_date       (chunk: 3 months)
--
-- Continuous Aggregates (2):
--   sbti_app.annual_progress_summary       - Annual progress aggregation
--   sbti_app.quarterly_temperature_trends  - Quarterly temperature trends
--
-- Also includes: 100+ indexes (B-tree, GIN), update triggers, security
-- grants, retention policies, compression policies, permissions, and comments.
-- Previous: V086__tcfd_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS sbti_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION sbti_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: sbti_app.gl_sbti_organizations
-- =============================================================================
-- Organization profiles for SBTi target setting.  Each organization represents
-- the top-level entity that sets science-based targets, including sector
-- classification codes (ISIC/NACE/NAICS), financial metrics, and OECD status
-- for pathway determination.

CREATE TABLE sbti_app.gl_sbti_organizations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    name                    VARCHAR(500)    NOT NULL,
    sector_classification   VARCHAR(50),
    isic_code               VARCHAR(10),
    nace_code               VARCHAR(10),
    naics_code              VARCHAR(10),
    country                 VARCHAR(3),
    oecd_status             BOOLEAN,
    annual_revenue          DECIMAL(18,2),
    market_cap              DECIMAL(18,2),
    enterprise_value        DECIMAL(18,2),
    total_assets            DECIMAL(18,2),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_org_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_sbti_org_country_length CHECK (
        country IS NULL OR (LENGTH(TRIM(country)) >= 2 AND LENGTH(TRIM(country)) <= 3)
    ),
    CONSTRAINT chk_sbti_org_revenue_non_neg CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_sbti_org_market_cap_non_neg CHECK (
        market_cap IS NULL OR market_cap >= 0
    ),
    CONSTRAINT chk_sbti_org_ev_non_neg CHECK (
        enterprise_value IS NULL OR enterprise_value >= 0
    ),
    CONSTRAINT chk_sbti_org_assets_non_neg CHECK (
        total_assets IS NULL OR total_assets >= 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_org_tenant ON sbti_app.gl_sbti_organizations(tenant_id);
CREATE INDEX idx_sbti_org_name ON sbti_app.gl_sbti_organizations(name);
CREATE INDEX idx_sbti_org_sector ON sbti_app.gl_sbti_organizations(sector_classification);
CREATE INDEX idx_sbti_org_isic ON sbti_app.gl_sbti_organizations(isic_code);
CREATE INDEX idx_sbti_org_nace ON sbti_app.gl_sbti_organizations(nace_code);
CREATE INDEX idx_sbti_org_naics ON sbti_app.gl_sbti_organizations(naics_code);
CREATE INDEX idx_sbti_org_country ON sbti_app.gl_sbti_organizations(country);
CREATE INDEX idx_sbti_org_oecd ON sbti_app.gl_sbti_organizations(oecd_status);
CREATE INDEX idx_sbti_org_created_at ON sbti_app.gl_sbti_organizations(created_at DESC);
CREATE INDEX idx_sbti_org_metadata ON sbti_app.gl_sbti_organizations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_org_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_organizations
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 2: sbti_app.gl_sbti_emissions_inventories
-- =============================================================================
-- Base year and annual GHG emissions inventories.  Stores Scope 1, Scope 2
-- (location and market), Scope 3 total and per-category (Cat 1-15) emissions,
-- FLAG and bioenergy emissions, data quality tier, and verification status.

CREATE TABLE sbti_app.gl_sbti_emissions_inventories (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    reporting_year          INTEGER         NOT NULL,
    is_base_year            BOOLEAN         NOT NULL DEFAULT FALSE,
    scope1_emissions        DECIMAL(18,6)   DEFAULT 0,
    scope2_location         DECIMAL(18,6)   DEFAULT 0,
    scope2_market           DECIMAL(18,6)   DEFAULT 0,
    scope3_total            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat1             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat2             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat3             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat4             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat5             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat6             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat7             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat8             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat9             DECIMAL(18,6)   DEFAULT 0,
    scope3_cat10            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat11            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat12            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat13            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat14            DECIMAL(18,6)   DEFAULT 0,
    scope3_cat15            DECIMAL(18,6)   DEFAULT 0,
    flag_emissions          DECIMAL(18,6)   DEFAULT 0,
    bioenergy_emissions     DECIMAL(18,6)   DEFAULT 0,
    total_emissions         DECIMAL(18,6)   DEFAULT 0,
    scope3_percentage       DECIMAL(6,2)    DEFAULT 0,
    flag_percentage         DECIMAL(6,2)    DEFAULT 0,
    data_quality_tier       VARCHAR(20),
    verification_status     VARCHAR(30)     NOT NULL DEFAULT 'unverified',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_ei_year_range CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_sbti_ei_scope1_non_neg CHECK (
        scope1_emissions IS NULL OR scope1_emissions >= 0
    ),
    CONSTRAINT chk_sbti_ei_scope2l_non_neg CHECK (
        scope2_location IS NULL OR scope2_location >= 0
    ),
    CONSTRAINT chk_sbti_ei_scope2m_non_neg CHECK (
        scope2_market IS NULL OR scope2_market >= 0
    ),
    CONSTRAINT chk_sbti_ei_scope3_non_neg CHECK (
        scope3_total IS NULL OR scope3_total >= 0
    ),
    CONSTRAINT chk_sbti_ei_total_non_neg CHECK (
        total_emissions IS NULL OR total_emissions >= 0
    ),
    CONSTRAINT chk_sbti_ei_s3_pct_range CHECK (
        scope3_percentage >= 0 AND scope3_percentage <= 100
    ),
    CONSTRAINT chk_sbti_ei_flag_pct_range CHECK (
        flag_percentage >= 0 AND flag_percentage <= 100
    ),
    CONSTRAINT chk_sbti_ei_quality CHECK (
        data_quality_tier IS NULL OR data_quality_tier IN ('tier1', 'tier2', 'tier3', 'estimated', 'screening')
    ),
    CONSTRAINT chk_sbti_ei_verification CHECK (
        verification_status IN ('unverified', 'limited_assurance', 'reasonable_assurance', 'verified', 'third_party')
    ),
    UNIQUE(org_id, reporting_year)
);

-- Indexes
CREATE INDEX idx_sbti_ei_org ON sbti_app.gl_sbti_emissions_inventories(org_id);
CREATE INDEX idx_sbti_ei_tenant ON sbti_app.gl_sbti_emissions_inventories(tenant_id);
CREATE INDEX idx_sbti_ei_year ON sbti_app.gl_sbti_emissions_inventories(reporting_year);
CREATE INDEX idx_sbti_ei_base_year ON sbti_app.gl_sbti_emissions_inventories(org_id, is_base_year);
CREATE INDEX idx_sbti_ei_verification ON sbti_app.gl_sbti_emissions_inventories(verification_status);
CREATE INDEX idx_sbti_ei_created_at ON sbti_app.gl_sbti_emissions_inventories(created_at DESC);
CREATE INDEX idx_sbti_ei_metadata ON sbti_app.gl_sbti_emissions_inventories USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_ei_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_emissions_inventories
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 3: sbti_app.gl_sbti_targets
-- =============================================================================
-- SBTi target definitions covering near-term, long-term, and net-zero targets.
-- Tracks target method (ACA/SDA/economic_intensity/physical_intensity/
-- supplier_engagement), pathway alignment (1.5C/well-below-2C), scope
-- coverage percentages, and full lifecycle dates (commitment through review).

CREATE TABLE sbti_app.gl_sbti_targets (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    target_type             VARCHAR(20)     NOT NULL,
    target_status           VARCHAR(20)     NOT NULL DEFAULT 'committed',
    target_method           VARCHAR(30)     NOT NULL,
    pathway_alignment       VARCHAR(20)     NOT NULL DEFAULT '1_5c',
    base_year               INTEGER         NOT NULL,
    target_year             INTEGER         NOT NULL,
    base_year_emissions     DECIMAL(18,6)   NOT NULL DEFAULT 0,
    target_year_emissions   DECIMAL(18,6)   DEFAULT 0,
    reduction_percentage    DECIMAL(6,2)    DEFAULT 0,
    annual_reduction_rate   DECIMAL(8,4)    DEFAULT 0,
    scope1_2_coverage_pct   DECIMAL(6,2)    DEFAULT 0,
    scope3_coverage_pct     DECIMAL(6,2)    DEFAULT 0,
    is_flag_target          BOOLEAN         NOT NULL DEFAULT FALSE,
    commitment_date         DATE,
    submission_date         DATE,
    validation_date         DATE,
    publication_date        DATE,
    next_review_date        DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_tgt_type CHECK (
        target_type IN ('near_term', 'long_term', 'net_zero')
    ),
    CONSTRAINT chk_sbti_tgt_status CHECK (
        target_status IN ('committed', 'submitted', 'validated', 'published', 'expired', 'withdrawn')
    ),
    CONSTRAINT chk_sbti_tgt_method CHECK (
        target_method IN ('aca', 'sda', 'economic_intensity', 'physical_intensity', 'supplier_engagement')
    ),
    CONSTRAINT chk_sbti_tgt_pathway CHECK (
        pathway_alignment IN ('1_5c', 'well_below_2c')
    ),
    CONSTRAINT chk_sbti_tgt_base_year_range CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_sbti_tgt_target_year_range CHECK (
        target_year >= 2000 AND target_year <= 2100
    ),
    CONSTRAINT chk_sbti_tgt_target_after_base CHECK (
        target_year > base_year
    ),
    CONSTRAINT chk_sbti_tgt_base_emissions_non_neg CHECK (
        base_year_emissions >= 0
    ),
    CONSTRAINT chk_sbti_tgt_reduction_range CHECK (
        reduction_percentage >= 0 AND reduction_percentage <= 100
    ),
    CONSTRAINT chk_sbti_tgt_annual_rate_non_neg CHECK (
        annual_reduction_rate >= 0
    ),
    CONSTRAINT chk_sbti_tgt_s12_coverage_range CHECK (
        scope1_2_coverage_pct >= 0 AND scope1_2_coverage_pct <= 100
    ),
    CONSTRAINT chk_sbti_tgt_s3_coverage_range CHECK (
        scope3_coverage_pct >= 0 AND scope3_coverage_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_sbti_tgt_org ON sbti_app.gl_sbti_targets(org_id);
CREATE INDEX idx_sbti_tgt_tenant ON sbti_app.gl_sbti_targets(tenant_id);
CREATE INDEX idx_sbti_tgt_type ON sbti_app.gl_sbti_targets(target_type);
CREATE INDEX idx_sbti_tgt_status ON sbti_app.gl_sbti_targets(target_status);
CREATE INDEX idx_sbti_tgt_method ON sbti_app.gl_sbti_targets(target_method);
CREATE INDEX idx_sbti_tgt_pathway ON sbti_app.gl_sbti_targets(pathway_alignment);
CREATE INDEX idx_sbti_tgt_base_year ON sbti_app.gl_sbti_targets(base_year);
CREATE INDEX idx_sbti_tgt_target_year ON sbti_app.gl_sbti_targets(target_year);
CREATE INDEX idx_sbti_tgt_flag ON sbti_app.gl_sbti_targets(is_flag_target);
CREATE INDEX idx_sbti_tgt_created_at ON sbti_app.gl_sbti_targets(created_at DESC);
CREATE INDEX idx_sbti_tgt_metadata ON sbti_app.gl_sbti_targets USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_tgt_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_targets
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 4: sbti_app.gl_sbti_target_scopes
-- =============================================================================
-- Per-scope target details breaking down each target into scope-level
-- components.  Tracks base/target emissions, coverage percentage, reduction
-- percentage, and intensity metrics where applicable.

CREATE TABLE sbti_app.gl_sbti_target_scopes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id               UUID            NOT NULL REFERENCES sbti_app.gl_sbti_targets(id) ON DELETE CASCADE,
    tenant_id               UUID,
    scope                   VARCHAR(20)     NOT NULL,
    base_emissions          DECIMAL(18,6)   DEFAULT 0,
    target_emissions        DECIMAL(18,6)   DEFAULT 0,
    coverage_percentage     DECIMAL(6,2)    DEFAULT 0,
    reduction_percentage    DECIMAL(6,2)    DEFAULT 0,
    intensity_metric        VARCHAR(100),
    intensity_base_value    DECIMAL(18,6),
    intensity_target_value  DECIMAL(18,6),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_ts_scope CHECK (
        scope IN ('scope1', 'scope2', 'scope1_2', 'scope3', 'flag')
    ),
    CONSTRAINT chk_sbti_ts_base_non_neg CHECK (
        base_emissions IS NULL OR base_emissions >= 0
    ),
    CONSTRAINT chk_sbti_ts_target_non_neg CHECK (
        target_emissions IS NULL OR target_emissions >= 0
    ),
    CONSTRAINT chk_sbti_ts_coverage_range CHECK (
        coverage_percentage >= 0 AND coverage_percentage <= 100
    ),
    CONSTRAINT chk_sbti_ts_reduction_range CHECK (
        reduction_percentage >= 0 AND reduction_percentage <= 100
    )
);

-- Indexes
CREATE INDEX idx_sbti_ts_target ON sbti_app.gl_sbti_target_scopes(target_id);
CREATE INDEX idx_sbti_ts_tenant ON sbti_app.gl_sbti_target_scopes(tenant_id);
CREATE INDEX idx_sbti_ts_scope ON sbti_app.gl_sbti_target_scopes(scope);
CREATE INDEX idx_sbti_ts_created_at ON sbti_app.gl_sbti_target_scopes(created_at DESC);
CREATE INDEX idx_sbti_ts_metadata ON sbti_app.gl_sbti_target_scopes USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_ts_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_target_scopes
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 5: sbti_app.gl_sbti_pathways
-- =============================================================================
-- Calculated reduction pathways for SBTi targets.  Stores the method
-- (ACA/SDA), sector, base/target values, annual linear rate, and full
-- pathway data as JSONB arrays with optional uncertainty bounds.

CREATE TABLE sbti_app.gl_sbti_pathways (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id               UUID            NOT NULL REFERENCES sbti_app.gl_sbti_targets(id) ON DELETE CASCADE,
    tenant_id               UUID,
    method                  VARCHAR(30)     NOT NULL,
    sector                  VARCHAR(50),
    base_year               INTEGER         NOT NULL,
    target_year             INTEGER         NOT NULL,
    base_value              DECIMAL(18,6)   NOT NULL DEFAULT 0,
    target_value            DECIMAL(18,6)   DEFAULT 0,
    annual_rate             DECIMAL(8,4)    DEFAULT 0,
    is_intensity            BOOLEAN         NOT NULL DEFAULT FALSE,
    pathway_data            JSONB           DEFAULT '[]',
    uncertainty_lower       JSONB           DEFAULT '[]',
    uncertainty_upper       JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_pw_method CHECK (
        method IN ('aca', 'sda', 'economic_intensity', 'physical_intensity', 'supplier_engagement')
    ),
    CONSTRAINT chk_sbti_pw_base_year_range CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_sbti_pw_target_year_range CHECK (
        target_year >= 2000 AND target_year <= 2100
    ),
    CONSTRAINT chk_sbti_pw_target_after_base CHECK (
        target_year > base_year
    ),
    CONSTRAINT chk_sbti_pw_base_value_non_neg CHECK (
        base_value >= 0
    ),
    CONSTRAINT chk_sbti_pw_annual_rate_non_neg CHECK (
        annual_rate >= 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_pw_target ON sbti_app.gl_sbti_pathways(target_id);
CREATE INDEX idx_sbti_pw_tenant ON sbti_app.gl_sbti_pathways(tenant_id);
CREATE INDEX idx_sbti_pw_method ON sbti_app.gl_sbti_pathways(method);
CREATE INDEX idx_sbti_pw_sector ON sbti_app.gl_sbti_pathways(sector);
CREATE INDEX idx_sbti_pw_intensity ON sbti_app.gl_sbti_pathways(is_intensity);
CREATE INDEX idx_sbti_pw_created_at ON sbti_app.gl_sbti_pathways(created_at DESC);
CREATE INDEX idx_sbti_pw_pathway_data ON sbti_app.gl_sbti_pathways USING GIN(pathway_data);
CREATE INDEX idx_sbti_pw_metadata ON sbti_app.gl_sbti_pathways USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_pw_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_pathways
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 6: sbti_app.gl_sbti_pathway_milestones
-- =============================================================================
-- Annual milestone points along a reduction pathway.  Stores expected
-- absolute/intensity values and cumulative reduction for each year between
-- base year and target year.

CREATE TABLE sbti_app.gl_sbti_pathway_milestones (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    pathway_id              UUID            NOT NULL REFERENCES sbti_app.gl_sbti_pathways(id) ON DELETE CASCADE,
    tenant_id               UUID,
    milestone_year          INTEGER         NOT NULL,
    expected_value          DECIMAL(18,6)   DEFAULT 0,
    intensity_value         DECIMAL(18,6),
    cumulative_reduction    DECIMAL(6,2)    DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_pm_year_range CHECK (
        milestone_year >= 2000 AND milestone_year <= 2100
    ),
    CONSTRAINT chk_sbti_pm_expected_non_neg CHECK (
        expected_value IS NULL OR expected_value >= 0
    ),
    CONSTRAINT chk_sbti_pm_cumulative_range CHECK (
        cumulative_reduction >= 0 AND cumulative_reduction <= 100
    ),
    UNIQUE(pathway_id, milestone_year)
);

-- Indexes
CREATE INDEX idx_sbti_pm_pathway ON sbti_app.gl_sbti_pathway_milestones(pathway_id);
CREATE INDEX idx_sbti_pm_tenant ON sbti_app.gl_sbti_pathway_milestones(tenant_id);
CREATE INDEX idx_sbti_pm_year ON sbti_app.gl_sbti_pathway_milestones(milestone_year);
CREATE INDEX idx_sbti_pm_created_at ON sbti_app.gl_sbti_pathway_milestones(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_pm_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_pathway_milestones
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 7: sbti_app.gl_sbti_validation_results (HYPERTABLE)
-- =============================================================================
-- Criteria validation results partitioned by validation_date for time-series
-- querying.  Stores aggregate pass/fail/not-applicable counts, readiness
-- percentage, overall result, and detailed criterion-level data.

CREATE TABLE sbti_app.gl_sbti_validation_results (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    target_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID,
    validation_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_criteria          INTEGER         NOT NULL DEFAULT 0,
    passed                  INTEGER         NOT NULL DEFAULT 0,
    failed                  INTEGER         NOT NULL DEFAULT 0,
    not_applicable          INTEGER         NOT NULL DEFAULT 0,
    readiness_percentage    DECIMAL(6,2)    DEFAULT 0,
    overall_result          VARCHAR(20)     NOT NULL DEFAULT 'fail',
    details                 JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_vr_total_non_neg CHECK (total_criteria >= 0),
    CONSTRAINT chk_sbti_vr_passed_non_neg CHECK (passed >= 0),
    CONSTRAINT chk_sbti_vr_failed_non_neg CHECK (failed >= 0),
    CONSTRAINT chk_sbti_vr_na_non_neg CHECK (not_applicable >= 0),
    CONSTRAINT chk_sbti_vr_readiness_range CHECK (
        readiness_percentage >= 0 AND readiness_percentage <= 100
    ),
    CONSTRAINT chk_sbti_vr_result CHECK (
        overall_result IN ('pass', 'fail', 'partial')
    ),
    CONSTRAINT chk_sbti_vr_counts_consistent CHECK (
        total_criteria = passed + failed + not_applicable
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('sbti_app.gl_sbti_validation_results', 'validation_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_sbti_vr_target ON sbti_app.gl_sbti_validation_results(target_id, validation_date DESC);
CREATE INDEX idx_sbti_vr_org ON sbti_app.gl_sbti_validation_results(org_id, validation_date DESC);
CREATE INDEX idx_sbti_vr_tenant ON sbti_app.gl_sbti_validation_results(tenant_id, validation_date DESC);
CREATE INDEX idx_sbti_vr_result ON sbti_app.gl_sbti_validation_results(overall_result, validation_date DESC);
CREATE INDEX idx_sbti_vr_org_target ON sbti_app.gl_sbti_validation_results(org_id, target_id, validation_date DESC);
CREATE INDEX idx_sbti_vr_details ON sbti_app.gl_sbti_validation_results USING GIN(details);

-- =============================================================================
-- Table 8: sbti_app.gl_sbti_criteria_checks
-- =============================================================================
-- Individual criterion validation results linked to a validation run.
-- Covers SBTi corporate criteria (C1-C28) and net-zero criteria (NZ-C1
-- through NZ-C14) with pass/fail/not_applicable/insufficient_data outcomes.

CREATE TABLE sbti_app.gl_sbti_criteria_checks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_id           UUID            NOT NULL,
    tenant_id               UUID,
    criterion_id            VARCHAR(10)     NOT NULL,
    criterion_name          VARCHAR(200)    NOT NULL,
    result                  VARCHAR(20)     NOT NULL,
    message                 TEXT,
    details                 JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_cc_criterion_not_empty CHECK (
        LENGTH(TRIM(criterion_id)) > 0
    ),
    CONSTRAINT chk_sbti_cc_name_not_empty CHECK (
        LENGTH(TRIM(criterion_name)) > 0
    ),
    CONSTRAINT chk_sbti_cc_result CHECK (
        result IN ('pass', 'fail', 'not_applicable', 'insufficient_data')
    )
);

-- Indexes
CREATE INDEX idx_sbti_cc_validation ON sbti_app.gl_sbti_criteria_checks(validation_id);
CREATE INDEX idx_sbti_cc_tenant ON sbti_app.gl_sbti_criteria_checks(tenant_id);
CREATE INDEX idx_sbti_cc_criterion ON sbti_app.gl_sbti_criteria_checks(criterion_id);
CREATE INDEX idx_sbti_cc_result ON sbti_app.gl_sbti_criteria_checks(result);
CREATE INDEX idx_sbti_cc_created_at ON sbti_app.gl_sbti_criteria_checks(created_at DESC);
CREATE INDEX idx_sbti_cc_details ON sbti_app.gl_sbti_criteria_checks USING GIN(details);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_cc_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_criteria_checks
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 9: sbti_app.gl_sbti_scope3_screenings
-- =============================================================================
-- Scope 3 screening results determining whether Scope 3 targets are required
-- (>40% threshold) and which categories to include.  Tracks total Scope 3
-- as a percentage of total emissions, near-term and long-term coverage
-- percentages, and recommended categories.

CREATE TABLE sbti_app.gl_sbti_scope3_screenings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    screening_date          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    scope3_total            DECIMAL(18,6)   DEFAULT 0,
    total_emissions         DECIMAL(18,6)   DEFAULT 0,
    scope3_percentage       DECIMAL(6,2)    DEFAULT 0,
    trigger_met             BOOLEAN         NOT NULL DEFAULT FALSE,
    near_term_coverage      DECIMAL(6,2)    DEFAULT 0,
    long_term_coverage      DECIMAL(6,2)    DEFAULT 0,
    recommended_categories  JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_s3s_scope3_non_neg CHECK (
        scope3_total IS NULL OR scope3_total >= 0
    ),
    CONSTRAINT chk_sbti_s3s_total_non_neg CHECK (
        total_emissions IS NULL OR total_emissions >= 0
    ),
    CONSTRAINT chk_sbti_s3s_pct_range CHECK (
        scope3_percentage >= 0 AND scope3_percentage <= 100
    ),
    CONSTRAINT chk_sbti_s3s_nt_coverage_range CHECK (
        near_term_coverage >= 0 AND near_term_coverage <= 100
    ),
    CONSTRAINT chk_sbti_s3s_lt_coverage_range CHECK (
        long_term_coverage >= 0 AND long_term_coverage <= 100
    )
);

-- Indexes
CREATE INDEX idx_sbti_s3s_org ON sbti_app.gl_sbti_scope3_screenings(org_id);
CREATE INDEX idx_sbti_s3s_tenant ON sbti_app.gl_sbti_scope3_screenings(tenant_id);
CREATE INDEX idx_sbti_s3s_date ON sbti_app.gl_sbti_scope3_screenings(org_id, screening_date DESC);
CREATE INDEX idx_sbti_s3s_trigger ON sbti_app.gl_sbti_scope3_screenings(trigger_met);
CREATE INDEX idx_sbti_s3s_created_at ON sbti_app.gl_sbti_scope3_screenings(created_at DESC);
CREATE INDEX idx_sbti_s3s_categories ON sbti_app.gl_sbti_scope3_screenings USING GIN(recommended_categories);
CREATE INDEX idx_sbti_s3s_metadata ON sbti_app.gl_sbti_scope3_screenings USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_s3s_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_scope3_screenings
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 10: sbti_app.gl_sbti_scope3_categories
-- =============================================================================
-- Per-category breakdown within a Scope 3 screening.  Stores emissions,
-- percentage of total, data quality, and whether the category is included
-- in the target boundary.

CREATE TABLE sbti_app.gl_sbti_scope3_categories (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    screening_id            UUID            NOT NULL REFERENCES sbti_app.gl_sbti_scope3_screenings(id) ON DELETE CASCADE,
    tenant_id               UUID,
    category                VARCHAR(10)     NOT NULL,
    category_name           VARCHAR(200)    NOT NULL,
    emissions               DECIMAL(18,6)   DEFAULT 0,
    percentage              DECIMAL(6,2)    DEFAULT 0,
    data_quality            VARCHAR(20),
    included_in_target      BOOLEAN         NOT NULL DEFAULT FALSE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_s3c_category CHECK (
        category IN ('cat1','cat2','cat3','cat4','cat5','cat6','cat7','cat8',
                      'cat9','cat10','cat11','cat12','cat13','cat14','cat15')
    ),
    CONSTRAINT chk_sbti_s3c_name_not_empty CHECK (
        LENGTH(TRIM(category_name)) > 0
    ),
    CONSTRAINT chk_sbti_s3c_emissions_non_neg CHECK (
        emissions IS NULL OR emissions >= 0
    ),
    CONSTRAINT chk_sbti_s3c_pct_range CHECK (
        percentage >= 0 AND percentage <= 100
    ),
    CONSTRAINT chk_sbti_s3c_quality CHECK (
        data_quality IS NULL OR data_quality IN ('high', 'medium', 'low', 'estimated', 'screening')
    ),
    UNIQUE(screening_id, category)
);

-- Indexes
CREATE INDEX idx_sbti_s3c_screening ON sbti_app.gl_sbti_scope3_categories(screening_id);
CREATE INDEX idx_sbti_s3c_tenant ON sbti_app.gl_sbti_scope3_categories(tenant_id);
CREATE INDEX idx_sbti_s3c_category ON sbti_app.gl_sbti_scope3_categories(category);
CREATE INDEX idx_sbti_s3c_included ON sbti_app.gl_sbti_scope3_categories(included_in_target);
CREATE INDEX idx_sbti_s3c_created_at ON sbti_app.gl_sbti_scope3_categories(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_s3c_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_scope3_categories
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 11: sbti_app.gl_sbti_flag_assessments
-- =============================================================================
-- Forest, Land and Agriculture (FLAG) assessments determining whether FLAG
-- targets are required (FLAG emissions >= 20% of total) and tracking
-- deforestation commitments, pathway type, and commodity-level data.

CREATE TABLE sbti_app.gl_sbti_flag_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id                   UUID,
    assessment_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    flag_emissions              DECIMAL(18,6)   DEFAULT 0,
    total_emissions             DECIMAL(18,6)   DEFAULT 0,
    flag_percentage             DECIMAL(6,2)    DEFAULT 0,
    trigger_met                 BOOLEAN         NOT NULL DEFAULT FALSE,
    flag_sector                 VARCHAR(100),
    deforestation_commitment    BOOLEAN         NOT NULL DEFAULT FALSE,
    deforestation_target_date   DATE,
    pathway_type                VARCHAR(20),
    commodity_data              JSONB           DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fa_flag_non_neg CHECK (
        flag_emissions IS NULL OR flag_emissions >= 0
    ),
    CONSTRAINT chk_sbti_fa_total_non_neg CHECK (
        total_emissions IS NULL OR total_emissions >= 0
    ),
    CONSTRAINT chk_sbti_fa_pct_range CHECK (
        flag_percentage >= 0 AND flag_percentage <= 100
    ),
    CONSTRAINT chk_sbti_fa_pathway CHECK (
        pathway_type IS NULL OR pathway_type IN ('flag_commodity', 'flag_sector', 'flag_combined')
    )
);

-- Indexes
CREATE INDEX idx_sbti_fa_org ON sbti_app.gl_sbti_flag_assessments(org_id);
CREATE INDEX idx_sbti_fa_tenant ON sbti_app.gl_sbti_flag_assessments(tenant_id);
CREATE INDEX idx_sbti_fa_date ON sbti_app.gl_sbti_flag_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_sbti_fa_trigger ON sbti_app.gl_sbti_flag_assessments(trigger_met);
CREATE INDEX idx_sbti_fa_sector ON sbti_app.gl_sbti_flag_assessments(flag_sector);
CREATE INDEX idx_sbti_fa_deforestation ON sbti_app.gl_sbti_flag_assessments(deforestation_commitment);
CREATE INDEX idx_sbti_fa_created_at ON sbti_app.gl_sbti_flag_assessments(created_at DESC);
CREATE INDEX idx_sbti_fa_commodity ON sbti_app.gl_sbti_flag_assessments USING GIN(commodity_data);
CREATE INDEX idx_sbti_fa_metadata ON sbti_app.gl_sbti_flag_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fa_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_flag_assessments
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 12: sbti_app.gl_sbti_flag_commodities
-- =============================================================================
-- Commodity-level FLAG data within a FLAG assessment.  Tracks commodity
-- intensity metrics (base, target, current), production volumes, and units
-- for commodity-specific pathway calculations.

CREATE TABLE sbti_app.gl_sbti_flag_commodities (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES sbti_app.gl_sbti_flag_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID,
    commodity               VARCHAR(30)     NOT NULL,
    base_intensity          DECIMAL(18,6)   DEFAULT 0,
    target_intensity        DECIMAL(18,6)   DEFAULT 0,
    current_intensity       DECIMAL(18,6)   DEFAULT 0,
    production_volume       DECIMAL(18,6)   DEFAULT 0,
    unit                    VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fc_commodity_not_empty CHECK (
        LENGTH(TRIM(commodity)) > 0
    ),
    CONSTRAINT chk_sbti_fc_base_non_neg CHECK (
        base_intensity IS NULL OR base_intensity >= 0
    ),
    CONSTRAINT chk_sbti_fc_target_non_neg CHECK (
        target_intensity IS NULL OR target_intensity >= 0
    ),
    CONSTRAINT chk_sbti_fc_current_non_neg CHECK (
        current_intensity IS NULL OR current_intensity >= 0
    ),
    CONSTRAINT chk_sbti_fc_volume_non_neg CHECK (
        production_volume IS NULL OR production_volume >= 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_fc_assessment ON sbti_app.gl_sbti_flag_commodities(assessment_id);
CREATE INDEX idx_sbti_fc_tenant ON sbti_app.gl_sbti_flag_commodities(tenant_id);
CREATE INDEX idx_sbti_fc_commodity ON sbti_app.gl_sbti_flag_commodities(commodity);
CREATE INDEX idx_sbti_fc_created_at ON sbti_app.gl_sbti_flag_commodities(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fc_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_flag_commodities
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 13: sbti_app.gl_sbti_sector_pathways
-- =============================================================================
-- SBTi sector-specific intensity pathways (SDA).  Stores sector name,
-- intensity metric, geography, year-by-year pathway data, source reference,
-- and version for pathway lookup during SDA target calculations.

CREATE TABLE sbti_app.gl_sbti_sector_pathways (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    sector                  VARCHAR(50)     NOT NULL,
    intensity_metric        VARCHAR(100)    NOT NULL,
    geography               VARCHAR(100)    NOT NULL DEFAULT 'Global',
    pathway_data            JSONB           DEFAULT '{}',
    source                  VARCHAR(200),
    version                 VARCHAR(20),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_sp_sector_not_empty CHECK (
        LENGTH(TRIM(sector)) > 0
    ),
    CONSTRAINT chk_sbti_sp_metric_not_empty CHECK (
        LENGTH(TRIM(intensity_metric)) > 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_sp_tenant ON sbti_app.gl_sbti_sector_pathways(tenant_id);
CREATE INDEX idx_sbti_sp_sector ON sbti_app.gl_sbti_sector_pathways(sector);
CREATE INDEX idx_sbti_sp_metric ON sbti_app.gl_sbti_sector_pathways(intensity_metric);
CREATE INDEX idx_sbti_sp_geography ON sbti_app.gl_sbti_sector_pathways(geography);
CREATE INDEX idx_sbti_sp_version ON sbti_app.gl_sbti_sector_pathways(version);
CREATE INDEX idx_sbti_sp_created_at ON sbti_app.gl_sbti_sector_pathways(created_at DESC);
CREATE INDEX idx_sbti_sp_pathway_data ON sbti_app.gl_sbti_sector_pathways USING GIN(pathway_data);
CREATE INDEX idx_sbti_sp_metadata ON sbti_app.gl_sbti_sector_pathways USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_sp_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_sector_pathways
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 14: sbti_app.gl_sbti_sector_benchmarks
-- =============================================================================
-- Sector benchmark data linked to sector pathways.  Stores percentile
-- distribution (25th, 50th, 75th), best-in-class, and sector average
-- values per benchmark year for peer comparison.

CREATE TABLE sbti_app.gl_sbti_sector_benchmarks (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    sector_pathway_id       UUID            NOT NULL REFERENCES sbti_app.gl_sbti_sector_pathways(id) ON DELETE CASCADE,
    tenant_id               UUID,
    benchmark_year          INTEGER         NOT NULL,
    percentile_25           DECIMAL(18,6)   DEFAULT 0,
    percentile_50           DECIMAL(18,6)   DEFAULT 0,
    percentile_75           DECIMAL(18,6)   DEFAULT 0,
    best_in_class           DECIMAL(18,6)   DEFAULT 0,
    sector_average          DECIMAL(18,6)   DEFAULT 0,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_sb_year_range CHECK (
        benchmark_year >= 2000 AND benchmark_year <= 2100
    ),
    UNIQUE(sector_pathway_id, benchmark_year)
);

-- Indexes
CREATE INDEX idx_sbti_sb_pathway ON sbti_app.gl_sbti_sector_benchmarks(sector_pathway_id);
CREATE INDEX idx_sbti_sb_tenant ON sbti_app.gl_sbti_sector_benchmarks(tenant_id);
CREATE INDEX idx_sbti_sb_year ON sbti_app.gl_sbti_sector_benchmarks(benchmark_year);
CREATE INDEX idx_sbti_sb_created_at ON sbti_app.gl_sbti_sector_benchmarks(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_sb_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_sector_benchmarks
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 15: sbti_app.gl_sbti_progress_records (HYPERTABLE)
-- =============================================================================
-- Annual progress tracking records partitioned by tracking_date for
-- time-series querying.  Stores actual emissions by scope, comparison to
-- expected pathway, variance analysis, on-track status, and cumulative
-- reduction percentage.

CREATE TABLE sbti_app.gl_sbti_progress_records (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    target_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID,
    tracking_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    tracking_year           INTEGER         NOT NULL,
    actual_scope1           DECIMAL(18,6)   DEFAULT 0,
    actual_scope2           DECIMAL(18,6)   DEFAULT 0,
    actual_scope3           DECIMAL(18,6)   DEFAULT 0,
    actual_total            DECIMAL(18,6)   DEFAULT 0,
    expected_pathway        DECIMAL(18,6)   DEFAULT 0,
    variance                DECIMAL(18,6)   DEFAULT 0,
    variance_percentage     DECIMAL(8,4)    DEFAULT 0,
    on_track                BOOLEAN         NOT NULL DEFAULT FALSE,
    cumulative_reduction    DECIMAL(6,2)    DEFAULT 0,
    current_annual_rate     DECIMAL(8,4)    DEFAULT 0,
    scope_breakdown         JSONB           DEFAULT '{}',
    category_progress       JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_pr_year_range CHECK (
        tracking_year >= 2000 AND tracking_year <= 2100
    ),
    CONSTRAINT chk_sbti_pr_scope1_non_neg CHECK (
        actual_scope1 IS NULL OR actual_scope1 >= 0
    ),
    CONSTRAINT chk_sbti_pr_scope2_non_neg CHECK (
        actual_scope2 IS NULL OR actual_scope2 >= 0
    ),
    CONSTRAINT chk_sbti_pr_scope3_non_neg CHECK (
        actual_scope3 IS NULL OR actual_scope3 >= 0
    ),
    CONSTRAINT chk_sbti_pr_total_non_neg CHECK (
        actual_total IS NULL OR actual_total >= 0
    ),
    CONSTRAINT chk_sbti_pr_cumulative_range CHECK (
        cumulative_reduction >= 0 AND cumulative_reduction <= 100
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('sbti_app.gl_sbti_progress_records', 'tracking_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_sbti_pr_target ON sbti_app.gl_sbti_progress_records(target_id, tracking_date DESC);
CREATE INDEX idx_sbti_pr_org ON sbti_app.gl_sbti_progress_records(org_id, tracking_date DESC);
CREATE INDEX idx_sbti_pr_tenant ON sbti_app.gl_sbti_progress_records(tenant_id, tracking_date DESC);
CREATE INDEX idx_sbti_pr_year ON sbti_app.gl_sbti_progress_records(tracking_year, tracking_date DESC);
CREATE INDEX idx_sbti_pr_on_track ON sbti_app.gl_sbti_progress_records(on_track, tracking_date DESC);
CREATE INDEX idx_sbti_pr_org_target ON sbti_app.gl_sbti_progress_records(org_id, target_id, tracking_date DESC);
CREATE INDEX idx_sbti_pr_scope ON sbti_app.gl_sbti_progress_records USING GIN(scope_breakdown);
CREATE INDEX idx_sbti_pr_category ON sbti_app.gl_sbti_progress_records USING GIN(category_progress);

-- =============================================================================
-- Table 16: sbti_app.gl_sbti_temperature_scores (HYPERTABLE)
-- =============================================================================
-- Temperature alignment scores partitioned by score_date for time-series
-- querying.  Stores company-level, short/long-term, and per-scope temperature
-- scores using SBTi temperature rating methodology.

CREATE TABLE sbti_app.gl_sbti_temperature_scores (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID,
    score_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    company_score           DECIMAL(4,2)    DEFAULT 0,
    short_term_score        DECIMAL(4,2)    DEFAULT 0,
    long_term_score         DECIMAL(4,2)    DEFAULT 0,
    scope1_temp             DECIMAL(4,2)    DEFAULT 0,
    scope2_temp             DECIMAL(4,2)    DEFAULT 0,
    scope3_temp             DECIMAL(4,2)    DEFAULT 0,
    method                  VARCHAR(50),
    details                 JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_ts2_company_range CHECK (
        company_score >= 0 AND company_score <= 10
    ),
    CONSTRAINT chk_sbti_ts2_short_range CHECK (
        short_term_score >= 0 AND short_term_score <= 10
    ),
    CONSTRAINT chk_sbti_ts2_long_range CHECK (
        long_term_score >= 0 AND long_term_score <= 10
    ),
    CONSTRAINT chk_sbti_ts2_s1_range CHECK (
        scope1_temp >= 0 AND scope1_temp <= 10
    ),
    CONSTRAINT chk_sbti_ts2_s2_range CHECK (
        scope2_temp >= 0 AND scope2_temp <= 10
    ),
    CONSTRAINT chk_sbti_ts2_s3_range CHECK (
        scope3_temp >= 0 AND scope3_temp <= 10
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('sbti_app.gl_sbti_temperature_scores', 'score_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_sbti_ts2_org ON sbti_app.gl_sbti_temperature_scores(org_id, score_date DESC);
CREATE INDEX idx_sbti_ts2_tenant ON sbti_app.gl_sbti_temperature_scores(tenant_id, score_date DESC);
CREATE INDEX idx_sbti_ts2_company ON sbti_app.gl_sbti_temperature_scores(company_score, score_date DESC);
CREATE INDEX idx_sbti_ts2_method ON sbti_app.gl_sbti_temperature_scores(method, score_date DESC);
CREATE INDEX idx_sbti_ts2_details ON sbti_app.gl_sbti_temperature_scores USING GIN(details);

-- =============================================================================
-- Table 17: sbti_app.gl_sbti_recalculations
-- =============================================================================
-- Base year recalculation records.  Tracks trigger type (structural_change,
-- methodology_change, error_correction, etc.), original vs recalculated
-- emissions, percentage change, revalidation requirement, affected targets,
-- and full audit trail.

CREATE TABLE sbti_app.gl_sbti_recalculations (
    id                                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                              UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id                           UUID,
    recalculation_date                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    trigger_type                        VARCHAR(30)     NOT NULL,
    original_base_year_emissions        DECIMAL(18,6)   NOT NULL DEFAULT 0,
    recalculated_base_year_emissions    DECIMAL(18,6)   NOT NULL DEFAULT 0,
    percentage_change                   DECIMAL(8,4)    DEFAULT 0,
    revalidation_required               BOOLEAN         NOT NULL DEFAULT FALSE,
    affected_targets                    JSONB           DEFAULT '[]',
    audit_trail                         JSONB           DEFAULT '{}',
    metadata                            JSONB           DEFAULT '{}',
    created_at                          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_rc_trigger CHECK (
        trigger_type IN ('structural_change', 'methodology_change', 'error_correction',
                          'outsourcing', 'insourcing', 'acquisition', 'divestiture', 'organic_growth')
    ),
    CONSTRAINT chk_sbti_rc_original_non_neg CHECK (
        original_base_year_emissions >= 0
    ),
    CONSTRAINT chk_sbti_rc_recalc_non_neg CHECK (
        recalculated_base_year_emissions >= 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_rc_org ON sbti_app.gl_sbti_recalculations(org_id);
CREATE INDEX idx_sbti_rc_tenant ON sbti_app.gl_sbti_recalculations(tenant_id);
CREATE INDEX idx_sbti_rc_date ON sbti_app.gl_sbti_recalculations(org_id, recalculation_date DESC);
CREATE INDEX idx_sbti_rc_trigger ON sbti_app.gl_sbti_recalculations(trigger_type);
CREATE INDEX idx_sbti_rc_revalidation ON sbti_app.gl_sbti_recalculations(revalidation_required);
CREATE INDEX idx_sbti_rc_created_at ON sbti_app.gl_sbti_recalculations(created_at DESC);
CREATE INDEX idx_sbti_rc_targets ON sbti_app.gl_sbti_recalculations USING GIN(affected_targets);
CREATE INDEX idx_sbti_rc_audit ON sbti_app.gl_sbti_recalculations USING GIN(audit_trail);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_rc_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_recalculations
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 18: sbti_app.gl_sbti_five_year_reviews
-- =============================================================================
-- Five-year target review records per SBTi requirements.  Tracks the
-- original validation date, trigger date (5 years from validation),
-- deadline, review status, readiness score, and outcome (renewed/updated/
-- expired).

CREATE TABLE sbti_app.gl_sbti_five_year_reviews (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    target_id               UUID            NOT NULL REFERENCES sbti_app.gl_sbti_targets(id) ON DELETE CASCADE,
    tenant_id               UUID,
    validation_date         DATE            NOT NULL,
    trigger_date            DATE            NOT NULL,
    deadline                DATE            NOT NULL,
    review_status           VARCHAR(20)     NOT NULL DEFAULT 'upcoming',
    readiness_score         DECIMAL(6,2)    DEFAULT 0,
    outcome                 VARCHAR(20),
    review_data             JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fyr_status CHECK (
        review_status IN ('upcoming', 'in_progress', 'completed', 'expired')
    ),
    CONSTRAINT chk_sbti_fyr_readiness_range CHECK (
        readiness_score >= 0 AND readiness_score <= 100
    ),
    CONSTRAINT chk_sbti_fyr_outcome CHECK (
        outcome IS NULL OR outcome IN ('renewed', 'updated', 'expired')
    ),
    CONSTRAINT chk_sbti_fyr_trigger_after_validation CHECK (
        trigger_date >= validation_date
    ),
    CONSTRAINT chk_sbti_fyr_deadline_after_trigger CHECK (
        deadline >= trigger_date
    )
);

-- Indexes
CREATE INDEX idx_sbti_fyr_org ON sbti_app.gl_sbti_five_year_reviews(org_id);
CREATE INDEX idx_sbti_fyr_target ON sbti_app.gl_sbti_five_year_reviews(target_id);
CREATE INDEX idx_sbti_fyr_tenant ON sbti_app.gl_sbti_five_year_reviews(tenant_id);
CREATE INDEX idx_sbti_fyr_status ON sbti_app.gl_sbti_five_year_reviews(review_status);
CREATE INDEX idx_sbti_fyr_deadline ON sbti_app.gl_sbti_five_year_reviews(deadline);
CREATE INDEX idx_sbti_fyr_outcome ON sbti_app.gl_sbti_five_year_reviews(outcome);
CREATE INDEX idx_sbti_fyr_created_at ON sbti_app.gl_sbti_five_year_reviews(created_at DESC);
CREATE INDEX idx_sbti_fyr_review_data ON sbti_app.gl_sbti_five_year_reviews USING GIN(review_data);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fyr_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_five_year_reviews
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 19: sbti_app.gl_sbti_fi_portfolios
-- =============================================================================
-- Financial institution portfolio records for SBTi for Financial Institutions
-- (SBTi-FI).  Tracks portfolio-level financed emissions, coverage percentage,
-- portfolio temperature score, target coverage by year, and WACI.

CREATE TABLE sbti_app.gl_sbti_fi_portfolios (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                      UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id                   UUID,
    portfolio_name              VARCHAR(300)    NOT NULL,
    total_financed_emissions    DECIMAL(18,6)   DEFAULT 0,
    portfolio_coverage_pct      DECIMAL(6,2)    DEFAULT 0,
    portfolio_temperature       DECIMAL(4,2)    DEFAULT 0,
    target_coverage_by_year     JSONB           DEFAULT '{}',
    waci                        DECIMAL(18,6)   DEFAULT 0,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fip_name_not_empty CHECK (
        LENGTH(TRIM(portfolio_name)) > 0
    ),
    CONSTRAINT chk_sbti_fip_financed_non_neg CHECK (
        total_financed_emissions IS NULL OR total_financed_emissions >= 0
    ),
    CONSTRAINT chk_sbti_fip_coverage_range CHECK (
        portfolio_coverage_pct >= 0 AND portfolio_coverage_pct <= 100
    ),
    CONSTRAINT chk_sbti_fip_temp_range CHECK (
        portfolio_temperature >= 0 AND portfolio_temperature <= 10
    ),
    CONSTRAINT chk_sbti_fip_waci_non_neg CHECK (
        waci IS NULL OR waci >= 0
    )
);

-- Indexes
CREATE INDEX idx_sbti_fip_org ON sbti_app.gl_sbti_fi_portfolios(org_id);
CREATE INDEX idx_sbti_fip_tenant ON sbti_app.gl_sbti_fi_portfolios(tenant_id);
CREATE INDEX idx_sbti_fip_name ON sbti_app.gl_sbti_fi_portfolios(portfolio_name);
CREATE INDEX idx_sbti_fip_created_at ON sbti_app.gl_sbti_fi_portfolios(created_at DESC);
CREATE INDEX idx_sbti_fip_coverage_year ON sbti_app.gl_sbti_fi_portfolios USING GIN(target_coverage_by_year);
CREATE INDEX idx_sbti_fip_metadata ON sbti_app.gl_sbti_fi_portfolios USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fip_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_fi_portfolios
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 20: sbti_app.gl_sbti_fi_portfolio_holdings
-- =============================================================================
-- Individual portfolio holdings within an FI portfolio.  Tracks company
-- details, asset class, investment value, financed emissions, attribution
-- method, PCAF data quality score, and SBTi target status of the investee.

CREATE TABLE sbti_app.gl_sbti_fi_portfolio_holdings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id            UUID            NOT NULL REFERENCES sbti_app.gl_sbti_fi_portfolios(id) ON DELETE CASCADE,
    tenant_id               UUID,
    company_name            VARCHAR(500)    NOT NULL,
    company_id              VARCHAR(100),
    asset_class             VARCHAR(30)     NOT NULL,
    investment_value        DECIMAL(18,2)   DEFAULT 0,
    financed_emissions      DECIMAL(18,6)   DEFAULT 0,
    attribution_method      VARCHAR(30),
    pcaf_data_quality       INTEGER,
    has_sbti_target         BOOLEAN         NOT NULL DEFAULT FALSE,
    target_status           VARCHAR(20),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fih_company_not_empty CHECK (
        LENGTH(TRIM(company_name)) > 0
    ),
    CONSTRAINT chk_sbti_fih_asset_class CHECK (
        asset_class IN ('listed_equity', 'corporate_bond', 'private_equity',
                         'project_finance', 'commercial_real_estate', 'mortgage',
                         'motor_vehicle_loan', 'sovereign_bond')
    ),
    CONSTRAINT chk_sbti_fih_investment_non_neg CHECK (
        investment_value IS NULL OR investment_value >= 0
    ),
    CONSTRAINT chk_sbti_fih_financed_non_neg CHECK (
        financed_emissions IS NULL OR financed_emissions >= 0
    ),
    CONSTRAINT chk_sbti_fih_attribution CHECK (
        attribution_method IS NULL OR attribution_method IN ('evic', 'revenue', 'balance_sheet', 'pcaf')
    ),
    CONSTRAINT chk_sbti_fih_pcaf_range CHECK (
        pcaf_data_quality IS NULL OR (pcaf_data_quality >= 1 AND pcaf_data_quality <= 5)
    ),
    CONSTRAINT chk_sbti_fih_target_status CHECK (
        target_status IS NULL OR target_status IN ('committed', 'submitted', 'validated', 'published', 'expired', 'none')
    )
);

-- Indexes
CREATE INDEX idx_sbti_fih_portfolio ON sbti_app.gl_sbti_fi_portfolio_holdings(portfolio_id);
CREATE INDEX idx_sbti_fih_tenant ON sbti_app.gl_sbti_fi_portfolio_holdings(tenant_id);
CREATE INDEX idx_sbti_fih_company ON sbti_app.gl_sbti_fi_portfolio_holdings(company_name);
CREATE INDEX idx_sbti_fih_asset_class ON sbti_app.gl_sbti_fi_portfolio_holdings(asset_class);
CREATE INDEX idx_sbti_fih_sbti_target ON sbti_app.gl_sbti_fi_portfolio_holdings(has_sbti_target);
CREATE INDEX idx_sbti_fih_target_status ON sbti_app.gl_sbti_fi_portfolio_holdings(target_status);
CREATE INDEX idx_sbti_fih_pcaf ON sbti_app.gl_sbti_fi_portfolio_holdings(pcaf_data_quality);
CREATE INDEX idx_sbti_fih_created_at ON sbti_app.gl_sbti_fi_portfolio_holdings(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fih_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_fi_portfolio_holdings
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 21: sbti_app.gl_sbti_fi_engagement
-- =============================================================================
-- Investee engagement records for SBTi-FI portfolio alignment.  Tracks
-- engagement type (direct, collaborative, escalation, proxy_voting),
-- engagement date, status, target date, notes, and outcome data.

CREATE TABLE sbti_app.gl_sbti_fi_engagement (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id            UUID            NOT NULL REFERENCES sbti_app.gl_sbti_fi_portfolios(id) ON DELETE CASCADE,
    holding_id              UUID            NOT NULL REFERENCES sbti_app.gl_sbti_fi_portfolio_holdings(id) ON DELETE CASCADE,
    tenant_id               UUID,
    engagement_type         VARCHAR(50)     NOT NULL,
    engagement_date         DATE            NOT NULL,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'initiated',
    target_date             DATE,
    notes                   TEXT,
    outcome                 JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fie_type CHECK (
        engagement_type IN ('direct', 'collaborative', 'escalation', 'proxy_voting', 'shareholder_resolution')
    ),
    CONSTRAINT chk_sbti_fie_status CHECK (
        status IN ('initiated', 'in_progress', 'successful', 'unsuccessful', 'ongoing', 'closed')
    )
);

-- Indexes
CREATE INDEX idx_sbti_fie_portfolio ON sbti_app.gl_sbti_fi_engagement(portfolio_id);
CREATE INDEX idx_sbti_fie_holding ON sbti_app.gl_sbti_fi_engagement(holding_id);
CREATE INDEX idx_sbti_fie_tenant ON sbti_app.gl_sbti_fi_engagement(tenant_id);
CREATE INDEX idx_sbti_fie_type ON sbti_app.gl_sbti_fi_engagement(engagement_type);
CREATE INDEX idx_sbti_fie_date ON sbti_app.gl_sbti_fi_engagement(engagement_date);
CREATE INDEX idx_sbti_fie_status ON sbti_app.gl_sbti_fi_engagement(status);
CREATE INDEX idx_sbti_fie_target_date ON sbti_app.gl_sbti_fi_engagement(target_date);
CREATE INDEX idx_sbti_fie_created_at ON sbti_app.gl_sbti_fi_engagement(created_at DESC);
CREATE INDEX idx_sbti_fie_outcome ON sbti_app.gl_sbti_fi_engagement USING GIN(outcome);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fie_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_fi_engagement
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 22: sbti_app.gl_sbti_framework_mappings
-- =============================================================================
-- Cross-framework alignment mappings linking SBTi targets and data to
-- other reporting frameworks (CDP, TCFD, CSRD, ISO14064, GHG Protocol,
-- ISSB).  Tracks coverage percentage, identified gaps, and last assessment.

CREATE TABLE sbti_app.gl_sbti_framework_mappings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    framework               VARCHAR(30)     NOT NULL,
    mapping_data            JSONB           DEFAULT '{}',
    coverage_percentage     DECIMAL(6,2)    DEFAULT 0,
    gaps                    JSONB           DEFAULT '[]',
    last_assessed           TIMESTAMPTZ,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_fm_framework CHECK (
        framework IN ('cdp', 'tcfd', 'csrd', 'iso14064', 'ghg_protocol', 'issb', 'sec_climate', 'nzba')
    ),
    CONSTRAINT chk_sbti_fm_coverage_range CHECK (
        coverage_percentage >= 0 AND coverage_percentage <= 100
    )
);

-- Indexes
CREATE INDEX idx_sbti_fm_org ON sbti_app.gl_sbti_framework_mappings(org_id);
CREATE INDEX idx_sbti_fm_tenant ON sbti_app.gl_sbti_framework_mappings(tenant_id);
CREATE INDEX idx_sbti_fm_framework ON sbti_app.gl_sbti_framework_mappings(framework);
CREATE INDEX idx_sbti_fm_created_at ON sbti_app.gl_sbti_framework_mappings(created_at DESC);
CREATE INDEX idx_sbti_fm_mapping ON sbti_app.gl_sbti_framework_mappings USING GIN(mapping_data);
CREATE INDEX idx_sbti_fm_gaps ON sbti_app.gl_sbti_framework_mappings USING GIN(gaps);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_fm_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_framework_mappings
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 23: sbti_app.gl_sbti_reports
-- =============================================================================
-- Generated SBTi reports including target submissions, validation reports,
-- progress reports, gap analyses, and executive summaries.  Tracks report
-- type, format, content, and generation metadata.

CREATE TABLE sbti_app.gl_sbti_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    report_type             VARCHAR(50)     NOT NULL,
    report_format           VARCHAR(10)     NOT NULL DEFAULT 'pdf',
    title                   VARCHAR(500)    NOT NULL,
    content                 JSONB           DEFAULT '{}',
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by            VARCHAR(200),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_rpt_title_not_empty CHECK (
        LENGTH(TRIM(title)) > 0
    ),
    CONSTRAINT chk_sbti_rpt_type CHECK (
        report_type IN ('target_submission', 'validation_report', 'progress_report',
                         'gap_analysis', 'executive_summary', 'fi_portfolio',
                         'temperature_alignment', 'flag_assessment', 'framework_mapping')
    ),
    CONSTRAINT chk_sbti_rpt_format CHECK (
        report_format IN ('pdf', 'excel', 'json', 'html', 'csv')
    )
);

-- Indexes
CREATE INDEX idx_sbti_rpt_org ON sbti_app.gl_sbti_reports(org_id);
CREATE INDEX idx_sbti_rpt_tenant ON sbti_app.gl_sbti_reports(tenant_id);
CREATE INDEX idx_sbti_rpt_type ON sbti_app.gl_sbti_reports(report_type);
CREATE INDEX idx_sbti_rpt_format ON sbti_app.gl_sbti_reports(report_format);
CREATE INDEX idx_sbti_rpt_generated_at ON sbti_app.gl_sbti_reports(generated_at DESC);
CREATE INDEX idx_sbti_rpt_created_at ON sbti_app.gl_sbti_reports(created_at DESC);
CREATE INDEX idx_sbti_rpt_content ON sbti_app.gl_sbti_reports USING GIN(content);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_rpt_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_reports
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 24: sbti_app.gl_sbti_gap_assessments
-- =============================================================================
-- Gap analysis results evaluating organizational readiness for SBTi target
-- validation.  Tracks overall readiness score, total/critical gap counts,
-- and action plan.

CREATE TABLE sbti_app.gl_sbti_gap_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES sbti_app.gl_sbti_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    overall_readiness       DECIMAL(6,2)    DEFAULT 0,
    total_gaps              INTEGER         NOT NULL DEFAULT 0,
    critical_gaps           INTEGER         NOT NULL DEFAULT 0,
    action_plan             JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_ga_readiness_range CHECK (
        overall_readiness >= 0 AND overall_readiness <= 100
    ),
    CONSTRAINT chk_sbti_ga_total_non_neg CHECK (
        total_gaps >= 0
    ),
    CONSTRAINT chk_sbti_ga_critical_non_neg CHECK (
        critical_gaps >= 0
    ),
    CONSTRAINT chk_sbti_ga_critical_lte_total CHECK (
        critical_gaps <= total_gaps
    )
);

-- Indexes
CREATE INDEX idx_sbti_ga_org ON sbti_app.gl_sbti_gap_assessments(org_id);
CREATE INDEX idx_sbti_ga_tenant ON sbti_app.gl_sbti_gap_assessments(tenant_id);
CREATE INDEX idx_sbti_ga_date ON sbti_app.gl_sbti_gap_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_sbti_ga_readiness ON sbti_app.gl_sbti_gap_assessments(overall_readiness DESC);
CREATE INDEX idx_sbti_ga_created_at ON sbti_app.gl_sbti_gap_assessments(created_at DESC);
CREATE INDEX idx_sbti_ga_action_plan ON sbti_app.gl_sbti_gap_assessments USING GIN(action_plan);
CREATE INDEX idx_sbti_ga_metadata ON sbti_app.gl_sbti_gap_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_ga_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_gap_assessments
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Table 25: sbti_app.gl_sbti_gap_items
-- =============================================================================
-- Individual gap items within a gap assessment.  Tracks criterion reference,
-- gap type, severity, description, recommendation, estimated effort, and
-- resolution status.

CREATE TABLE sbti_app.gl_sbti_gap_items (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES sbti_app.gl_sbti_gap_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID,
    criterion               VARCHAR(10)     NOT NULL,
    gap_type                VARCHAR(30)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
    description             TEXT,
    recommendation          TEXT,
    estimated_effort        VARCHAR(50),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'open',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_sbti_gi_criterion_not_empty CHECK (
        LENGTH(TRIM(criterion)) > 0
    ),
    CONSTRAINT chk_sbti_gi_gap_type CHECK (
        gap_type IN ('data_missing', 'data_quality', 'methodology', 'coverage',
                      'boundary', 'verification', 'governance', 'reporting', 'other')
    ),
    CONSTRAINT chk_sbti_gi_severity CHECK (
        severity IN ('critical', 'high', 'medium', 'low')
    ),
    CONSTRAINT chk_sbti_gi_effort CHECK (
        estimated_effort IS NULL OR estimated_effort IN (
            '1_week', '2_weeks', '1_month', '2_months', '3_months', '6_months', '12_months'
        )
    ),
    CONSTRAINT chk_sbti_gi_status CHECK (
        status IN ('open', 'in_progress', 'resolved', 'deferred', 'not_applicable')
    )
);

-- Indexes
CREATE INDEX idx_sbti_gi_assessment ON sbti_app.gl_sbti_gap_items(assessment_id);
CREATE INDEX idx_sbti_gi_tenant ON sbti_app.gl_sbti_gap_items(tenant_id);
CREATE INDEX idx_sbti_gi_criterion ON sbti_app.gl_sbti_gap_items(criterion);
CREATE INDEX idx_sbti_gi_gap_type ON sbti_app.gl_sbti_gap_items(gap_type);
CREATE INDEX idx_sbti_gi_severity ON sbti_app.gl_sbti_gap_items(severity);
CREATE INDEX idx_sbti_gi_status ON sbti_app.gl_sbti_gap_items(status);
CREATE INDEX idx_sbti_gi_created_at ON sbti_app.gl_sbti_gap_items(created_at DESC);

-- Updated_at trigger
CREATE TRIGGER trg_sbti_gi_updated_at
    BEFORE UPDATE ON sbti_app.gl_sbti_gap_items
    FOR EACH ROW
    EXECUTE FUNCTION sbti_app.set_updated_at();

-- =============================================================================
-- Continuous Aggregate: sbti_app.annual_progress_summary
-- =============================================================================
-- Precomputed annual progress aggregation by organization and target
-- derived from the progress_records hypertable for dashboard and
-- year-over-year comparison.

CREATE MATERIALIZED VIEW sbti_app.annual_progress_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 year', tracking_date)    AS bucket,
    org_id,
    target_id,
    AVG(actual_total)                       AS avg_actual_total,
    MIN(actual_total)                       AS min_actual_total,
    MAX(actual_total)                       AS max_actual_total,
    AVG(expected_pathway)                   AS avg_expected_pathway,
    AVG(variance_percentage)                AS avg_variance_pct,
    AVG(cumulative_reduction)               AS avg_cumulative_reduction,
    COUNT(*)                                AS record_count
FROM sbti_app.gl_sbti_progress_records
GROUP BY bucket, org_id, target_id
WITH NO DATA;

-- Refresh policy: every hour, covering last 3 days
SELECT add_continuous_aggregate_policy('sbti_app.annual_progress_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- =============================================================================
-- Continuous Aggregate: sbti_app.quarterly_temperature_trends
-- =============================================================================
-- Precomputed quarterly temperature score trends by organization for
-- portfolio alignment monitoring and trend analysis.

CREATE MATERIALIZED VIEW sbti_app.quarterly_temperature_trends
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('3 months', score_date)     AS bucket,
    org_id,
    AVG(company_score)                      AS avg_company_score,
    MIN(company_score)                      AS min_company_score,
    MAX(company_score)                      AS max_company_score,
    AVG(short_term_score)                   AS avg_short_term,
    AVG(long_term_score)                    AS avg_long_term,
    AVG(scope1_temp)                        AS avg_scope1_temp,
    AVG(scope2_temp)                        AS avg_scope2_temp,
    AVG(scope3_temp)                        AS avg_scope3_temp,
    COUNT(*)                                AS score_count
FROM sbti_app.gl_sbti_temperature_scores
GROUP BY bucket, org_id
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('sbti_app.quarterly_temperature_trends',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep validation results for 3650 days (10 years, regulatory retention)
SELECT add_retention_policy('sbti_app.gl_sbti_validation_results', INTERVAL '3650 days');

-- Keep progress records for 3650 days (10 years)
SELECT add_retention_policy('sbti_app.gl_sbti_progress_records', INTERVAL '3650 days');

-- Keep temperature scores for 3650 days (10 years)
SELECT add_retention_policy('sbti_app.gl_sbti_temperature_scores', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on validation_results after 90 days
ALTER TABLE sbti_app.gl_sbti_validation_results SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'validation_date DESC'
);

SELECT add_compression_policy('sbti_app.gl_sbti_validation_results', INTERVAL '90 days');

-- Enable compression on progress_records after 90 days
ALTER TABLE sbti_app.gl_sbti_progress_records SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'tracking_date DESC'
);

SELECT add_compression_policy('sbti_app.gl_sbti_progress_records', INTERVAL '90 days');

-- Enable compression on temperature_scores after 90 days
ALTER TABLE sbti_app.gl_sbti_temperature_scores SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'score_date DESC'
);

SELECT add_compression_policy('sbti_app.gl_sbti_temperature_scores', INTERVAL '90 days');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA sbti_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA sbti_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA sbti_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON sbti_app.annual_progress_summary TO greenlang_app;
GRANT SELECT ON sbti_app.quarterly_temperature_trends TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA sbti_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA sbti_app TO greenlang_readonly;
        GRANT SELECT ON sbti_app.annual_progress_summary TO greenlang_readonly;
        GRANT SELECT ON sbti_app.quarterly_temperature_trends TO greenlang_readonly;
    END IF;
END
$$;

-- Add SBTi app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'sbti_app:organizations:read', 'sbti_app', 'organizations_read', 'View SBTi organization profiles'),
    (gen_random_uuid(), 'sbti_app:organizations:write', 'sbti_app', 'organizations_write', 'Create and manage SBTi organization profiles'),
    (gen_random_uuid(), 'sbti_app:inventories:read', 'sbti_app', 'inventories_read', 'View SBTi emissions inventories'),
    (gen_random_uuid(), 'sbti_app:inventories:write', 'sbti_app', 'inventories_write', 'Create and manage SBTi emissions inventories'),
    (gen_random_uuid(), 'sbti_app:targets:read', 'sbti_app', 'targets_read', 'View SBTi target definitions and scope details'),
    (gen_random_uuid(), 'sbti_app:targets:write', 'sbti_app', 'targets_write', 'Create and manage SBTi targets'),
    (gen_random_uuid(), 'sbti_app:targets:submit', 'sbti_app', 'targets_submit', 'Submit SBTi targets for validation'),
    (gen_random_uuid(), 'sbti_app:pathways:read', 'sbti_app', 'pathways_read', 'View SBTi reduction pathways and milestones'),
    (gen_random_uuid(), 'sbti_app:pathways:write', 'sbti_app', 'pathways_write', 'Calculate and manage SBTi reduction pathways'),
    (gen_random_uuid(), 'sbti_app:validation:read', 'sbti_app', 'validation_read', 'View SBTi criteria validation results'),
    (gen_random_uuid(), 'sbti_app:validation:run', 'sbti_app', 'validation_run', 'Run SBTi criteria validation checks'),
    (gen_random_uuid(), 'sbti_app:scope3:read', 'sbti_app', 'scope3_read', 'View SBTi Scope 3 screenings and categories'),
    (gen_random_uuid(), 'sbti_app:scope3:write', 'sbti_app', 'scope3_write', 'Run and manage SBTi Scope 3 screenings'),
    (gen_random_uuid(), 'sbti_app:flag:read', 'sbti_app', 'flag_read', 'View SBTi FLAG assessments and commodity data'),
    (gen_random_uuid(), 'sbti_app:flag:write', 'sbti_app', 'flag_write', 'Run and manage SBTi FLAG assessments'),
    (gen_random_uuid(), 'sbti_app:sectors:read', 'sbti_app', 'sectors_read', 'View SBTi sector pathways and benchmarks'),
    (gen_random_uuid(), 'sbti_app:sectors:write', 'sbti_app', 'sectors_write', 'Manage SBTi sector pathway data'),
    (gen_random_uuid(), 'sbti_app:progress:read', 'sbti_app', 'progress_read', 'View SBTi progress tracking records'),
    (gen_random_uuid(), 'sbti_app:progress:write', 'sbti_app', 'progress_write', 'Create and manage SBTi progress records'),
    (gen_random_uuid(), 'sbti_app:temperature:read', 'sbti_app', 'temperature_read', 'View SBTi temperature alignment scores'),
    (gen_random_uuid(), 'sbti_app:temperature:calculate', 'sbti_app', 'temperature_calculate', 'Calculate SBTi temperature scores'),
    (gen_random_uuid(), 'sbti_app:recalculations:read', 'sbti_app', 'recalculations_read', 'View SBTi base year recalculations'),
    (gen_random_uuid(), 'sbti_app:recalculations:write', 'sbti_app', 'recalculations_write', 'Perform SBTi base year recalculations'),
    (gen_random_uuid(), 'sbti_app:reviews:read', 'sbti_app', 'reviews_read', 'View SBTi five-year review records'),
    (gen_random_uuid(), 'sbti_app:reviews:write', 'sbti_app', 'reviews_write', 'Manage SBTi five-year review processes'),
    (gen_random_uuid(), 'sbti_app:fi_portfolios:read', 'sbti_app', 'fi_portfolios_read', 'View SBTi-FI portfolio data'),
    (gen_random_uuid(), 'sbti_app:fi_portfolios:write', 'sbti_app', 'fi_portfolios_write', 'Create and manage SBTi-FI portfolios'),
    (gen_random_uuid(), 'sbti_app:fi_engagement:read', 'sbti_app', 'fi_engagement_read', 'View SBTi-FI investee engagement records'),
    (gen_random_uuid(), 'sbti_app:fi_engagement:write', 'sbti_app', 'fi_engagement_write', 'Manage SBTi-FI investee engagement'),
    (gen_random_uuid(), 'sbti_app:frameworks:read', 'sbti_app', 'frameworks_read', 'View SBTi cross-framework mappings'),
    (gen_random_uuid(), 'sbti_app:frameworks:write', 'sbti_app', 'frameworks_write', 'Manage SBTi cross-framework mappings'),
    (gen_random_uuid(), 'sbti_app:reports:read', 'sbti_app', 'reports_read', 'View SBTi generated reports'),
    (gen_random_uuid(), 'sbti_app:reports:generate', 'sbti_app', 'reports_generate', 'Generate SBTi reports and exports'),
    (gen_random_uuid(), 'sbti_app:gaps:read', 'sbti_app', 'gaps_read', 'View SBTi gap assessment results'),
    (gen_random_uuid(), 'sbti_app:gaps:write', 'sbti_app', 'gaps_write', 'Run and manage SBTi gap assessments'),
    (gen_random_uuid(), 'sbti_app:dashboard:read', 'sbti_app', 'dashboard_read', 'View SBTi dashboards and analytics'),
    (gen_random_uuid(), 'sbti_app:admin', 'sbti_app', 'admin', 'SBTi application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA sbti_app IS 'GL-SBTi-APP v1.0 Application Schema - SBTi Target Setting & Validation Platform with organization profiling, emissions inventories, near-term/long-term/net-zero target definitions, ACA/SDA pathway calculation, criteria validation (C1-C28, NZ-C1-NZ-C14), Scope 3 screening, FLAG assessments, sector pathways, progress tracking, temperature scoring, base year recalculations, five-year reviews, FI portfolios, framework mappings, gap analysis, and reporting';

COMMENT ON TABLE sbti_app.gl_sbti_organizations IS 'Organization profiles for SBTi target setting with sector classification (ISIC/NACE/NAICS), financial metrics, and OECD status';
COMMENT ON TABLE sbti_app.gl_sbti_emissions_inventories IS 'Base year and annual GHG emissions inventories with Scope 1/2/3, per-category (Cat 1-15), FLAG, bioenergy, data quality, and verification status';
COMMENT ON TABLE sbti_app.gl_sbti_targets IS 'SBTi target definitions (near-term/long-term/net-zero) with method (ACA/SDA), pathway alignment (1.5C/well-below-2C), scope coverage, and lifecycle dates';
COMMENT ON TABLE sbti_app.gl_sbti_target_scopes IS 'Per-scope target details with base/target emissions, coverage percentage, reduction percentage, and intensity metrics';
COMMENT ON TABLE sbti_app.gl_sbti_pathways IS 'Calculated reduction pathways with method, sector, annual rate, pathway data (JSONB), and uncertainty bounds';
COMMENT ON TABLE sbti_app.gl_sbti_pathway_milestones IS 'Annual milestone points along reduction pathways with expected absolute/intensity values and cumulative reduction';
COMMENT ON TABLE sbti_app.gl_sbti_validation_results IS 'TimescaleDB hypertable: SBTi criteria validation results with pass/fail/not-applicable counts, readiness percentage, and overall result';
COMMENT ON TABLE sbti_app.gl_sbti_criteria_checks IS 'Individual criterion validation results for SBTi corporate criteria (C1-C28) and net-zero criteria (NZ-C1 through NZ-C14)';
COMMENT ON TABLE sbti_app.gl_sbti_scope3_screenings IS 'Scope 3 screening results determining target requirement (>40% threshold) with coverage percentages and recommended categories';
COMMENT ON TABLE sbti_app.gl_sbti_scope3_categories IS 'Per-category Scope 3 breakdown with emissions, percentage, data quality, and target inclusion status';
COMMENT ON TABLE sbti_app.gl_sbti_flag_assessments IS 'FLAG (Forest, Land and Agriculture) assessments determining target requirement (>=20% threshold) with deforestation commitments and commodity data';
COMMENT ON TABLE sbti_app.gl_sbti_flag_commodities IS 'Commodity-level FLAG data with base/target/current intensity, production volume, and unit for commodity-specific pathway calculations';
COMMENT ON TABLE sbti_app.gl_sbti_sector_pathways IS 'SBTi sector-specific intensity pathways (SDA) with sector, metric, geography, year-by-year pathway data, source, and version';
COMMENT ON TABLE sbti_app.gl_sbti_sector_benchmarks IS 'Sector benchmark data with percentile distribution (25th/50th/75th), best-in-class, and sector average per year';
COMMENT ON TABLE sbti_app.gl_sbti_progress_records IS 'TimescaleDB hypertable: Annual progress tracking with actual emissions by scope, pathway comparison, variance analysis, on-track status, and cumulative reduction';
COMMENT ON TABLE sbti_app.gl_sbti_temperature_scores IS 'TimescaleDB hypertable: Temperature alignment scores (company, short/long-term, per-scope) using SBTi temperature rating methodology';
COMMENT ON TABLE sbti_app.gl_sbti_recalculations IS 'Base year recalculation records with trigger type, original/recalculated emissions, percentage change, and affected targets';
COMMENT ON TABLE sbti_app.gl_sbti_five_year_reviews IS 'Five-year target review records with validation/trigger/deadline dates, review status, readiness score, and outcome';
COMMENT ON TABLE sbti_app.gl_sbti_fi_portfolios IS 'SBTi-FI portfolio records with financed emissions, coverage percentage, portfolio temperature, target coverage by year, and WACI';
COMMENT ON TABLE sbti_app.gl_sbti_fi_portfolio_holdings IS 'FI portfolio holdings with company details, asset class, investment value, financed emissions, PCAF data quality, and investee SBTi target status';
COMMENT ON TABLE sbti_app.gl_sbti_fi_engagement IS 'Investee engagement records for SBTi-FI portfolio alignment with engagement type, status, target date, and outcome tracking';
COMMENT ON TABLE sbti_app.gl_sbti_framework_mappings IS 'Cross-framework alignment mappings (CDP/TCFD/CSRD/ISO14064/GHG Protocol/ISSB) with coverage percentage and identified gaps';
COMMENT ON TABLE sbti_app.gl_sbti_reports IS 'Generated SBTi reports (target submission/validation/progress/gap analysis/executive summary) with type, format, content, and generation metadata';
COMMENT ON TABLE sbti_app.gl_sbti_gap_assessments IS 'Gap analysis results evaluating SBTi validation readiness with overall readiness score, total/critical gap counts, and action plan';
COMMENT ON TABLE sbti_app.gl_sbti_gap_items IS 'Individual gap items with criterion reference, gap type, severity, description, recommendation, estimated effort, and resolution status';

COMMENT ON MATERIALIZED VIEW sbti_app.annual_progress_summary IS 'Continuous aggregate: annual progress aggregation by organization and target for year-over-year comparison';
COMMENT ON MATERIALIZED VIEW sbti_app.quarterly_temperature_trends IS 'Continuous aggregate: quarterly temperature score trends by organization for portfolio alignment monitoring';

COMMENT ON COLUMN sbti_app.gl_sbti_targets.target_type IS 'SBTi target type: near_term, long_term, net_zero';
COMMENT ON COLUMN sbti_app.gl_sbti_targets.target_status IS 'Target lifecycle: committed, submitted, validated, published, expired, withdrawn';
COMMENT ON COLUMN sbti_app.gl_sbti_targets.target_method IS 'SBTi target method: aca (absolute contraction approach), sda (sectoral decarbonization approach), economic_intensity, physical_intensity, supplier_engagement';
COMMENT ON COLUMN sbti_app.gl_sbti_targets.pathway_alignment IS 'Temperature pathway alignment: 1_5c (1.5 degrees C), well_below_2c';
COMMENT ON COLUMN sbti_app.gl_sbti_target_scopes.scope IS 'Target scope: scope1, scope2, scope1_2, scope3, flag';
COMMENT ON COLUMN sbti_app.gl_sbti_validation_results.overall_result IS 'Validation result: pass, fail, partial';
COMMENT ON COLUMN sbti_app.gl_sbti_criteria_checks.criterion_id IS 'SBTi criterion ID: c1-c28 (corporate), nz_c1-nz_c14 (net-zero)';
COMMENT ON COLUMN sbti_app.gl_sbti_criteria_checks.result IS 'Criterion result: pass, fail, not_applicable, insufficient_data';
COMMENT ON COLUMN sbti_app.gl_sbti_scope3_categories.category IS 'Scope 3 category: cat1-cat15 per GHG Protocol';
COMMENT ON COLUMN sbti_app.gl_sbti_recalculations.trigger_type IS 'Recalculation trigger: structural_change, methodology_change, error_correction, outsourcing, insourcing, acquisition, divestiture, organic_growth';
COMMENT ON COLUMN sbti_app.gl_sbti_five_year_reviews.review_status IS 'Review status: upcoming, in_progress, completed, expired';
COMMENT ON COLUMN sbti_app.gl_sbti_five_year_reviews.outcome IS 'Review outcome: renewed, updated, expired';
COMMENT ON COLUMN sbti_app.gl_sbti_fi_portfolio_holdings.asset_class IS 'PCAF asset class: listed_equity, corporate_bond, private_equity, project_finance, commercial_real_estate, mortgage, motor_vehicle_loan, sovereign_bond';
COMMENT ON COLUMN sbti_app.gl_sbti_fi_portfolio_holdings.pcaf_data_quality IS 'PCAF data quality score 1-5 (1 = highest quality, 5 = lowest)';
COMMENT ON COLUMN sbti_app.gl_sbti_fi_engagement.engagement_type IS 'Engagement type: direct, collaborative, escalation, proxy_voting, shareholder_resolution';
COMMENT ON COLUMN sbti_app.gl_sbti_framework_mappings.framework IS 'Target framework: cdp, tcfd, csrd, iso14064, ghg_protocol, issb, sec_climate, nzba';
COMMENT ON COLUMN sbti_app.gl_sbti_gap_items.severity IS 'Gap severity: critical, high, medium, low';
COMMENT ON COLUMN sbti_app.gl_sbti_gap_items.gap_type IS 'Gap type: data_missing, data_quality, methodology, coverage, boundary, verification, governance, reporting, other';
COMMENT ON COLUMN sbti_app.gl_sbti_gap_items.status IS 'Gap status: open, in_progress, resolved, deferred, not_applicable';
COMMENT ON COLUMN sbti_app.gl_sbti_reports.report_type IS 'Report type: target_submission, validation_report, progress_report, gap_analysis, executive_summary, fi_portfolio, temperature_alignment, flag_assessment, framework_mapping';

-- =============================================================================
-- End of V087: GL-SBTi-APP v1.0 Application Service Schema
-- =============================================================================
-- Summary:
--   25 tables created
--   3 hypertables (validation_results, progress_records, temperature_scores)
--   2 continuous aggregates (annual_progress_summary, quarterly_temperature_trends)
--   22 update triggers
--   100+ B-tree indexes
--   20+ GIN indexes on JSONB columns
--   3 retention policies (10-year retention)
--   3 compression policies (90-day threshold)
--   37 security permissions
--   Security grants for greenlang_app and greenlang_readonly
-- Previous: V086__tcfd_app_service.sql
-- =============================================================================
