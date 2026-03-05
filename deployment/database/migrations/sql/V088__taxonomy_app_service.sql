-- =============================================================================
-- V088: GL-Taxonomy-APP v1.0 Application Service Schema
-- =============================================================================
-- Application: GL-Taxonomy-APP (EU Taxonomy Alignment & Green Investment Ratio)
-- Date:        March 2026
--
-- Application-level tables for the EU Taxonomy Regulation alignment platform.
-- Covers economic activity eligibility screening, Substantial Contribution (SC)
-- assessment per environmental objective, Do No Significant Harm (DNSH) matrix
-- evaluation, Minimum Safeguards (MS) checks (human rights, anti-corruption,
-- taxation, fair competition), KPI calculation (turnover/CapEx/OpEx), Green
-- Asset Ratio (GAR) and Banking-book Taxonomy Alignment Ratio (BTAR) for
-- financial institutions, exposure classification, portfolio alignment,
-- CapEx plans, regulatory version management, evidence tracking, data quality,
-- gap analysis, and disclosure reporting (Article 8, EBA Pillar 3).
--
-- EXTENDS:
--   V051-V070: AGENT-MRV calculation agents (Scope 1-3)
--   V080: Scope 3 Category Mapper Service
--   V081: Audit Trail & Lineage Service
--   V083: GL-GHG-APP v1.0
--   V084: GL-ISO14064-APP v1.0
--   V085: GL-CDP-APP v1.0
--   V086: GL-TCFD-APP v1.0
--   V087: GL-SBTi-APP v1.0
--
-- These tables sit in the taxonomy_app schema and integrate with the
-- underlying MRV agent data for auto-population of environmental metrics
-- into eligibility screening, SC/DNSH assessment, and KPI calculations.
-- =============================================================================
-- Tables (25):
--   1.  gl_tax_organizations                 - Organization profiles
--   2.  gl_tax_economic_activities           - EU Taxonomy activity catalog
--   3.  gl_tax_nace_mappings                 - NACE code to activity mappings
--   4.  gl_tax_eligibility_screenings        - Eligibility screening runs
--   5.  gl_tax_screening_results             - Per-activity screening results
--   6.  gl_tax_sc_assessments                - Substantial contribution (HT)
--   7.  gl_tax_tsc_evaluations               - Technical screening criteria
--   8.  gl_tax_dnsh_assessments              - Do No Significant Harm
--   9.  gl_tax_dnsh_objective_results        - Per-objective DNSH results
--  10.  gl_tax_climate_risk_assessments      - Climate risk (adaptation DNSH)
--  11.  gl_tax_minimum_safeguard_assessments - Minimum safeguards
--  12.  gl_tax_safeguard_topic_results       - Per-topic safeguard results
--  13.  gl_tax_kpi_calculations              - KPI (turnover/capex/opex)
--  14.  gl_tax_activity_financials           - Per-activity financial data
--  15.  gl_tax_capex_plans                   - CapEx plan tracking
--  16.  gl_tax_gar_calculations              - Green Asset Ratio (HT)
--  17.  gl_tax_exposures                     - FI exposure records
--  18.  gl_tax_alignment_results             - Full alignment results (HT)
--  19.  gl_tax_portfolio_alignments          - Portfolio-level alignment
--  20.  gl_tax_reports                       - Generated reports
--  21.  gl_tax_evidence_items                - Evidence/document tracking
--  22.  gl_tax_regulatory_versions           - Delegated act versions
--  23.  gl_tax_data_quality_scores           - Data quality assessment
--  24.  gl_tax_gap_assessments               - Gap analysis results
--  25.  gl_tax_gap_items                     - Individual gap items
--
-- Hypertables (3):
--   gl_tax_sc_assessments       on assessment_date  (chunk: 3 months)
--   gl_tax_gar_calculations     on calculation_date (chunk: 3 months)
--   gl_tax_alignment_results    on alignment_date   (chunk: 3 months)
--
-- Continuous Aggregates (2):
--   taxonomy_app.quarterly_alignment_summary  - Quarterly alignment trends
--   taxonomy_app.annual_gar_trends            - Annual GAR trends
--
-- Also includes: 170+ indexes (B-tree, GIN), update triggers, security
-- grants, retention policies, compression policies, permissions, and comments.
-- Previous: V087__sbti_app_service.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS taxonomy_app;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION taxonomy_app.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: taxonomy_app.gl_tax_organizations
-- =============================================================================
-- Organization profiles for EU Taxonomy alignment.  Each organization
-- represents the reporting entity, with entity type (financial vs non-
-- financial), sector classification, LEI, and NFRD/CSRD reporting flags
-- that determine applicable disclosure templates and KPI requirements.

CREATE TABLE taxonomy_app.gl_tax_organizations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID,
    name                    VARCHAR(500)    NOT NULL,
    entity_type             VARCHAR(30)     NOT NULL DEFAULT 'non_financial',
    sector                  VARCHAR(100),
    country                 VARCHAR(3),
    lei                     VARCHAR(20),
    nfrd_reporting          BOOLEAN         NOT NULL DEFAULT FALSE,
    csrd_reporting          BOOLEAN         NOT NULL DEFAULT FALSE,
    employee_count          INTEGER,
    annual_revenue          DECIMAL(18,2),
    total_assets            DECIMAL(18,2),
    settings                JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_org_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tax_org_entity_type CHECK (
        entity_type IN ('financial', 'non_financial', 'insurance', 'asset_manager')
    ),
    CONSTRAINT chk_tax_org_country_length CHECK (
        country IS NULL OR (LENGTH(TRIM(country)) >= 2 AND LENGTH(TRIM(country)) <= 3)
    ),
    CONSTRAINT chk_tax_org_revenue_non_neg CHECK (
        annual_revenue IS NULL OR annual_revenue >= 0
    ),
    CONSTRAINT chk_tax_org_assets_non_neg CHECK (
        total_assets IS NULL OR total_assets >= 0
    ),
    CONSTRAINT chk_tax_org_employees_non_neg CHECK (
        employee_count IS NULL OR employee_count >= 0
    )
);

-- Indexes
CREATE INDEX idx_tax_org_tenant ON taxonomy_app.gl_tax_organizations(tenant_id);
CREATE INDEX idx_tax_org_name ON taxonomy_app.gl_tax_organizations(name);
CREATE INDEX idx_tax_org_entity_type ON taxonomy_app.gl_tax_organizations(entity_type);
CREATE INDEX idx_tax_org_sector ON taxonomy_app.gl_tax_organizations(sector);
CREATE INDEX idx_tax_org_country ON taxonomy_app.gl_tax_organizations(country);
CREATE INDEX idx_tax_org_lei ON taxonomy_app.gl_tax_organizations(lei);
CREATE INDEX idx_tax_org_nfrd ON taxonomy_app.gl_tax_organizations(nfrd_reporting);
CREATE INDEX idx_tax_org_csrd ON taxonomy_app.gl_tax_organizations(csrd_reporting);
CREATE INDEX idx_tax_org_created_at ON taxonomy_app.gl_tax_organizations(created_at DESC);
CREATE INDEX idx_tax_org_settings ON taxonomy_app.gl_tax_organizations USING GIN(settings);
CREATE INDEX idx_tax_org_metadata ON taxonomy_app.gl_tax_organizations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_org_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_organizations
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 2: taxonomy_app.gl_tax_economic_activities
-- =============================================================================
-- EU Taxonomy economic activity catalog.  Each activity has a unique
-- activity_code (e.g., "CCM_4.1"), associated NACE codes, environmental
-- objectives it can contribute to, activity type classification
-- (own_performance, enabling, transitional), and embedded SC/DNSH criteria
-- from the Climate Delegated Act or Environmental Delegated Act.

CREATE TABLE taxonomy_app.gl_tax_economic_activities (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    activity_code           VARCHAR(20)     NOT NULL UNIQUE,
    nace_codes              TEXT[]          DEFAULT '{}',
    sector                  VARCHAR(100)    NOT NULL,
    name                    TEXT            NOT NULL,
    description             TEXT,
    objectives              TEXT[]          DEFAULT '{}',
    activity_type           VARCHAR(20)     NOT NULL DEFAULT 'own_performance',
    delegated_act           VARCHAR(50)     NOT NULL DEFAULT 'climate',
    sc_criteria             JSONB           DEFAULT '{}',
    dnsh_criteria           JSONB           DEFAULT '{}',
    effective_date          DATE,
    version                 VARCHAR(20)     DEFAULT '1.0',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_ea_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_ea_name_not_empty CHECK (
        LENGTH(TRIM(name)) > 0
    ),
    CONSTRAINT chk_tax_ea_sector_not_empty CHECK (
        LENGTH(TRIM(sector)) > 0
    ),
    CONSTRAINT chk_tax_ea_activity_type CHECK (
        activity_type IN ('own_performance', 'enabling', 'transitional')
    ),
    CONSTRAINT chk_tax_ea_delegated_act CHECK (
        delegated_act IN ('climate', 'environmental', 'climate_amending', 'complementary')
    )
);

-- Indexes
CREATE INDEX idx_tax_ea_code ON taxonomy_app.gl_tax_economic_activities(activity_code);
CREATE INDEX idx_tax_ea_sector ON taxonomy_app.gl_tax_economic_activities(sector);
CREATE INDEX idx_tax_ea_type ON taxonomy_app.gl_tax_economic_activities(activity_type);
CREATE INDEX idx_tax_ea_delegated_act ON taxonomy_app.gl_tax_economic_activities(delegated_act);
CREATE INDEX idx_tax_ea_effective_date ON taxonomy_app.gl_tax_economic_activities(effective_date);
CREATE INDEX idx_tax_ea_version ON taxonomy_app.gl_tax_economic_activities(version);
CREATE INDEX idx_tax_ea_created_at ON taxonomy_app.gl_tax_economic_activities(created_at DESC);
CREATE INDEX idx_tax_ea_nace ON taxonomy_app.gl_tax_economic_activities USING GIN(nace_codes);
CREATE INDEX idx_tax_ea_objectives ON taxonomy_app.gl_tax_economic_activities USING GIN(objectives);
CREATE INDEX idx_tax_ea_sc_criteria ON taxonomy_app.gl_tax_economic_activities USING GIN(sc_criteria);
CREATE INDEX idx_tax_ea_dnsh_criteria ON taxonomy_app.gl_tax_economic_activities USING GIN(dnsh_criteria);
CREATE INDEX idx_tax_ea_metadata ON taxonomy_app.gl_tax_economic_activities USING GIN(metadata);

-- =============================================================================
-- Table 3: taxonomy_app.gl_tax_nace_mappings
-- =============================================================================
-- NACE code to EU Taxonomy activity mappings.  Provides the hierarchical
-- NACE structure (levels 1-4) with parent references and the list of
-- taxonomy activities that each NACE code maps to.  Used by the eligibility
-- screening engine to match corporate activities to taxonomy activities.

CREATE TABLE taxonomy_app.gl_tax_nace_mappings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    nace_code               VARCHAR(10)     NOT NULL,
    nace_description        TEXT            NOT NULL,
    nace_level              INTEGER         NOT NULL,
    parent_code             VARCHAR(10),
    taxonomy_activities     TEXT[]          DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_nm_code_not_empty CHECK (
        LENGTH(TRIM(nace_code)) > 0
    ),
    CONSTRAINT chk_tax_nm_desc_not_empty CHECK (
        LENGTH(TRIM(nace_description)) > 0
    ),
    CONSTRAINT chk_tax_nm_level_range CHECK (
        nace_level >= 1 AND nace_level <= 4
    ),
    UNIQUE(nace_code)
);

-- Indexes
CREATE INDEX idx_tax_nm_code ON taxonomy_app.gl_tax_nace_mappings(nace_code);
CREATE INDEX idx_tax_nm_level ON taxonomy_app.gl_tax_nace_mappings(nace_level);
CREATE INDEX idx_tax_nm_parent ON taxonomy_app.gl_tax_nace_mappings(parent_code);
CREATE INDEX idx_tax_nm_created_at ON taxonomy_app.gl_tax_nace_mappings(created_at DESC);
CREATE INDEX idx_tax_nm_activities ON taxonomy_app.gl_tax_nace_mappings USING GIN(taxonomy_activities);

-- =============================================================================
-- Table 4: taxonomy_app.gl_tax_eligibility_screenings
-- =============================================================================
-- Eligibility screening runs that evaluate which of an organization's
-- economic activities are taxonomy-eligible.  Tracks total/eligible/not-
-- eligible/de-minimis-excluded counts and overall screening status.

CREATE TABLE taxonomy_app.gl_tax_eligibility_screenings (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    screening_date          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_activities        INTEGER         NOT NULL DEFAULT 0,
    eligible_count          INTEGER         NOT NULL DEFAULT 0,
    not_eligible_count      INTEGER         NOT NULL DEFAULT 0,
    de_minimis_excluded     INTEGER         NOT NULL DEFAULT 0,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_es_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_es_total_non_neg CHECK (total_activities >= 0),
    CONSTRAINT chk_tax_es_eligible_non_neg CHECK (eligible_count >= 0),
    CONSTRAINT chk_tax_es_not_eligible_non_neg CHECK (not_eligible_count >= 0),
    CONSTRAINT chk_tax_es_de_minimis_non_neg CHECK (de_minimis_excluded >= 0),
    CONSTRAINT chk_tax_es_counts_consistent CHECK (
        total_activities = eligible_count + not_eligible_count + de_minimis_excluded
    ),
    CONSTRAINT chk_tax_es_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'approved')
    )
);

-- Indexes
CREATE INDEX idx_tax_es_org ON taxonomy_app.gl_tax_eligibility_screenings(org_id);
CREATE INDEX idx_tax_es_tenant ON taxonomy_app.gl_tax_eligibility_screenings(tenant_id);
CREATE INDEX idx_tax_es_period ON taxonomy_app.gl_tax_eligibility_screenings(period);
CREATE INDEX idx_tax_es_org_period ON taxonomy_app.gl_tax_eligibility_screenings(org_id, period);
CREATE INDEX idx_tax_es_status ON taxonomy_app.gl_tax_eligibility_screenings(status);
CREATE INDEX idx_tax_es_date ON taxonomy_app.gl_tax_eligibility_screenings(screening_date DESC);
CREATE INDEX idx_tax_es_created_at ON taxonomy_app.gl_tax_eligibility_screenings(created_at DESC);
CREATE INDEX idx_tax_es_metadata ON taxonomy_app.gl_tax_eligibility_screenings USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_es_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_eligibility_screenings
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 5: taxonomy_app.gl_tax_screening_results
-- =============================================================================
-- Per-activity eligibility screening results linked to a screening run.
-- Records whether each activity is taxonomy-eligible, which objectives it
-- may contribute to, the applicable delegated act, confidence score, and
-- whether the activity was excluded under de minimis thresholds.

CREATE TABLE taxonomy_app.gl_tax_screening_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    screening_id            UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_eligibility_screenings(id) ON DELETE CASCADE,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    eligible                BOOLEAN         NOT NULL DEFAULT FALSE,
    objectives              TEXT[]          DEFAULT '{}',
    delegated_act           VARCHAR(50),
    confidence              DECIMAL(5,2)    DEFAULT 0,
    de_minimis              BOOLEAN         NOT NULL DEFAULT FALSE,
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_sr_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_sr_confidence_range CHECK (
        confidence >= 0 AND confidence <= 100
    ),
    CONSTRAINT chk_tax_sr_delegated_act CHECK (
        delegated_act IS NULL OR delegated_act IN ('climate', 'environmental', 'climate_amending', 'complementary')
    )
);

-- Indexes
CREATE INDEX idx_tax_sr_screening ON taxonomy_app.gl_tax_screening_results(screening_id);
CREATE INDEX idx_tax_sr_tenant ON taxonomy_app.gl_tax_screening_results(tenant_id);
CREATE INDEX idx_tax_sr_activity ON taxonomy_app.gl_tax_screening_results(activity_code);
CREATE INDEX idx_tax_sr_eligible ON taxonomy_app.gl_tax_screening_results(eligible);
CREATE INDEX idx_tax_sr_de_minimis ON taxonomy_app.gl_tax_screening_results(de_minimis);
CREATE INDEX idx_tax_sr_created_at ON taxonomy_app.gl_tax_screening_results(created_at DESC);
CREATE INDEX idx_tax_sr_objectives ON taxonomy_app.gl_tax_screening_results USING GIN(objectives);
CREATE INDEX idx_tax_sr_metadata ON taxonomy_app.gl_tax_screening_results USING GIN(metadata);

-- =============================================================================
-- Table 6: taxonomy_app.gl_tax_sc_assessments (HYPERTABLE)
-- =============================================================================
-- Substantial Contribution (SC) assessments partitioned by assessment_date
-- for time-series querying.  Each record evaluates whether an economic
-- activity makes a substantial contribution to one of the six environmental
-- objectives through own performance, enabling, or transitional criteria.

CREATE TABLE taxonomy_app.gl_tax_sc_assessments (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    objective               VARCHAR(50)     NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    sc_type                 VARCHAR(20)     NOT NULL DEFAULT 'own_performance',
    overall_pass            BOOLEAN         NOT NULL DEFAULT FALSE,
    threshold_checks        JSONB           DEFAULT '{}',
    evidence_items          JSONB           DEFAULT '[]',
    assessor                VARCHAR(200),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_sca_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_sca_objective CHECK (
        objective IN (
            'climate_mitigation', 'climate_adaptation',
            'water_marine', 'circular_economy',
            'pollution_prevention', 'biodiversity'
        )
    ),
    CONSTRAINT chk_tax_sca_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'approved', 'rejected')
    ),
    CONSTRAINT chk_tax_sca_sc_type CHECK (
        sc_type IN ('own_performance', 'enabling', 'transitional')
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('taxonomy_app.gl_tax_sc_assessments', 'assessment_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tax_sca_org ON taxonomy_app.gl_tax_sc_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_tax_sca_tenant ON taxonomy_app.gl_tax_sc_assessments(tenant_id, assessment_date DESC);
CREATE INDEX idx_tax_sca_activity ON taxonomy_app.gl_tax_sc_assessments(activity_code, assessment_date DESC);
CREATE INDEX idx_tax_sca_objective ON taxonomy_app.gl_tax_sc_assessments(objective, assessment_date DESC);
CREATE INDEX idx_tax_sca_status ON taxonomy_app.gl_tax_sc_assessments(status, assessment_date DESC);
CREATE INDEX idx_tax_sca_sc_type ON taxonomy_app.gl_tax_sc_assessments(sc_type, assessment_date DESC);
CREATE INDEX idx_tax_sca_pass ON taxonomy_app.gl_tax_sc_assessments(overall_pass, assessment_date DESC);
CREATE INDEX idx_tax_sca_org_activity ON taxonomy_app.gl_tax_sc_assessments(org_id, activity_code, assessment_date DESC);
CREATE INDEX idx_tax_sca_threshold ON taxonomy_app.gl_tax_sc_assessments USING GIN(threshold_checks);
CREATE INDEX idx_tax_sca_evidence ON taxonomy_app.gl_tax_sc_assessments USING GIN(evidence_items);
CREATE INDEX idx_tax_sca_metadata ON taxonomy_app.gl_tax_sc_assessments USING GIN(metadata);

-- =============================================================================
-- Table 7: taxonomy_app.gl_tax_tsc_evaluations
-- =============================================================================
-- Technical Screening Criteria (TSC) evaluations linked to an SC assessment.
-- Each record evaluates a single criterion against its threshold, recording
-- the actual measured value, unit, pass/fail result, and evidence reference.

CREATE TABLE taxonomy_app.gl_tax_tsc_evaluations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL,
    tenant_id               UUID,
    criterion_id            VARCHAR(50)     NOT NULL,
    description             TEXT            NOT NULL,
    threshold_value         DECIMAL(15,4),
    actual_value            DECIMAL(15,4),
    unit                    VARCHAR(50),
    pass_result             BOOLEAN         NOT NULL DEFAULT FALSE,
    evidence_ref            TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_tsc_criterion_not_empty CHECK (
        LENGTH(TRIM(criterion_id)) > 0
    ),
    CONSTRAINT chk_tax_tsc_desc_not_empty CHECK (
        LENGTH(TRIM(description)) > 0
    )
);

-- Indexes
CREATE INDEX idx_tax_tsc_assessment ON taxonomy_app.gl_tax_tsc_evaluations(assessment_id);
CREATE INDEX idx_tax_tsc_tenant ON taxonomy_app.gl_tax_tsc_evaluations(tenant_id);
CREATE INDEX idx_tax_tsc_criterion ON taxonomy_app.gl_tax_tsc_evaluations(criterion_id);
CREATE INDEX idx_tax_tsc_pass ON taxonomy_app.gl_tax_tsc_evaluations(pass_result);
CREATE INDEX idx_tax_tsc_created_at ON taxonomy_app.gl_tax_tsc_evaluations(created_at DESC);
CREATE INDEX idx_tax_tsc_metadata ON taxonomy_app.gl_tax_tsc_evaluations USING GIN(metadata);

-- =============================================================================
-- Table 8: taxonomy_app.gl_tax_dnsh_assessments
-- =============================================================================
-- Do No Significant Harm (DNSH) assessments evaluating whether an eligible
-- activity passes DNSH checks against all other environmental objectives.
-- Stores aggregate pass/fail, per-objective results as JSONB, and evidence.

CREATE TABLE taxonomy_app.gl_tax_dnsh_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    sc_objective            VARCHAR(50)     NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    overall_pass            BOOLEAN         NOT NULL DEFAULT FALSE,
    objective_results       JSONB           DEFAULT '{}',
    evidence_items          JSONB           DEFAULT '[]',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_dnsh_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_dnsh_sc_objective CHECK (
        sc_objective IN (
            'climate_mitigation', 'climate_adaptation',
            'water_marine', 'circular_economy',
            'pollution_prevention', 'biodiversity'
        )
    ),
    CONSTRAINT chk_tax_dnsh_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'approved', 'rejected')
    )
);

-- Indexes
CREATE INDEX idx_tax_dnsh_org ON taxonomy_app.gl_tax_dnsh_assessments(org_id);
CREATE INDEX idx_tax_dnsh_tenant ON taxonomy_app.gl_tax_dnsh_assessments(tenant_id);
CREATE INDEX idx_tax_dnsh_activity ON taxonomy_app.gl_tax_dnsh_assessments(activity_code);
CREATE INDEX idx_tax_dnsh_sc_objective ON taxonomy_app.gl_tax_dnsh_assessments(sc_objective);
CREATE INDEX idx_tax_dnsh_pass ON taxonomy_app.gl_tax_dnsh_assessments(overall_pass);
CREATE INDEX idx_tax_dnsh_status ON taxonomy_app.gl_tax_dnsh_assessments(status);
CREATE INDEX idx_tax_dnsh_date ON taxonomy_app.gl_tax_dnsh_assessments(assessment_date DESC);
CREATE INDEX idx_tax_dnsh_org_activity ON taxonomy_app.gl_tax_dnsh_assessments(org_id, activity_code);
CREATE INDEX idx_tax_dnsh_created_at ON taxonomy_app.gl_tax_dnsh_assessments(created_at DESC);
CREATE INDEX idx_tax_dnsh_obj_results ON taxonomy_app.gl_tax_dnsh_assessments USING GIN(objective_results);
CREATE INDEX idx_tax_dnsh_evidence ON taxonomy_app.gl_tax_dnsh_assessments USING GIN(evidence_items);
CREATE INDEX idx_tax_dnsh_metadata ON taxonomy_app.gl_tax_dnsh_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_dnsh_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_dnsh_assessments
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 9: taxonomy_app.gl_tax_dnsh_objective_results
-- =============================================================================
-- Per-objective DNSH results linked to a DNSH assessment.  Each record
-- evaluates DNSH against one environmental objective (the ones that the
-- activity does NOT substantially contribute to) with pass/fail/not_applicable
-- status, criteria checks, and evidence items.

CREATE TABLE taxonomy_app.gl_tax_dnsh_objective_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_dnsh_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID,
    objective               VARCHAR(50)     NOT NULL,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'pending',
    criteria_checks         JSONB           DEFAULT '{}',
    evidence_items          JSONB           DEFAULT '[]',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_dor_objective CHECK (
        objective IN (
            'climate_mitigation', 'climate_adaptation',
            'water_marine', 'circular_economy',
            'pollution_prevention', 'biodiversity'
        )
    ),
    CONSTRAINT chk_tax_dor_status CHECK (
        status IN ('pass', 'fail', 'not_applicable', 'pending')
    ),
    UNIQUE(assessment_id, objective)
);

-- Indexes
CREATE INDEX idx_tax_dor_assessment ON taxonomy_app.gl_tax_dnsh_objective_results(assessment_id);
CREATE INDEX idx_tax_dor_tenant ON taxonomy_app.gl_tax_dnsh_objective_results(tenant_id);
CREATE INDEX idx_tax_dor_objective ON taxonomy_app.gl_tax_dnsh_objective_results(objective);
CREATE INDEX idx_tax_dor_status ON taxonomy_app.gl_tax_dnsh_objective_results(status);
CREATE INDEX idx_tax_dor_created_at ON taxonomy_app.gl_tax_dnsh_objective_results(created_at DESC);
CREATE INDEX idx_tax_dor_criteria ON taxonomy_app.gl_tax_dnsh_objective_results USING GIN(criteria_checks);
CREATE INDEX idx_tax_dor_evidence ON taxonomy_app.gl_tax_dnsh_objective_results USING GIN(evidence_items);

-- =============================================================================
-- Table 10: taxonomy_app.gl_tax_climate_risk_assessments
-- =============================================================================
-- Climate risk assessments supporting DNSH for climate adaptation objective.
-- Identifies physical climate risks at specific locations and time horizons,
-- records adaptation solutions, residual risks, and overall assessment status.

CREATE TABLE taxonomy_app.gl_tax_climate_risk_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    location                TEXT,
    time_horizon            VARCHAR(20)     NOT NULL DEFAULT 'long_term',
    physical_risks          JSONB           DEFAULT '{}',
    adaptation_solutions    JSONB           DEFAULT '[]',
    residual_risks          JSONB           DEFAULT '{}',
    overall_status          VARCHAR(20)     NOT NULL DEFAULT 'pending',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_cra_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_cra_time_horizon CHECK (
        time_horizon IN ('short_term', 'medium_term', 'long_term')
    ),
    CONSTRAINT chk_tax_cra_status CHECK (
        overall_status IN ('pending', 'low_risk', 'managed', 'high_risk', 'unmanaged')
    )
);

-- Indexes
CREATE INDEX idx_tax_cra_org ON taxonomy_app.gl_tax_climate_risk_assessments(org_id);
CREATE INDEX idx_tax_cra_tenant ON taxonomy_app.gl_tax_climate_risk_assessments(tenant_id);
CREATE INDEX idx_tax_cra_activity ON taxonomy_app.gl_tax_climate_risk_assessments(activity_code);
CREATE INDEX idx_tax_cra_horizon ON taxonomy_app.gl_tax_climate_risk_assessments(time_horizon);
CREATE INDEX idx_tax_cra_status ON taxonomy_app.gl_tax_climate_risk_assessments(overall_status);
CREATE INDEX idx_tax_cra_org_activity ON taxonomy_app.gl_tax_climate_risk_assessments(org_id, activity_code);
CREATE INDEX idx_tax_cra_created_at ON taxonomy_app.gl_tax_climate_risk_assessments(created_at DESC);
CREATE INDEX idx_tax_cra_physical ON taxonomy_app.gl_tax_climate_risk_assessments USING GIN(physical_risks);
CREATE INDEX idx_tax_cra_adaptation ON taxonomy_app.gl_tax_climate_risk_assessments USING GIN(adaptation_solutions);
CREATE INDEX idx_tax_cra_residual ON taxonomy_app.gl_tax_climate_risk_assessments USING GIN(residual_risks);
CREATE INDEX idx_tax_cra_metadata ON taxonomy_app.gl_tax_climate_risk_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_cra_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_climate_risk_assessments
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 11: taxonomy_app.gl_tax_minimum_safeguard_assessments
-- =============================================================================
-- Minimum Safeguard (MS) assessments evaluating alignment with OECD
-- Guidelines, UN Guiding Principles, ILO core conventions, and ICCPR.
-- Covers four topics: human rights, anti-corruption, taxation, and fair
-- competition with procedural and outcome-based checks.

CREATE TABLE taxonomy_app.gl_tax_minimum_safeguard_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    overall_pass            BOOLEAN         NOT NULL DEFAULT FALSE,
    topics                  JSONB           DEFAULT '{}',
    evidence_items          JSONB           DEFAULT '[]',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_msa_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'approved', 'rejected')
    )
);

-- Indexes
CREATE INDEX idx_tax_msa_org ON taxonomy_app.gl_tax_minimum_safeguard_assessments(org_id);
CREATE INDEX idx_tax_msa_tenant ON taxonomy_app.gl_tax_minimum_safeguard_assessments(tenant_id);
CREATE INDEX idx_tax_msa_date ON taxonomy_app.gl_tax_minimum_safeguard_assessments(assessment_date DESC);
CREATE INDEX idx_tax_msa_pass ON taxonomy_app.gl_tax_minimum_safeguard_assessments(overall_pass);
CREATE INDEX idx_tax_msa_status ON taxonomy_app.gl_tax_minimum_safeguard_assessments(status);
CREATE INDEX idx_tax_msa_org_date ON taxonomy_app.gl_tax_minimum_safeguard_assessments(org_id, assessment_date DESC);
CREATE INDEX idx_tax_msa_created_at ON taxonomy_app.gl_tax_minimum_safeguard_assessments(created_at DESC);
CREATE INDEX idx_tax_msa_topics ON taxonomy_app.gl_tax_minimum_safeguard_assessments USING GIN(topics);
CREATE INDEX idx_tax_msa_evidence ON taxonomy_app.gl_tax_minimum_safeguard_assessments USING GIN(evidence_items);
CREATE INDEX idx_tax_msa_metadata ON taxonomy_app.gl_tax_minimum_safeguard_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_msa_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_minimum_safeguard_assessments
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 12: taxonomy_app.gl_tax_safeguard_topic_results
-- =============================================================================
-- Per-topic minimum safeguard results linked to an MS assessment.  Each
-- record evaluates one of the four safeguard topics with separate procedural
-- (due diligence processes) and outcome-based (no adverse findings) checks.

CREATE TABLE taxonomy_app.gl_tax_safeguard_topic_results (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_minimum_safeguard_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID,
    topic                   VARCHAR(30)     NOT NULL,
    procedural_pass         BOOLEAN         NOT NULL DEFAULT FALSE,
    outcome_pass            BOOLEAN         NOT NULL DEFAULT FALSE,
    overall_pass            BOOLEAN         NOT NULL DEFAULT FALSE,
    checks                  JSONB           DEFAULT '{}',
    evidence_items          JSONB           DEFAULT '[]',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_str_topic CHECK (
        topic IN ('human_rights', 'anti_corruption', 'taxation', 'fair_competition')
    ),
    UNIQUE(assessment_id, topic)
);

-- Indexes
CREATE INDEX idx_tax_str_assessment ON taxonomy_app.gl_tax_safeguard_topic_results(assessment_id);
CREATE INDEX idx_tax_str_tenant ON taxonomy_app.gl_tax_safeguard_topic_results(tenant_id);
CREATE INDEX idx_tax_str_topic ON taxonomy_app.gl_tax_safeguard_topic_results(topic);
CREATE INDEX idx_tax_str_procedural ON taxonomy_app.gl_tax_safeguard_topic_results(procedural_pass);
CREATE INDEX idx_tax_str_outcome ON taxonomy_app.gl_tax_safeguard_topic_results(outcome_pass);
CREATE INDEX idx_tax_str_overall ON taxonomy_app.gl_tax_safeguard_topic_results(overall_pass);
CREATE INDEX idx_tax_str_created_at ON taxonomy_app.gl_tax_safeguard_topic_results(created_at DESC);
CREATE INDEX idx_tax_str_checks ON taxonomy_app.gl_tax_safeguard_topic_results USING GIN(checks);
CREATE INDEX idx_tax_str_evidence ON taxonomy_app.gl_tax_safeguard_topic_results USING GIN(evidence_items);

-- =============================================================================
-- Table 13: taxonomy_app.gl_tax_kpi_calculations
-- =============================================================================
-- KPI calculations for the three mandatory disclosure metrics: turnover,
-- CapEx, and OpEx.  Each record computes eligible/aligned/total amounts
-- and the resulting KPI percentage, with an objective-level breakdown.

CREATE TABLE taxonomy_app.gl_tax_kpi_calculations (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    kpi_type                VARCHAR(10)     NOT NULL,
    calculation_date        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    eligible_amount         DECIMAL(20,2)   NOT NULL DEFAULT 0,
    aligned_amount          DECIMAL(20,2)   NOT NULL DEFAULT 0,
    total_amount            DECIMAL(20,2)   NOT NULL DEFAULT 0,
    kpi_percentage          DECIMAL(7,4)    NOT NULL DEFAULT 0,
    objective_breakdown     JSONB           DEFAULT '{}',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_kpi_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_kpi_type CHECK (
        kpi_type IN ('turnover', 'capex', 'opex')
    ),
    CONSTRAINT chk_tax_kpi_eligible_non_neg CHECK (eligible_amount >= 0),
    CONSTRAINT chk_tax_kpi_aligned_non_neg CHECK (aligned_amount >= 0),
    CONSTRAINT chk_tax_kpi_total_non_neg CHECK (total_amount >= 0),
    CONSTRAINT chk_tax_kpi_pct_range CHECK (
        kpi_percentage >= 0 AND kpi_percentage <= 100
    ),
    CONSTRAINT chk_tax_kpi_aligned_le_eligible CHECK (
        aligned_amount <= eligible_amount
    ),
    CONSTRAINT chk_tax_kpi_eligible_le_total CHECK (
        eligible_amount <= total_amount
    ),
    CONSTRAINT chk_tax_kpi_status CHECK (
        status IN ('draft', 'calculated', 'approved', 'submitted')
    ),
    UNIQUE(org_id, period, kpi_type)
);

-- Indexes
CREATE INDEX idx_tax_kpi_org ON taxonomy_app.gl_tax_kpi_calculations(org_id);
CREATE INDEX idx_tax_kpi_tenant ON taxonomy_app.gl_tax_kpi_calculations(tenant_id);
CREATE INDEX idx_tax_kpi_period ON taxonomy_app.gl_tax_kpi_calculations(period);
CREATE INDEX idx_tax_kpi_type ON taxonomy_app.gl_tax_kpi_calculations(kpi_type);
CREATE INDEX idx_tax_kpi_org_period ON taxonomy_app.gl_tax_kpi_calculations(org_id, period);
CREATE INDEX idx_tax_kpi_org_type ON taxonomy_app.gl_tax_kpi_calculations(org_id, kpi_type);
CREATE INDEX idx_tax_kpi_status ON taxonomy_app.gl_tax_kpi_calculations(status);
CREATE INDEX idx_tax_kpi_date ON taxonomy_app.gl_tax_kpi_calculations(calculation_date DESC);
CREATE INDEX idx_tax_kpi_created_at ON taxonomy_app.gl_tax_kpi_calculations(created_at DESC);
CREATE INDEX idx_tax_kpi_obj_breakdown ON taxonomy_app.gl_tax_kpi_calculations USING GIN(objective_breakdown);
CREATE INDEX idx_tax_kpi_metadata ON taxonomy_app.gl_tax_kpi_calculations USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_kpi_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_kpi_calculations
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 14: taxonomy_app.gl_tax_activity_financials
-- =============================================================================
-- Per-activity financial data linking economic activities to their turnover,
-- CapEx, and OpEx amounts for KPI calculation.  Tracks eligibility and
-- alignment status per activity, enabling activity-level KPI drill-down.

CREATE TABLE taxonomy_app.gl_tax_activity_financials (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    period                  VARCHAR(10)     NOT NULL,
    turnover                DECIMAL(20,2)   DEFAULT 0,
    capex                   DECIMAL(20,2)   DEFAULT 0,
    opex                    DECIMAL(20,2)   DEFAULT 0,
    eligible                BOOLEAN         NOT NULL DEFAULT FALSE,
    aligned                 BOOLEAN         NOT NULL DEFAULT FALSE,
    objective               VARCHAR(50),
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_af_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_af_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_af_turnover_non_neg CHECK (
        turnover IS NULL OR turnover >= 0
    ),
    CONSTRAINT chk_tax_af_capex_non_neg CHECK (
        capex IS NULL OR capex >= 0
    ),
    CONSTRAINT chk_tax_af_opex_non_neg CHECK (
        opex IS NULL OR opex >= 0
    ),
    CONSTRAINT chk_tax_af_objective CHECK (
        objective IS NULL OR objective IN (
            'climate_mitigation', 'climate_adaptation',
            'water_marine', 'circular_economy',
            'pollution_prevention', 'biodiversity'
        )
    ),
    UNIQUE(org_id, activity_code, period)
);

-- Indexes
CREATE INDEX idx_tax_af_org ON taxonomy_app.gl_tax_activity_financials(org_id);
CREATE INDEX idx_tax_af_tenant ON taxonomy_app.gl_tax_activity_financials(tenant_id);
CREATE INDEX idx_tax_af_activity ON taxonomy_app.gl_tax_activity_financials(activity_code);
CREATE INDEX idx_tax_af_period ON taxonomy_app.gl_tax_activity_financials(period);
CREATE INDEX idx_tax_af_org_period ON taxonomy_app.gl_tax_activity_financials(org_id, period);
CREATE INDEX idx_tax_af_eligible ON taxonomy_app.gl_tax_activity_financials(eligible);
CREATE INDEX idx_tax_af_aligned ON taxonomy_app.gl_tax_activity_financials(aligned);
CREATE INDEX idx_tax_af_objective ON taxonomy_app.gl_tax_activity_financials(objective);
CREATE INDEX idx_tax_af_created_at ON taxonomy_app.gl_tax_activity_financials(created_at DESC);
CREATE INDEX idx_tax_af_metadata ON taxonomy_app.gl_tax_activity_financials USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_af_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_activity_financials
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 15: taxonomy_app.gl_tax_capex_plans
-- =============================================================================
-- CapEx plan tracking for activities that are taxonomy-eligible but not yet
-- aligned.  Stores planned vs actual CapEx amounts per year, management
-- approval status, and plan timeline for transitional CapEx reporting.

CREATE TABLE taxonomy_app.gl_tax_capex_plans (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    start_year              INTEGER         NOT NULL,
    end_year                INTEGER         NOT NULL,
    planned_amounts         JSONB           DEFAULT '{}',
    actual_amounts          JSONB           DEFAULT '{}',
    management_approved     BOOLEAN         NOT NULL DEFAULT FALSE,
    approved_date           DATE,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    notes                   TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_cp_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_cp_start_range CHECK (
        start_year >= 2020 AND start_year <= 2100
    ),
    CONSTRAINT chk_tax_cp_end_range CHECK (
        end_year >= 2020 AND end_year <= 2100
    ),
    CONSTRAINT chk_tax_cp_end_after_start CHECK (
        end_year >= start_year
    ),
    CONSTRAINT chk_tax_cp_status CHECK (
        status IN ('draft', 'submitted', 'approved', 'active', 'completed', 'cancelled')
    )
);

-- Indexes
CREATE INDEX idx_tax_cp_org ON taxonomy_app.gl_tax_capex_plans(org_id);
CREATE INDEX idx_tax_cp_tenant ON taxonomy_app.gl_tax_capex_plans(tenant_id);
CREATE INDEX idx_tax_cp_activity ON taxonomy_app.gl_tax_capex_plans(activity_code);
CREATE INDEX idx_tax_cp_org_activity ON taxonomy_app.gl_tax_capex_plans(org_id, activity_code);
CREATE INDEX idx_tax_cp_years ON taxonomy_app.gl_tax_capex_plans(start_year, end_year);
CREATE INDEX idx_tax_cp_approved ON taxonomy_app.gl_tax_capex_plans(management_approved);
CREATE INDEX idx_tax_cp_status ON taxonomy_app.gl_tax_capex_plans(status);
CREATE INDEX idx_tax_cp_created_at ON taxonomy_app.gl_tax_capex_plans(created_at DESC);
CREATE INDEX idx_tax_cp_planned ON taxonomy_app.gl_tax_capex_plans USING GIN(planned_amounts);
CREATE INDEX idx_tax_cp_actual ON taxonomy_app.gl_tax_capex_plans USING GIN(actual_amounts);
CREATE INDEX idx_tax_cp_metadata ON taxonomy_app.gl_tax_capex_plans USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_cp_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_capex_plans
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 16: taxonomy_app.gl_tax_gar_calculations (HYPERTABLE)
-- =============================================================================
-- Green Asset Ratio (GAR) calculations for financial institutions partitioned
-- by calculation_date.  Computes aligned assets as a proportion of covered
-- assets with stock and flow variants, sector and exposure breakdowns.

CREATE TABLE taxonomy_app.gl_tax_gar_calculations (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    institution_id          UUID            NOT NULL,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    gar_type                VARCHAR(10)     NOT NULL DEFAULT 'stock',
    calculation_date        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    aligned_assets          DECIMAL(20,2)   NOT NULL DEFAULT 0,
    covered_assets          DECIMAL(20,2)   NOT NULL DEFAULT 0,
    gar_percentage          DECIMAL(7,4)    NOT NULL DEFAULT 0,
    sector_breakdown        JSONB           DEFAULT '{}',
    exposure_breakdown      JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_gar_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_gar_type CHECK (
        gar_type IN ('stock', 'flow')
    ),
    CONSTRAINT chk_tax_gar_aligned_non_neg CHECK (aligned_assets >= 0),
    CONSTRAINT chk_tax_gar_covered_non_neg CHECK (covered_assets >= 0),
    CONSTRAINT chk_tax_gar_pct_range CHECK (
        gar_percentage >= 0 AND gar_percentage <= 100
    ),
    CONSTRAINT chk_tax_gar_aligned_le_covered CHECK (
        aligned_assets <= covered_assets
    ),
    CONSTRAINT chk_tax_gar_status CHECK (
        status IN ('draft', 'calculated', 'approved', 'submitted')
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('taxonomy_app.gl_tax_gar_calculations', 'calculation_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tax_gar_inst ON taxonomy_app.gl_tax_gar_calculations(institution_id, calculation_date DESC);
CREATE INDEX idx_tax_gar_tenant ON taxonomy_app.gl_tax_gar_calculations(tenant_id, calculation_date DESC);
CREATE INDEX idx_tax_gar_period ON taxonomy_app.gl_tax_gar_calculations(period, calculation_date DESC);
CREATE INDEX idx_tax_gar_type ON taxonomy_app.gl_tax_gar_calculations(gar_type, calculation_date DESC);
CREATE INDEX idx_tax_gar_status ON taxonomy_app.gl_tax_gar_calculations(status, calculation_date DESC);
CREATE INDEX idx_tax_gar_inst_period ON taxonomy_app.gl_tax_gar_calculations(institution_id, period, calculation_date DESC);
CREATE INDEX idx_tax_gar_inst_type ON taxonomy_app.gl_tax_gar_calculations(institution_id, gar_type, calculation_date DESC);
CREATE INDEX idx_tax_gar_sector ON taxonomy_app.gl_tax_gar_calculations USING GIN(sector_breakdown);
CREATE INDEX idx_tax_gar_exposure ON taxonomy_app.gl_tax_gar_calculations USING GIN(exposure_breakdown);
CREATE INDEX idx_tax_gar_metadata ON taxonomy_app.gl_tax_gar_calculations USING GIN(metadata);

-- =============================================================================
-- Table 17: taxonomy_app.gl_tax_exposures
-- =============================================================================
-- Financial institution exposure records detailing individual counterparty
-- exposures with NACE classification, exposure type, amount, EPC rating
-- (for mortgages), CO2 emissions intensity (for auto loans), and taxonomy
-- alignment status with alignment percentage.

CREATE TABLE taxonomy_app.gl_tax_exposures (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    institution_id          UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    portfolio_id            UUID,
    counterparty_name       TEXT            NOT NULL,
    nace_code               VARCHAR(10),
    exposure_type           VARCHAR(30)     NOT NULL,
    exposure_amount         DECIMAL(20,2)   NOT NULL DEFAULT 0,
    currency                VARCHAR(3)      NOT NULL DEFAULT 'EUR',
    epc_rating              VARCHAR(1),
    co2_gkm                 DECIMAL(8,2),
    taxonomy_aligned        BOOLEAN         NOT NULL DEFAULT FALSE,
    alignment_pct           DECIMAL(5,2)    DEFAULT 0,
    reporting_date          DATE            NOT NULL DEFAULT CURRENT_DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_exp_name_not_empty CHECK (
        LENGTH(TRIM(counterparty_name)) > 0
    ),
    CONSTRAINT chk_tax_exp_type CHECK (
        exposure_type IN (
            'corporate_loan', 'debt_security', 'equity',
            'retail_mortgage', 'auto_loan', 'project_finance',
            'green_bond'
        )
    ),
    CONSTRAINT chk_tax_exp_amount_non_neg CHECK (exposure_amount >= 0),
    CONSTRAINT chk_tax_exp_currency_length CHECK (
        LENGTH(TRIM(currency)) = 3
    ),
    CONSTRAINT chk_tax_exp_epc CHECK (
        epc_rating IS NULL OR epc_rating IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_tax_exp_co2_non_neg CHECK (
        co2_gkm IS NULL OR co2_gkm >= 0
    ),
    CONSTRAINT chk_tax_exp_alignment_range CHECK (
        alignment_pct >= 0 AND alignment_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_tax_exp_inst ON taxonomy_app.gl_tax_exposures(institution_id);
CREATE INDEX idx_tax_exp_tenant ON taxonomy_app.gl_tax_exposures(tenant_id);
CREATE INDEX idx_tax_exp_portfolio ON taxonomy_app.gl_tax_exposures(portfolio_id);
CREATE INDEX idx_tax_exp_counterparty ON taxonomy_app.gl_tax_exposures(counterparty_name);
CREATE INDEX idx_tax_exp_nace ON taxonomy_app.gl_tax_exposures(nace_code);
CREATE INDEX idx_tax_exp_type ON taxonomy_app.gl_tax_exposures(exposure_type);
CREATE INDEX idx_tax_exp_aligned ON taxonomy_app.gl_tax_exposures(taxonomy_aligned);
CREATE INDEX idx_tax_exp_epc ON taxonomy_app.gl_tax_exposures(epc_rating);
CREATE INDEX idx_tax_exp_date ON taxonomy_app.gl_tax_exposures(reporting_date DESC);
CREATE INDEX idx_tax_exp_inst_type ON taxonomy_app.gl_tax_exposures(institution_id, exposure_type);
CREATE INDEX idx_tax_exp_inst_date ON taxonomy_app.gl_tax_exposures(institution_id, reporting_date DESC);
CREATE INDEX idx_tax_exp_created_at ON taxonomy_app.gl_tax_exposures(created_at DESC);
CREATE INDEX idx_tax_exp_metadata ON taxonomy_app.gl_tax_exposures USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_exp_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_exposures
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 18: taxonomy_app.gl_tax_alignment_results (HYPERTABLE)
-- =============================================================================
-- Full four-step alignment results partitioned by alignment_date.  Combines
-- eligibility, SC pass, DNSH pass, and MS pass into a single aligned/not-
-- aligned determination per activity, with the SC objective and detail JSONB.

CREATE TABLE taxonomy_app.gl_tax_alignment_results (
    id                      UUID            NOT NULL DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID,
    activity_code           VARCHAR(20)     NOT NULL,
    period                  VARCHAR(10)     NOT NULL,
    alignment_date          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    eligible                BOOLEAN         NOT NULL DEFAULT FALSE,
    sc_pass                 BOOLEAN         NOT NULL DEFAULT FALSE,
    dnsh_pass               BOOLEAN         NOT NULL DEFAULT FALSE,
    ms_pass                 BOOLEAN         NOT NULL DEFAULT FALSE,
    aligned                 BOOLEAN         NOT NULL DEFAULT FALSE,
    sc_objective            VARCHAR(50),
    alignment_details       JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_ar_code_not_empty CHECK (
        LENGTH(TRIM(activity_code)) > 0
    ),
    CONSTRAINT chk_tax_ar_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_ar_sc_objective CHECK (
        sc_objective IS NULL OR sc_objective IN (
            'climate_mitigation', 'climate_adaptation',
            'water_marine', 'circular_economy',
            'pollution_prevention', 'biodiversity'
        )
    ),
    CONSTRAINT chk_tax_ar_aligned_requires_all CHECK (
        aligned = FALSE OR (eligible = TRUE AND sc_pass = TRUE AND dnsh_pass = TRUE AND ms_pass = TRUE)
    )
);

-- Convert to hypertable (3-month chunks)
SELECT create_hypertable('taxonomy_app.gl_tax_alignment_results', 'alignment_date',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- Indexes (hypertable-aware)
CREATE INDEX idx_tax_ar_org ON taxonomy_app.gl_tax_alignment_results(org_id, alignment_date DESC);
CREATE INDEX idx_tax_ar_tenant ON taxonomy_app.gl_tax_alignment_results(tenant_id, alignment_date DESC);
CREATE INDEX idx_tax_ar_activity ON taxonomy_app.gl_tax_alignment_results(activity_code, alignment_date DESC);
CREATE INDEX idx_tax_ar_period ON taxonomy_app.gl_tax_alignment_results(period, alignment_date DESC);
CREATE INDEX idx_tax_ar_aligned ON taxonomy_app.gl_tax_alignment_results(aligned, alignment_date DESC);
CREATE INDEX idx_tax_ar_eligible ON taxonomy_app.gl_tax_alignment_results(eligible, alignment_date DESC);
CREATE INDEX idx_tax_ar_sc_pass ON taxonomy_app.gl_tax_alignment_results(sc_pass, alignment_date DESC);
CREATE INDEX idx_tax_ar_dnsh_pass ON taxonomy_app.gl_tax_alignment_results(dnsh_pass, alignment_date DESC);
CREATE INDEX idx_tax_ar_ms_pass ON taxonomy_app.gl_tax_alignment_results(ms_pass, alignment_date DESC);
CREATE INDEX idx_tax_ar_org_period ON taxonomy_app.gl_tax_alignment_results(org_id, period, alignment_date DESC);
CREATE INDEX idx_tax_ar_org_activity ON taxonomy_app.gl_tax_alignment_results(org_id, activity_code, alignment_date DESC);
CREATE INDEX idx_tax_ar_details ON taxonomy_app.gl_tax_alignment_results USING GIN(alignment_details);
CREATE INDEX idx_tax_ar_metadata ON taxonomy_app.gl_tax_alignment_results USING GIN(metadata);

-- =============================================================================
-- Table 19: taxonomy_app.gl_tax_portfolio_alignments
-- =============================================================================
-- Portfolio-level alignment summaries aggregating all activity-level alignment
-- results for an organization and period.  Provides headline counts,
-- alignment percentage, KPI summary, and sector breakdown.

CREATE TABLE taxonomy_app.gl_tax_portfolio_alignments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    total_activities        INTEGER         NOT NULL DEFAULT 0,
    eligible_count          INTEGER         NOT NULL DEFAULT 0,
    aligned_count           INTEGER         NOT NULL DEFAULT 0,
    alignment_percentage    DECIMAL(7,4)    NOT NULL DEFAULT 0,
    kpi_summary             JSONB           DEFAULT '{}',
    sector_breakdown        JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_pa_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_pa_total_non_neg CHECK (total_activities >= 0),
    CONSTRAINT chk_tax_pa_eligible_non_neg CHECK (eligible_count >= 0),
    CONSTRAINT chk_tax_pa_aligned_non_neg CHECK (aligned_count >= 0),
    CONSTRAINT chk_tax_pa_aligned_le_eligible CHECK (
        aligned_count <= eligible_count
    ),
    CONSTRAINT chk_tax_pa_eligible_le_total CHECK (
        eligible_count <= total_activities
    ),
    CONSTRAINT chk_tax_pa_pct_range CHECK (
        alignment_percentage >= 0 AND alignment_percentage <= 100
    ),
    UNIQUE(org_id, period)
);

-- Indexes
CREATE INDEX idx_tax_pa_org ON taxonomy_app.gl_tax_portfolio_alignments(org_id);
CREATE INDEX idx_tax_pa_tenant ON taxonomy_app.gl_tax_portfolio_alignments(tenant_id);
CREATE INDEX idx_tax_pa_period ON taxonomy_app.gl_tax_portfolio_alignments(period);
CREATE INDEX idx_tax_pa_org_period ON taxonomy_app.gl_tax_portfolio_alignments(org_id, period);
CREATE INDEX idx_tax_pa_pct ON taxonomy_app.gl_tax_portfolio_alignments(alignment_percentage);
CREATE INDEX idx_tax_pa_created_at ON taxonomy_app.gl_tax_portfolio_alignments(created_at DESC);
CREATE INDEX idx_tax_pa_kpi ON taxonomy_app.gl_tax_portfolio_alignments USING GIN(kpi_summary);
CREATE INDEX idx_tax_pa_sector ON taxonomy_app.gl_tax_portfolio_alignments USING GIN(sector_breakdown);
CREATE INDEX idx_tax_pa_metadata ON taxonomy_app.gl_tax_portfolio_alignments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_pa_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_portfolio_alignments
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 20: taxonomy_app.gl_tax_reports
-- =============================================================================
-- Generated taxonomy disclosure reports covering Article 8 templates (for
-- non-financial undertakings), EBA Pillar 3 templates (for credit
-- institutions), and multi-format exports (PDF, Excel, CSV, XBRL).

CREATE TABLE taxonomy_app.gl_tax_reports (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    template                VARCHAR(40)     NOT NULL,
    format                  VARCHAR(10)     NOT NULL DEFAULT 'pdf',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    generated_at            TIMESTAMPTZ,
    download_url            TEXT,
    content                 JSONB           DEFAULT '{}',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_rpt_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_rpt_template CHECK (
        template IN (
            'article_8_turnover', 'article_8_capex', 'article_8_opex',
            'article_8_combined', 'eba_template_0', 'eba_template_1',
            'eba_template_2', 'eba_template_3', 'eba_template_4',
            'eba_template_5', 'eba_template_6', 'eba_template_7',
            'eba_template_8', 'eba_template_9', 'eba_template_10',
            'gar_summary', 'btar_summary', 'qualitative_disclosure',
            'executive_summary'
        )
    ),
    CONSTRAINT chk_tax_rpt_format CHECK (
        format IN ('pdf', 'excel', 'csv', 'xbrl', 'json')
    ),
    CONSTRAINT chk_tax_rpt_status CHECK (
        status IN ('draft', 'generated', 'approved', 'submitted')
    )
);

-- Indexes
CREATE INDEX idx_tax_rpt_org ON taxonomy_app.gl_tax_reports(org_id);
CREATE INDEX idx_tax_rpt_tenant ON taxonomy_app.gl_tax_reports(tenant_id);
CREATE INDEX idx_tax_rpt_period ON taxonomy_app.gl_tax_reports(period);
CREATE INDEX idx_tax_rpt_template ON taxonomy_app.gl_tax_reports(template);
CREATE INDEX idx_tax_rpt_format ON taxonomy_app.gl_tax_reports(format);
CREATE INDEX idx_tax_rpt_status ON taxonomy_app.gl_tax_reports(status);
CREATE INDEX idx_tax_rpt_org_period ON taxonomy_app.gl_tax_reports(org_id, period);
CREATE INDEX idx_tax_rpt_generated ON taxonomy_app.gl_tax_reports(generated_at DESC);
CREATE INDEX idx_tax_rpt_created_at ON taxonomy_app.gl_tax_reports(created_at DESC);
CREATE INDEX idx_tax_rpt_content ON taxonomy_app.gl_tax_reports USING GIN(content);
CREATE INDEX idx_tax_rpt_metadata ON taxonomy_app.gl_tax_reports USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_rpt_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_reports
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 21: taxonomy_app.gl_tax_evidence_items
-- =============================================================================
-- Evidence items (documents, certifications, audit reports, declarations)
-- linked to SC, DNSH, or minimum safeguard assessments.  Tracks upload,
-- verification status, and verifier identity for audit trail purposes.

CREATE TABLE taxonomy_app.gl_tax_evidence_items (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    assessment_type         VARCHAR(20)     NOT NULL,
    assessment_id           UUID            NOT NULL,
    evidence_type           VARCHAR(20)     NOT NULL,
    description             TEXT            NOT NULL,
    document_ref            TEXT,
    uploaded_at             TIMESTAMPTZ     DEFAULT NOW(),
    verified                BOOLEAN         NOT NULL DEFAULT FALSE,
    verified_by             TEXT,
    verified_at             TIMESTAMPTZ,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_ev_assessment_type CHECK (
        assessment_type IN ('sc', 'dnsh', 'safeguard', 'dq', 'gap', 'climate_risk')
    ),
    CONSTRAINT chk_tax_ev_evidence_type CHECK (
        evidence_type IN ('document', 'certification', 'report', 'audit', 'declaration', 'data_extract')
    ),
    CONSTRAINT chk_tax_ev_desc_not_empty CHECK (
        LENGTH(TRIM(description)) > 0
    )
);

-- Indexes
CREATE INDEX idx_tax_ev_org ON taxonomy_app.gl_tax_evidence_items(org_id);
CREATE INDEX idx_tax_ev_tenant ON taxonomy_app.gl_tax_evidence_items(tenant_id);
CREATE INDEX idx_tax_ev_atype ON taxonomy_app.gl_tax_evidence_items(assessment_type);
CREATE INDEX idx_tax_ev_aid ON taxonomy_app.gl_tax_evidence_items(assessment_id);
CREATE INDEX idx_tax_ev_etype ON taxonomy_app.gl_tax_evidence_items(evidence_type);
CREATE INDEX idx_tax_ev_verified ON taxonomy_app.gl_tax_evidence_items(verified);
CREATE INDEX idx_tax_ev_org_atype ON taxonomy_app.gl_tax_evidence_items(org_id, assessment_type);
CREATE INDEX idx_tax_ev_aid_atype ON taxonomy_app.gl_tax_evidence_items(assessment_id, assessment_type);
CREATE INDEX idx_tax_ev_uploaded ON taxonomy_app.gl_tax_evidence_items(uploaded_at DESC);
CREATE INDEX idx_tax_ev_created_at ON taxonomy_app.gl_tax_evidence_items(created_at DESC);
CREATE INDEX idx_tax_ev_metadata ON taxonomy_app.gl_tax_evidence_items USING GIN(metadata);

-- =============================================================================
-- Table 22: taxonomy_app.gl_tax_regulatory_versions
-- =============================================================================
-- EU Taxonomy delegated act version tracking.  Records effective dates,
-- amendment details, affected activities, and status (active/superseded/draft)
-- to support multi-version SC/DNSH criteria application.

CREATE TABLE taxonomy_app.gl_tax_regulatory_versions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    delegated_act           VARCHAR(50)     NOT NULL,
    version_number          VARCHAR(20)     NOT NULL,
    effective_date          DATE            NOT NULL,
    amendment_details       JSONB           DEFAULT '{}',
    activities_affected     TEXT[]          DEFAULT '{}',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'active',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_rv_act_not_empty CHECK (
        LENGTH(TRIM(delegated_act)) > 0
    ),
    CONSTRAINT chk_tax_rv_version_not_empty CHECK (
        LENGTH(TRIM(version_number)) > 0
    ),
    CONSTRAINT chk_tax_rv_status CHECK (
        status IN ('active', 'superseded', 'draft')
    ),
    UNIQUE(delegated_act, version_number)
);

-- Indexes
CREATE INDEX idx_tax_rv_act ON taxonomy_app.gl_tax_regulatory_versions(delegated_act);
CREATE INDEX idx_tax_rv_version ON taxonomy_app.gl_tax_regulatory_versions(version_number);
CREATE INDEX idx_tax_rv_effective ON taxonomy_app.gl_tax_regulatory_versions(effective_date);
CREATE INDEX idx_tax_rv_status ON taxonomy_app.gl_tax_regulatory_versions(status);
CREATE INDEX idx_tax_rv_act_status ON taxonomy_app.gl_tax_regulatory_versions(delegated_act, status);
CREATE INDEX idx_tax_rv_created_at ON taxonomy_app.gl_tax_regulatory_versions(created_at DESC);
CREATE INDEX idx_tax_rv_activities ON taxonomy_app.gl_tax_regulatory_versions USING GIN(activities_affected);
CREATE INDEX idx_tax_rv_amendment ON taxonomy_app.gl_tax_regulatory_versions USING GIN(amendment_details);
CREATE INDEX idx_tax_rv_metadata ON taxonomy_app.gl_tax_regulatory_versions USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_rv_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_regulatory_versions
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 23: taxonomy_app.gl_tax_data_quality_scores
-- =============================================================================
-- Data quality assessments scoring the completeness, accuracy, timeliness,
-- and consistency of taxonomy disclosure data across multiple dimensions.

CREATE TABLE taxonomy_app.gl_tax_data_quality_scores (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    overall_score           DECIMAL(5,2)    NOT NULL DEFAULT 0,
    grade                   VARCHAR(2)      NOT NULL DEFAULT 'C',
    dimensions              JSONB           DEFAULT '{}',
    improvement_actions     JSONB           DEFAULT '[]',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_dq_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_dq_score_range CHECK (
        overall_score >= 0 AND overall_score <= 100
    ),
    CONSTRAINT chk_tax_dq_grade CHECK (
        grade IN ('A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F')
    ),
    UNIQUE(org_id, period)
);

-- Indexes
CREATE INDEX idx_tax_dq_org ON taxonomy_app.gl_tax_data_quality_scores(org_id);
CREATE INDEX idx_tax_dq_tenant ON taxonomy_app.gl_tax_data_quality_scores(tenant_id);
CREATE INDEX idx_tax_dq_period ON taxonomy_app.gl_tax_data_quality_scores(period);
CREATE INDEX idx_tax_dq_org_period ON taxonomy_app.gl_tax_data_quality_scores(org_id, period);
CREATE INDEX idx_tax_dq_score ON taxonomy_app.gl_tax_data_quality_scores(overall_score);
CREATE INDEX idx_tax_dq_grade ON taxonomy_app.gl_tax_data_quality_scores(grade);
CREATE INDEX idx_tax_dq_date ON taxonomy_app.gl_tax_data_quality_scores(assessment_date DESC);
CREATE INDEX idx_tax_dq_created_at ON taxonomy_app.gl_tax_data_quality_scores(created_at DESC);
CREATE INDEX idx_tax_dq_dimensions ON taxonomy_app.gl_tax_data_quality_scores USING GIN(dimensions);
CREATE INDEX idx_tax_dq_actions ON taxonomy_app.gl_tax_data_quality_scores USING GIN(improvement_actions);
CREATE INDEX idx_tax_dq_metadata ON taxonomy_app.gl_tax_data_quality_scores USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_dq_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_data_quality_scores
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 24: taxonomy_app.gl_tax_gap_assessments
-- =============================================================================
-- Gap analysis results evaluating an organization's readiness for full
-- taxonomy disclosure.  Tracks total/priority-weighted gap counts, gap
-- categories, and action items with deadlines and assignments.

CREATE TABLE taxonomy_app.gl_tax_gap_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_organizations(id) ON DELETE CASCADE,
    tenant_id               UUID,
    period                  VARCHAR(10)     NOT NULL,
    assessment_date         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_gaps              INTEGER         NOT NULL DEFAULT 0,
    high_priority           INTEGER         NOT NULL DEFAULT 0,
    gap_categories          JSONB           DEFAULT '{}',
    action_items            JSONB           DEFAULT '[]',
    status                  VARCHAR(20)     NOT NULL DEFAULT 'draft',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_ga_period_not_empty CHECK (
        LENGTH(TRIM(period)) > 0
    ),
    CONSTRAINT chk_tax_ga_total_non_neg CHECK (total_gaps >= 0),
    CONSTRAINT chk_tax_ga_high_non_neg CHECK (high_priority >= 0),
    CONSTRAINT chk_tax_ga_high_le_total CHECK (
        high_priority <= total_gaps
    ),
    CONSTRAINT chk_tax_ga_status CHECK (
        status IN ('draft', 'in_progress', 'completed', 'approved')
    )
);

-- Indexes
CREATE INDEX idx_tax_ga_org ON taxonomy_app.gl_tax_gap_assessments(org_id);
CREATE INDEX idx_tax_ga_tenant ON taxonomy_app.gl_tax_gap_assessments(tenant_id);
CREATE INDEX idx_tax_ga_period ON taxonomy_app.gl_tax_gap_assessments(period);
CREATE INDEX idx_tax_ga_org_period ON taxonomy_app.gl_tax_gap_assessments(org_id, period);
CREATE INDEX idx_tax_ga_status ON taxonomy_app.gl_tax_gap_assessments(status);
CREATE INDEX idx_tax_ga_date ON taxonomy_app.gl_tax_gap_assessments(assessment_date DESC);
CREATE INDEX idx_tax_ga_created_at ON taxonomy_app.gl_tax_gap_assessments(created_at DESC);
CREATE INDEX idx_tax_ga_categories ON taxonomy_app.gl_tax_gap_assessments USING GIN(gap_categories);
CREATE INDEX idx_tax_ga_actions ON taxonomy_app.gl_tax_gap_assessments USING GIN(action_items);
CREATE INDEX idx_tax_ga_metadata ON taxonomy_app.gl_tax_gap_assessments USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_ga_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_gap_assessments
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Table 25: taxonomy_app.gl_tax_gap_items
-- =============================================================================
-- Individual gap items linked to a gap assessment.  Each record describes
-- a specific gap with category, priority, current/target status, required
-- action, deadline, and assignment for remediation tracking.

CREATE TABLE taxonomy_app.gl_tax_gap_items (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           UUID            NOT NULL REFERENCES taxonomy_app.gl_tax_gap_assessments(id) ON DELETE CASCADE,
    tenant_id               UUID,
    category                VARCHAR(20)     NOT NULL,
    description             TEXT            NOT NULL,
    priority                VARCHAR(10)     NOT NULL DEFAULT 'medium',
    current_status          TEXT,
    target_status           TEXT,
    action_required         TEXT,
    deadline                DATE,
    assigned_to             TEXT,
    status                  VARCHAR(20)     NOT NULL DEFAULT 'open',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_tax_gi_category CHECK (
        category IN ('sc', 'dnsh', 'safeguard', 'data', 'regulatory', 'governance', 'reporting')
    ),
    CONSTRAINT chk_tax_gi_desc_not_empty CHECK (
        LENGTH(TRIM(description)) > 0
    ),
    CONSTRAINT chk_tax_gi_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low')
    ),
    CONSTRAINT chk_tax_gi_status CHECK (
        status IN ('open', 'in_progress', 'resolved', 'deferred', 'not_applicable')
    )
);

-- Indexes
CREATE INDEX idx_tax_gi_assessment ON taxonomy_app.gl_tax_gap_items(assessment_id);
CREATE INDEX idx_tax_gi_tenant ON taxonomy_app.gl_tax_gap_items(tenant_id);
CREATE INDEX idx_tax_gi_category ON taxonomy_app.gl_tax_gap_items(category);
CREATE INDEX idx_tax_gi_priority ON taxonomy_app.gl_tax_gap_items(priority);
CREATE INDEX idx_tax_gi_status ON taxonomy_app.gl_tax_gap_items(status);
CREATE INDEX idx_tax_gi_deadline ON taxonomy_app.gl_tax_gap_items(deadline);
CREATE INDEX idx_tax_gi_assigned ON taxonomy_app.gl_tax_gap_items(assigned_to);
CREATE INDEX idx_tax_gi_assessment_cat ON taxonomy_app.gl_tax_gap_items(assessment_id, category);
CREATE INDEX idx_tax_gi_assessment_priority ON taxonomy_app.gl_tax_gap_items(assessment_id, priority);
CREATE INDEX idx_tax_gi_created_at ON taxonomy_app.gl_tax_gap_items(created_at DESC);
CREATE INDEX idx_tax_gi_metadata ON taxonomy_app.gl_tax_gap_items USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_tax_gi_updated_at
    BEFORE UPDATE ON taxonomy_app.gl_tax_gap_items
    FOR EACH ROW
    EXECUTE FUNCTION taxonomy_app.set_updated_at();

-- =============================================================================
-- Continuous Aggregate 1: taxonomy_app.quarterly_alignment_summary
-- =============================================================================
-- Quarterly aggregation of alignment metrics by organization showing average
-- alignment percentage, eligible and aligned counts over three-month buckets.

CREATE MATERIALIZED VIEW taxonomy_app.quarterly_alignment_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('3 months', alignment_date)     AS bucket,
    org_id,
    COUNT(*)                                    AS total_results,
    COUNT(*) FILTER (WHERE eligible = TRUE)     AS eligible_count,
    COUNT(*) FILTER (WHERE aligned = TRUE)      AS aligned_count,
    ROUND(
        (COUNT(*) FILTER (WHERE aligned = TRUE))::NUMERIC /
        NULLIF(COUNT(*) FILTER (WHERE eligible = TRUE), 0) * 100,
        2
    )                                           AS alignment_percentage
FROM taxonomy_app.gl_tax_alignment_results
GROUP BY bucket, org_id
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('taxonomy_app.quarterly_alignment_summary',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Continuous Aggregate 2: taxonomy_app.annual_gar_trends
-- =============================================================================
-- Annual aggregation of GAR metrics by institution and GAR type showing
-- average GAR percentage and total aligned/covered asset sums.

CREATE MATERIALIZED VIEW taxonomy_app.annual_gar_trends
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 year', calculation_date)     AS bucket,
    institution_id,
    gar_type,
    AVG(gar_percentage)                         AS avg_gar_percentage,
    SUM(aligned_assets)                         AS total_aligned_assets,
    SUM(covered_assets)                         AS total_covered_assets,
    COUNT(*)                                    AS calculation_count
FROM taxonomy_app.gl_tax_gar_calculations
GROUP BY bucket, institution_id, gar_type
WITH NO DATA;

-- Refresh policy: every 30 minutes, covering last 6 hours
SELECT add_continuous_aggregate_policy('taxonomy_app.annual_gar_trends',
    start_offset => INTERVAL '6 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '30 minutes'
);

-- =============================================================================
-- Retention Policies
-- =============================================================================

-- Keep SC assessments for 3650 days (10 years, regulatory retention)
SELECT add_retention_policy('taxonomy_app.gl_tax_sc_assessments', INTERVAL '3650 days');

-- Keep GAR calculations for 3650 days (10 years)
SELECT add_retention_policy('taxonomy_app.gl_tax_gar_calculations', INTERVAL '3650 days');

-- Keep alignment results for 3650 days (10 years)
SELECT add_retention_policy('taxonomy_app.gl_tax_alignment_results', INTERVAL '3650 days');

-- =============================================================================
-- Compression Policies
-- =============================================================================

-- Enable compression on sc_assessments after 90 days
ALTER TABLE taxonomy_app.gl_tax_sc_assessments SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'assessment_date DESC'
);

SELECT add_compression_policy('taxonomy_app.gl_tax_sc_assessments', INTERVAL '90 days');

-- Enable compression on gar_calculations after 90 days
ALTER TABLE taxonomy_app.gl_tax_gar_calculations SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'calculation_date DESC'
);

SELECT add_compression_policy('taxonomy_app.gl_tax_gar_calculations', INTERVAL '90 days');

-- Enable compression on alignment_results after 90 days
ALTER TABLE taxonomy_app.gl_tax_alignment_results SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'alignment_date DESC'
);

SELECT add_compression_policy('taxonomy_app.gl_tax_alignment_results', INTERVAL '90 days');

-- =============================================================================
-- Permissions
-- =============================================================================

GRANT USAGE ON SCHEMA taxonomy_app TO greenlang_app;
GRANT ALL ON ALL TABLES IN SCHEMA taxonomy_app TO greenlang_app;
GRANT ALL ON ALL SEQUENCES IN SCHEMA taxonomy_app TO greenlang_app;

-- Grant SELECT on continuous aggregates
GRANT SELECT ON taxonomy_app.quarterly_alignment_summary TO greenlang_app;
GRANT SELECT ON taxonomy_app.annual_gar_trends TO greenlang_app;

-- Read-only role
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'greenlang_readonly') THEN
        GRANT USAGE ON SCHEMA taxonomy_app TO greenlang_readonly;
        GRANT SELECT ON ALL TABLES IN SCHEMA taxonomy_app TO greenlang_readonly;
        GRANT SELECT ON taxonomy_app.quarterly_alignment_summary TO greenlang_readonly;
        GRANT SELECT ON taxonomy_app.annual_gar_trends TO greenlang_readonly;
    END IF;
END
$$;

-- Add Taxonomy app service permissions to security.permissions
INSERT INTO security.permissions (permission_id, name, resource, action, description) VALUES
    (gen_random_uuid(), 'taxonomy_app:organizations:read', 'taxonomy_app', 'organizations_read', 'View taxonomy organization profiles'),
    (gen_random_uuid(), 'taxonomy_app:organizations:write', 'taxonomy_app', 'organizations_write', 'Create and manage taxonomy organization profiles'),
    (gen_random_uuid(), 'taxonomy_app:activities:read', 'taxonomy_app', 'activities_read', 'View EU Taxonomy economic activity catalog'),
    (gen_random_uuid(), 'taxonomy_app:activities:write', 'taxonomy_app', 'activities_write', 'Manage EU Taxonomy economic activity catalog'),
    (gen_random_uuid(), 'taxonomy_app:nace:read', 'taxonomy_app', 'nace_read', 'View NACE code to taxonomy activity mappings'),
    (gen_random_uuid(), 'taxonomy_app:nace:write', 'taxonomy_app', 'nace_write', 'Manage NACE code mappings'),
    (gen_random_uuid(), 'taxonomy_app:eligibility:read', 'taxonomy_app', 'eligibility_read', 'View eligibility screening results'),
    (gen_random_uuid(), 'taxonomy_app:eligibility:run', 'taxonomy_app', 'eligibility_run', 'Run eligibility screening assessments'),
    (gen_random_uuid(), 'taxonomy_app:sc:read', 'taxonomy_app', 'sc_read', 'View substantial contribution assessments'),
    (gen_random_uuid(), 'taxonomy_app:sc:write', 'taxonomy_app', 'sc_write', 'Create and manage SC assessments'),
    (gen_random_uuid(), 'taxonomy_app:tsc:read', 'taxonomy_app', 'tsc_read', 'View technical screening criteria evaluations'),
    (gen_random_uuid(), 'taxonomy_app:tsc:write', 'taxonomy_app', 'tsc_write', 'Manage TSC evaluations'),
    (gen_random_uuid(), 'taxonomy_app:dnsh:read', 'taxonomy_app', 'dnsh_read', 'View DNSH assessment results'),
    (gen_random_uuid(), 'taxonomy_app:dnsh:write', 'taxonomy_app', 'dnsh_write', 'Create and manage DNSH assessments'),
    (gen_random_uuid(), 'taxonomy_app:climate_risk:read', 'taxonomy_app', 'climate_risk_read', 'View climate risk assessments'),
    (gen_random_uuid(), 'taxonomy_app:climate_risk:write', 'taxonomy_app', 'climate_risk_write', 'Create and manage climate risk assessments'),
    (gen_random_uuid(), 'taxonomy_app:safeguards:read', 'taxonomy_app', 'safeguards_read', 'View minimum safeguard assessments'),
    (gen_random_uuid(), 'taxonomy_app:safeguards:write', 'taxonomy_app', 'safeguards_write', 'Create and manage minimum safeguard assessments'),
    (gen_random_uuid(), 'taxonomy_app:kpi:read', 'taxonomy_app', 'kpi_read', 'View KPI calculations (turnover/CapEx/OpEx)'),
    (gen_random_uuid(), 'taxonomy_app:kpi:calculate', 'taxonomy_app', 'kpi_calculate', 'Calculate taxonomy KPIs'),
    (gen_random_uuid(), 'taxonomy_app:financials:read', 'taxonomy_app', 'financials_read', 'View activity-level financial data'),
    (gen_random_uuid(), 'taxonomy_app:financials:write', 'taxonomy_app', 'financials_write', 'Manage activity financial data'),
    (gen_random_uuid(), 'taxonomy_app:capex_plans:read', 'taxonomy_app', 'capex_plans_read', 'View CapEx plan data'),
    (gen_random_uuid(), 'taxonomy_app:capex_plans:write', 'taxonomy_app', 'capex_plans_write', 'Create and manage CapEx plans'),
    (gen_random_uuid(), 'taxonomy_app:gar:read', 'taxonomy_app', 'gar_read', 'View Green Asset Ratio calculations'),
    (gen_random_uuid(), 'taxonomy_app:gar:calculate', 'taxonomy_app', 'gar_calculate', 'Calculate GAR and BTAR'),
    (gen_random_uuid(), 'taxonomy_app:exposures:read', 'taxonomy_app', 'exposures_read', 'View financial exposure records'),
    (gen_random_uuid(), 'taxonomy_app:exposures:write', 'taxonomy_app', 'exposures_write', 'Manage financial exposure records'),
    (gen_random_uuid(), 'taxonomy_app:alignment:read', 'taxonomy_app', 'alignment_read', 'View alignment results and portfolio alignment'),
    (gen_random_uuid(), 'taxonomy_app:alignment:run', 'taxonomy_app', 'alignment_run', 'Run alignment assessments'),
    (gen_random_uuid(), 'taxonomy_app:reports:read', 'taxonomy_app', 'reports_read', 'View generated taxonomy reports'),
    (gen_random_uuid(), 'taxonomy_app:reports:generate', 'taxonomy_app', 'reports_generate', 'Generate taxonomy disclosure reports'),
    (gen_random_uuid(), 'taxonomy_app:evidence:read', 'taxonomy_app', 'evidence_read', 'View evidence items'),
    (gen_random_uuid(), 'taxonomy_app:evidence:write', 'taxonomy_app', 'evidence_write', 'Upload and manage evidence items'),
    (gen_random_uuid(), 'taxonomy_app:regulatory:read', 'taxonomy_app', 'regulatory_read', 'View regulatory version data'),
    (gen_random_uuid(), 'taxonomy_app:regulatory:write', 'taxonomy_app', 'regulatory_write', 'Manage regulatory versions'),
    (gen_random_uuid(), 'taxonomy_app:data_quality:read', 'taxonomy_app', 'data_quality_read', 'View data quality scores'),
    (gen_random_uuid(), 'taxonomy_app:data_quality:assess', 'taxonomy_app', 'data_quality_assess', 'Run data quality assessments'),
    (gen_random_uuid(), 'taxonomy_app:gaps:read', 'taxonomy_app', 'gaps_read', 'View gap assessment results'),
    (gen_random_uuid(), 'taxonomy_app:gaps:write', 'taxonomy_app', 'gaps_write', 'Run and manage gap assessments'),
    (gen_random_uuid(), 'taxonomy_app:dashboard:read', 'taxonomy_app', 'dashboard_read', 'View taxonomy dashboards and analytics'),
    (gen_random_uuid(), 'taxonomy_app:admin', 'taxonomy_app', 'admin', 'Taxonomy application administration')
ON CONFLICT DO NOTHING;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON SCHEMA taxonomy_app IS 'GL-Taxonomy-APP v1.0 Application Schema - EU Taxonomy Alignment & Green Investment Ratio Platform with economic activity eligibility screening, Substantial Contribution (SC) assessment per environmental objective, DNSH matrix evaluation, Minimum Safeguards (human rights, anti-corruption, taxation, fair competition), KPI calculation (turnover/CapEx/OpEx), Green Asset Ratio (GAR), BTAR, exposure classification, portfolio alignment, CapEx plans, regulatory version management, evidence tracking, data quality scoring, gap analysis, and Article 8 / EBA Pillar 3 disclosure reporting';

COMMENT ON TABLE taxonomy_app.gl_tax_organizations IS 'Organization profiles for EU Taxonomy alignment with entity type (financial/non-financial), sector, LEI, NFRD/CSRD reporting status';
COMMENT ON TABLE taxonomy_app.gl_tax_economic_activities IS 'EU Taxonomy economic activity catalog with activity codes, NACE mappings, objectives, SC/DNSH criteria, and delegated act reference';
COMMENT ON TABLE taxonomy_app.gl_tax_nace_mappings IS 'NACE code hierarchy (levels 1-4) with parent references and associated taxonomy activity codes';
COMMENT ON TABLE taxonomy_app.gl_tax_eligibility_screenings IS 'Eligibility screening runs with total/eligible/not-eligible/de-minimis counts';
COMMENT ON TABLE taxonomy_app.gl_tax_screening_results IS 'Per-activity eligibility screening results with confidence scores and objective mappings';
COMMENT ON TABLE taxonomy_app.gl_tax_sc_assessments IS 'TimescaleDB hypertable: Substantial Contribution assessments per activity per objective with threshold checks and evidence';
COMMENT ON TABLE taxonomy_app.gl_tax_tsc_evaluations IS 'Technical Screening Criteria evaluations with threshold vs actual value comparison';
COMMENT ON TABLE taxonomy_app.gl_tax_dnsh_assessments IS 'Do No Significant Harm assessments with per-objective results and evidence';
COMMENT ON TABLE taxonomy_app.gl_tax_dnsh_objective_results IS 'Per-objective DNSH results (pass/fail/not_applicable) with criteria checks';
COMMENT ON TABLE taxonomy_app.gl_tax_climate_risk_assessments IS 'Climate risk assessments for DNSH adaptation with physical risks, adaptation solutions, and residual risks';
COMMENT ON TABLE taxonomy_app.gl_tax_minimum_safeguard_assessments IS 'Minimum Safeguard assessments (OECD/UNGP/ILO/ICCPR) with 4-topic evaluation';
COMMENT ON TABLE taxonomy_app.gl_tax_safeguard_topic_results IS 'Per-topic safeguard results (human rights/anti-corruption/taxation/fair competition) with procedural and outcome checks';
COMMENT ON TABLE taxonomy_app.gl_tax_kpi_calculations IS 'KPI calculations (turnover/CapEx/OpEx) with eligible/aligned/total amounts and percentage';
COMMENT ON TABLE taxonomy_app.gl_tax_activity_financials IS 'Per-activity financial data (turnover/CapEx/OpEx) with eligibility and alignment status';
COMMENT ON TABLE taxonomy_app.gl_tax_capex_plans IS 'CapEx plan tracking with planned vs actual amounts, management approval, and timeline';
COMMENT ON TABLE taxonomy_app.gl_tax_gar_calculations IS 'TimescaleDB hypertable: Green Asset Ratio (GAR) calculations with stock/flow types, sector and exposure breakdowns';
COMMENT ON TABLE taxonomy_app.gl_tax_exposures IS 'Financial institution exposure records with counterparty, NACE code, exposure type, EPC rating, CO2 intensity, and alignment';
COMMENT ON TABLE taxonomy_app.gl_tax_alignment_results IS 'TimescaleDB hypertable: Full 4-step alignment results (eligible + SC + DNSH + MS = aligned) per activity';
COMMENT ON TABLE taxonomy_app.gl_tax_portfolio_alignments IS 'Portfolio-level alignment summaries with total/eligible/aligned counts, KPI summary, and sector breakdown';
COMMENT ON TABLE taxonomy_app.gl_tax_reports IS 'Generated taxonomy disclosure reports (Article 8, EBA templates, GAR/BTAR summaries) in PDF/Excel/CSV/XBRL formats';
COMMENT ON TABLE taxonomy_app.gl_tax_evidence_items IS 'Evidence items (documents, certifications, audits) linked to SC/DNSH/safeguard assessments';
COMMENT ON TABLE taxonomy_app.gl_tax_regulatory_versions IS 'EU Taxonomy delegated act version tracking with effective dates and affected activities';
COMMENT ON TABLE taxonomy_app.gl_tax_data_quality_scores IS 'Data quality assessments with overall score, grade, dimension breakdown, and improvement actions';
COMMENT ON TABLE taxonomy_app.gl_tax_gap_assessments IS 'Gap analysis results with total/high-priority gap counts, categories, and action items';
COMMENT ON TABLE taxonomy_app.gl_tax_gap_items IS 'Individual gap items with category, priority, current/target status, action required, and deadline';

COMMENT ON MATERIALIZED VIEW taxonomy_app.quarterly_alignment_summary IS 'Continuous aggregate: quarterly alignment trends by organization showing eligible/aligned counts and alignment percentage';
COMMENT ON MATERIALIZED VIEW taxonomy_app.annual_gar_trends IS 'Continuous aggregate: annual GAR trends by institution and type showing average GAR percentage and aligned/covered asset totals';

COMMENT ON COLUMN taxonomy_app.gl_tax_organizations.entity_type IS 'Entity type: financial (credit institution), non_financial (general undertaking), insurance, asset_manager';
COMMENT ON COLUMN taxonomy_app.gl_tax_economic_activities.activity_type IS 'Activity classification: own_performance (direct), enabling (enables others), transitional (interim pathway)';
COMMENT ON COLUMN taxonomy_app.gl_tax_economic_activities.delegated_act IS 'Delegated act: climate (CCM/CCA), environmental (other 4 objectives), climate_amending, complementary';
COMMENT ON COLUMN taxonomy_app.gl_tax_sc_assessments.objective IS 'Environmental objective: climate_mitigation, climate_adaptation, water_marine, circular_economy, pollution_prevention, biodiversity';
COMMENT ON COLUMN taxonomy_app.gl_tax_sc_assessments.sc_type IS 'SC type: own_performance, enabling, transitional';
COMMENT ON COLUMN taxonomy_app.gl_tax_dnsh_objective_results.status IS 'DNSH result: pass, fail, not_applicable, pending';
COMMENT ON COLUMN taxonomy_app.gl_tax_safeguard_topic_results.topic IS 'Safeguard topic: human_rights, anti_corruption, taxation, fair_competition';
COMMENT ON COLUMN taxonomy_app.gl_tax_kpi_calculations.kpi_type IS 'KPI type: turnover, capex, opex';
COMMENT ON COLUMN taxonomy_app.gl_tax_gar_calculations.gar_type IS 'GAR type: stock (balance sheet), flow (new originations)';
COMMENT ON COLUMN taxonomy_app.gl_tax_exposures.exposure_type IS 'Exposure type: corporate_loan, debt_security, equity, retail_mortgage, auto_loan, project_finance, green_bond';
COMMENT ON COLUMN taxonomy_app.gl_tax_exposures.epc_rating IS 'Energy Performance Certificate rating: A-G (for mortgage exposures)';
COMMENT ON COLUMN taxonomy_app.gl_tax_reports.template IS 'Report template: article_8_turnover/capex/opex/combined, eba_template_0-10, gar_summary, btar_summary, qualitative_disclosure, executive_summary';
COMMENT ON COLUMN taxonomy_app.gl_tax_evidence_items.assessment_type IS 'Assessment type: sc, dnsh, safeguard, dq, gap, climate_risk';
COMMENT ON COLUMN taxonomy_app.gl_tax_evidence_items.evidence_type IS 'Evidence type: document, certification, report, audit, declaration, data_extract';
COMMENT ON COLUMN taxonomy_app.gl_tax_regulatory_versions.status IS 'Version status: active (current), superseded (replaced), draft (pending)';
COMMENT ON COLUMN taxonomy_app.gl_tax_data_quality_scores.grade IS 'Quality grade: A+ (excellent), A, B+, B, C+, C, D, F (failing)';
COMMENT ON COLUMN taxonomy_app.gl_tax_gap_items.category IS 'Gap category: sc, dnsh, safeguard, data, regulatory, governance, reporting';
COMMENT ON COLUMN taxonomy_app.gl_tax_gap_items.priority IS 'Gap priority: critical, high, medium, low';
COMMENT ON COLUMN taxonomy_app.gl_tax_gap_items.status IS 'Gap status: open, in_progress, resolved, deferred, not_applicable';

-- =============================================================================
-- End of V088: GL-Taxonomy-APP v1.0 Application Service Schema
-- =============================================================================
-- Summary:
--   25 tables created
--   3 hypertables (sc_assessments, gar_calculations, alignment_results)
--   2 continuous aggregates (quarterly_alignment_summary, annual_gar_trends)
--   20 update triggers
--   170+ B-tree indexes
--   40+ GIN indexes on JSONB columns
--   3 retention policies (10-year retention)
--   3 compression policies (90-day threshold)
--   42 security permissions
--   Security grants for greenlang_app and greenlang_readonly
-- Previous: V087__sbti_app_service.sql
-- =============================================================================
