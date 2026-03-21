-- =============================================================================
-- V138: PACK-024-carbon-neutral-001: Carbon Footprint Quantification Records
-- =============================================================================
-- Pack:         PACK-024 (Carbon Neutral Pack)
-- Date:         March 2026
--
-- Pack-level tables for carbon footprint quantification across Scope 1, 2, 3
-- emissions with baseline establishment, year-over-year tracking, uncertainty
-- assessment, methodology documentation, and reconciliation to MRV agents.
--
-- EXTENDS:
--   V051-V081: AGENT-MRV agents (all scopes and categories)
--   V088: GL-Taxonomy-APP v1.0
--
-- These tables sit in the pack024_carbon_neutral schema and provide the
-- foundational carbon quantification layer for carbon neutral pathway planning.
-- =============================================================================
-- Tables (4):
--   1. pack024_carbon_neutral.pack024_footprint_records        - Scope-level footprint records
--   2. pack024_carbon_neutral.pack024_footprint_components     - Component-level breakdown
--   3. pack024_carbon_neutral.pack024_baseline_establishment   - Baseline definition & tracking
--   4. pack024_carbon_neutral.pack024_uncertainty_assessment   - Uncertainty bounds
--
-- Also includes: 50+ indexes, update triggers, security grants, and comments.
-- Previous: V137__pack023_submission_readiness_009.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS pack024_carbon_neutral;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION pack024_carbon_neutral.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack024_carbon_neutral.pack024_footprint_records
-- =============================================================================
-- Scope-level carbon footprint records with emissions quantification,
-- source traceability, and methodology documentation.

CREATE TABLE pack024_carbon_neutral.pack024_footprint_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    reporting_year          INTEGER         NOT NULL,
    reporting_period_start  DATE            NOT NULL,
    reporting_period_end    DATE            NOT NULL,
    scope                   VARCHAR(30)     NOT NULL,
    scope_category          VARCHAR(100),
    scope_type              VARCHAR(50),
    emissions_source        VARCHAR(255),
    unit                    VARCHAR(20)     NOT NULL DEFAULT 'tCO2e',
    total_emissions         DECIMAL(18,4)   NOT NULL,
    emissions_value         DECIMAL(18,4),
    uncertainty_lower       DECIMAL(18,4),
    uncertainty_upper       DECIMAL(18,4),
    uncertainty_type        VARCHAR(50),
    uncertainty_percentage  DECIMAL(6,2),
    confidence_level        VARCHAR(30),
    calculation_methodology VARCHAR(500),
    data_quality_score      DECIMAL(5,2),
    completeness_percentage DECIMAL(6,2),
    mismatch_with_agents    BOOLEAN         DEFAULT FALSE,
    mismatch_reason         TEXT,
    reconciliation_status   VARCHAR(30)     DEFAULT 'pending',
    reconciled_by           VARCHAR(255),
    reconciliation_date     DATE,
    adjustments_applied     BOOLEAN         DEFAULT FALSE,
    adjustment_reason       TEXT,
    adjusted_emissions      DECIMAL(18,4),
    verification_status     VARCHAR(30)     DEFAULT 'unverified',
    verified_by             VARCHAR(255),
    verification_date       DATE,
    third_party_verified    BOOLEAN         DEFAULT FALSE,
    verifier_organization   VARCHAR(255),
    assurance_level         VARCHAR(30),
    peer_reviewed           BOOLEAN         DEFAULT FALSE,
    review_comments         TEXT,
    assumptions             JSONB           DEFAULT '{}',
    data_sources            TEXT[],
    agent_references        UUID[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_fp_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_pack024_fp_unit CHECK (
        unit IN ('tCO2e', 'kgCO2e', 'mtCO2e')
    ),
    CONSTRAINT chk_pack024_fp_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_pack024_fp_emissions_non_neg CHECK (
        total_emissions >= 0
    ),
    CONSTRAINT chk_pack024_fp_uncertainty_order CHECK (
        uncertainty_lower IS NULL OR uncertainty_upper IS NULL OR uncertainty_lower <= uncertainty_upper
    ),
    CONSTRAINT chk_pack024_fp_data_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_pack024_fp_completeness CHECK (
        completeness_percentage IS NULL OR (completeness_percentage >= 0 AND completeness_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_fp_org ON pack024_carbon_neutral.pack024_footprint_records(org_id);
CREATE INDEX idx_pack024_fp_tenant ON pack024_carbon_neutral.pack024_footprint_records(tenant_id);
CREATE INDEX idx_pack024_fp_year ON pack024_carbon_neutral.pack024_footprint_records(reporting_year);
CREATE INDEX idx_pack024_fp_scope ON pack024_carbon_neutral.pack024_footprint_records(scope);
CREATE INDEX idx_pack024_fp_scope_cat ON pack024_carbon_neutral.pack024_footprint_records(scope_category);
CREATE INDEX idx_pack024_fp_source ON pack024_carbon_neutral.pack024_footprint_records(emissions_source);
CREATE INDEX idx_pack024_fp_total ON pack024_carbon_neutral.pack024_footprint_records(total_emissions);
CREATE INDEX idx_pack024_fp_quality ON pack024_carbon_neutral.pack024_footprint_records(data_quality_score);
CREATE INDEX idx_pack024_fp_reconcile_status ON pack024_carbon_neutral.pack024_footprint_records(reconciliation_status);
CREATE INDEX idx_pack024_fp_verify_status ON pack024_carbon_neutral.pack024_footprint_records(verification_status);
CREATE INDEX idx_pack024_fp_verified ON pack024_carbon_neutral.pack024_footprint_records(third_party_verified);
CREATE INDEX idx_pack024_fp_mismatch ON pack024_carbon_neutral.pack024_footprint_records(mismatch_with_agents);
CREATE INDEX idx_pack024_fp_period ON pack024_carbon_neutral.pack024_footprint_records(reporting_period_start, reporting_period_end);
CREATE INDEX idx_pack024_fp_created_at ON pack024_carbon_neutral.pack024_footprint_records(created_at DESC);
CREATE INDEX idx_pack024_fp_assumptions ON pack024_carbon_neutral.pack024_footprint_records USING GIN(assumptions);
CREATE INDEX idx_pack024_fp_data_sources ON pack024_carbon_neutral.pack024_footprint_records USING GIN(data_sources);
CREATE INDEX idx_pack024_fp_agents ON pack024_carbon_neutral.pack024_footprint_records USING GIN(agent_references);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_fp_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_footprint_records
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 2: pack024_carbon_neutral.pack024_footprint_components
-- =============================================================================
-- Component-level breakdown of emissions showing individual activity/category
-- contributions to total scope emissions with sensitivity analysis capability.

CREATE TABLE pack024_carbon_neutral.pack024_footprint_components (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    footprint_record_id     UUID            NOT NULL REFERENCES pack024_carbon_neutral.pack024_footprint_records(id) ON DELETE CASCADE,
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    component_type          VARCHAR(100)    NOT NULL,
    category_code           VARCHAR(50),
    category_name           VARCHAR(255),
    activity_description    TEXT,
    quantity                DECIMAL(18,4),
    quantity_unit           VARCHAR(50),
    emission_factor         DECIMAL(18,6),
    ef_unit                 VARCHAR(100),
    ef_source               VARCHAR(255),
    ef_year                 INTEGER,
    emissions_contribution  DECIMAL(18,4)   NOT NULL,
    percentage_of_scope     DECIMAL(6,2),
    percentage_of_total     DECIMAL(6,2),
    data_quality_level      VARCHAR(30),
    sensitivity_rank        INTEGER,
    sensitivity_percentage  DECIMAL(6,2),
    aggregation_level       VARCHAR(30),
    level_hierarchy         INTEGER,
    related_agent_id        UUID,
    agent_calculation_id    VARCHAR(100),
    manual_override          BOOLEAN         DEFAULT FALSE,
    override_reason         TEXT,
    override_approval       VARCHAR(255),
    includes_uncertainty    BOOLEAN         DEFAULT TRUE,
    uncertainty_min         DECIMAL(18,4),
    uncertainty_max         DECIMAL(18,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_fpc_quantity_non_neg CHECK (
        quantity IS NULL OR quantity >= 0
    ),
    CONSTRAINT chk_pack024_fpc_ef_non_neg CHECK (
        emission_factor IS NULL OR emission_factor >= 0
    ),
    CONSTRAINT chk_pack024_fpc_emission_non_neg CHECK (
        emissions_contribution >= 0
    ),
    CONSTRAINT chk_pack024_fpc_percentage_valid CHECK (
        percentage_of_scope IS NULL OR (percentage_of_scope >= 0 AND percentage_of_scope <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_fpc_footprint_id ON pack024_carbon_neutral.pack024_footprint_components(footprint_record_id);
CREATE INDEX idx_pack024_fpc_org ON pack024_carbon_neutral.pack024_footprint_components(org_id);
CREATE INDEX idx_pack024_fpc_tenant ON pack024_carbon_neutral.pack024_footprint_components(tenant_id);
CREATE INDEX idx_pack024_fpc_type ON pack024_carbon_neutral.pack024_footprint_components(component_type);
CREATE INDEX idx_pack024_fpc_category ON pack024_carbon_neutral.pack024_footprint_components(category_code);
CREATE INDEX idx_pack024_fpc_contribution ON pack024_carbon_neutral.pack024_footprint_components(emissions_contribution DESC);
CREATE INDEX idx_pack024_fpc_percentage ON pack024_carbon_neutral.pack024_footprint_components(percentage_of_scope);
CREATE INDEX idx_pack024_fpc_sensitivity ON pack024_carbon_neutral.pack024_footprint_components(sensitivity_rank);
CREATE INDEX idx_pack024_fpc_agent_id ON pack024_carbon_neutral.pack024_footprint_components(related_agent_id);
CREATE INDEX idx_pack024_fpc_override ON pack024_carbon_neutral.pack024_footprint_components(manual_override);
CREATE INDEX idx_pack024_fpc_quality ON pack024_carbon_neutral.pack024_footprint_components(data_quality_level);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_fpc_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_footprint_components
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 3: pack024_carbon_neutral.pack024_baseline_establishment
-- =============================================================================
-- Baseline establishment and revision tracking for carbon neutral pathway.
-- Baseline year selection, normalization factors, and adjustment justifications.

CREATE TABLE pack024_carbon_neutral.pack024_baseline_establishment (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    baseline_year           INTEGER         NOT NULL,
    baseline_established_date DATE          NOT NULL,
    baseline_rationale      TEXT,
    baseline_scope          VARCHAR(50)     NOT NULL,
    total_baseline_emissions DECIMAL(18,4)  NOT NULL,
    unit                    VARCHAR(20)     DEFAULT 'tCO2e',
    scope1_baseline         DECIMAL(18,4),
    scope2_baseline         DECIMAL(18,4),
    scope3_baseline         DECIMAL(18,4),
    normalization_applied   BOOLEAN         DEFAULT FALSE,
    normalization_type      VARCHAR(100),
    normalization_factors   JSONB           DEFAULT '{}',
    normalization_basis     TEXT,
    baseline_adjustment_applied BOOLEAN     DEFAULT FALSE,
    adjustment_reason       TEXT,
    adjusted_baseline       DECIMAL(18,4),
    adjustment_percentage   DECIMAL(6,2),
    adjustment_approval     VARCHAR(255),
    materiality_assessment  DECIMAL(5,2),
    baseline_version        VARCHAR(20),
    baseline_revision_count INTEGER         DEFAULT 0,
    revision_history        JSONB           DEFAULT '{}',
    verification_status     VARCHAR(30),
    verifier_organization   VARCHAR(255),
    verification_date       DATE,
    peer_reviewed           BOOLEAN         DEFAULT FALSE,
    review_date             DATE,
    baseline_method         VARCHAR(100),
    method_description      TEXT,
    assumptions_documented  BOOLEAN         DEFAULT TRUE,
    assumptions_detail      JSONB           DEFAULT '{}',
    baseline_data_sources   TEXT[],
    baseline_quality_score  DECIMAL(5,2),
    materiality_threshold   DECIMAL(6,2),
    official_baseline_status BOOLEAN        DEFAULT FALSE,
    status_approval_date    DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_base_year CHECK (
        baseline_year >= 2000 AND baseline_year <= 2100
    ),
    CONSTRAINT chk_pack024_base_emissions_non_neg CHECK (
        total_baseline_emissions >= 0
    ),
    CONSTRAINT chk_pack024_base_materiality CHECK (
        materiality_assessment IS NULL OR (materiality_assessment >= 0 AND materiality_assessment <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_base_org ON pack024_carbon_neutral.pack024_baseline_establishment(org_id);
CREATE INDEX idx_pack024_base_tenant ON pack024_carbon_neutral.pack024_baseline_establishment(tenant_id);
CREATE INDEX idx_pack024_base_year ON pack024_carbon_neutral.pack024_baseline_establishment(baseline_year);
CREATE INDEX idx_pack024_base_established_date ON pack024_carbon_neutral.pack024_baseline_establishment(baseline_established_date);
CREATE INDEX idx_pack024_base_scope ON pack024_carbon_neutral.pack024_baseline_establishment(baseline_scope);
CREATE INDEX idx_pack024_base_official ON pack024_carbon_neutral.pack024_baseline_establishment(official_baseline_status);
CREATE INDEX idx_pack024_base_verify ON pack024_carbon_neutral.pack024_baseline_establishment(verification_status);
CREATE INDEX idx_pack024_base_quality ON pack024_carbon_neutral.pack024_baseline_establishment(baseline_quality_score);
CREATE INDEX idx_pack024_base_revision ON pack024_carbon_neutral.pack024_baseline_establishment(baseline_revision_count);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_base_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_baseline_establishment
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Table 4: pack024_carbon_neutral.pack024_uncertainty_assessment
-- =============================================================================
-- Detailed uncertainty quantification for emissions data with Monte Carlo
-- simulation support and sensitivity analysis results.

CREATE TABLE pack024_carbon_neutral.pack024_uncertainty_assessment (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  UUID            NOT NULL,
    tenant_id               UUID            NOT NULL,
    footprint_record_id     UUID            REFERENCES pack024_carbon_neutral.pack024_footprint_records(id) ON DELETE CASCADE,
    assessment_year         INTEGER         NOT NULL,
    assessment_date         DATE            NOT NULL,
    uncertainty_methodology VARCHAR(100),
    methodology_description TEXT,
    overall_uncertainty_min DECIMAL(6,2),
    overall_uncertainty_max DECIMAL(6,2),
    overall_uncertainty_95  DECIMAL(6,2),
    uncertainty_percentage  DECIMAL(6,2),
    confidence_interval     VARCHAR(30),
    quantitative_uncer      BOOLEAN         DEFAULT FALSE,
    quantitative_method     VARCHAR(100),
    qualitative_factors     TEXT[],
    monte_carlo_performed   BOOLEAN         DEFAULT FALSE,
    simulation_iterations   INTEGER,
    simulation_confidence   DECIMAL(5,2),
    sensitivity_analysis    BOOLEAN         DEFAULT FALSE,
    most_sensitive_factor   VARCHAR(255),
    sensitivity_percentage  DECIMAL(6,2),
    second_sensitive        VARCHAR(255),
    second_sensitivity_pct  DECIMAL(6,2),
    third_sensitive         VARCHAR(255),
    third_sensitivity_pct   DECIMAL(6,2),
    key_uncertainty_sources TEXT[],
    data_gaps               TEXT[],
    gap_mitigation_plan     TEXT,
    uncertainty_reduction_target DECIMAL(6,2),
    reduction_timeline      VARCHAR(100),
    improvement_actions     TEXT[],
    expert_judgment_applied BOOLEAN         DEFAULT FALSE,
    expert_names            TEXT[],
    expert_documentation    TEXT,
    uncertainty_register    JSONB           DEFAULT '{}',
    distribution_type       VARCHAR(50),
    distribution_params     JSONB           DEFAULT '{}',
    documented_by           VARCHAR(255),
    reviewed_by             VARCHAR(255),
    review_date             DATE,
    approved_by             VARCHAR(255),
    approval_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pack024_unc_year CHECK (
        assessment_year >= 2000 AND assessment_year <= 2100
    ),
    CONSTRAINT chk_pack024_unc_uncertainty_order CHECK (
        overall_uncertainty_min IS NULL OR overall_uncertainty_max IS NULL OR overall_uncertainty_min <= overall_uncertainty_max
    ),
    CONSTRAINT chk_pack024_unc_percentage CHECK (
        uncertainty_percentage IS NULL OR (uncertainty_percentage >= 0 AND uncertainty_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pack024_unc_org ON pack024_carbon_neutral.pack024_uncertainty_assessment(org_id);
CREATE INDEX idx_pack024_unc_tenant ON pack024_carbon_neutral.pack024_uncertainty_assessment(tenant_id);
CREATE INDEX idx_pack024_unc_footprint_id ON pack024_carbon_neutral.pack024_uncertainty_assessment(footprint_record_id);
CREATE INDEX idx_pack024_unc_year ON pack024_carbon_neutral.pack024_uncertainty_assessment(assessment_year);
CREATE INDEX idx_pack024_unc_date ON pack024_carbon_neutral.pack024_uncertainty_assessment(assessment_date);
CREATE INDEX idx_pack024_unc_methodology ON pack024_carbon_neutral.pack024_uncertainty_assessment(uncertainty_methodology);
CREATE INDEX idx_pack024_unc_monte_carlo ON pack024_carbon_neutral.pack024_uncertainty_assessment(monte_carlo_performed);
CREATE INDEX idx_pack024_unc_sensitivity ON pack024_carbon_neutral.pack024_uncertainty_assessment(sensitivity_analysis);
CREATE INDEX idx_pack024_unc_approved ON pack024_carbon_neutral.pack024_uncertainty_assessment(approval_date);

-- Updated_at trigger
CREATE TRIGGER trg_pack024_unc_updated_at
    BEFORE UPDATE ON pack024_carbon_neutral.pack024_uncertainty_assessment
    FOR EACH ROW
    EXECUTE FUNCTION pack024_carbon_neutral.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack024_carbon_neutral TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack024_carbon_neutral TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack024_carbon_neutral TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack024_carbon_neutral.pack024_footprint_records IS
'Scope-level carbon footprint records with emissions quantification, source traceability, methodology documentation, and reconciliation to MRV agents.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_footprint_components IS
'Component-level breakdown of emissions showing individual activity/category contributions to total scope emissions with sensitivity analysis.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_baseline_establishment IS
'Baseline establishment and revision tracking for carbon neutral pathway planning with normalization factors and adjustment justifications.';

COMMENT ON TABLE pack024_carbon_neutral.pack024_uncertainty_assessment IS
'Detailed uncertainty quantification for emissions data with Monte Carlo simulation support, sensitivity analysis, and data gap identification.';
