-- =============================================================================
-- V129: PACK-023-sbti-alignment-001: SBTi Target Definitions and Pathways
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for SBTi target definition and pathway management.
-- Covers near-term, long-term, and net-zero target definitions with ACA/SDA/FLAG
-- pathway selection, ambition assessment (1.5C/WB2C/2C), base year validation,
-- and target boundary specification (operational/financial control, equity share).
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (target base structures)
--   PACK-021: Net Zero Starter Pack (baseline emissions)
--   PACK-022: Net Zero Acceleration Pack (scenario pathways)
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the foundational SBTi target definition layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_sbti_target_definitions    - Target definitions (near/long/net-zero)
--   2. pack023_sbti_target_boundaries     - Target boundary specifications
--   3. pack023_sbti_pathway_selections    - Pathway selection (ACA/SDA/FLAG)
--   4. pack023_sbti_ambition_assessments  - Temperature alignment assessment
--
-- Hypertables (1):
--   pack023_sbti_target_definitions on created_at (chunk: 3 months)
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V128__agent_eudr_authority_communication_manager.sql
-- =============================================================================

-- Schema
CREATE SCHEMA IF NOT EXISTS pack023_sbti_alignment;

-- =============================================================================
-- Function: Auto-update updated_at timestamp
-- =============================================================================

CREATE OR REPLACE FUNCTION pack023_sbti_alignment.set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_sbti_target_definitions
-- =============================================================================
-- SBTi target definitions for an organization. Stores near-term (5-10yr),
-- long-term (>2035), and net-zero (2050 max) target specifications with
-- reduction percentage, base year, target year, scope coverage, and status.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_target_definitions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    target_type             VARCHAR(30)     NOT NULL,
    target_name             VARCHAR(500),
    description             TEXT,
    base_year               INTEGER         NOT NULL,
    target_year             INTEGER         NOT NULL,
    target_scope            VARCHAR(50)     NOT NULL,
    reduction_percentage    DECIMAL(8,4)    NOT NULL,
    absolute_reduction_mt   DECIMAL(18,6),
    intensity_reduction_pc  DECIMAL(8,4),
    intensity_unit          VARCHAR(100),
    scope1_include          BOOLEAN         DEFAULT TRUE,
    scope2_include          BOOLEAN         DEFAULT TRUE,
    scope3_include          BOOLEAN         DEFAULT FALSE,
    scope3_percentage       DECIMAL(6,2),
    scope3_min_coverage     DECIMAL(6,2),
    boundary_method         VARCHAR(50),
    equity_share_percentage DECIMAL(6,2),
    external_verification   BOOLEAN         DEFAULT FALSE,
    verified_at             TIMESTAMPTZ,
    status                  VARCHAR(30)     DEFAULT 'draft',
    validation_errors       TEXT[],
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_target_type CHECK (
        target_type IN ('NEAR_TERM', 'LONG_TERM', 'NET_ZERO')
    ),
    CONSTRAINT chk_pk_base_year CHECK (
        base_year >= 2015 AND base_year <= EXTRACT(YEAR FROM NOW())::INTEGER
    ),
    CONSTRAINT chk_pk_target_year CHECK (
        target_year > base_year AND target_year <= 2050
    ),
    CONSTRAINT chk_pk_reduction_range CHECK (
        reduction_percentage >= 0 AND reduction_percentage <= 100
    ),
    CONSTRAINT chk_pk_status CHECK (
        status IN ('draft', 'submitted', 'validated', 'approved', 'rejected', 'expired')
    ),
    CONSTRAINT chk_pk_boundary_method CHECK (
        boundary_method IN ('operational_control', 'financial_control', 'equity_share')
    ),
    CONSTRAINT chk_pk_scope_select CHECK (
        (scope1_include OR scope2_include OR scope3_include)
    ),
    CONSTRAINT chk_pk_equity_share CHECK (
        equity_share_percentage IS NULL OR (equity_share_percentage > 0 AND equity_share_percentage <= 100)
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_sbti_target_definitions',
    'created_at',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_targets_tenant ON pack023_sbti_alignment.pack023_sbti_target_definitions(tenant_id);
CREATE INDEX idx_pk_targets_org ON pack023_sbti_alignment.pack023_sbti_target_definitions(org_id);
CREATE INDEX idx_pk_targets_type ON pack023_sbti_alignment.pack023_sbti_target_definitions(target_type);
CREATE INDEX idx_pk_targets_scope ON pack023_sbti_alignment.pack023_sbti_target_definitions(target_scope);
CREATE INDEX idx_pk_targets_base_year ON pack023_sbti_alignment.pack023_sbti_target_definitions(base_year);
CREATE INDEX idx_pk_targets_target_year ON pack023_sbti_alignment.pack023_sbti_target_definitions(target_year);
CREATE INDEX idx_pk_targets_status ON pack023_sbti_alignment.pack023_sbti_target_definitions(status);
CREATE INDEX idx_pk_targets_verified ON pack023_sbti_alignment.pack023_sbti_target_definitions(external_verification);
CREATE INDEX idx_pk_targets_org_type ON pack023_sbti_alignment.pack023_sbti_target_definitions(org_id, target_type);
CREATE INDEX idx_pk_targets_created_at ON pack023_sbti_alignment.pack023_sbti_target_definitions(created_at DESC);
CREATE INDEX idx_pk_targets_updated_at ON pack023_sbti_alignment.pack023_sbti_target_definitions(updated_at DESC);
CREATE INDEX idx_pk_targets_metadata ON pack023_sbti_alignment.pack023_sbti_target_definitions USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_targets_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_target_definitions
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_sbti_target_boundaries
-- =============================================================================
-- Target boundary details specifying operational scope, organizational units,
-- facilities, divisions, or geographic regions included in the target.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_target_boundaries (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    boundary_type           VARCHAR(50)     NOT NULL,
    boundary_description    TEXT,
    include_scopes          VARCHAR(30)[],
    scope1_facilities       TEXT[],
    scope2_sources          TEXT[],
    geographic_regions      VARCHAR(100)[],
    business_units          VARCHAR(500)[],
    subsidiaries_included   BOOLEAN         DEFAULT FALSE,
    jv_equity_share         DECIMAL(6,2),
    exclusions              TEXT[],
    exclusion_rationale     TEXT,
    completeness_percentage DECIMAL(6,2),
    status                  VARCHAR(30)     DEFAULT 'draft',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_boundary_type CHECK (
        boundary_type IN ('operational', 'financial', 'equity_share', 'combined')
    ),
    CONSTRAINT chk_pk_completeness CHECK (
        completeness_percentage >= 0 AND completeness_percentage <= 100
    ),
    CONSTRAINT chk_pk_jv_share CHECK (
        jv_equity_share IS NULL OR (jv_equity_share > 0 AND jv_equity_share <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pk_boundary_target ON pack023_sbti_alignment.pack023_sbti_target_boundaries(target_definition_id);
CREATE INDEX idx_pk_boundary_tenant ON pack023_sbti_alignment.pack023_sbti_target_boundaries(tenant_id);
CREATE INDEX idx_pk_boundary_type ON pack023_sbti_alignment.pack023_sbti_target_boundaries(boundary_type);
CREATE INDEX idx_pk_boundary_created_at ON pack023_sbti_alignment.pack023_sbti_target_boundaries(created_at DESC);
CREATE INDEX idx_pk_boundary_scope1 ON pack023_sbti_alignment.pack023_sbti_target_boundaries USING GIN(scope1_facilities);
CREATE INDEX idx_pk_boundary_regions ON pack023_sbti_alignment.pack023_sbti_target_boundaries USING GIN(geographic_regions);

-- Updated_at trigger
CREATE TRIGGER trg_pk_boundary_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_target_boundaries
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_sbti_pathway_selections
-- =============================================================================
-- Pathway selection records showing which reduction pathway (ACA/SDA/FLAG)
-- was chosen for each target, with rationale, benchmarks, and alignment data.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_pathway_selections (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    pathway_type            VARCHAR(50)     NOT NULL,
    pathway_name            VARCHAR(500),
    scope1_pathway          VARCHAR(50),
    scope2_pathway          VARCHAR(50),
    scope3_pathway          VARCHAR(50),
    sector_code             VARCHAR(20),
    sector_name             VARCHAR(500),
    aca_annual_reduction    DECIMAL(6,4),
    aca_ambition            VARCHAR(50),
    sda_sector              VARCHAR(100),
    sda_baseline_intensity  DECIMAL(12,6),
    sda_2050_intensity      DECIMAL(12,6),
    sda_intensity_unit      VARCHAR(100),
    flag_flag_percentage    DECIMAL(6,2),
    flag_commodities        VARCHAR(50)[],
    flag_annual_reduction   DECIMAL(6,4),
    rationale               TEXT,
    selection_method        VARCHAR(100),
    documented_by           VARCHAR(255),
    approved_by             VARCHAR(255),
    approval_date           DATE,
    status                  VARCHAR(30)     DEFAULT 'draft',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_pathway_type CHECK (
        pathway_type IN ('ACA', 'SDA', 'FLAG', 'COMBINED')
    ),
    CONSTRAINT chk_pk_aca_reduction CHECK (
        aca_annual_reduction IS NULL OR (aca_annual_reduction >= 0 AND aca_annual_reduction <= 10)
    ),
    CONSTRAINT chk_pk_flag_percentage CHECK (
        flag_flag_percentage IS NULL OR (flag_flag_percentage >= 0 AND flag_flag_percentage <= 100)
    )
);

-- Indexes
CREATE INDEX idx_pk_pathway_target ON pack023_sbti_alignment.pack023_sbti_pathway_selections(target_definition_id);
CREATE INDEX idx_pk_pathway_tenant ON pack023_sbti_alignment.pack023_sbti_pathway_selections(tenant_id);
CREATE INDEX idx_pk_pathway_org ON pack023_sbti_alignment.pack023_sbti_pathway_selections(org_id);
CREATE INDEX idx_pk_pathway_type ON pack023_sbti_alignment.pack023_sbti_pathway_selections(pathway_type);
CREATE INDEX idx_pk_pathway_sector ON pack023_sbti_alignment.pack023_sbti_pathway_selections(sector_code);
CREATE INDEX idx_pk_pathway_sda_sector ON pack023_sbti_alignment.pack023_sbti_pathway_selections(sda_sector);
CREATE INDEX idx_pk_pathway_created_at ON pack023_sbti_alignment.pack023_sbti_pathway_selections(created_at DESC);
CREATE INDEX idx_pk_pathway_status ON pack023_sbti_alignment.pack023_sbti_pathway_selections(status);
CREATE INDEX idx_pk_pathway_commodities ON pack023_sbti_alignment.pack023_sbti_pathway_selections USING GIN(flag_commodities);

-- Updated_at trigger
CREATE TRIGGER trg_pk_pathway_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_pathway_selections
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_sbti_ambition_assessments
-- =============================================================================
-- Ambition level assessment results showing temperature alignment of selected
-- targets against 1.5C, Well-Below-2C, and 2C scenarios.

CREATE TABLE pack023_sbti_alignment.pack023_sbti_ambition_assessments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_definition_id    UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_sbti_target_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    ambition_level          VARCHAR(30)     NOT NULL,
    implied_temperature     DECIMAL(4,2),
    reduction_rate_annual   DECIMAL(6,4),
    target_aligned_1_5c     BOOLEAN         DEFAULT FALSE,
    target_aligned_2c       BOOLEAN         DEFAULT FALSE,
    ipcc_carbon_budget      DECIMAL(18,6),
    carbon_budget_used_pct  DECIMAL(6,2),
    comparison_aca_1_5c     DECIMAL(6,4),
    comparison_aca_2c       DECIMAL(6,4),
    gap_to_1_5c             DECIMAL(8,4),
    gap_analysis            TEXT,
    assumptions             JSONB           DEFAULT '{}',
    evidence_links          TEXT[],
    validation_status       VARCHAR(30)     DEFAULT 'pending',
    validated_at            TIMESTAMPTZ,
    validated_by            VARCHAR(255),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_ambition_level CHECK (
        ambition_level IN ('1.5C', 'WB2C', '2C', 'UNALIGNED')
    ),
    CONSTRAINT chk_pk_temp_range CHECK (
        implied_temperature IS NULL OR (implied_temperature >= 1.0 AND implied_temperature <= 6.0)
    ),
    CONSTRAINT chk_pk_reduction_rate CHECK (
        reduction_rate_annual IS NULL OR (reduction_rate_annual >= 0 AND reduction_rate_annual <= 10)
    )
);

-- Indexes
CREATE INDEX idx_pk_ambition_target ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(target_definition_id);
CREATE INDEX idx_pk_ambition_tenant ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(tenant_id);
CREATE INDEX idx_pk_ambition_org ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(org_id);
CREATE INDEX idx_pk_ambition_level ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(ambition_level);
CREATE INDEX idx_pk_ambition_validation ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(validation_status);
CREATE INDEX idx_pk_ambition_temp ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(implied_temperature);
CREATE INDEX idx_pk_ambition_created_at ON pack023_sbti_alignment.pack023_sbti_ambition_assessments(created_at DESC);
CREATE INDEX idx_pk_ambition_assumptions ON pack023_sbti_alignment.pack023_sbti_ambition_assessments USING GIN(assumptions);

-- Updated_at trigger
CREATE TRIGGER trg_pk_ambition_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_sbti_ambition_assessments
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

COMMENT ON SCHEMA pack023_sbti_alignment IS 'PACK-023 SBTi Alignment Pack schema';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_target_definitions IS
'SBTi target definitions covering near-term (5-10yr), long-term (>2035), and net-zero (2050 max) targets with reduction percentages and scope specifications.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_target_boundaries IS
'Detailed target boundary specifications defining organizational units, facilities, geographic regions, and subsidiaries included in each target.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_pathway_selections IS
'Pathway selection records tracking which reduction methodology (ACA/SDA/FLAG) was chosen for each target with benchmarks and rationale.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_sbti_ambition_assessments IS
'Ambition assessment results showing temperature alignment of targets against 1.5C, WB2C, and 2C carbon budget scenarios.';
