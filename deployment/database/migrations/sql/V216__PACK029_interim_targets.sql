-- =============================================================================
-- V196: PACK-029 Interim Targets Pack - Schema & Interim Targets
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    001 of 015
-- Date:         March 2026
--
-- Creates the pack029_interim_targets schema and the core interim targets
-- table with baseline/target year tracking, scope-level reduction targets,
-- SBTi pathway alignment (1.5C/WB2C), and validation status.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_interim_targets
--
-- Also includes: schema, update trigger function, indexes, RLS, comments.
-- Previous: V195__PACK028_views_and_indexes.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack029_interim_targets;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack029_interim_targets.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_interim_targets
-- =============================================================================
-- Core interim target definitions with baseline year emissions, target year
-- reduction percentages, SBTi pathway alignment, and validation tracking
-- for near-term (2030) and mid-term (2035) science-based targets.

CREATE TABLE pack029_interim_targets.gl_interim_targets (
    target_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Baseline definition
    baseline_year               INTEGER         NOT NULL,
    baseline_emissions_tco2e    DECIMAL(18,4)   NOT NULL,
    baseline_revenue_usd        DECIMAL(18,2),
    baseline_intensity_value    DECIMAL(18,8),
    baseline_intensity_unit     VARCHAR(80),
    baseline_verified           BOOLEAN         DEFAULT FALSE,
    baseline_verification_date  DATE,
    -- Target definition
    target_year                 INTEGER         NOT NULL,
    target_type                 VARCHAR(30)     NOT NULL DEFAULT 'ABSOLUTE',
    scope                       VARCHAR(20)     NOT NULL,
    target_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    reduction_pct               DECIMAL(8,4)    NOT NULL,
    target_intensity_value      DECIMAL(18,8),
    target_intensity_unit       VARCHAR(80),
    -- SBTi pathway alignment
    sbti_pathway                VARCHAR(20)     NOT NULL DEFAULT '1_5C',
    sbti_method                 VARCHAR(30)     DEFAULT 'ABSOLUTE_CONTRACTION',
    sbti_near_term              BOOLEAN         DEFAULT FALSE,
    sbti_long_term              BOOLEAN         DEFAULT FALSE,
    sbti_committed              BOOLEAN         DEFAULT FALSE,
    sbti_validated              BOOLEAN         DEFAULT FALSE,
    sbti_target_id              VARCHAR(50),
    -- Coverage
    coverage_pct                DECIMAL(5,2)    DEFAULT 100.00,
    boundary_description        TEXT,
    exclusions                  JSONB           DEFAULT '[]',
    -- Validation
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    validation_notes            TEXT,
    -- Approval workflow
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    approval_status             VARCHAR(20)     DEFAULT 'PENDING',
    -- Status tracking
    is_active                   BOOLEAN         DEFAULT TRUE,
    superseded_by               UUID,
    superseded_at               TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_it_target_lt_baseline CHECK (
        target_emissions_tco2e <= baseline_emissions_tco2e
    ),
    CONSTRAINT chk_p029_it_reduction_pct CHECK (
        reduction_pct >= 0 AND reduction_pct <= 100
    ),
    CONSTRAINT chk_p029_it_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_it_target_type CHECK (
        target_type IN ('ABSOLUTE', 'INTENSITY', 'ABSOLUTE_AND_INTENSITY')
    ),
    CONSTRAINT chk_p029_it_sbti_pathway CHECK (
        sbti_pathway IN ('1_5C', 'WB2C', '2C', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_it_sbti_method CHECK (
        sbti_method IS NULL OR sbti_method IN (
            'ABSOLUTE_CONTRACTION', 'SECTORAL_DECARBONIZATION',
            'ECONOMIC_INTENSITY', 'PHYSICAL_INTENSITY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p029_it_validation_status CHECK (
        validation_status IN ('DRAFT', 'PENDING', 'VALIDATED', 'REJECTED', 'EXPIRED', 'REVIEW_REQUIRED')
    ),
    CONSTRAINT chk_p029_it_approval_status CHECK (
        approval_status IN ('PENDING', 'APPROVED', 'REJECTED', 'WITHDRAWN')
    ),
    CONSTRAINT chk_p029_it_coverage_pct CHECK (
        coverage_pct >= 0 AND coverage_pct <= 100
    ),
    CONSTRAINT chk_p029_it_baseline_year CHECK (
        baseline_year >= 2000 AND baseline_year <= 2100
    ),
    CONSTRAINT chk_p029_it_target_year CHECK (
        target_year >= 2025 AND target_year <= 2100
    ),
    CONSTRAINT chk_p029_it_target_after_baseline CHECK (
        target_year > baseline_year
    ),
    CONSTRAINT chk_p029_it_baseline_emissions CHECK (
        baseline_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_it_target_emissions CHECK (
        target_emissions_tco2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_it_tenant             ON pack029_interim_targets.gl_interim_targets(tenant_id);
CREATE INDEX idx_p029_it_org                ON pack029_interim_targets.gl_interim_targets(organization_id);
CREATE INDEX idx_p029_it_org_baseline       ON pack029_interim_targets.gl_interim_targets(organization_id, baseline_year);
CREATE INDEX idx_p029_it_org_target         ON pack029_interim_targets.gl_interim_targets(organization_id, target_year);
CREATE INDEX idx_p029_it_org_scope          ON pack029_interim_targets.gl_interim_targets(organization_id, scope);
CREATE INDEX idx_p029_it_scope              ON pack029_interim_targets.gl_interim_targets(scope);
CREATE INDEX idx_p029_it_target_year        ON pack029_interim_targets.gl_interim_targets(target_year);
CREATE INDEX idx_p029_it_baseline_year      ON pack029_interim_targets.gl_interim_targets(baseline_year);
CREATE INDEX idx_p029_it_sbti_pathway       ON pack029_interim_targets.gl_interim_targets(sbti_pathway);
CREATE INDEX idx_p029_it_validation         ON pack029_interim_targets.gl_interim_targets(validation_status);
CREATE INDEX idx_p029_it_approval           ON pack029_interim_targets.gl_interim_targets(approval_status);
CREATE INDEX idx_p029_it_active             ON pack029_interim_targets.gl_interim_targets(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_it_sbti_validated     ON pack029_interim_targets.gl_interim_targets(organization_id, sbti_validated) WHERE sbti_validated = TRUE;
CREATE INDEX idx_p029_it_sbti_near_term     ON pack029_interim_targets.gl_interim_targets(organization_id, sbti_near_term) WHERE sbti_near_term = TRUE;
CREATE INDEX idx_p029_it_reduction_pct      ON pack029_interim_targets.gl_interim_targets(reduction_pct DESC);
CREATE INDEX idx_p029_it_created            ON pack029_interim_targets.gl_interim_targets(created_at DESC);
CREATE INDEX idx_p029_it_exclusions         ON pack029_interim_targets.gl_interim_targets USING GIN(exclusions);
CREATE INDEX idx_p029_it_metadata           ON pack029_interim_targets.gl_interim_targets USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_interim_targets_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_interim_targets
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_interim_targets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_it_tenant_isolation
    ON pack029_interim_targets.gl_interim_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_it_service_bypass
    ON pack029_interim_targets.gl_interim_targets
    TO greenlang_service
    USING (TRUE)
    WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack029_interim_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_interim_targets TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack029_interim_targets.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack029_interim_targets IS
    'PACK-029 Interim Targets Pack - Interim target tracking, annual pathway decomposition, quarterly milestone monitoring, variance analysis, corrective actions, and SBTi submission management for science-based net-zero transition.';

COMMENT ON TABLE pack029_interim_targets.gl_interim_targets IS
    'Core interim target definitions with baseline/target year emissions, scope-level reduction percentages, SBTi pathway alignment (1.5C/WB2C), validation status, and approval workflow for near-term and mid-term targets.';

COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.target_id IS 'Unique interim target identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.organization_id IS 'Reference to the organization owning this target.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.baseline_year IS 'Base year for emissions baseline (e.g., 2019, 2020).';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.baseline_emissions_tco2e IS 'Total baseline emissions in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.target_year IS 'Target year for emissions reduction (e.g., 2030, 2035).';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.scope IS 'GHG Protocol scope coverage: SCOPE_1, SCOPE_2, SCOPE_3, SCOPE_1_2, SCOPE_1_2_3.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.target_emissions_tco2e IS 'Target emissions level in tonnes CO2 equivalent at the target year.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.reduction_pct IS 'Required reduction percentage from baseline (0-100).';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.sbti_pathway IS 'SBTi temperature pathway alignment: 1_5C, WB2C, 2C, CUSTOM.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.sbti_method IS 'SBTi target-setting method: ABSOLUTE_CONTRACTION, SECTORAL_DECARBONIZATION, etc.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.validation_status IS 'Target validation status: DRAFT, PENDING, VALIDATED, REJECTED, EXPIRED, REVIEW_REQUIRED.';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.coverage_pct IS 'Percentage of organizational emissions covered by this target (0-100).';
COMMENT ON COLUMN pack029_interim_targets.gl_interim_targets.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
