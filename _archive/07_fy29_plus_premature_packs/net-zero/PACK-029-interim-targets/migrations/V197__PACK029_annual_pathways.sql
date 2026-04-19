-- =============================================================================
-- V197: PACK-029 Interim Targets Pack - Annual Pathways
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    002 of 015
-- Date:         March 2026
--
-- Annual emission reduction pathways from baseline year to net-zero year
-- with linear, milestone, and accelerating pathway types, carbon budget
-- allocation per year, and cumulative reduction tracking.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_annual_pathways
--
-- TimescaleDB hypertable partitioned by year for time-series query performance.
-- Previous: V196__PACK029_interim_targets.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_annual_pathways
-- =============================================================================
-- Annual emission reduction pathway with year-by-year target emissions,
-- annual/cumulative reduction percentages, carbon budget allocation,
-- and pathway type for linear, milestone, or accelerating trajectories.

CREATE TABLE pack029_interim_targets.gl_annual_pathways (
    pathway_id                  UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE CASCADE,
    -- Time dimension
    year                        INTEGER         NOT NULL,
    -- Scope and pathway
    scope                       VARCHAR(20)     NOT NULL,
    pathway_type                VARCHAR(30)     NOT NULL DEFAULT 'LINEAR',
    pathway_label               VARCHAR(100),
    -- Emissions targets
    target_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    baseline_emissions_tco2e    DECIMAL(18,4)   NOT NULL,
    -- Reduction tracking
    annual_reduction_tco2e      DECIMAL(18,4),
    annual_reduction_pct        DECIMAL(8,4),
    cumulative_reduction_tco2e  DECIMAL(18,4),
    cumulative_reduction_pct    DECIMAL(8,4),
    -- Carbon budget
    carbon_budget_allocated_tco2e DECIMAL(18,4),
    carbon_budget_consumed_tco2e  DECIMAL(18,4)   DEFAULT 0,
    carbon_budget_remaining_tco2e DECIMAL(18,4),
    -- Intensity pathway
    target_intensity_value      DECIMAL(18,8),
    target_intensity_unit       VARCHAR(80),
    -- Pathway parameters
    pathway_slope               DECIMAL(12,8),
    acceleration_factor         DECIMAL(8,4)    DEFAULT 1.0,
    milestone_flag              BOOLEAN         DEFAULT FALSE,
    -- SBTi alignment
    sbti_minimum_reduction_pct  DECIMAL(8,4),
    sbti_compliant              BOOLEAN         DEFAULT FALSE,
    -- Data quality
    data_quality_score          DECIMAL(5,2),
    confidence_level            VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_locked                   BOOLEAN         DEFAULT FALSE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Primary key for hypertable
    PRIMARY KEY (pathway_id, year),
    -- Constraints
    CONSTRAINT chk_p029_ap_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_ap_pathway_type CHECK (
        pathway_type IN ('LINEAR', 'MILESTONE', 'ACCELERATING', 'FRONT_LOADED',
                         'BACK_LOADED', 'S_CURVE', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_ap_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_ap_target_emissions CHECK (
        target_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ap_baseline_emissions CHECK (
        baseline_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_ap_annual_reduction_pct CHECK (
        annual_reduction_pct IS NULL OR (annual_reduction_pct >= -100 AND annual_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p029_ap_cumulative_reduction_pct CHECK (
        cumulative_reduction_pct IS NULL OR (cumulative_reduction_pct >= -100 AND cumulative_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p029_ap_confidence CHECK (
        confidence_level IN ('HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p029_ap_data_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
SELECT create_hypertable(
    'pack029_interim_targets.gl_annual_pathways',
    'year',
    chunk_time_interval => 5,
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ap_tenant             ON pack029_interim_targets.gl_annual_pathways(tenant_id);
CREATE INDEX idx_p029_ap_org                ON pack029_interim_targets.gl_annual_pathways(organization_id);
CREATE INDEX idx_p029_ap_target             ON pack029_interim_targets.gl_annual_pathways(target_id);
CREATE INDEX idx_p029_ap_org_year           ON pack029_interim_targets.gl_annual_pathways(organization_id, year);
CREATE INDEX idx_p029_ap_scope_year         ON pack029_interim_targets.gl_annual_pathways(scope, year);
CREATE INDEX idx_p029_ap_org_scope_year     ON pack029_interim_targets.gl_annual_pathways(organization_id, scope, year);
CREATE INDEX idx_p029_ap_pathway_type       ON pack029_interim_targets.gl_annual_pathways(pathway_type);
CREATE INDEX idx_p029_ap_sbti_compliant     ON pack029_interim_targets.gl_annual_pathways(organization_id, sbti_compliant) WHERE sbti_compliant = TRUE;
CREATE INDEX idx_p029_ap_active             ON pack029_interim_targets.gl_annual_pathways(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_ap_milestone          ON pack029_interim_targets.gl_annual_pathways(organization_id, year) WHERE milestone_flag = TRUE;
CREATE INDEX idx_p029_ap_cumulative_red     ON pack029_interim_targets.gl_annual_pathways(cumulative_reduction_pct DESC NULLS LAST);
CREATE INDEX idx_p029_ap_created            ON pack029_interim_targets.gl_annual_pathways(created_at DESC);
CREATE INDEX idx_p029_ap_metadata           ON pack029_interim_targets.gl_annual_pathways USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_annual_pathways_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_annual_pathways
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_annual_pathways ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_ap_tenant_isolation
    ON pack029_interim_targets.gl_annual_pathways
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_ap_service_bypass
    ON pack029_interim_targets.gl_annual_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_annual_pathways TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_annual_pathways IS
    'Annual emission reduction pathway with year-by-year target emissions, reduction percentages, carbon budget allocation, and pathway type (linear/milestone/accelerating) for baseline-to-net-zero trajectory planning.';

COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.pathway_id IS 'Unique annual pathway record identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.organization_id IS 'Reference to the organization owning this pathway.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.target_id IS 'Foreign key to the interim target this pathway supports.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.year IS 'Calendar year for this pathway data point (baseline_year to net_zero_year).';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.pathway_type IS 'Pathway shape: LINEAR, MILESTONE, ACCELERATING, FRONT_LOADED, BACK_LOADED, S_CURVE, CUSTOM.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.target_emissions_tco2e IS 'Target emissions for this year in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.annual_reduction_pct IS 'Year-over-year reduction percentage from previous year target.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.cumulative_reduction_pct IS 'Cumulative reduction from baseline year.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.carbon_budget_allocated_tco2e IS 'Carbon budget allocated for this year in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.sbti_compliant IS 'Whether this year pathway point meets SBTi minimum reduction requirements.';
COMMENT ON COLUMN pack029_interim_targets.gl_annual_pathways.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
