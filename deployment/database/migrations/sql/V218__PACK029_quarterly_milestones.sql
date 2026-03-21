-- =============================================================================
-- V198: PACK-029 Interim Targets Pack - Quarterly Milestones
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    003 of 015
-- Date:         March 2026
--
-- Quarterly milestone breakdowns for interim target tracking with
-- per-quarter emission targets, reduction percentages, and milestone
-- achievement status for granular progress monitoring.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_quarterly_milestones
--
-- TimescaleDB hypertable partitioned by year for time-series performance.
-- Previous: V197__PACK029_annual_pathways.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_quarterly_milestones
-- =============================================================================
-- Quarterly milestone breakdowns for interim targets with per-quarter
-- emission targets, reduction percentages, milestone achievement tracking,
-- and seasonal adjustment factors.

CREATE TABLE pack029_interim_targets.gl_quarterly_milestones (
    milestone_id                UUID            NOT NULL DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE CASCADE,
    -- Time dimension
    year                        INTEGER         NOT NULL,
    quarter                     VARCHAR(2)      NOT NULL,
    quarter_start_date          DATE,
    quarter_end_date            DATE,
    -- Scope
    scope                       VARCHAR(20)     NOT NULL,
    -- Milestone targets
    milestone_emissions_tco2e   DECIMAL(18,4)   NOT NULL,
    milestone_reduction_pct     DECIMAL(8,4),
    annual_target_share_pct     DECIMAL(6,2)    DEFAULT 25.00,
    -- Seasonal adjustment
    seasonal_factor             DECIMAL(6,4)    DEFAULT 1.0000,
    seasonally_adjusted         BOOLEAN         DEFAULT FALSE,
    -- Achievement tracking
    milestone_status            VARCHAR(20)     DEFAULT 'PENDING',
    achieved                    BOOLEAN         DEFAULT FALSE,
    achieved_at                 TIMESTAMPTZ,
    -- Prior quarter comparison
    prior_quarter_emissions     DECIMAL(18,4),
    qoq_change_pct              DECIMAL(8,4),
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
    PRIMARY KEY (milestone_id, year),
    -- Constraints
    CONSTRAINT chk_p029_qm_quarter CHECK (
        quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_qm_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_qm_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_qm_milestone_emissions CHECK (
        milestone_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_qm_milestone_reduction CHECK (
        milestone_reduction_pct IS NULL OR (milestone_reduction_pct >= -100 AND milestone_reduction_pct <= 100)
    ),
    CONSTRAINT chk_p029_qm_annual_share CHECK (
        annual_target_share_pct >= 0 AND annual_target_share_pct <= 100
    ),
    CONSTRAINT chk_p029_qm_seasonal_factor CHECK (
        seasonal_factor >= 0.01 AND seasonal_factor <= 10.0
    ),
    CONSTRAINT chk_p029_qm_milestone_status CHECK (
        milestone_status IN ('PENDING', 'ON_TRACK', 'AT_RISK', 'MISSED', 'ACHIEVED', 'DEFERRED')
    ),
    CONSTRAINT chk_p029_qm_confidence CHECK (
        confidence_level IN ('HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p029_qm_data_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- TimescaleDB Hypertable
-- ---------------------------------------------------------------------------
SELECT create_hypertable(
    'pack029_interim_targets.gl_quarterly_milestones',
    'year',
    chunk_time_interval => 5,
    if_not_exists => TRUE,
    migrate_data => TRUE
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_qm_tenant             ON pack029_interim_targets.gl_quarterly_milestones(tenant_id);
CREATE INDEX idx_p029_qm_org                ON pack029_interim_targets.gl_quarterly_milestones(organization_id);
CREATE INDEX idx_p029_qm_target             ON pack029_interim_targets.gl_quarterly_milestones(target_id);
CREATE INDEX idx_p029_qm_org_year_quarter   ON pack029_interim_targets.gl_quarterly_milestones(organization_id, year, quarter);
CREATE INDEX idx_p029_qm_scope_year         ON pack029_interim_targets.gl_quarterly_milestones(scope, year);
CREATE INDEX idx_p029_qm_org_scope_year     ON pack029_interim_targets.gl_quarterly_milestones(organization_id, scope, year, quarter);
CREATE INDEX idx_p029_qm_status             ON pack029_interim_targets.gl_quarterly_milestones(milestone_status);
CREATE INDEX idx_p029_qm_missed             ON pack029_interim_targets.gl_quarterly_milestones(organization_id, year) WHERE milestone_status = 'MISSED';
CREATE INDEX idx_p029_qm_at_risk            ON pack029_interim_targets.gl_quarterly_milestones(organization_id, year) WHERE milestone_status = 'AT_RISK';
CREATE INDEX idx_p029_qm_achieved           ON pack029_interim_targets.gl_quarterly_milestones(organization_id, achieved) WHERE achieved = TRUE;
CREATE INDEX idx_p029_qm_active             ON pack029_interim_targets.gl_quarterly_milestones(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_qm_created            ON pack029_interim_targets.gl_quarterly_milestones(created_at DESC);
CREATE INDEX idx_p029_qm_metadata           ON pack029_interim_targets.gl_quarterly_milestones USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_quarterly_milestones_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_quarterly_milestones
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_quarterly_milestones ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_qm_tenant_isolation
    ON pack029_interim_targets.gl_quarterly_milestones
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_qm_service_bypass
    ON pack029_interim_targets.gl_quarterly_milestones
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_quarterly_milestones TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_quarterly_milestones IS
    'Quarterly milestone breakdowns for interim targets with per-quarter emission targets, reduction percentages, seasonal adjustment factors, and milestone achievement tracking for granular progress monitoring.';

COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.milestone_id IS 'Unique quarterly milestone identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.organization_id IS 'Reference to the organization owning this milestone.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.year IS 'Calendar year for this milestone.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.quarter IS 'Quarter designation: Q1, Q2, Q3, Q4.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.milestone_emissions_tco2e IS 'Target emissions for this quarter in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.milestone_reduction_pct IS 'Reduction percentage from same quarter in baseline year.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.annual_target_share_pct IS 'Percentage of annual target allocated to this quarter (default 25%).';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.seasonal_factor IS 'Seasonal adjustment factor for quarter (e.g., heating season Q1/Q4 > Q2/Q3).';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.milestone_status IS 'Milestone status: PENDING, ON_TRACK, AT_RISK, MISSED, ACHIEVED, DEFERRED.';
COMMENT ON COLUMN pack029_interim_targets.gl_quarterly_milestones.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
