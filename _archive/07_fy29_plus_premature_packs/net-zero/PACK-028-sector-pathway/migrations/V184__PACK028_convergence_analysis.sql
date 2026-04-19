-- =============================================================================
-- V184: PACK-028 Sector Pathway Pack - Convergence Analysis
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    004 of 015
-- Date:         March 2026
--
-- Sector intensity convergence analysis with gap calculations,
-- time-to-convergence modeling, acceleration requirements, and
-- risk-level assessments for SBTi SDA pathway alignment tracking.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_convergence
--
-- Previous: V183__PACK028_sector_pathways.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_convergence
-- =============================================================================

CREATE TABLE pack028_sector_pathway.gl_sector_convergence (
    convergence_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    pathway_id                  UUID            REFERENCES pack028_sector_pathway.gl_sector_pathways(pathway_id) ON DELETE SET NULL,
    metric_id                   UUID            REFERENCES pack028_sector_pathway.gl_sector_intensity_metrics(metric_id) ON DELETE SET NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    scenario                    VARCHAR(30)     NOT NULL DEFAULT 'NZE_1_5C',
    intensity_metric            VARCHAR(60)     NOT NULL,
    -- Analysis period
    analysis_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    analysis_year               INTEGER         NOT NULL,
    base_year                   INTEGER         NOT NULL,
    target_year                 INTEGER         NOT NULL DEFAULT 2050,
    -- Current position
    current_intensity           DECIMAL(18,8)   NOT NULL,
    current_intensity_unit      VARCHAR(80)     NOT NULL,
    current_year_pathway_target DECIMAL(18,8)   NOT NULL,
    -- Gap analysis
    intensity_gap               DECIMAL(18,8)   NOT NULL,
    intensity_gap_pct           DECIMAL(8,4)    NOT NULL,
    gap_direction               VARCHAR(10)     NOT NULL,
    gap_severity                VARCHAR(20)     NOT NULL,
    -- Trajectory analysis
    current_annual_reduction_pct DECIMAL(8,4),
    required_annual_reduction_pct DECIMAL(8,4),
    acceleration_needed_pct     DECIMAL(8,4),
    trajectory_intensity_2030   DECIMAL(18,8),
    trajectory_intensity_2050   DECIMAL(18,8),
    -- Time-to-convergence
    time_to_convergence_years   DECIMAL(6,2),
    convergence_year            INTEGER,
    convergence_feasible        BOOLEAN         DEFAULT TRUE,
    -- Historical trend
    intensity_trend_3yr         DECIMAL(8,4),
    intensity_trend_5yr         DECIMAL(8,4),
    trend_direction             VARCHAR(20),
    trend_consistency_score     DECIMAL(5,2),
    -- Year-by-year gap projection
    projected_gap_2025          DECIMAL(8,4),
    projected_gap_2030          DECIMAL(8,4),
    projected_gap_2035          DECIMAL(8,4),
    projected_gap_2040          DECIMAL(8,4),
    projected_gap_2045          DECIMAL(8,4),
    projected_gap_2050          DECIMAL(8,4),
    annual_gap_projection       JSONB           DEFAULT '{}',
    -- Required actions
    required_absolute_reduction DECIMAL(18,4),
    required_actions_summary    JSONB           DEFAULT '{}',
    recommended_levers          JSONB           DEFAULT '[]',
    -- Risk assessment
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    risk_score                  DECIMAL(5,2),
    risk_factors                JSONB           DEFAULT '[]',
    probability_on_track        DECIMAL(5,2),
    probability_convergence     DECIMAL(5,2),
    -- Comparison benchmarks
    sector_avg_intensity        DECIMAL(18,8),
    sector_leader_intensity     DECIMAL(18,8),
    peer_median_intensity       DECIMAL(18,8),
    position_vs_sector_avg_pct  DECIMAL(8,4),
    position_vs_leader_pct      DECIMAL(8,4),
    percentile_rank             DECIMAL(5,2),
    -- SBTi alignment
    sbti_aligned                BOOLEAN         DEFAULT FALSE,
    sbti_gap_pct                DECIMAL(8,4),
    sbti_alignment_score        DECIMAL(5,2),
    -- IEA alignment
    iea_aligned                 BOOLEAN         DEFAULT FALSE,
    iea_gap_pct                 DECIMAL(8,4),
    iea_milestone_status        VARCHAR(20),
    -- Investment gap
    investment_gap_usd          DECIMAL(18,2),
    annual_investment_needed    DECIMAL(18,2),
    technology_gap_description  TEXT,
    -- Metadata
    analysis_version            INTEGER         DEFAULT 1,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_cv_scenario CHECK (
        scenario IN ('NZE_1_5C', 'WB2C', '2C', 'APS', 'STEPS', 'NDC_ALIGNED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_cv_gap_direction CHECK (
        gap_direction IN ('ABOVE', 'BELOW', 'ON_TARGET')
    ),
    CONSTRAINT chk_p028_cv_gap_severity CHECK (
        gap_severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'ON_TRACK', 'AHEAD')
    ),
    CONSTRAINT chk_p028_cv_risk_level CHECK (
        risk_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL')
    ),
    CONSTRAINT chk_p028_cv_trend_direction CHECK (
        trend_direction IS NULL OR trend_direction IN (
            'STRONGLY_IMPROVING', 'IMPROVING', 'STABLE', 'DETERIORATING', 'STRONGLY_DETERIORATING'
        )
    ),
    CONSTRAINT chk_p028_cv_iea_milestone CHECK (
        iea_milestone_status IS NULL OR iea_milestone_status IN (
            'ON_TRACK', 'BEHIND', 'WELL_BEHIND', 'AHEAD', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p028_cv_analysis_year CHECK (
        analysis_year >= 2020 AND analysis_year <= 2100
    ),
    CONSTRAINT chk_p028_cv_base_year CHECK (
        base_year >= 2000 AND base_year <= 2030
    ),
    CONSTRAINT chk_p028_cv_target_year CHECK (
        target_year >= 2030 AND target_year <= 2100
    ),
    CONSTRAINT chk_p028_cv_risk_score CHECK (
        risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)
    ),
    CONSTRAINT chk_p028_cv_probability_track CHECK (
        probability_on_track IS NULL OR (probability_on_track >= 0 AND probability_on_track <= 100)
    ),
    CONSTRAINT chk_p028_cv_probability_conv CHECK (
        probability_convergence IS NULL OR (probability_convergence >= 0 AND probability_convergence <= 100)
    ),
    CONSTRAINT chk_p028_cv_percentile CHECK (
        percentile_rank IS NULL OR (percentile_rank >= 0 AND percentile_rank <= 100)
    ),
    CONSTRAINT chk_p028_cv_sbti_alignment CHECK (
        sbti_alignment_score IS NULL OR (sbti_alignment_score >= 0 AND sbti_alignment_score <= 100)
    ),
    CONSTRAINT chk_p028_cv_trend_consistency CHECK (
        trend_consistency_score IS NULL OR (trend_consistency_score >= 0 AND trend_consistency_score <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_cv_tenant             ON pack028_sector_pathway.gl_sector_convergence(tenant_id);
CREATE INDEX idx_p028_cv_company            ON pack028_sector_pathway.gl_sector_convergence(company_id);
CREATE INDEX idx_p028_cv_pathway            ON pack028_sector_pathway.gl_sector_convergence(pathway_id);
CREATE INDEX idx_p028_cv_metric             ON pack028_sector_pathway.gl_sector_convergence(metric_id);
CREATE INDEX idx_p028_cv_classification     ON pack028_sector_pathway.gl_sector_convergence(classification_id);
CREATE INDEX idx_p028_cv_sector             ON pack028_sector_pathway.gl_sector_convergence(sector_code);
CREATE INDEX idx_p028_cv_scenario           ON pack028_sector_pathway.gl_sector_convergence(scenario);
CREATE INDEX idx_p028_cv_analysis_date      ON pack028_sector_pathway.gl_sector_convergence(analysis_date DESC);
CREATE INDEX idx_p028_cv_analysis_year      ON pack028_sector_pathway.gl_sector_convergence(analysis_year);
CREATE INDEX idx_p028_cv_gap_severity       ON pack028_sector_pathway.gl_sector_convergence(gap_severity);
CREATE INDEX idx_p028_cv_risk_level         ON pack028_sector_pathway.gl_sector_convergence(risk_level);
CREATE INDEX idx_p028_cv_gap_direction      ON pack028_sector_pathway.gl_sector_convergence(gap_direction);
CREATE INDEX idx_p028_cv_company_scenario   ON pack028_sector_pathway.gl_sector_convergence(company_id, scenario, analysis_year);
CREATE INDEX idx_p028_cv_company_sector     ON pack028_sector_pathway.gl_sector_convergence(company_id, sector_code, analysis_year);
CREATE INDEX idx_p028_cv_sbti_aligned       ON pack028_sector_pathway.gl_sector_convergence(sbti_aligned) WHERE sbti_aligned = FALSE;
CREATE INDEX idx_p028_cv_iea_aligned        ON pack028_sector_pathway.gl_sector_convergence(iea_aligned) WHERE iea_aligned = FALSE;
CREATE INDEX idx_p028_cv_critical           ON pack028_sector_pathway.gl_sector_convergence(risk_level) WHERE risk_level IN ('CRITICAL', 'HIGH');
CREATE INDEX idx_p028_cv_convergence_year   ON pack028_sector_pathway.gl_sector_convergence(convergence_year);
CREATE INDEX idx_p028_cv_not_feasible       ON pack028_sector_pathway.gl_sector_convergence(convergence_feasible) WHERE convergence_feasible = FALSE;
CREATE INDEX idx_p028_cv_percentile         ON pack028_sector_pathway.gl_sector_convergence(percentile_rank);
CREATE INDEX idx_p028_cv_created            ON pack028_sector_pathway.gl_sector_convergence(created_at DESC);
CREATE INDEX idx_p028_cv_risk_factors       ON pack028_sector_pathway.gl_sector_convergence USING GIN(risk_factors);
CREATE INDEX idx_p028_cv_gap_projection     ON pack028_sector_pathway.gl_sector_convergence USING GIN(annual_gap_projection);
CREATE INDEX idx_p028_cv_metadata           ON pack028_sector_pathway.gl_sector_convergence USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_sector_convergence_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_convergence
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_convergence ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_cv_tenant_isolation
    ON pack028_sector_pathway.gl_sector_convergence
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_cv_service_bypass
    ON pack028_sector_pathway.gl_sector_convergence
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_convergence TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_convergence IS
    'Sector intensity convergence analysis with gap calculations, time-to-convergence, acceleration requirements, risk assessments, and benchmark comparisons for SBTi SDA pathway alignment.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.convergence_id IS 'Unique convergence analysis record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.intensity_gap IS 'Absolute gap between current intensity and pathway target (positive = above target).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.intensity_gap_pct IS 'Percentage gap between current intensity and pathway target.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.gap_severity IS 'Gap severity classification: CRITICAL, HIGH, MEDIUM, LOW, ON_TRACK, AHEAD.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.time_to_convergence_years IS 'Estimated years until intensity converges with pathway target at current reduction rate.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.acceleration_needed_pct IS 'Additional annual reduction rate needed beyond current trajectory to converge.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.risk_level IS 'Overall convergence risk: CRITICAL, HIGH, MEDIUM, LOW, MINIMAL.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.percentile_rank IS 'Company percentile rank within sector (100 = best performer).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_convergence.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
