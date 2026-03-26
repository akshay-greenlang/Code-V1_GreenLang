-- =============================================================================
-- V381: PACK-046 Intensity Metrics Pack - SBTi SDA Targets & Tracking
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    006 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for SBTi Sectoral Decarbonization Approach (SDA) intensity
-- targets, annual pathway milestones, and actual-vs-target progress tracking.
-- Supports 1.5C, Well Below 2C, and Net Zero pathways with sector-specific
-- convergence targets. Pathway milestones define the annual intensity budget.
-- Progress tracking compares actual intensity against the pathway with
-- ON_TRACK, AT_RISK, OFF_TRACK, and AHEAD status indicators.
--
-- Tables (3):
--   1. ghg_intensity.gl_im_targets
--   2. ghg_intensity.gl_im_target_pathways
--   3. ghg_intensity.gl_im_target_progress
--
-- Also includes: indexes, RLS, comments.
-- Previous: V380__pack046_benchmarking.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_targets
-- =============================================================================
-- Intensity reduction targets per organisation. Supports SBTi SDA (sector-
-- specific convergence), custom intensity targets (e.g., board-mandated),
-- and internal operational targets. Each target defines a base year
-- intensity, target year intensity, and the required annual reduction rate.
-- SBTi-specific fields track submission and validation status.

CREATE TABLE ghg_intensity.gl_im_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    target_name                 VARCHAR(255)    NOT NULL,
    target_type                 VARCHAR(30)     NOT NULL,
    sector                      VARCHAR(100)    NOT NULL,
    pathway                     VARCHAR(30)     NOT NULL DEFAULT 'ONE_POINT_FIVE_C',
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL,
    base_year                   INTEGER         NOT NULL,
    base_year_intensity         NUMERIC(20,10)  NOT NULL,
    target_year                 INTEGER         NOT NULL,
    target_intensity            NUMERIC(20,10)  NOT NULL,
    annual_reduction_rate_pct   NUMERIC(10,6)   NOT NULL,
    sector_2050_target          NUMERIC(20,10),
    sbti_submission_date        DATE,
    sbti_validation_status      VARCHAR(30),
    sbti_target_id              VARCHAR(100),
    near_term_year              INTEGER,
    near_term_intensity         NUMERIC(20,10),
    long_term_year              INTEGER,
    long_term_intensity         NUMERIC(20,10),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    target_metadata             JSONB           NOT NULL DEFAULT '{}',
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p046_tgt_type CHECK (
        target_type IN ('SBTI_SDA', 'CUSTOM_INTENSITY', 'INTERNAL', 'REGULATORY')
    ),
    CONSTRAINT chk_p046_tgt_pathway CHECK (
        pathway IN (
            'WELL_BELOW_2C', 'ONE_POINT_FIVE_C', 'NET_ZERO', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p046_tgt_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_tgt_base_year CHECK (
        base_year >= 2000 AND base_year <= 2100
    ),
    CONSTRAINT chk_p046_tgt_target_year CHECK (
        target_year > base_year AND target_year <= 2100
    ),
    CONSTRAINT chk_p046_tgt_base_intensity CHECK (
        base_year_intensity >= 0
    ),
    CONSTRAINT chk_p046_tgt_target_intensity CHECK (
        target_intensity >= 0
    ),
    CONSTRAINT chk_p046_tgt_reduction_rate CHECK (
        annual_reduction_rate_pct > 0 AND annual_reduction_rate_pct <= 100
    ),
    CONSTRAINT chk_p046_tgt_sbti_status CHECK (
        sbti_validation_status IS NULL OR sbti_validation_status IN (
            'DRAFT', 'SUBMITTED', 'UNDER_REVIEW', 'APPROVED', 'REJECTED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p046_tgt_near_term CHECK (
        near_term_year IS NULL OR (near_term_year > base_year AND near_term_year <= target_year)
    ),
    CONSTRAINT chk_p046_tgt_long_term CHECK (
        long_term_year IS NULL OR long_term_year >= target_year
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_tgt_tenant           ON ghg_intensity.gl_im_targets(tenant_id);
CREATE INDEX idx_p046_tgt_org              ON ghg_intensity.gl_im_targets(org_id);
CREATE INDEX idx_p046_tgt_config           ON ghg_intensity.gl_im_targets(config_id);
CREATE INDEX idx_p046_tgt_type             ON ghg_intensity.gl_im_targets(target_type);
CREATE INDEX idx_p046_tgt_sector           ON ghg_intensity.gl_im_targets(sector);
CREATE INDEX idx_p046_tgt_pathway          ON ghg_intensity.gl_im_targets(pathway);
CREATE INDEX idx_p046_tgt_denom            ON ghg_intensity.gl_im_targets(denominator_code);
CREATE INDEX idx_p046_tgt_scope            ON ghg_intensity.gl_im_targets(scope_inclusion);
CREATE INDEX idx_p046_tgt_active           ON ghg_intensity.gl_im_targets(is_active) WHERE is_active = true;
CREATE INDEX idx_p046_tgt_sbti_status      ON ghg_intensity.gl_im_targets(sbti_validation_status);
CREATE INDEX idx_p046_tgt_created          ON ghg_intensity.gl_im_targets(created_at DESC);
CREATE INDEX idx_p046_tgt_metadata         ON ghg_intensity.gl_im_targets USING GIN(target_metadata);

-- Composite: org + active targets
CREATE INDEX idx_p046_tgt_org_active       ON ghg_intensity.gl_im_targets(org_id, is_active) WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_tgt_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_targets
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_target_pathways
-- =============================================================================
-- Annual pathway milestones for each target. Defines the expected intensity
-- value and cumulative reduction percentage for each year between base year
-- and target year. Used for linear interpolation and gap analysis. Pathway
-- values are pre-computed from the annual reduction rate (SDA convergence
-- formula for SBTi, linear for custom).

CREATE TABLE ghg_intensity.gl_im_target_pathways (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_targets(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    pathway_intensity           NUMERIC(20,10)  NOT NULL,
    cumulative_reduction_pct    NUMERIC(10,6)   NOT NULL,
    annual_budget_tco2e         NUMERIC(20,6),
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_tp_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p046_tp_intensity CHECK (
        pathway_intensity >= 0
    ),
    CONSTRAINT chk_p046_tp_reduction CHECK (
        cumulative_reduction_pct >= 0 AND cumulative_reduction_pct <= 100
    ),
    CONSTRAINT uq_p046_tp_target_year UNIQUE (target_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_tp_target            ON ghg_intensity.gl_im_target_pathways(target_id);
CREATE INDEX idx_p046_tp_year              ON ghg_intensity.gl_im_target_pathways(year);
CREATE INDEX idx_p046_tp_created           ON ghg_intensity.gl_im_target_pathways(created_at DESC);

-- Composite: target + year for ordered retrieval
CREATE INDEX idx_p046_tp_target_year_ord   ON ghg_intensity.gl_im_target_pathways(target_id, year ASC);

-- =============================================================================
-- Table 3: ghg_intensity.gl_im_target_progress
-- =============================================================================
-- Actual vs target tracking per reporting period. Compares the actual
-- calculated intensity against the pathway target for each year. Status
-- indicators (ON_TRACK, AT_RISK, OFF_TRACK, AHEAD) are determined by
-- configurable variance thresholds. Tracks percentage of total target
-- reduction achieved to date.

CREATE TABLE ghg_intensity.gl_im_target_progress (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    target_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_targets(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id),
    year                        INTEGER         NOT NULL,
    actual_intensity            NUMERIC(20,10),
    target_intensity            NUMERIC(20,10)  NOT NULL,
    variance                    NUMERIC(20,10),
    variance_pct                NUMERIC(10,6),
    status                      VARCHAR(30)     NOT NULL,
    pct_of_target_achieved      NUMERIC(10,6),
    cumulative_reduction_pct    NUMERIC(10,6),
    remaining_annual_rate_pct   NUMERIC(10,6),
    carbon_budget_remaining     NUMERIC(20,6),
    assessment_notes            TEXT,
    provenance_hash             VARCHAR(64),
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_prg_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p046_prg_status CHECK (
        status IN ('ON_TRACK', 'AT_RISK', 'OFF_TRACK', 'AHEAD', 'NOT_ASSESSED')
    ),
    CONSTRAINT chk_p046_prg_pct CHECK (
        pct_of_target_achieved IS NULL OR (pct_of_target_achieved >= 0 AND pct_of_target_achieved <= 200)
    ),
    CONSTRAINT uq_p046_prg_target_period UNIQUE (target_id, period_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_prg_tenant           ON ghg_intensity.gl_im_target_progress(tenant_id);
CREATE INDEX idx_p046_prg_target           ON ghg_intensity.gl_im_target_progress(target_id);
CREATE INDEX idx_p046_prg_period           ON ghg_intensity.gl_im_target_progress(period_id);
CREATE INDEX idx_p046_prg_year             ON ghg_intensity.gl_im_target_progress(year);
CREATE INDEX idx_p046_prg_status           ON ghg_intensity.gl_im_target_progress(status);
CREATE INDEX idx_p046_prg_assessed         ON ghg_intensity.gl_im_target_progress(assessed_at DESC);
CREATE INDEX idx_p046_prg_created          ON ghg_intensity.gl_im_target_progress(created_at DESC);

-- Composite: target + year for ordered tracking
CREATE INDEX idx_p046_prg_target_year      ON ghg_intensity.gl_im_target_progress(target_id, year ASC);

-- Composite: status filter for dashboard alerts
CREATE INDEX idx_p046_prg_at_risk          ON ghg_intensity.gl_im_target_progress(status)
    WHERE status IN ('AT_RISK', 'OFF_TRACK');

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_target_pathways ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_target_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_tgt_tenant_isolation
    ON ghg_intensity.gl_im_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_tgt_service_bypass
    ON ghg_intensity.gl_im_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Pathways inherit access via target_id FK
CREATE POLICY p046_tp_service_bypass
    ON ghg_intensity.gl_im_target_pathways
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_prg_tenant_isolation
    ON ghg_intensity.gl_im_target_progress
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_prg_service_bypass
    ON ghg_intensity.gl_im_target_progress
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_target_pathways TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_target_progress TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_targets IS
    'SBTi SDA and custom intensity reduction targets with sector-specific convergence pathways and validation status.';
COMMENT ON TABLE ghg_intensity.gl_im_target_pathways IS
    'Annual pathway milestones defining expected intensity and cumulative reduction for each year between base and target year.';
COMMENT ON TABLE ghg_intensity.gl_im_target_progress IS
    'Actual vs target tracking with ON_TRACK/AT_RISK/OFF_TRACK/AHEAD status indicators and remaining carbon budget.';

COMMENT ON COLUMN ghg_intensity.gl_im_targets.target_type IS 'SBTI_SDA (sector convergence), CUSTOM_INTENSITY (board-mandated), INTERNAL (operational), REGULATORY (compliance).';
COMMENT ON COLUMN ghg_intensity.gl_im_targets.pathway IS 'Temperature pathway: WELL_BELOW_2C, ONE_POINT_FIVE_C, NET_ZERO, or CUSTOM.';
COMMENT ON COLUMN ghg_intensity.gl_im_targets.annual_reduction_rate_pct IS 'Required annual compound reduction rate to achieve target. E.g., 4.2% for well-below-2C.';
COMMENT ON COLUMN ghg_intensity.gl_im_targets.sector_2050_target IS 'Sector-specific 2050 convergence intensity value from SBTi SDA model.';
COMMENT ON COLUMN ghg_intensity.gl_im_targets.sbti_validation_status IS 'SBTi target validation lifecycle: DRAFT, SUBMITTED, UNDER_REVIEW, APPROVED, REJECTED, EXPIRED.';
COMMENT ON COLUMN ghg_intensity.gl_im_target_pathways.pathway_intensity IS 'Expected intensity at this year based on the reduction pathway (SDA convergence or linear interpolation).';
COMMENT ON COLUMN ghg_intensity.gl_im_target_pathways.cumulative_reduction_pct IS 'Cumulative reduction from base year to this year as percentage (0% at base year, 100% at target year).';
COMMENT ON COLUMN ghg_intensity.gl_im_target_progress.status IS 'ON_TRACK (within threshold), AT_RISK (slightly off), OFF_TRACK (significantly off), AHEAD (better than pathway).';
COMMENT ON COLUMN ghg_intensity.gl_im_target_progress.remaining_annual_rate_pct IS 'Required annual reduction rate from current year to meet target, accounting for over/under-performance.';
COMMENT ON COLUMN ghg_intensity.gl_im_target_progress.carbon_budget_remaining IS 'Remaining cumulative emission budget (tCO2e) from current year to target year on the pathway.';
