-- =============================================================================
-- V373: PACK-045 Base Year Management Pack - Target Tracking
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates target tracking tables that link emission reduction targets to the
-- base year. Targets define scope, base year emissions, target year, and
-- required reduction percentage. Progress tracking records actual vs expected
-- emissions each year to monitor whether the organisation is on track to meet
-- its targets. Supports absolute, intensity, and SBTi-aligned targets.
--
-- Tables (2):
--   1. ghg_base_year.gl_by_targets
--   2. ghg_base_year.gl_by_target_progress
--
-- Also includes: indexes, RLS, comments.
-- Previous: V372__pack045_time_series.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_targets
-- =============================================================================
-- Emission reduction targets anchored to the base year. Each target specifies
-- the scope coverage, base year emissions, target year, and the required
-- reduction percentage. SBTi alignment fields capture the ambition level
-- (1.5C, well-below 2C) and sector pathway where applicable.

CREATE TABLE ghg_base_year.gl_by_targets (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    target_name                 VARCHAR(255)    NOT NULL,
    target_type                 VARCHAR(30)     NOT NULL DEFAULT 'ABSOLUTE',
    scope                       VARCHAR(60)     NOT NULL,
    base_year                   INTEGER         NOT NULL,
    base_year_tco2e             NUMERIC(14,3)   NOT NULL,
    base_year_intensity         NUMERIC(14,6),
    intensity_unit              VARCHAR(50),
    intensity_denominator       VARCHAR(100),
    target_year                 INTEGER         NOT NULL,
    target_reduction_pct        NUMERIC(6,2)    NOT NULL,
    target_tco2e                NUMERIC(14,3),
    target_intensity            NUMERIC(14,6),
    annual_linear_rate_pct      NUMERIC(6,3),
    sbti_ambition               VARCHAR(30),
    sbti_status                 VARCHAR(30),
    sbti_validation_date        DATE,
    sbti_target_id              VARCHAR(100),
    sector_pathway              VARCHAR(60),
    net_zero_target_year        INTEGER,
    includes_offsets             BOOLEAN         NOT NULL DEFAULT false,
    offset_limit_pct            NUMERIC(5,2),
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_date               DATE,
    published_date              DATE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p045_tgt_type CHECK (
        target_type IN ('ABSOLUTE', 'INTENSITY', 'ABSOLUTE_CONTRACTION', 'SECTOR_DECARBONISATION')
    ),
    CONSTRAINT chk_p045_tgt_scope CHECK (
        scope IN (
            'SCOPE_1', 'SCOPE_2', 'SCOPE_1_2', 'SCOPE_3', 'SCOPE_1_2_3',
            'SCOPE_1_2_PARTIAL_3', 'NEAR_TERM', 'LONG_TERM'
        )
    ),
    CONSTRAINT chk_p045_tgt_base_year CHECK (
        base_year >= 1990 AND base_year <= 2100
    ),
    CONSTRAINT chk_p045_tgt_base_tco2e CHECK (
        base_year_tco2e > 0
    ),
    CONSTRAINT chk_p045_tgt_target_year CHECK (
        target_year > base_year AND target_year <= 2100
    ),
    CONSTRAINT chk_p045_tgt_reduction CHECK (
        target_reduction_pct > 0 AND target_reduction_pct <= 100
    ),
    CONSTRAINT chk_p045_tgt_target_tco2e CHECK (
        target_tco2e IS NULL OR target_tco2e >= 0
    ),
    CONSTRAINT chk_p045_tgt_annual_rate CHECK (
        annual_linear_rate_pct IS NULL OR (annual_linear_rate_pct > 0 AND annual_linear_rate_pct <= 50)
    ),
    CONSTRAINT chk_p045_tgt_sbti_ambition CHECK (
        sbti_ambition IS NULL OR sbti_ambition IN ('1_5C', 'WELL_BELOW_2C', '2C', 'FLOOR')
    ),
    CONSTRAINT chk_p045_tgt_sbti_status CHECK (
        sbti_status IS NULL OR sbti_status IN (
            'COMMITTED', 'TARGETS_SET', 'VALIDATED', 'TARGETS_PUBLISHED', 'REMOVED'
        )
    ),
    CONSTRAINT chk_p045_tgt_status CHECK (
        status IN ('DRAFT', 'UNDER_REVIEW', 'APPROVED', 'PUBLISHED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p045_tgt_offset_limit CHECK (
        offset_limit_pct IS NULL OR (offset_limit_pct >= 0 AND offset_limit_pct <= 100)
    ),
    CONSTRAINT chk_p045_tgt_net_zero_year CHECK (
        net_zero_target_year IS NULL OR (net_zero_target_year >= 2030 AND net_zero_target_year <= 2100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_tgt_tenant         ON ghg_base_year.gl_by_targets(tenant_id);
CREATE INDEX idx_p045_tgt_org            ON ghg_base_year.gl_by_targets(org_id);
CREATE INDEX idx_p045_tgt_type           ON ghg_base_year.gl_by_targets(target_type);
CREATE INDEX idx_p045_tgt_scope          ON ghg_base_year.gl_by_targets(scope);
CREATE INDEX idx_p045_tgt_base_year      ON ghg_base_year.gl_by_targets(base_year);
CREATE INDEX idx_p045_tgt_target_year    ON ghg_base_year.gl_by_targets(target_year);
CREATE INDEX idx_p045_tgt_sbti_ambition  ON ghg_base_year.gl_by_targets(sbti_ambition);
CREATE INDEX idx_p045_tgt_sbti_status    ON ghg_base_year.gl_by_targets(sbti_status);
CREATE INDEX idx_p045_tgt_status         ON ghg_base_year.gl_by_targets(status);
CREATE INDEX idx_p045_tgt_created        ON ghg_base_year.gl_by_targets(created_at DESC);
CREATE INDEX idx_p045_tgt_metadata       ON ghg_base_year.gl_by_targets USING GIN(metadata);

-- Composite: org + active targets
CREATE INDEX idx_p045_tgt_org_active     ON ghg_base_year.gl_by_targets(org_id, scope)
    WHERE status IN ('APPROVED', 'PUBLISHED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_tgt_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_targets
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_target_progress
-- =============================================================================
-- Annual progress tracking against emission reduction targets. Each row
-- records the actual tCO2e for a year, the expected tCO2e based on the
-- linear reduction pathway, and the resulting status (on track, behind,
-- ahead, exceeded).

CREATE TABLE ghg_base_year.gl_by_target_progress (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    target_id                   UUID            NOT NULL REFERENCES ghg_base_year.gl_by_targets(id) ON DELETE CASCADE,
    year                        INTEGER         NOT NULL,
    actual_tco2e                NUMERIC(14,3),
    actual_intensity            NUMERIC(14,6),
    expected_tco2e              NUMERIC(14,3)   NOT NULL,
    expected_intensity          NUMERIC(14,6),
    gap_tco2e                   NUMERIC(14,3)   GENERATED ALWAYS AS (
        CASE WHEN actual_tco2e IS NOT NULL
             THEN actual_tco2e - expected_tco2e
             ELSE NULL
        END
    ) STORED,
    gap_pct                     NUMERIC(8,4),
    cumulative_reduction_pct    NUMERIC(8,4),
    required_reduction_pct      NUMERIC(8,4)    NOT NULL,
    status                      VARCHAR(30)     NOT NULL DEFAULT 'PENDING',
    carbon_budget_remaining     NUMERIC(14,3),
    offsets_applied_tco2e       NUMERIC(14,3),
    net_actual_tco2e            NUMERIC(14,3),
    data_quality                VARCHAR(20),
    is_projected                BOOLEAN         NOT NULL DEFAULT false,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_tp_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p045_tp_actual CHECK (
        actual_tco2e IS NULL OR actual_tco2e >= 0
    ),
    CONSTRAINT chk_p045_tp_expected CHECK (
        expected_tco2e >= 0
    ),
    CONSTRAINT chk_p045_tp_status CHECK (
        status IN (
            'PENDING', 'ON_TRACK', 'AHEAD', 'BEHIND', 'SIGNIFICANTLY_BEHIND',
            'TARGET_MET', 'TARGET_EXCEEDED', 'NOT_REPORTED'
        )
    ),
    CONSTRAINT chk_p045_tp_quality CHECK (
        data_quality IS NULL OR data_quality IN ('HIGH', 'MEDIUM', 'LOW', 'ESTIMATED', 'PROJECTED')
    ),
    CONSTRAINT chk_p045_tp_offsets CHECK (
        offsets_applied_tco2e IS NULL OR offsets_applied_tco2e >= 0
    ),
    CONSTRAINT chk_p045_tp_budget CHECK (
        carbon_budget_remaining IS NULL OR carbon_budget_remaining >= 0
    ),
    CONSTRAINT uq_p045_tp_target_year UNIQUE (target_id, year)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_tp_tenant          ON ghg_base_year.gl_by_target_progress(tenant_id);
CREATE INDEX idx_p045_tp_target          ON ghg_base_year.gl_by_target_progress(target_id);
CREATE INDEX idx_p045_tp_year            ON ghg_base_year.gl_by_target_progress(year);
CREATE INDEX idx_p045_tp_status          ON ghg_base_year.gl_by_target_progress(status);
CREATE INDEX idx_p045_tp_projected       ON ghg_base_year.gl_by_target_progress(is_projected) WHERE is_projected = true;
CREATE INDEX idx_p045_tp_created         ON ghg_base_year.gl_by_target_progress(created_at DESC);
CREATE INDEX idx_p045_tp_metadata        ON ghg_base_year.gl_by_target_progress USING GIN(metadata);

-- Composite: target + year for progress query
CREATE INDEX idx_p045_tp_target_year     ON ghg_base_year.gl_by_target_progress(target_id, year);

-- Composite: behind targets for alerts
CREATE INDEX idx_p045_tp_behind          ON ghg_base_year.gl_by_target_progress(status)
    WHERE status IN ('BEHIND', 'SIGNIFICANTLY_BEHIND');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_tp_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_target_progress
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_target_progress ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_tgt_tenant_isolation
    ON ghg_base_year.gl_by_targets
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_tgt_service_bypass
    ON ghg_base_year.gl_by_targets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_tp_tenant_isolation
    ON ghg_base_year.gl_by_target_progress
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_tp_service_bypass
    ON ghg_base_year.gl_by_target_progress
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_targets TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_target_progress TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_targets IS
    'Emission reduction targets anchored to the base year with SBTi alignment, scope coverage, and reduction pathway parameters.';
COMMENT ON TABLE ghg_base_year.gl_by_target_progress IS
    'Annual progress tracking against targets with actual vs expected emissions, gap analysis, and carbon budget remaining.';

COMMENT ON COLUMN ghg_base_year.gl_by_targets.target_reduction_pct IS 'Required reduction from base year as percentage (e.g., 42.0 means 42% reduction by target year).';
COMMENT ON COLUMN ghg_base_year.gl_by_targets.sbti_ambition IS 'SBTi temperature alignment: 1_5C, WELL_BELOW_2C, 2C, or FLOOR (minimum ambition).';
COMMENT ON COLUMN ghg_base_year.gl_by_targets.annual_linear_rate_pct IS 'Annual linear reduction rate to achieve the target (total_reduction / years).';
COMMENT ON COLUMN ghg_base_year.gl_by_target_progress.gap_tco2e IS 'Auto-calculated: actual_tco2e - expected_tco2e. Positive = behind target, negative = ahead.';
COMMENT ON COLUMN ghg_base_year.gl_by_target_progress.status IS 'Progress status: ON_TRACK, AHEAD, BEHIND, SIGNIFICANTLY_BEHIND, TARGET_MET, TARGET_EXCEEDED.';
COMMENT ON COLUMN ghg_base_year.gl_by_target_progress.carbon_budget_remaining IS 'Remaining cumulative carbon budget from current year to target year.';
