-- =============================================================================
-- V372: PACK-045 Base Year Management Pack - Time Series & Consistency
-- =============================================================================
-- Pack:         PACK-045 (Base Year Management Pack)
-- Migration:    007 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates time series and consistency tables that track annual emissions
-- relative to the base year and identify methodological or boundary
-- inconsistencies across the reporting time series. The time series table
-- stores each year's total alongside base year flags and normalization values.
-- The consistency findings table records issues detected during time series
-- validation (e.g., scope gaps, methodology changes, boundary differences).
--
-- Tables (2):
--   1. ghg_base_year.gl_by_time_series
--   2. ghg_base_year.gl_by_consistency_findings
--
-- Also includes: indexes, RLS, comments.
-- Previous: V371__pack045_adjustments.sql
-- =============================================================================

SET search_path TO ghg_base_year, public;

-- =============================================================================
-- Table 1: ghg_base_year.gl_by_time_series
-- =============================================================================
-- Annual emission totals forming the reporting time series. Each row represents
-- one year for one organisation at one scope level. Flags whether the year is
-- the base year and whether it has been recalculated. Normalized values enable
-- fair comparison when intensity metrics or weather normalization are applied.

CREATE TABLE ghg_base_year.gl_by_time_series (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    year                        INTEGER         NOT NULL,
    scope                       VARCHAR(10)     NOT NULL,
    category                    VARCHAR(60),
    total_tco2e                 NUMERIC(14,3)   NOT NULL,
    is_base_year                BOOLEAN         NOT NULL DEFAULT false,
    is_recalculated             BOOLEAN         NOT NULL DEFAULT false,
    recalculation_package_id    UUID            REFERENCES ghg_base_year.gl_by_adjustment_packages(id) ON DELETE SET NULL,
    original_tco2e              NUMERIC(14,3),
    normalized_tco2e            NUMERIC(14,3),
    normalization_method        VARCHAR(60),
    normalization_denominator   NUMERIC(18,6),
    normalization_unit          VARCHAR(50),
    intensity_value             NUMERIC(14,6),
    intensity_unit              VARCHAR(50),
    yoy_change_tco2e            NUMERIC(14,3),
    yoy_change_pct              NUMERIC(8,4),
    base_year_change_tco2e      NUMERIC(14,3),
    base_year_change_pct        NUMERIC(8,4),
    gwp_version                 VARCHAR(10)     DEFAULT 'AR5',
    methodology_tier            VARCHAR(20),
    data_completeness_pct       NUMERIC(5,2),
    data_quality_score          NUMERIC(5,2),
    consistency_status          VARCHAR(30)     NOT NULL DEFAULT 'CONSISTENT',
    consistency_notes           TEXT,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_ts_year CHECK (
        year >= 1990 AND year <= 2100
    ),
    CONSTRAINT chk_p045_ts_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2_LOCATION', 'SCOPE_2_MARKET', 'SCOPE_3', 'TOTAL')
    ),
    CONSTRAINT chk_p045_ts_total CHECK (
        total_tco2e >= 0
    ),
    CONSTRAINT chk_p045_ts_original CHECK (
        original_tco2e IS NULL OR original_tco2e >= 0
    ),
    CONSTRAINT chk_p045_ts_normalized CHECK (
        normalized_tco2e IS NULL OR normalized_tco2e >= 0
    ),
    CONSTRAINT chk_p045_ts_gwp CHECK (
        gwp_version IN ('SAR', 'TAR', 'AR4', 'AR5', 'AR6')
    ),
    CONSTRAINT chk_p045_ts_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p045_ts_quality CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p045_ts_consistency CHECK (
        consistency_status IN ('CONSISTENT', 'MINOR_INCONSISTENCY', 'MAJOR_INCONSISTENCY', 'NOT_ASSESSED')
    ),
    CONSTRAINT uq_p045_ts_org_year_scope_cat UNIQUE (org_id, year, scope, category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_ts_tenant          ON ghg_base_year.gl_by_time_series(tenant_id);
CREATE INDEX idx_p045_ts_org             ON ghg_base_year.gl_by_time_series(org_id);
CREATE INDEX idx_p045_ts_year            ON ghg_base_year.gl_by_time_series(year);
CREATE INDEX idx_p045_ts_scope           ON ghg_base_year.gl_by_time_series(scope);
CREATE INDEX idx_p045_ts_category        ON ghg_base_year.gl_by_time_series(category);
CREATE INDEX idx_p045_ts_base_year       ON ghg_base_year.gl_by_time_series(is_base_year) WHERE is_base_year = true;
CREATE INDEX idx_p045_ts_recalculated    ON ghg_base_year.gl_by_time_series(is_recalculated) WHERE is_recalculated = true;
CREATE INDEX idx_p045_ts_consistency     ON ghg_base_year.gl_by_time_series(consistency_status);
CREATE INDEX idx_p045_ts_created         ON ghg_base_year.gl_by_time_series(created_at DESC);
CREATE INDEX idx_p045_ts_metadata        ON ghg_base_year.gl_by_time_series USING GIN(metadata);

-- Composite: org + year for time series query
CREATE INDEX idx_p045_ts_org_year        ON ghg_base_year.gl_by_time_series(org_id, year);

-- Composite: org + scope + year for scope-specific trends
CREATE INDEX idx_p045_ts_org_scope_year  ON ghg_base_year.gl_by_time_series(org_id, scope, year);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_ts_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_time_series
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_base_year.gl_by_consistency_findings
-- =============================================================================
-- Records methodological, boundary, or data inconsistencies detected across
-- the reporting time series. Each finding identifies the years affected,
-- the type of inconsistency, its severity, and a recommended remediation.
-- Used to flag time series breaks that compromise trend comparability.

CREATE TABLE ghg_base_year.gl_by_consistency_findings (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    finding_code                VARCHAR(30)     NOT NULL,
    finding_type                VARCHAR(40)     NOT NULL,
    severity                    VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    years_affected              INTEGER[]       NOT NULL,
    scope_affected              VARCHAR(10),
    category_affected           VARCHAR(60),
    description                 TEXT            NOT NULL,
    impact_description          TEXT,
    impact_tco2e                NUMERIC(14,3),
    recommendation              TEXT,
    remediation_status          VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    remediation_date            DATE,
    remediation_notes           TEXT,
    detected_by                 VARCHAR(255),
    detected_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    evidence_refs               TEXT[],
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p045_cf_type CHECK (
        finding_type IN (
            'SCOPE_GAP', 'BOUNDARY_CHANGE', 'METHODOLOGY_CHANGE',
            'GWP_VERSION_CHANGE', 'EMISSION_FACTOR_CHANGE', 'DATA_QUALITY_DROP',
            'COMPLETENESS_DROP', 'ANOMALOUS_TREND', 'MISSING_YEAR',
            'CONSOLIDATION_CHANGE', 'CATEGORY_RECLASSIFICATION', 'OTHER'
        )
    ),
    CONSTRAINT chk_p045_cf_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p045_cf_scope CHECK (
        scope_affected IS NULL OR scope_affected IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'ALL')
    ),
    CONSTRAINT chk_p045_cf_remediation CHECK (
        remediation_status IN ('OPEN', 'IN_PROGRESS', 'RESOLVED', 'ACCEPTED', 'DEFERRED', 'NOT_APPLICABLE')
    ),
    CONSTRAINT chk_p045_cf_impact CHECK (
        impact_tco2e IS NULL OR impact_tco2e >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p045_cf_tenant          ON ghg_base_year.gl_by_consistency_findings(tenant_id);
CREATE INDEX idx_p045_cf_org             ON ghg_base_year.gl_by_consistency_findings(org_id);
CREATE INDEX idx_p045_cf_code            ON ghg_base_year.gl_by_consistency_findings(finding_code);
CREATE INDEX idx_p045_cf_type            ON ghg_base_year.gl_by_consistency_findings(finding_type);
CREATE INDEX idx_p045_cf_severity        ON ghg_base_year.gl_by_consistency_findings(severity);
CREATE INDEX idx_p045_cf_remediation     ON ghg_base_year.gl_by_consistency_findings(remediation_status);
CREATE INDEX idx_p045_cf_detected_date   ON ghg_base_year.gl_by_consistency_findings(detected_date);
CREATE INDEX idx_p045_cf_created         ON ghg_base_year.gl_by_consistency_findings(created_at DESC);
CREATE INDEX idx_p045_cf_years           ON ghg_base_year.gl_by_consistency_findings USING GIN(years_affected);
CREATE INDEX idx_p045_cf_metadata        ON ghg_base_year.gl_by_consistency_findings USING GIN(metadata);

-- Composite: org + open findings for dashboard
CREATE INDEX idx_p045_cf_org_open        ON ghg_base_year.gl_by_consistency_findings(org_id, severity)
    WHERE remediation_status IN ('OPEN', 'IN_PROGRESS');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p045_cf_updated
    BEFORE UPDATE ON ghg_base_year.gl_by_consistency_findings
    FOR EACH ROW EXECUTE FUNCTION ghg_base_year.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_base_year.gl_by_time_series ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_base_year.gl_by_consistency_findings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p045_ts_tenant_isolation
    ON ghg_base_year.gl_by_time_series
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_ts_service_bypass
    ON ghg_base_year.gl_by_time_series
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p045_cf_tenant_isolation
    ON ghg_base_year.gl_by_consistency_findings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p045_cf_service_bypass
    ON ghg_base_year.gl_by_consistency_findings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_time_series TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_base_year.gl_by_consistency_findings TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_base_year.gl_by_time_series IS
    'Annual emission totals per scope forming the reporting time series, with base year flags, normalization, and YoY change calculations.';
COMMENT ON TABLE ghg_base_year.gl_by_consistency_findings IS
    'Methodological, boundary, and data inconsistencies detected across the time series that may compromise trend comparability.';

COMMENT ON COLUMN ghg_base_year.gl_by_time_series.yoy_change_pct IS 'Year-over-year change: ((current - previous) / previous) * 100.';
COMMENT ON COLUMN ghg_base_year.gl_by_time_series.base_year_change_pct IS 'Change relative to base year: ((current - base) / base) * 100.';
COMMENT ON COLUMN ghg_base_year.gl_by_time_series.normalized_tco2e IS 'Weather or occupancy normalised emission value for fair comparison.';
COMMENT ON COLUMN ghg_base_year.gl_by_consistency_findings.years_affected IS 'Array of years where this inconsistency applies.';
COMMENT ON COLUMN ghg_base_year.gl_by_consistency_findings.finding_type IS 'Type of inconsistency: SCOPE_GAP, BOUNDARY_CHANGE, METHODOLOGY_CHANGE, etc.';
