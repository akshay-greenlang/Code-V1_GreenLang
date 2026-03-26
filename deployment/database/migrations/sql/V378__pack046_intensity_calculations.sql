-- =============================================================================
-- V378: PACK-046 Intensity Metrics Pack - Intensity Calculations
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates the core intensity calculation results table and pre-computed time
-- series table. The calculations table stores individual intensity metric
-- results (emissions / denominator) with full provenance, data quality
-- scoring, year-on-year change tracking, and scope coverage metadata.
-- The time series table stores pre-computed series for efficient trending
-- with compound annual reduction rate (CARR) and Mann-Kendall trend analysis.
--
-- Tables (2):
--   1. ghg_intensity.gl_im_calculations
--   2. ghg_intensity.gl_im_time_series
--
-- Also includes: indexes, RLS, comments.
-- Previous: V377__pack046_denominator_registry.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_calculations
-- =============================================================================
-- Individual intensity metric calculation results. Each row represents
-- emissions / denominator for a specific organisation, period, denominator,
-- scope inclusion, and optionally entity. The intensity_value may be NULL
-- if the denominator is zero (division by zero protection). Year-on-year
-- change and combined data quality are tracked for trend analysis and
-- assurance reporting.

CREATE TABLE ghg_intensity.gl_im_calculations (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_reporting_periods(id) ON DELETE CASCADE,
    denominator_id              UUID            NOT NULL REFERENCES ghg_intensity.gl_im_denominator_definitions(id),
    entity_id                   UUID,
    entity_name                 VARCHAR(255),
    scope_inclusion             VARCHAR(50)     NOT NULL,
    emissions_tco2e             NUMERIC(20,6)   NOT NULL,
    denominator_value           NUMERIC(20,6)   NOT NULL,
    intensity_value             NUMERIC(20,10),
    intensity_unit              VARCHAR(100)    NOT NULL,
    yoy_change_pct              NUMERIC(10,6),
    data_quality_numerator      INTEGER,
    data_quality_denominator    INTEGER,
    data_quality_combined       INTEGER,
    scope_coverage_pct          NUMERIC(10,6)   DEFAULT 100.0,
    calculation_metadata        JSONB           NOT NULL DEFAULT '{}',
    provenance_hash             VARCHAR(64)     NOT NULL,
    calculated_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_calc_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_calc_emissions CHECK (
        emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p046_calc_denominator CHECK (
        denominator_value >= 0
    ),
    CONSTRAINT chk_p046_calc_intensity CHECK (
        intensity_value IS NULL OR intensity_value >= 0
    ),
    CONSTRAINT chk_p046_calc_dq_num CHECK (
        data_quality_numerator IS NULL OR (data_quality_numerator BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_calc_dq_den CHECK (
        data_quality_denominator IS NULL OR (data_quality_denominator BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_calc_dq_comb CHECK (
        data_quality_combined IS NULL OR (data_quality_combined BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_calc_coverage CHECK (
        scope_coverage_pct IS NULL OR (scope_coverage_pct >= 0 AND scope_coverage_pct <= 100)
    ),
    CONSTRAINT chk_p046_calc_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_calc_tenant          ON ghg_intensity.gl_im_calculations(tenant_id);
CREATE INDEX idx_p046_calc_org             ON ghg_intensity.gl_im_calculations(org_id);
CREATE INDEX idx_p046_calc_config          ON ghg_intensity.gl_im_calculations(config_id);
CREATE INDEX idx_p046_calc_period          ON ghg_intensity.gl_im_calculations(period_id);
CREATE INDEX idx_p046_calc_denominator     ON ghg_intensity.gl_im_calculations(denominator_id);
CREATE INDEX idx_p046_calc_entity          ON ghg_intensity.gl_im_calculations(entity_id);
CREATE INDEX idx_p046_calc_scope           ON ghg_intensity.gl_im_calculations(scope_inclusion);
CREATE INDEX idx_p046_calc_intensity       ON ghg_intensity.gl_im_calculations(intensity_value);
CREATE INDEX idx_p046_calc_calculated      ON ghg_intensity.gl_im_calculations(calculated_at DESC);
CREATE INDEX idx_p046_calc_created         ON ghg_intensity.gl_im_calculations(created_at DESC);
CREATE INDEX idx_p046_calc_provenance      ON ghg_intensity.gl_im_calculations(provenance_hash);
CREATE INDEX idx_p046_calc_metadata        ON ghg_intensity.gl_im_calculations USING GIN(calculation_metadata);

-- Composite: primary query pattern (org + config + period)
CREATE INDEX idx_p046_calc_org_period      ON ghg_intensity.gl_im_calculations(org_id, config_id, period_id);

-- Composite: scope + denominator for filtered analysis
CREATE INDEX idx_p046_calc_scope_denom     ON ghg_intensity.gl_im_calculations(scope_inclusion, denominator_id);

-- Composite: tenant + org for multi-tenant queries
CREATE INDEX idx_p046_calc_tenant_org      ON ghg_intensity.gl_im_calculations(tenant_id, org_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p046_calc_updated
    BEFORE UPDATE ON ghg_intensity.gl_im_calculations
    FOR EACH ROW EXECUTE FUNCTION ghg_intensity.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_time_series
-- =============================================================================
-- Pre-computed time series for efficient trending and dashboards. Each row
-- holds the complete series data as JSONB for a specific organisation,
-- config, denominator, scope, and optionally entity. Includes compound
-- annual reduction rate (CARR) and Mann-Kendall trend significance test.
-- Recalculated on each new period publication.

CREATE TABLE ghg_intensity.gl_im_time_series (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    denominator_code            VARCHAR(50)     NOT NULL,
    scope_inclusion             VARCHAR(50)     NOT NULL,
    entity_id                   UUID,
    series_data                 JSONB           NOT NULL,
    data_points                 INTEGER         NOT NULL DEFAULT 0,
    carr_pct                    NUMERIC(10,6),
    trend_direction             VARCHAR(20),
    trend_significance_p        NUMERIC(10,6),
    first_period_label          VARCHAR(50),
    last_period_label           VARCHAR(50),
    last_updated                TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    provenance_hash             VARCHAR(64)     NOT NULL,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_ts_scope CHECK (
        scope_inclusion IN (
            'SCOPE_1_ONLY', 'SCOPE_2_LOCATION_ONLY', 'SCOPE_2_MARKET_ONLY',
            'SCOPE_1_2_LOCATION', 'SCOPE_1_2_MARKET', 'SCOPE_1_2_3',
            'ALL_SCOPES'
        )
    ),
    CONSTRAINT chk_p046_ts_trend CHECK (
        trend_direction IS NULL OR trend_direction IN (
            'DECREASING', 'INCREASING', 'STABLE', 'INSUFFICIENT_DATA'
        )
    ),
    CONSTRAINT chk_p046_ts_significance CHECK (
        trend_significance_p IS NULL OR (trend_significance_p >= 0 AND trend_significance_p <= 1)
    ),
    CONSTRAINT chk_p046_ts_data_points CHECK (
        data_points >= 0
    ),
    CONSTRAINT uq_p046_ts_series UNIQUE (org_id, config_id, denominator_code, scope_inclusion, entity_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_ts_tenant            ON ghg_intensity.gl_im_time_series(tenant_id);
CREATE INDEX idx_p046_ts_org               ON ghg_intensity.gl_im_time_series(org_id);
CREATE INDEX idx_p046_ts_config            ON ghg_intensity.gl_im_time_series(config_id);
CREATE INDEX idx_p046_ts_denom_code        ON ghg_intensity.gl_im_time_series(denominator_code);
CREATE INDEX idx_p046_ts_scope             ON ghg_intensity.gl_im_time_series(scope_inclusion);
CREATE INDEX idx_p046_ts_entity            ON ghg_intensity.gl_im_time_series(entity_id);
CREATE INDEX idx_p046_ts_trend             ON ghg_intensity.gl_im_time_series(trend_direction);
CREATE INDEX idx_p046_ts_updated           ON ghg_intensity.gl_im_time_series(last_updated DESC);
CREATE INDEX idx_p046_ts_created           ON ghg_intensity.gl_im_time_series(created_at DESC);
CREATE INDEX idx_p046_ts_series_data       ON ghg_intensity.gl_im_time_series USING GIN(series_data);

-- Composite: denominator + scope for filtered analysis
CREATE INDEX idx_p046_ts_denom_scope       ON ghg_intensity.gl_im_time_series(denominator_code, scope_inclusion);

-- Composite: org + config for batch retrieval
CREATE INDEX idx_p046_ts_org_config        ON ghg_intensity.gl_im_time_series(org_id, config_id);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_calculations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_time_series ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_calc_tenant_isolation
    ON ghg_intensity.gl_im_calculations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_calc_service_bypass
    ON ghg_intensity.gl_im_calculations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_ts_tenant_isolation
    ON ghg_intensity.gl_im_time_series
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_ts_service_bypass
    ON ghg_intensity.gl_im_time_series
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_calculations TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_time_series TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_calculations IS
    'Individual intensity metric calculation results (emissions / denominator) with full provenance, data quality scoring, and year-on-year change tracking.';
COMMENT ON TABLE ghg_intensity.gl_im_time_series IS
    'Pre-computed intensity time series for efficient trending with compound annual reduction rate (CARR) and Mann-Kendall trend significance.';

COMMENT ON COLUMN ghg_intensity.gl_im_calculations.intensity_value IS 'Calculated intensity (emissions_tco2e / denominator_value). NULL when denominator is zero (division by zero protection).';
COMMENT ON COLUMN ghg_intensity.gl_im_calculations.intensity_unit IS 'Unit string for the intensity metric, e.g. tCO2e/MEUR, tCO2e/FTE, kgCO2e/m2.';
COMMENT ON COLUMN ghg_intensity.gl_im_calculations.yoy_change_pct IS 'Year-on-year percentage change in intensity. Negative means improvement (decreasing intensity).';
COMMENT ON COLUMN ghg_intensity.gl_im_calculations.scope_coverage_pct IS 'Percentage of emission sources covered within the declared scope inclusion.';
COMMENT ON COLUMN ghg_intensity.gl_im_calculations.provenance_hash IS 'SHA-256 hash of (input emissions + denominator + config) for complete audit trail.';
COMMENT ON COLUMN ghg_intensity.gl_im_time_series.series_data IS 'JSON array of [{period, intensity, emissions, denominator, quality}] for charting.';
COMMENT ON COLUMN ghg_intensity.gl_im_time_series.carr_pct IS 'Compound Annual Reduction Rate as percentage. Negative indicates increasing intensity.';
COMMENT ON COLUMN ghg_intensity.gl_im_time_series.trend_significance_p IS 'Mann-Kendall trend test p-value. Values < 0.05 indicate statistically significant trend.';
