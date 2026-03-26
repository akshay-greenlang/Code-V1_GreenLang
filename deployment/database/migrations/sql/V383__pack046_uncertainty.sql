-- =============================================================================
-- V383: PACK-046 Intensity Metrics Pack - Uncertainty & Data Quality
-- =============================================================================
-- Pack:         PACK-046 (Intensity Metrics Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for uncertainty quantification and data quality logging.
-- Uncertainty assessments propagate numerator and denominator uncertainties
-- using IPCC Tier 1 (error propagation) or Tier 2 (Monte Carlo) methods to
-- produce confidence intervals for intensity values. Data quality logging
-- tracks individual data element quality scores with source type
-- classification for continuous improvement and assurance support.
--
-- Tables (2):
--   1. ghg_intensity.gl_im_uncertainty_assessments
--   2. ghg_intensity.gl_im_data_quality_log
--
-- Also includes: indexes, RLS, comments.
-- Previous: V382__pack046_scenarios.sql
-- =============================================================================

SET search_path TO ghg_intensity, public;

-- =============================================================================
-- Table 1: ghg_intensity.gl_im_uncertainty_assessments
-- =============================================================================
-- Uncertainty quantification for intensity calculations. Propagates
-- numerator (emissions) and denominator uncertainties to derive combined
-- intensity uncertainty. Supports IPCC Tier 1 (simple error propagation
-- assuming uncorrelated uncertainties) and Tier 2 (Monte Carlo simulation).
-- Produces confidence intervals (lower/upper bounds) at the specified
-- confidence level (default 90%). Improvement recommendations are stored
-- as JSON for data quality enhancement planning.

CREATE TABLE ghg_intensity.gl_im_uncertainty_assessments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    calculation_id              UUID            REFERENCES ghg_intensity.gl_im_calculations(id) ON DELETE CASCADE,
    numerator_uncertainty_pct   NUMERIC(10,6)   NOT NULL,
    denominator_uncertainty_pct NUMERIC(10,6)   NOT NULL,
    combined_uncertainty_pct    NUMERIC(10,6)   NOT NULL,
    propagation_method          VARCHAR(30)     NOT NULL DEFAULT 'IPCC_TIER1',
    confidence_level            NUMERIC(10,6)   NOT NULL DEFAULT 90.0,
    intensity_central           NUMERIC(20,10),
    lower_bound                 NUMERIC(20,10),
    upper_bound                 NUMERIC(20,10),
    data_quality_numerator      INTEGER,
    data_quality_denominator    INTEGER,
    data_quality_combined       INTEGER,
    correlation_factor          NUMERIC(10,6)   DEFAULT 0.0,
    sensitivity_analysis        JSONB           DEFAULT '{}',
    improvement_recommendations JSONB           DEFAULT '[]',
    provenance_hash             VARCHAR(64)     NOT NULL,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    processing_time_ms          INTEGER,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_ua_num_unc CHECK (
        numerator_uncertainty_pct >= 0 AND numerator_uncertainty_pct <= 200
    ),
    CONSTRAINT chk_p046_ua_den_unc CHECK (
        denominator_uncertainty_pct >= 0 AND denominator_uncertainty_pct <= 200
    ),
    CONSTRAINT chk_p046_ua_comb_unc CHECK (
        combined_uncertainty_pct >= 0 AND combined_uncertainty_pct <= 300
    ),
    CONSTRAINT chk_p046_ua_method CHECK (
        propagation_method IN ('IPCC_TIER1', 'IPCC_TIER2_MC', 'ANALYTICAL', 'EXPERT_JUDGEMENT')
    ),
    CONSTRAINT chk_p046_ua_confidence CHECK (
        confidence_level > 0 AND confidence_level < 100
    ),
    CONSTRAINT chk_p046_ua_bounds CHECK (
        (lower_bound IS NULL AND upper_bound IS NULL) OR
        (lower_bound IS NOT NULL AND upper_bound IS NOT NULL AND upper_bound >= lower_bound)
    ),
    CONSTRAINT chk_p046_ua_dq_num CHECK (
        data_quality_numerator IS NULL OR (data_quality_numerator BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_ua_dq_den CHECK (
        data_quality_denominator IS NULL OR (data_quality_denominator BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_ua_dq_comb CHECK (
        data_quality_combined IS NULL OR (data_quality_combined BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_ua_correlation CHECK (
        correlation_factor IS NULL OR (correlation_factor >= -1 AND correlation_factor <= 1)
    ),
    CONSTRAINT chk_p046_ua_processing CHECK (
        processing_time_ms IS NULL OR processing_time_ms >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_ua_tenant            ON ghg_intensity.gl_im_uncertainty_assessments(tenant_id);
CREATE INDEX idx_p046_ua_org               ON ghg_intensity.gl_im_uncertainty_assessments(org_id);
CREATE INDEX idx_p046_ua_calculation       ON ghg_intensity.gl_im_uncertainty_assessments(calculation_id);
CREATE INDEX idx_p046_ua_method            ON ghg_intensity.gl_im_uncertainty_assessments(propagation_method);
CREATE INDEX idx_p046_ua_confidence        ON ghg_intensity.gl_im_uncertainty_assessments(confidence_level);
CREATE INDEX idx_p046_ua_dq_combined       ON ghg_intensity.gl_im_uncertainty_assessments(data_quality_combined);
CREATE INDEX idx_p046_ua_assessed          ON ghg_intensity.gl_im_uncertainty_assessments(assessed_at DESC);
CREATE INDEX idx_p046_ua_created           ON ghg_intensity.gl_im_uncertainty_assessments(created_at DESC);
CREATE INDEX idx_p046_ua_provenance        ON ghg_intensity.gl_im_uncertainty_assessments(provenance_hash);
CREATE INDEX idx_p046_ua_sensitivity       ON ghg_intensity.gl_im_uncertainty_assessments USING GIN(sensitivity_analysis);
CREATE INDEX idx_p046_ua_recommendations   ON ghg_intensity.gl_im_uncertainty_assessments USING GIN(improvement_recommendations);

-- Composite: org + calculation for lookup
CREATE INDEX idx_p046_ua_org_calc          ON ghg_intensity.gl_im_uncertainty_assessments(org_id, calculation_id);

-- =============================================================================
-- Table 2: ghg_intensity.gl_im_data_quality_log
-- =============================================================================
-- Granular data quality logging per data element. Each record scores a
-- specific data element (emissions by scope, denominator by type) on the
-- GHG Protocol 1-5 scale, classifies the source type (measured, calculated,
-- estimated, proxy), and tracks uncertainty percentage. Used for data
-- quality trend analysis, improvement planning, and assurance evidence.

CREATE TABLE ghg_intensity.gl_im_data_quality_log (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    org_id                      UUID            NOT NULL,
    config_id                   UUID            NOT NULL REFERENCES ghg_intensity.gl_im_configurations(id) ON DELETE CASCADE,
    period_id                   UUID            REFERENCES ghg_intensity.gl_im_reporting_periods(id),
    data_element                VARCHAR(100)    NOT NULL,
    quality_score               INTEGER         NOT NULL,
    uncertainty_pct             NUMERIC(10,6),
    source_type                 VARCHAR(50),
    completeness_pct            NUMERIC(10,6),
    timeliness_days             INTEGER,
    consistency_score           INTEGER,
    previous_quality_score      INTEGER,
    improvement_action          TEXT,
    notes                       TEXT,
    assessed_by                 UUID,
    assessed_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    metadata                    JSONB           DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p046_dql_quality CHECK (
        quality_score BETWEEN 1 AND 5
    ),
    CONSTRAINT chk_p046_dql_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)
    ),
    CONSTRAINT chk_p046_dql_source CHECK (
        source_type IS NULL OR source_type IN (
            'MEASURED', 'CALCULATED', 'ESTIMATED', 'PROXY',
            'MODELLED', 'SUPPLIER_PROVIDED', 'INDUSTRY_AVERAGE'
        )
    ),
    CONSTRAINT chk_p046_dql_completeness CHECK (
        completeness_pct IS NULL OR (completeness_pct >= 0 AND completeness_pct <= 100)
    ),
    CONSTRAINT chk_p046_dql_timeliness CHECK (
        timeliness_days IS NULL OR timeliness_days >= 0
    ),
    CONSTRAINT chk_p046_dql_consistency CHECK (
        consistency_score IS NULL OR (consistency_score BETWEEN 1 AND 5)
    ),
    CONSTRAINT chk_p046_dql_previous CHECK (
        previous_quality_score IS NULL OR (previous_quality_score BETWEEN 1 AND 5)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p046_dql_tenant           ON ghg_intensity.gl_im_data_quality_log(tenant_id);
CREATE INDEX idx_p046_dql_org              ON ghg_intensity.gl_im_data_quality_log(org_id);
CREATE INDEX idx_p046_dql_config           ON ghg_intensity.gl_im_data_quality_log(config_id);
CREATE INDEX idx_p046_dql_period           ON ghg_intensity.gl_im_data_quality_log(period_id);
CREATE INDEX idx_p046_dql_element          ON ghg_intensity.gl_im_data_quality_log(data_element);
CREATE INDEX idx_p046_dql_quality          ON ghg_intensity.gl_im_data_quality_log(quality_score);
CREATE INDEX idx_p046_dql_source           ON ghg_intensity.gl_im_data_quality_log(source_type);
CREATE INDEX idx_p046_dql_assessed         ON ghg_intensity.gl_im_data_quality_log(assessed_at DESC);
CREATE INDEX idx_p046_dql_created          ON ghg_intensity.gl_im_data_quality_log(created_at DESC);

-- Composite: org + period for batch retrieval
CREATE INDEX idx_p046_dql_org_period       ON ghg_intensity.gl_im_data_quality_log(org_id, period_id);

-- Composite: org + element for trend analysis
CREATE INDEX idx_p046_dql_org_element      ON ghg_intensity.gl_im_data_quality_log(org_id, data_element);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE ghg_intensity.gl_im_uncertainty_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_intensity.gl_im_data_quality_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY p046_ua_tenant_isolation
    ON ghg_intensity.gl_im_uncertainty_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_ua_service_bypass
    ON ghg_intensity.gl_im_uncertainty_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p046_dql_tenant_isolation
    ON ghg_intensity.gl_im_data_quality_log
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p046_dql_service_bypass
    ON ghg_intensity.gl_im_data_quality_log
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_uncertainty_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_intensity.gl_im_data_quality_log TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_intensity.gl_im_uncertainty_assessments IS
    'Uncertainty quantification for intensity calculations using IPCC Tier 1/2 error propagation with confidence intervals and improvement recommendations.';
COMMENT ON TABLE ghg_intensity.gl_im_data_quality_log IS
    'Granular data quality scoring per data element with source type classification for trend analysis and assurance support.';

COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.propagation_method IS 'IPCC_TIER1 (quadrature), IPCC_TIER2_MC (Monte Carlo), ANALYTICAL (closed-form), EXPERT_JUDGEMENT.';
COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.confidence_level IS 'Confidence level for the interval (default 90%). Use 95% for IPCC reporting.';
COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.lower_bound IS 'Lower bound of the confidence interval for the intensity value.';
COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.upper_bound IS 'Upper bound of the confidence interval for the intensity value.';
COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.correlation_factor IS 'Correlation between numerator and denominator uncertainties (-1 to +1). Default 0 assumes independence.';
COMMENT ON COLUMN ghg_intensity.gl_im_uncertainty_assessments.improvement_recommendations IS 'JSON array of recommendations to reduce uncertainty: [{"element": "scope1_mobile", "action": "install_telematics", "impact_pct": -15}].';
COMMENT ON COLUMN ghg_intensity.gl_im_data_quality_log.data_element IS 'Identifier for the data element, e.g. emissions_scope1, emissions_scope2_location, denominator_revenue, denominator_fte.';
COMMENT ON COLUMN ghg_intensity.gl_im_data_quality_log.quality_score IS 'GHG Protocol data quality scale: 1=highest (measured/metered), 2=high, 3=medium, 4=low, 5=lowest (proxy/default).';
COMMENT ON COLUMN ghg_intensity.gl_im_data_quality_log.source_type IS 'MEASURED, CALCULATED, ESTIMATED, PROXY, MODELLED, SUPPLIER_PROVIDED, INDUSTRY_AVERAGE.';
COMMENT ON COLUMN ghg_intensity.gl_im_data_quality_log.completeness_pct IS 'Percentage of expected data points that are available (0-100).';
