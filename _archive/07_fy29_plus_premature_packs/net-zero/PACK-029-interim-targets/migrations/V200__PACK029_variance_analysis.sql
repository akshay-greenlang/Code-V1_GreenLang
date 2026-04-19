-- =============================================================================
-- V200: PACK-029 Interim Targets Pack - Variance Analysis
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    005 of 015
-- Date:         March 2026
--
-- Variance analysis between target and actual emissions with decomposition
-- methods (LMDI/Kaya), activity/intensity/structural effect breakdown,
-- and root cause classification for off-track identification.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_variance_analysis
--
-- Previous: V199__PACK029_actual_performance.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_variance_analysis
-- =============================================================================
-- Variance analysis records with target-vs-actual decomposition using
-- LMDI or Kaya methods, activity/intensity/structural effect quantification,
-- root cause classification, and severity scoring.

CREATE TABLE pack029_interim_targets.gl_variance_analysis (
    variance_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Time dimension
    year                        INTEGER         NOT NULL,
    quarter                     VARCHAR(2),
    analysis_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    -- Scope
    scope                       VARCHAR(20)     NOT NULL,
    category                    VARCHAR(60),
    -- Target vs Actual
    target_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    actual_emissions_tco2e      DECIMAL(18,4)   NOT NULL,
    variance_absolute_tco2e     DECIMAL(18,4)   NOT NULL,
    variance_pct                DECIMAL(8,4)    NOT NULL,
    -- Variance direction
    variance_direction          VARCHAR(20)     NOT NULL DEFAULT 'OVER_TARGET',
    on_track                    BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Decomposition
    decomposition_method        VARCHAR(20)     NOT NULL DEFAULT 'LMDI',
    activity_effect_tco2e       DECIMAL(18,4),
    intensity_effect_tco2e      DECIMAL(18,4),
    structural_effect_tco2e     DECIMAL(18,4),
    fuel_mix_effect_tco2e       DECIMAL(18,4),
    weather_effect_tco2e        DECIMAL(18,4),
    -- Activity decomposition detail
    activity_change_pct         DECIMAL(8,4),
    activity_metric             VARCHAR(80),
    activity_metric_unit        VARCHAR(50),
    -- Intensity decomposition detail
    intensity_change_pct        DECIMAL(8,4),
    intensity_metric            VARCHAR(80),
    intensity_metric_unit       VARCHAR(50),
    -- Root cause analysis
    root_cause_classification   VARCHAR(20)     NOT NULL DEFAULT 'UNKNOWN',
    root_cause_category         VARCHAR(50),
    root_cause_description      TEXT,
    root_causes                 JSONB           DEFAULT '[]',
    -- Severity and impact
    severity_score              DECIMAL(5,2),
    severity_level              VARCHAR(20)     DEFAULT 'MODERATE',
    cumulative_impact_tco2e     DECIMAL(18,4),
    budget_impact_pct           DECIMAL(8,4),
    -- Trend analysis
    consecutive_quarters_off    INTEGER         DEFAULT 0,
    trend_direction             VARCHAR(20),
    trend_acceleration          DECIMAL(8,4),
    -- Corrective action needed
    corrective_action_required  BOOLEAN         DEFAULT FALSE,
    corrective_action_deadline  DATE,
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    reviewed                    BOOLEAN         DEFAULT FALSE,
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_va_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_va_quarter CHECK (
        quarter IS NULL OR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_va_year CHECK (
        year >= 2000 AND year <= 2100
    ),
    CONSTRAINT chk_p029_va_target_emissions CHECK (
        target_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_va_actual_emissions CHECK (
        actual_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_va_variance_check CHECK (
        ABS(variance_absolute_tco2e - (actual_emissions_tco2e - target_emissions_tco2e)) < 0.01
    ),
    CONSTRAINT chk_p029_va_variance_direction CHECK (
        variance_direction IN ('OVER_TARGET', 'UNDER_TARGET', 'ON_TARGET')
    ),
    CONSTRAINT chk_p029_va_decomposition_method CHECK (
        decomposition_method IN ('LMDI', 'KAYA', 'SDA', 'ADDITIVE', 'MULTIPLICATIVE', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_va_root_cause CHECK (
        root_cause_classification IN ('INTERNAL', 'EXTERNAL', 'METHODOLOGICAL', 'DATA_QUALITY', 'UNKNOWN')
    ),
    CONSTRAINT chk_p029_va_root_cause_category CHECK (
        root_cause_category IS NULL OR root_cause_category IN (
            'PRODUCTION_INCREASE', 'PRODUCTION_DECREASE', 'EFFICIENCY_GAIN', 'EFFICIENCY_LOSS',
            'FUEL_SWITCHING', 'GRID_DECARBONIZATION', 'WEATHER', 'ACQUISITION', 'DIVESTITURE',
            'METHODOLOGY_CHANGE', 'EF_UPDATE', 'SCOPE_CHANGE', 'BOUNDARY_CHANGE',
            'MARKET_CONDITIONS', 'REGULATORY', 'SUPPLY_CHAIN', 'OTHER'
        )
    ),
    CONSTRAINT chk_p029_va_severity_level CHECK (
        severity_level IN ('CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'NEGLIGIBLE')
    ),
    CONSTRAINT chk_p029_va_severity_score CHECK (
        severity_score IS NULL OR (severity_score >= 0 AND severity_score <= 100)
    ),
    CONSTRAINT chk_p029_va_trend_direction CHECK (
        trend_direction IS NULL OR trend_direction IN ('IMPROVING', 'STABLE', 'DETERIORATING', 'VOLATILE')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_va_tenant             ON pack029_interim_targets.gl_variance_analysis(tenant_id);
CREATE INDEX idx_p029_va_org                ON pack029_interim_targets.gl_variance_analysis(organization_id);
CREATE INDEX idx_p029_va_target             ON pack029_interim_targets.gl_variance_analysis(target_id);
CREATE INDEX idx_p029_va_org_year           ON pack029_interim_targets.gl_variance_analysis(organization_id, year);
CREATE INDEX idx_p029_va_org_year_scope     ON pack029_interim_targets.gl_variance_analysis(organization_id, year, scope);
CREATE INDEX idx_p029_va_org_year_qtr       ON pack029_interim_targets.gl_variance_analysis(organization_id, year, quarter);
CREATE INDEX idx_p029_va_variance_pct       ON pack029_interim_targets.gl_variance_analysis(variance_pct DESC);
CREATE INDEX idx_p029_va_off_track          ON pack029_interim_targets.gl_variance_analysis(organization_id, year) WHERE on_track = FALSE;
CREATE INDEX idx_p029_va_severity           ON pack029_interim_targets.gl_variance_analysis(severity_level, organization_id);
CREATE INDEX idx_p029_va_critical           ON pack029_interim_targets.gl_variance_analysis(organization_id) WHERE severity_level = 'CRITICAL';
CREATE INDEX idx_p029_va_root_cause         ON pack029_interim_targets.gl_variance_analysis(root_cause_classification);
CREATE INDEX idx_p029_va_root_cause_cat     ON pack029_interim_targets.gl_variance_analysis(root_cause_category);
CREATE INDEX idx_p029_va_corrective_needed  ON pack029_interim_targets.gl_variance_analysis(organization_id, corrective_action_deadline) WHERE corrective_action_required = TRUE;
CREATE INDEX idx_p029_va_decomp_method      ON pack029_interim_targets.gl_variance_analysis(decomposition_method);
CREATE INDEX idx_p029_va_trend              ON pack029_interim_targets.gl_variance_analysis(trend_direction);
CREATE INDEX idx_p029_va_unreviewed         ON pack029_interim_targets.gl_variance_analysis(organization_id) WHERE reviewed = FALSE;
CREATE INDEX idx_p029_va_active             ON pack029_interim_targets.gl_variance_analysis(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_va_analysis_date      ON pack029_interim_targets.gl_variance_analysis(analysis_date DESC);
CREATE INDEX idx_p029_va_created            ON pack029_interim_targets.gl_variance_analysis(created_at DESC);
CREATE INDEX idx_p029_va_root_causes        ON pack029_interim_targets.gl_variance_analysis USING GIN(root_causes);
CREATE INDEX idx_p029_va_metadata           ON pack029_interim_targets.gl_variance_analysis USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_variance_analysis_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_variance_analysis
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_variance_analysis ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_va_tenant_isolation
    ON pack029_interim_targets.gl_variance_analysis
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_va_service_bypass
    ON pack029_interim_targets.gl_variance_analysis
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_variance_analysis TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_variance_analysis IS
    'Variance analysis between target and actual emissions with LMDI/Kaya decomposition, activity/intensity/structural effect quantification, root cause classification, and severity scoring for off-track identification.';

COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.variance_id IS 'Unique variance analysis record identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.organization_id IS 'Reference to the organization being analyzed.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.target_emissions_tco2e IS 'Target emissions for the period in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.actual_emissions_tco2e IS 'Actual reported emissions for the period in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.variance_absolute_tco2e IS 'Absolute variance (actual - target) in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.variance_pct IS 'Variance as percentage of target emissions.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.decomposition_method IS 'Decomposition method: LMDI, KAYA, SDA, ADDITIVE, MULTIPLICATIVE, CUSTOM.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.activity_effect_tco2e IS 'Activity effect contribution to variance in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.intensity_effect_tco2e IS 'Intensity effect contribution to variance in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.structural_effect_tco2e IS 'Structural effect contribution to variance in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.root_cause_classification IS 'Root cause type: INTERNAL, EXTERNAL, METHODOLOGICAL, DATA_QUALITY, UNKNOWN.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.severity_level IS 'Variance severity: CRITICAL, HIGH, MODERATE, LOW, NEGLIGIBLE.';
COMMENT ON COLUMN pack029_interim_targets.gl_variance_analysis.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
