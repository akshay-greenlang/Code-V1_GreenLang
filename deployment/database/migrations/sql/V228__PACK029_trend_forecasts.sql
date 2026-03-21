-- =============================================================================
-- V208: PACK-029 Interim Targets Pack - Trend Forecasts
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    013 of 015
-- Date:         March 2026
--
-- Trend forecasting for emission projections with multiple methods
-- (linear, exponential, ARIMA), confidence intervals, and historical
-- forecast accuracy tracking for predictive target monitoring.
--
-- Tables (1):
--   1. pack029_interim_targets.gl_trend_forecasts
--
-- Previous: V207__PACK029_assurance_evidence.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack029_interim_targets.gl_trend_forecasts
-- =============================================================================
-- Emission trend forecasts with multiple projection methods, confidence
-- intervals, model parameters, forecast accuracy tracking, and target
-- attainment probability estimates.

CREATE TABLE pack029_interim_targets.gl_trend_forecasts (
    forecast_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    target_id                   UUID            REFERENCES pack029_interim_targets.gl_interim_targets(target_id) ON DELETE SET NULL,
    -- Forecast context
    forecast_date               DATE            NOT NULL DEFAULT CURRENT_DATE,
    forecast_run_id             UUID,
    forecast_horizon            VARCHAR(20)     DEFAULT 'MEDIUM_TERM',
    -- Target period
    forecast_year               INTEGER         NOT NULL,
    forecast_quarter            VARCHAR(2),
    scope                       VARCHAR(20)     NOT NULL,
    -- Forecast values
    forecasted_emissions_tco2e  DECIMAL(18,4)   NOT NULL,
    confidence_interval_lower   DECIMAL(18,4),
    confidence_interval_upper   DECIMAL(18,4),
    confidence_level_pct        DECIMAL(5,2)    DEFAULT 95.00,
    -- Forecast vs target
    target_emissions_tco2e      DECIMAL(18,4),
    forecast_vs_target_tco2e    DECIMAL(18,4),
    forecast_vs_target_pct      DECIMAL(8,4),
    target_attainment_probability DECIMAL(5,2),
    -- Method
    method                      VARCHAR(30)     NOT NULL DEFAULT 'LINEAR',
    model_name                  VARCHAR(100),
    model_version               VARCHAR(20),
    -- Model parameters
    model_parameters            JSONB           DEFAULT '{}',
    training_data_start_year    INTEGER,
    training_data_end_year      INTEGER,
    data_points_used            INTEGER,
    -- Model quality
    r_squared                   DECIMAL(8,6),
    rmse                        DECIMAL(18,4),
    mape_pct                    DECIMAL(8,4),
    aic                         DECIMAL(12,4),
    bic                         DECIMAL(12,4),
    -- Decomposition (if applicable)
    trend_component_tco2e       DECIMAL(18,4),
    seasonal_component_tco2e    DECIMAL(18,4),
    residual_component_tco2e    DECIMAL(18,4),
    -- Historical accuracy
    prior_forecast_tco2e        DECIMAL(18,4),
    actual_emissions_tco2e      DECIMAL(18,4),
    forecast_error_tco2e        DECIMAL(18,4),
    forecast_error_pct          DECIMAL(8,4),
    forecast_bias               VARCHAR(20),
    -- Scenario
    scenario                    VARCHAR(30)     DEFAULT 'BASE_CASE',
    scenario_assumptions        JSONB           DEFAULT '{}',
    -- Status
    is_active                   BOOLEAN         DEFAULT TRUE,
    is_latest                   BOOLEAN         DEFAULT TRUE,
    superseded_by               UUID,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p029_tf_scope CHECK (
        scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3')
    ),
    CONSTRAINT chk_p029_tf_forecast_year CHECK (
        forecast_year >= 2000 AND forecast_year <= 2100
    ),
    CONSTRAINT chk_p029_tf_quarter CHECK (
        forecast_quarter IS NULL OR forecast_quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    ),
    CONSTRAINT chk_p029_tf_forecasted_emissions CHECK (
        forecasted_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p029_tf_confidence_level CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.99
    ),
    CONSTRAINT chk_p029_tf_confidence_interval CHECK (
        confidence_interval_lower IS NULL OR confidence_interval_upper IS NULL
        OR confidence_interval_lower <= confidence_interval_upper
    ),
    CONSTRAINT chk_p029_tf_target_probability CHECK (
        target_attainment_probability IS NULL OR (target_attainment_probability >= 0 AND target_attainment_probability <= 100)
    ),
    CONSTRAINT chk_p029_tf_method CHECK (
        method IN ('LINEAR', 'EXPONENTIAL', 'ARIMA', 'SARIMA', 'PROPHET',
                   'HOLT_WINTERS', 'POLYNOMIAL', 'MOVING_AVERAGE',
                   'BAYESIAN', 'ENSEMBLE', 'CUSTOM')
    ),
    CONSTRAINT chk_p029_tf_forecast_horizon CHECK (
        forecast_horizon IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM')
    ),
    CONSTRAINT chk_p029_tf_r_squared CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p029_tf_forecast_bias CHECK (
        forecast_bias IS NULL OR forecast_bias IN ('OPTIMISTIC', 'NEUTRAL', 'PESSIMISTIC', 'UNKNOWN')
    ),
    CONSTRAINT chk_p029_tf_scenario CHECK (
        scenario IN ('BASE_CASE', 'OPTIMISTIC', 'PESSIMISTIC', 'ACCELERATED',
                     'DELAYED', 'HIGH_GROWTH', 'LOW_GROWTH', 'CUSTOM')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_tf_tenant             ON pack029_interim_targets.gl_trend_forecasts(tenant_id);
CREATE INDEX idx_p029_tf_org                ON pack029_interim_targets.gl_trend_forecasts(organization_id);
CREATE INDEX idx_p029_tf_target             ON pack029_interim_targets.gl_trend_forecasts(target_id);
CREATE INDEX idx_p029_tf_org_forecast_year  ON pack029_interim_targets.gl_trend_forecasts(organization_id, forecast_year);
CREATE INDEX idx_p029_tf_org_scope_year     ON pack029_interim_targets.gl_trend_forecasts(organization_id, scope, forecast_year);
CREATE INDEX idx_p029_tf_forecast_date      ON pack029_interim_targets.gl_trend_forecasts(forecast_date DESC);
CREATE INDEX idx_p029_tf_org_forecast_date  ON pack029_interim_targets.gl_trend_forecasts(organization_id, forecast_date DESC);
CREATE INDEX idx_p029_tf_method             ON pack029_interim_targets.gl_trend_forecasts(method);
CREATE INDEX idx_p029_tf_scenario           ON pack029_interim_targets.gl_trend_forecasts(scenario);
CREATE INDEX idx_p029_tf_latest             ON pack029_interim_targets.gl_trend_forecasts(organization_id, forecast_year, scope) WHERE is_latest = TRUE;
CREATE INDEX idx_p029_tf_low_probability    ON pack029_interim_targets.gl_trend_forecasts(organization_id, target_attainment_probability) WHERE target_attainment_probability IS NOT NULL AND target_attainment_probability < 50;
CREATE INDEX idx_p029_tf_run_id             ON pack029_interim_targets.gl_trend_forecasts(forecast_run_id) WHERE forecast_run_id IS NOT NULL;
CREATE INDEX idx_p029_tf_active             ON pack029_interim_targets.gl_trend_forecasts(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_tf_created            ON pack029_interim_targets.gl_trend_forecasts(created_at DESC);
CREATE INDEX idx_p029_tf_model_params       ON pack029_interim_targets.gl_trend_forecasts USING GIN(model_parameters);
CREATE INDEX idx_p029_tf_scenario_assump    ON pack029_interim_targets.gl_trend_forecasts USING GIN(scenario_assumptions);
CREATE INDEX idx_p029_tf_metadata           ON pack029_interim_targets.gl_trend_forecasts USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p029_trend_forecasts_updated
    BEFORE UPDATE ON pack029_interim_targets.gl_trend_forecasts
    FOR EACH ROW EXECUTE FUNCTION pack029_interim_targets.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack029_interim_targets.gl_trend_forecasts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p029_tf_tenant_isolation
    ON pack029_interim_targets.gl_trend_forecasts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p029_tf_service_bypass
    ON pack029_interim_targets.gl_trend_forecasts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack029_interim_targets.gl_trend_forecasts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack029_interim_targets.gl_trend_forecasts IS
    'Emission trend forecasts with multiple projection methods (linear/exponential/ARIMA), confidence intervals, model quality metrics, forecast accuracy tracking, and target attainment probability for predictive monitoring.';

COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.forecast_id IS 'Unique trend forecast identifier.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.organization_id IS 'Reference to the organization this forecast pertains to.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.forecast_date IS 'Date when the forecast was generated.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.forecast_year IS 'Year being forecasted.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.forecasted_emissions_tco2e IS 'Forecasted emissions in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.confidence_interval_lower IS 'Lower bound of confidence interval in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.confidence_interval_upper IS 'Upper bound of confidence interval in tonnes CO2 equivalent.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.method IS 'Forecasting method: LINEAR, EXPONENTIAL, ARIMA, SARIMA, PROPHET, etc.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.target_attainment_probability IS 'Probability (0-100) of meeting the target based on current trends.';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.r_squared IS 'R-squared goodness of fit metric for the forecast model (0-1).';
COMMENT ON COLUMN pack029_interim_targets.gl_trend_forecasts.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
