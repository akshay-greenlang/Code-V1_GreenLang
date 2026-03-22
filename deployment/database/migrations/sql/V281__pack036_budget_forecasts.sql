-- =============================================================================
-- V281: PACK-036 Utility Analysis Pack - Budget Forecasts & Variance Analysis
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Tables for utility budget forecasting, monthly consumption and cost
-- projections, budget-vs-actual variance analysis with decomposition
-- (weather, rate, volume impacts), and scenario modelling.
--
-- Tables (4):
--   1. pack036_utility_analysis.gl_budget_forecasts
--   2. pack036_utility_analysis.gl_monthly_forecasts
--   3. pack036_utility_analysis.gl_budget_variances
--   4. pack036_utility_analysis.gl_forecast_scenarios
--
-- Previous: V280__pack036_cost_allocation.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_budget_forecasts
-- =============================================================================
-- Forecast model definitions per facility and commodity. Each forecast
-- captures the method used, forecast horizon, annual total, and model
-- accuracy metrics (R-squared, MAPE).

CREATE TABLE pack036_utility_analysis.gl_budget_forecasts (
    forecast_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    method                  VARCHAR(50)     NOT NULL,
    horizon_months          INTEGER         NOT NULL DEFAULT 12,
    base_period_start       DATE,
    base_period_end         DATE,
    annual_total_eur        NUMERIC(16,2),
    annual_consumption      NUMERIC(16,4),
    consumption_unit        VARCHAR(20)     DEFAULT 'kWh',
    model_r_squared         NUMERIC(6,4),
    model_mape              NUMERIC(8,4),
    model_cv_rmse           NUMERIC(8,4),
    model_parameters        JSONB           DEFAULT '{}',
    status                  VARCHAR(30)     NOT NULL DEFAULT 'DRAFT',
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_bf_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_bf_method CHECK (
        method IN (
            'REGRESSION', 'MOVING_AVERAGE', 'EXPONENTIAL_SMOOTHING',
            'ARIMA', 'SARIMA', 'PROPHET', 'DEGREE_DAY', 'BASELINE_ADJUSTMENT',
            'PERCENT_INCREASE', 'MANUAL', 'ENSEMBLE'
        )
    ),
    CONSTRAINT chk_p036_bf_horizon CHECK (
        horizon_months >= 1 AND horizon_months <= 120
    ),
    CONSTRAINT chk_p036_bf_r_squared CHECK (
        model_r_squared IS NULL OR (model_r_squared >= 0 AND model_r_squared <= 1)
    ),
    CONSTRAINT chk_p036_bf_mape CHECK (
        model_mape IS NULL OR model_mape >= 0
    ),
    CONSTRAINT chk_p036_bf_cv_rmse CHECK (
        model_cv_rmse IS NULL OR model_cv_rmse >= 0
    ),
    CONSTRAINT chk_p036_bf_status CHECK (
        status IN ('DRAFT', 'ACTIVE', 'APPROVED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_bf_base_period CHECK (
        base_period_start IS NULL OR base_period_end IS NULL OR base_period_end >= base_period_start
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_bf_tenant         ON pack036_utility_analysis.gl_budget_forecasts(tenant_id);
CREATE INDEX idx_p036_bf_facility       ON pack036_utility_analysis.gl_budget_forecasts(facility_id);
CREATE INDEX idx_p036_bf_commodity      ON pack036_utility_analysis.gl_budget_forecasts(commodity);
CREATE INDEX idx_p036_bf_method         ON pack036_utility_analysis.gl_budget_forecasts(method);
CREATE INDEX idx_p036_bf_status         ON pack036_utility_analysis.gl_budget_forecasts(status);
CREATE INDEX idx_p036_bf_created        ON pack036_utility_analysis.gl_budget_forecasts(created_at DESC);
CREATE INDEX idx_p036_bf_metadata       ON pack036_utility_analysis.gl_budget_forecasts USING GIN(metadata);

-- Composite: facility + commodity + status for active forecast lookup
CREATE INDEX idx_p036_bf_fac_comm_act   ON pack036_utility_analysis.gl_budget_forecasts(facility_id, commodity)
    WHERE status IN ('ACTIVE', 'APPROVED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_bf_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_budget_forecasts
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_monthly_forecasts
-- =============================================================================
-- Monthly granularity forecast values within a forecast model. Includes
-- consumption, cost, and demand projections with confidence intervals.

CREATE TABLE pack036_utility_analysis.gl_monthly_forecasts (
    monthly_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id             UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_budget_forecasts(forecast_id) ON DELETE CASCADE,
    month                   DATE            NOT NULL,
    consumption_forecast    NUMERIC(16,4),
    cost_forecast_eur       NUMERIC(14,2)   NOT NULL,
    demand_forecast_kw      NUMERIC(12,4),
    confidence_lower_eur    NUMERIC(14,2),
    confidence_upper_eur    NUMERIC(14,2),
    confidence_level        NUMERIC(5,2)    DEFAULT 0.95,
    degree_days_hdd         NUMERIC(10,2),
    degree_days_cdd         NUMERIC(10,2),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_mf_cost CHECK (
        cost_forecast_eur >= 0
    ),
    CONSTRAINT chk_p036_mf_consumption CHECK (
        consumption_forecast IS NULL OR consumption_forecast >= 0
    ),
    CONSTRAINT chk_p036_mf_demand CHECK (
        demand_forecast_kw IS NULL OR demand_forecast_kw >= 0
    ),
    CONSTRAINT chk_p036_mf_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 1)
    ),
    CONSTRAINT chk_p036_mf_interval CHECK (
        confidence_lower_eur IS NULL OR confidence_upper_eur IS NULL
        OR confidence_upper_eur >= confidence_lower_eur
    ),
    CONSTRAINT uq_p036_mf_forecast_month UNIQUE (forecast_id, month)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_mf_forecast       ON pack036_utility_analysis.gl_monthly_forecasts(forecast_id);
CREATE INDEX idx_p036_mf_month          ON pack036_utility_analysis.gl_monthly_forecasts(month);
CREATE INDEX idx_p036_mf_cost           ON pack036_utility_analysis.gl_monthly_forecasts(cost_forecast_eur DESC);

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_budget_variances
-- =============================================================================
-- Monthly budget-vs-actual variance analysis with decomposition into
-- weather impact, rate impact, and volume impact components.

CREATE TABLE pack036_utility_analysis.gl_budget_variances (
    variance_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    month                   DATE            NOT NULL,
    budgeted_consumption    NUMERIC(16,4),
    actual_consumption      NUMERIC(16,4),
    consumption_variance    NUMERIC(16,4),
    budgeted_eur            NUMERIC(14,2)   NOT NULL,
    actual_eur              NUMERIC(14,2)   NOT NULL,
    variance_eur            NUMERIC(14,2)   NOT NULL,
    variance_pct            NUMERIC(8,4),
    weather_impact_eur      NUMERIC(14,2),
    rate_impact_eur         NUMERIC(14,2),
    volume_impact_eur       NUMERIC(14,2),
    production_impact_eur   NUMERIC(14,2),
    other_impact_eur        NUMERIC(14,2),
    explanation             TEXT,
    status                  VARCHAR(30)     DEFAULT 'CALCULATED',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_bv_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_bv_budgeted CHECK (
        budgeted_eur >= 0
    ),
    CONSTRAINT chk_p036_bv_actual CHECK (
        actual_eur >= 0
    ),
    CONSTRAINT chk_p036_bv_status CHECK (
        status IN ('CALCULATED', 'REVIEWED', 'EXPLAINED', 'APPROVED')
    ),
    CONSTRAINT uq_p036_bv_fac_comm_month UNIQUE (facility_id, commodity, month)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_bv_tenant         ON pack036_utility_analysis.gl_budget_variances(tenant_id);
CREATE INDEX idx_p036_bv_facility       ON pack036_utility_analysis.gl_budget_variances(facility_id);
CREATE INDEX idx_p036_bv_commodity      ON pack036_utility_analysis.gl_budget_variances(commodity);
CREATE INDEX idx_p036_bv_month          ON pack036_utility_analysis.gl_budget_variances(month DESC);
CREATE INDEX idx_p036_bv_variance       ON pack036_utility_analysis.gl_budget_variances(variance_pct);
CREATE INDEX idx_p036_bv_status         ON pack036_utility_analysis.gl_budget_variances(status);
CREATE INDEX idx_p036_bv_created        ON pack036_utility_analysis.gl_budget_variances(created_at DESC);

-- Composite: facility + month for time-series variance lookup
CREATE INDEX idx_p036_bv_fac_month      ON pack036_utility_analysis.gl_budget_variances(facility_id, month DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_bv_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_budget_variances
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack036_utility_analysis.gl_forecast_scenarios
-- =============================================================================
-- What-if scenario definitions within a forecast model. Each scenario
-- modifies assumptions (e.g., rate increase, occupancy change, equipment
-- upgrade) and projects the resulting annual cost.

CREATE TABLE pack036_utility_analysis.gl_forecast_scenarios (
    scenario_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id             UUID            NOT NULL REFERENCES pack036_utility_analysis.gl_budget_forecasts(forecast_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    scenario_type           VARCHAR(30)     NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    description             TEXT,
    annual_consumption      NUMERIC(16,4),
    annual_cost_eur         NUMERIC(16,2)   NOT NULL,
    annual_demand_kw        NUMERIC(12,4),
    delta_vs_base_eur       NUMERIC(14,2),
    delta_vs_base_pct       NUMERIC(8,4),
    assumptions             JSONB           NOT NULL DEFAULT '{}',
    probability             NUMERIC(5,4),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_fs_scenario_type CHECK (
        scenario_type IN (
            'BASE_CASE', 'OPTIMISTIC', 'PESSIMISTIC', 'RATE_INCREASE',
            'RATE_DECREASE', 'EXPANSION', 'CONTRACTION', 'EFFICIENCY',
            'ELECTRIFICATION', 'FUEL_SWITCH', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p036_fs_cost CHECK (
        annual_cost_eur >= 0
    ),
    CONSTRAINT chk_p036_fs_probability CHECK (
        probability IS NULL OR (probability >= 0 AND probability <= 1)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_fs_forecast       ON pack036_utility_analysis.gl_forecast_scenarios(forecast_id);
CREATE INDEX idx_p036_fs_tenant         ON pack036_utility_analysis.gl_forecast_scenarios(tenant_id);
CREATE INDEX idx_p036_fs_type           ON pack036_utility_analysis.gl_forecast_scenarios(scenario_type);
CREATE INDEX idx_p036_fs_cost           ON pack036_utility_analysis.gl_forecast_scenarios(annual_cost_eur);
CREATE INDEX idx_p036_fs_created        ON pack036_utility_analysis.gl_forecast_scenarios(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_fs_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_forecast_scenarios
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_budget_forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_monthly_forecasts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_budget_variances ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_forecast_scenarios ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_bf_tenant_isolation
    ON pack036_utility_analysis.gl_budget_forecasts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_bf_service_bypass
    ON pack036_utility_analysis.gl_budget_forecasts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_mf_tenant_isolation
    ON pack036_utility_analysis.gl_monthly_forecasts
    USING (forecast_id IN (
        SELECT forecast_id FROM pack036_utility_analysis.gl_budget_forecasts
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p036_mf_service_bypass
    ON pack036_utility_analysis.gl_monthly_forecasts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_bv_tenant_isolation
    ON pack036_utility_analysis.gl_budget_variances
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_bv_service_bypass
    ON pack036_utility_analysis.gl_budget_variances
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_fs_tenant_isolation
    ON pack036_utility_analysis.gl_forecast_scenarios
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_fs_service_bypass
    ON pack036_utility_analysis.gl_forecast_scenarios
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_budget_forecasts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_monthly_forecasts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_budget_variances TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_forecast_scenarios TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_budget_forecasts IS
    'Forecast model definitions per facility and commodity with method, horizon, annual total, and model accuracy metrics.';

COMMENT ON TABLE pack036_utility_analysis.gl_monthly_forecasts IS
    'Monthly granularity forecast values with consumption, cost, and demand projections plus confidence intervals.';

COMMENT ON TABLE pack036_utility_analysis.gl_budget_variances IS
    'Monthly budget-vs-actual variance analysis with weather, rate, and volume impact decomposition.';

COMMENT ON TABLE pack036_utility_analysis.gl_forecast_scenarios IS
    'What-if scenario definitions modifying assumptions and projecting resulting annual cost impacts.';

COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.forecast_id IS
    'Unique identifier for the budget forecast model.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.method IS
    'Forecasting method: REGRESSION, MOVING_AVERAGE, ARIMA, SARIMA, PROPHET, DEGREE_DAY, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.model_r_squared IS
    'Coefficient of determination (R-squared) for the forecast model (0 to 1).';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.model_mape IS
    'Mean Absolute Percentage Error of the forecast model.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.model_cv_rmse IS
    'Coefficient of Variation of Root Mean Squared Error.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_forecasts.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_monthly_forecasts.confidence_level IS
    'Confidence level for the prediction interval (e.g., 0.95 = 95%).';
COMMENT ON COLUMN pack036_utility_analysis.gl_monthly_forecasts.degree_days_hdd IS
    'Heating degree days used in this month forecast calculation.';
COMMENT ON COLUMN pack036_utility_analysis.gl_monthly_forecasts.degree_days_cdd IS
    'Cooling degree days used in this month forecast calculation.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_variances.weather_impact_eur IS
    'Variance component attributable to weather deviation from budget assumptions.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_variances.rate_impact_eur IS
    'Variance component attributable to rate/tariff changes.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_variances.volume_impact_eur IS
    'Variance component attributable to consumption volume changes.';
COMMENT ON COLUMN pack036_utility_analysis.gl_budget_variances.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_forecast_scenarios.scenario_type IS
    'Scenario type: BASE_CASE, OPTIMISTIC, PESSIMISTIC, RATE_INCREASE, EFFICIENCY, ELECTRIFICATION, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_forecast_scenarios.assumptions IS
    'JSON object containing scenario assumptions (e.g., rate_change_pct, occupancy_factor).';
COMMENT ON COLUMN pack036_utility_analysis.gl_forecast_scenarios.probability IS
    'Probability weight assigned to this scenario (0 to 1).';
