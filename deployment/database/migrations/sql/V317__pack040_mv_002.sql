-- =============================================================================
-- V317: PACK-040 M&V Pack - Baseline Models, Regression Parameters, Validation
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    002 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for energy baseline modeling including regression baselines,
-- regression parameter storage, model diagnostic statistics, raw baseline
-- data points, and model comparison results. These tables support OLS,
-- change-point (3P/4P/5P), and TOWT regression models with full ASHRAE 14
-- validation statistics.
--
-- Tables (5):
--   1. pack040_mv.mv_baselines
--   2. pack040_mv.mv_regression_params
--   3. pack040_mv.mv_model_diagnostics
--   4. pack040_mv.mv_baseline_data
--   5. pack040_mv.mv_model_comparisons
--
-- Previous: V316__pack040_mv_001.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_baselines
-- =============================================================================
-- Energy baseline models developed for M&V projects. Each baseline represents
-- a regression model of energy consumption as a function of independent
-- variables (temperature, production, occupancy, etc.) during the baseline
-- period. Baselines are the foundation for savings calculations per IPMVP.

CREATE TABLE pack040_mv.mv_baselines (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    boundary_id                 UUID            REFERENCES pack040_mv.mv_measurement_boundaries(id) ON DELETE SET NULL,
    baseline_name               VARCHAR(255)    NOT NULL,
    baseline_version            INTEGER         NOT NULL DEFAULT 1,
    is_current                  BOOLEAN         NOT NULL DEFAULT true,
    model_type                  VARCHAR(30)     NOT NULL DEFAULT 'OLS',
    data_granularity            VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    baseline_period_start       DATE            NOT NULL,
    baseline_period_end         DATE            NOT NULL,
    num_data_points             INTEGER         NOT NULL,
    num_excluded_points         INTEGER         NOT NULL DEFAULT 0,
    dependent_variable          VARCHAR(50)     NOT NULL DEFAULT 'energy_kwh',
    dependent_unit              VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    independent_variables       VARCHAR(50)[]   NOT NULL DEFAULT '{temperature}',
    -- Model coefficients
    intercept                   NUMERIC(18,6),
    slope_1                     NUMERIC(18,6),
    slope_2                     NUMERIC(18,6),
    slope_3                     NUMERIC(18,6),
    slope_4                     NUMERIC(18,6),
    change_point_1              NUMERIC(10,3),
    change_point_2              NUMERIC(10,3),
    balance_point_heating_f     NUMERIC(6,2),
    balance_point_cooling_f     NUMERIC(6,2),
    -- ASHRAE 14 statistics
    r_squared                   NUMERIC(8,6),
    adjusted_r_squared          NUMERIC(8,6),
    cvrmse_pct                  NUMERIC(8,4),
    nmbe_pct                    NUMERIC(8,4),
    rmse                        NUMERIC(18,6),
    standard_error              NUMERIC(18,6),
    f_statistic                 NUMERIC(12,4),
    f_p_value                   NUMERIC(12,10),
    durbin_watson               NUMERIC(6,4),
    mean_dependent              NUMERIC(18,6),
    sum_squared_residuals       NUMERIC(22,6),
    total_sum_squares           NUMERIC(22,6),
    aic                         NUMERIC(12,4),
    bic                         NUMERIC(12,4),
    -- Validation
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    passes_cvrmse               BOOLEAN,
    passes_nmbe                 BOOLEAN,
    passes_r_squared            BOOLEAN,
    passes_all_criteria         BOOLEAN,
    -- Metadata
    model_equation              TEXT,
    model_description           TEXT,
    exclusion_reason            TEXT,
    weather_station_id          VARCHAR(50),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by                  UUID,
    -- Constraints
    CONSTRAINT chk_p040_bl_model_type CHECK (
        model_type IN (
            'OLS', '3P_COOLING', '3P_HEATING', '4P', '5P',
            'TOWT', 'MULTIVARIATE', 'MEAN', 'DEGREE_DAY',
            'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_bl_granularity CHECK (
        data_granularity IN (
            'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'BILLING_PERIOD'
        )
    ),
    CONSTRAINT chk_p040_bl_dates CHECK (
        baseline_period_start < baseline_period_end
    ),
    CONSTRAINT chk_p040_bl_data_points CHECK (
        num_data_points >= 3
    ),
    CONSTRAINT chk_p040_bl_excluded CHECK (
        num_excluded_points >= 0 AND num_excluded_points < num_data_points
    ),
    CONSTRAINT chk_p040_bl_r_squared CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p040_bl_adj_r_squared CHECK (
        adjusted_r_squared IS NULL OR (adjusted_r_squared >= -1 AND adjusted_r_squared <= 1)
    ),
    CONSTRAINT chk_p040_bl_cvrmse CHECK (
        cvrmse_pct IS NULL OR cvrmse_pct >= 0
    ),
    CONSTRAINT chk_p040_bl_nmbe CHECK (
        nmbe_pct IS NULL OR (nmbe_pct >= -100 AND nmbe_pct <= 100)
    ),
    CONSTRAINT chk_p040_bl_durbin_watson CHECK (
        durbin_watson IS NULL OR (durbin_watson >= 0 AND durbin_watson <= 4)
    ),
    CONSTRAINT chk_p040_bl_f_stat CHECK (
        f_statistic IS NULL OR f_statistic >= 0
    ),
    CONSTRAINT chk_p040_bl_f_pvalue CHECK (
        f_p_value IS NULL OR (f_p_value >= 0 AND f_p_value <= 1)
    ),
    CONSTRAINT chk_p040_bl_valid_status CHECK (
        validation_status IN (
            'PENDING', 'PASSED', 'FAILED', 'CONDITIONAL', 'OVERRIDE'
        )
    ),
    CONSTRAINT chk_p040_bl_version CHECK (
        baseline_version >= 1
    ),
    CONSTRAINT uq_p040_bl_project_version UNIQUE (project_id, baseline_name, baseline_version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_bl_tenant            ON pack040_mv.mv_baselines(tenant_id);
CREATE INDEX idx_p040_bl_project           ON pack040_mv.mv_baselines(project_id);
CREATE INDEX idx_p040_bl_ecm               ON pack040_mv.mv_baselines(ecm_id);
CREATE INDEX idx_p040_bl_boundary          ON pack040_mv.mv_baselines(boundary_id);
CREATE INDEX idx_p040_bl_model_type        ON pack040_mv.mv_baselines(model_type);
CREATE INDEX idx_p040_bl_granularity       ON pack040_mv.mv_baselines(data_granularity);
CREATE INDEX idx_p040_bl_valid_status      ON pack040_mv.mv_baselines(validation_status);
CREATE INDEX idx_p040_bl_current           ON pack040_mv.mv_baselines(is_current) WHERE is_current = true;
CREATE INDEX idx_p040_bl_created           ON pack040_mv.mv_baselines(created_at DESC);
CREATE INDEX idx_p040_bl_metadata          ON pack040_mv.mv_baselines USING GIN(metadata);

-- Composite: project + current baseline
CREATE INDEX idx_p040_bl_project_current   ON pack040_mv.mv_baselines(project_id, baseline_version DESC)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_bl_updated
    BEFORE UPDATE ON pack040_mv.mv_baselines
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_regression_params
-- =============================================================================
-- Detailed regression parameters for each baseline model. Stores individual
-- coefficient statistics including standard errors, t-statistics, p-values,
-- confidence intervals, and VIF for multicollinearity detection. Provides
-- the statistical detail needed for ASHRAE 14 compliance reporting.

CREATE TABLE pack040_mv.mv_regression_params (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    param_name                  VARCHAR(100)    NOT NULL,
    param_type                  VARCHAR(30)     NOT NULL DEFAULT 'SLOPE',
    param_index                 INTEGER         NOT NULL DEFAULT 0,
    coefficient                 NUMERIC(18,8)   NOT NULL,
    standard_error              NUMERIC(18,8),
    t_statistic                 NUMERIC(12,6),
    p_value                     NUMERIC(12,10),
    confidence_lower_90         NUMERIC(18,8),
    confidence_upper_90         NUMERIC(18,8),
    confidence_lower_95         NUMERIC(18,8),
    confidence_upper_95         NUMERIC(18,8),
    vif                         NUMERIC(10,4),
    is_significant              BOOLEAN,
    significance_level          NUMERIC(5,4)    DEFAULT 0.05,
    associated_variable         VARCHAR(100),
    unit                        VARCHAR(50),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_rp_type CHECK (
        param_type IN (
            'INTERCEPT', 'SLOPE', 'CHANGE_POINT', 'BALANCE_POINT',
            'TOWT_COEFFICIENT', 'INTERACTION', 'DUMMY', 'OFFSET'
        )
    ),
    CONSTRAINT chk_p040_rp_index CHECK (
        param_index >= 0 AND param_index <= 100
    ),
    CONSTRAINT chk_p040_rp_se CHECK (
        standard_error IS NULL OR standard_error >= 0
    ),
    CONSTRAINT chk_p040_rp_pvalue CHECK (
        p_value IS NULL OR (p_value >= 0 AND p_value <= 1)
    ),
    CONSTRAINT chk_p040_rp_vif CHECK (
        vif IS NULL OR vif >= 1
    ),
    CONSTRAINT chk_p040_rp_sig_level CHECK (
        significance_level > 0 AND significance_level < 1
    ),
    CONSTRAINT chk_p040_rp_confidence CHECK (
        confidence_lower_90 IS NULL OR confidence_upper_90 IS NULL OR
        confidence_lower_90 <= confidence_upper_90
    ),
    CONSTRAINT uq_p040_rp_baseline_param UNIQUE (baseline_id, param_name, param_index)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_rp_tenant            ON pack040_mv.mv_regression_params(tenant_id);
CREATE INDEX idx_p040_rp_baseline          ON pack040_mv.mv_regression_params(baseline_id);
CREATE INDEX idx_p040_rp_type              ON pack040_mv.mv_regression_params(param_type);
CREATE INDEX idx_p040_rp_significant       ON pack040_mv.mv_regression_params(is_significant) WHERE is_significant = true;
CREATE INDEX idx_p040_rp_variable          ON pack040_mv.mv_regression_params(associated_variable);
CREATE INDEX idx_p040_rp_created           ON pack040_mv.mv_regression_params(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_rp_updated
    BEFORE UPDATE ON pack040_mv.mv_regression_params
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_model_diagnostics
-- =============================================================================
-- Comprehensive model diagnostic statistics for baseline validation. Stores
-- residual analysis results, autocorrelation tests, heteroscedasticity
-- checks, normality tests, influential observation detection, and other
-- ASHRAE 14 required diagnostics. Used for model quality assurance.

CREATE TABLE pack040_mv.mv_model_diagnostics (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    diagnostic_type             VARCHAR(50)     NOT NULL,
    diagnostic_name             VARCHAR(100)    NOT NULL,
    test_statistic              NUMERIC(18,8),
    p_value                     NUMERIC(12,10),
    critical_value              NUMERIC(18,8),
    degrees_of_freedom          INTEGER,
    result                      VARCHAR(20)     NOT NULL DEFAULT 'PASS',
    result_description          TEXT,
    threshold_used              NUMERIC(12,6),
    -- Residual diagnostics
    residual_mean               NUMERIC(18,8),
    residual_std                NUMERIC(18,8),
    residual_skewness           NUMERIC(10,6),
    residual_kurtosis           NUMERIC(10,6),
    max_residual                NUMERIC(18,8),
    min_residual                NUMERIC(18,8),
    -- Influential observations
    num_influential_points      INTEGER,
    max_cooks_distance          NUMERIC(10,6),
    max_leverage                NUMERIC(10,6),
    max_dffits                  NUMERIC(10,6),
    influential_point_ids       INTEGER[]       DEFAULT '{}',
    -- Autocorrelation
    lag_1_autocorrelation       NUMERIC(8,6),
    lag_2_autocorrelation       NUMERIC(8,6),
    ljung_box_statistic         NUMERIC(12,6),
    ljung_box_p_value           NUMERIC(12,10),
    -- Heteroscedasticity
    breusch_pagan_statistic     NUMERIC(12,6),
    breusch_pagan_p_value       NUMERIC(12,10),
    white_test_statistic        NUMERIC(12,6),
    white_test_p_value          NUMERIC(12,10),
    -- Additional
    plot_data                   JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_md_diag_type CHECK (
        diagnostic_type IN (
            'RESIDUAL_ANALYSIS', 'AUTOCORRELATION', 'HETEROSCEDASTICITY',
            'NORMALITY', 'MULTICOLLINEARITY', 'INFLUENTIAL_OBSERVATIONS',
            'MODEL_FIT', 'PREDICTION_ACCURACY', 'CROSS_VALIDATION',
            'STABILITY', 'SPECIFICATION'
        )
    ),
    CONSTRAINT chk_p040_md_result CHECK (
        result IN ('PASS', 'FAIL', 'WARNING', 'NOT_APPLICABLE', 'INCONCLUSIVE')
    ),
    CONSTRAINT chk_p040_md_pvalue CHECK (
        p_value IS NULL OR (p_value >= 0 AND p_value <= 1)
    ),
    CONSTRAINT chk_p040_md_dof CHECK (
        degrees_of_freedom IS NULL OR degrees_of_freedom >= 0
    ),
    CONSTRAINT chk_p040_md_inf_points CHECK (
        num_influential_points IS NULL OR num_influential_points >= 0
    ),
    CONSTRAINT chk_p040_md_autocorr CHECK (
        lag_1_autocorrelation IS NULL OR
        (lag_1_autocorrelation >= -1 AND lag_1_autocorrelation <= 1)
    ),
    CONSTRAINT uq_p040_md_baseline_diag UNIQUE (baseline_id, diagnostic_type, diagnostic_name)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_md_tenant            ON pack040_mv.mv_model_diagnostics(tenant_id);
CREATE INDEX idx_p040_md_baseline          ON pack040_mv.mv_model_diagnostics(baseline_id);
CREATE INDEX idx_p040_md_diag_type         ON pack040_mv.mv_model_diagnostics(diagnostic_type);
CREATE INDEX idx_p040_md_result            ON pack040_mv.mv_model_diagnostics(result);
CREATE INDEX idx_p040_md_created           ON pack040_mv.mv_model_diagnostics(created_at DESC);

-- Composite: baseline + failed diagnostics
CREATE INDEX idx_p040_md_baseline_fail     ON pack040_mv.mv_model_diagnostics(baseline_id, diagnostic_type)
    WHERE result = 'FAIL';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_md_updated
    BEFORE UPDATE ON pack040_mv.mv_model_diagnostics
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_baseline_data
-- =============================================================================
-- Raw and processed data points used in baseline model development. Each row
-- represents one data point (interval) with the dependent variable value and
-- all independent variable values. Stores both original and cleaned/excluded
-- data for audit trail purposes.

CREATE TABLE pack040_mv.mv_baseline_data (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    data_point_index            INTEGER         NOT NULL,
    period_start                TIMESTAMPTZ     NOT NULL,
    period_end                  TIMESTAMPTZ     NOT NULL,
    -- Dependent variable
    actual_value                NUMERIC(18,6)   NOT NULL,
    predicted_value             NUMERIC(18,6),
    residual                    NUMERIC(18,6),
    standardized_residual       NUMERIC(10,6),
    -- Independent variables
    temperature_avg_f           NUMERIC(8,3),
    temperature_avg_c           NUMERIC(8,3),
    hdd                         NUMERIC(10,3),
    cdd                         NUMERIC(10,3),
    production_volume           NUMERIC(18,3),
    occupancy_pct               NUMERIC(7,4),
    operating_hours             NUMERIC(8,2),
    daylight_hours              NUMERIC(6,3),
    additional_variables        JSONB           DEFAULT '{}',
    -- Data quality
    is_excluded                 BOOLEAN         NOT NULL DEFAULT false,
    exclusion_reason            VARCHAR(100),
    is_estimated                BOOLEAN         NOT NULL DEFAULT false,
    estimation_method           VARCHAR(50),
    data_quality_flag           VARCHAR(20)     NOT NULL DEFAULT 'GOOD',
    -- Influence diagnostics
    cooks_distance              NUMERIC(10,6),
    leverage                    NUMERIC(10,6),
    dffits                      NUMERIC(10,6),
    is_influential              BOOLEAN         NOT NULL DEFAULT false,
    source_meter_id             UUID,
    source_data_ref             VARCHAR(255),
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_bd_dates CHECK (
        period_start < period_end
    ),
    CONSTRAINT chk_p040_bd_index CHECK (
        data_point_index >= 0
    ),
    CONSTRAINT chk_p040_bd_quality CHECK (
        data_quality_flag IN (
            'GOOD', 'ESTIMATED', 'SUSPECT', 'EXCLUDED', 'MISSING', 'CORRECTED'
        )
    ),
    CONSTRAINT chk_p040_bd_exclusion CHECK (
        exclusion_reason IS NULL OR exclusion_reason IN (
            'OUTLIER', 'MISSING_DATA', 'EQUIPMENT_FAILURE', 'ATYPICAL_EVENT',
            'DATA_ERROR', 'HOLIDAY', 'SHUTDOWN', 'CONSTRUCTION',
            'WEATHER_EXTREME', 'MANUAL_OVERRIDE'
        )
    ),
    CONSTRAINT chk_p040_bd_estimation CHECK (
        estimation_method IS NULL OR estimation_method IN (
            'LINEAR_INTERPOLATION', 'REGRESSION_PREDICT', 'HISTORICAL_AVERAGE',
            'DEGREE_DAY_SCALING', 'MANUAL_ENTRY', 'NEIGHBOR_AVERAGE'
        )
    ),
    CONSTRAINT chk_p040_bd_occupancy CHECK (
        occupancy_pct IS NULL OR (occupancy_pct >= 0 AND occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p040_bd_cooks CHECK (
        cooks_distance IS NULL OR cooks_distance >= 0
    ),
    CONSTRAINT chk_p040_bd_leverage CHECK (
        leverage IS NULL OR (leverage >= 0 AND leverage <= 1)
    ),
    CONSTRAINT uq_p040_bd_baseline_index UNIQUE (baseline_id, data_point_index)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_bd_tenant            ON pack040_mv.mv_baseline_data(tenant_id);
CREATE INDEX idx_p040_bd_baseline          ON pack040_mv.mv_baseline_data(baseline_id);
CREATE INDEX idx_p040_bd_period            ON pack040_mv.mv_baseline_data(period_start, period_end);
CREATE INDEX idx_p040_bd_excluded          ON pack040_mv.mv_baseline_data(is_excluded) WHERE is_excluded = true;
CREATE INDEX idx_p040_bd_influential       ON pack040_mv.mv_baseline_data(is_influential) WHERE is_influential = true;
CREATE INDEX idx_p040_bd_quality           ON pack040_mv.mv_baseline_data(data_quality_flag);
CREATE INDEX idx_p040_bd_source            ON pack040_mv.mv_baseline_data(source_meter_id);
CREATE INDEX idx_p040_bd_created           ON pack040_mv.mv_baseline_data(created_at DESC);

-- Composite: baseline + included data points
CREATE INDEX idx_p040_bd_baseline_incl     ON pack040_mv.mv_baseline_data(baseline_id, data_point_index)
    WHERE is_excluded = false;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_bd_updated
    BEFORE UPDATE ON pack040_mv.mv_baseline_data
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_model_comparisons
-- =============================================================================
-- Side-by-side comparison of multiple candidate baseline models for automated
-- model selection. Stores key statistics for each candidate to enable ranking
-- and selection of the optimal model per ASHRAE 14 criteria.

CREATE TABLE pack040_mv.mv_model_comparisons (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    comparison_name             VARCHAR(255)    NOT NULL,
    comparison_date             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    candidate_baseline_id       UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    candidate_rank              INTEGER         NOT NULL DEFAULT 0,
    is_selected                 BOOLEAN         NOT NULL DEFAULT false,
    selection_reason            TEXT,
    -- Key statistics for comparison
    model_type                  VARCHAR(30)     NOT NULL,
    num_parameters              INTEGER         NOT NULL,
    r_squared                   NUMERIC(8,6),
    adjusted_r_squared          NUMERIC(8,6),
    cvrmse_pct                  NUMERIC(8,4),
    nmbe_pct                    NUMERIC(8,4),
    aic                         NUMERIC(12,4),
    bic                         NUMERIC(12,4),
    f_statistic                 NUMERIC(12,4),
    f_p_value                   NUMERIC(12,10),
    durbin_watson               NUMERIC(6,4),
    -- Validation pass/fail
    passes_cvrmse               BOOLEAN,
    passes_nmbe                 BOOLEAN,
    passes_r_squared            BOOLEAN,
    passes_all_criteria         BOOLEAN,
    -- Scoring
    overall_score               NUMERIC(6,3),
    scoring_method              VARCHAR(30)     NOT NULL DEFAULT 'WEIGHTED',
    score_breakdown             JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_mc_model_type CHECK (
        model_type IN (
            'OLS', '3P_COOLING', '3P_HEATING', '4P', '5P',
            'TOWT', 'MULTIVARIATE', 'MEAN', 'DEGREE_DAY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p040_mc_rank CHECK (
        candidate_rank >= 0
    ),
    CONSTRAINT chk_p040_mc_num_params CHECK (
        num_parameters >= 1 AND num_parameters <= 50
    ),
    CONSTRAINT chk_p040_mc_r_squared CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p040_mc_cvrmse CHECK (
        cvrmse_pct IS NULL OR cvrmse_pct >= 0
    ),
    CONSTRAINT chk_p040_mc_nmbe CHECK (
        nmbe_pct IS NULL OR (nmbe_pct >= -100 AND nmbe_pct <= 100)
    ),
    CONSTRAINT chk_p040_mc_score CHECK (
        overall_score IS NULL OR (overall_score >= 0 AND overall_score <= 100)
    ),
    CONSTRAINT chk_p040_mc_scoring CHECK (
        scoring_method IN ('WEIGHTED', 'AIC_BASED', 'BIC_BASED', 'CUSTOM')
    ),
    CONSTRAINT uq_p040_mc_comparison_candidate UNIQUE (project_id, comparison_name, candidate_baseline_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_mc_tenant            ON pack040_mv.mv_model_comparisons(tenant_id);
CREATE INDEX idx_p040_mc_project           ON pack040_mv.mv_model_comparisons(project_id);
CREATE INDEX idx_p040_mc_baseline          ON pack040_mv.mv_model_comparisons(candidate_baseline_id);
CREATE INDEX idx_p040_mc_model_type        ON pack040_mv.mv_model_comparisons(model_type);
CREATE INDEX idx_p040_mc_selected          ON pack040_mv.mv_model_comparisons(is_selected) WHERE is_selected = true;
CREATE INDEX idx_p040_mc_rank              ON pack040_mv.mv_model_comparisons(candidate_rank);
CREATE INDEX idx_p040_mc_score             ON pack040_mv.mv_model_comparisons(overall_score DESC);
CREATE INDEX idx_p040_mc_created           ON pack040_mv.mv_model_comparisons(created_at DESC);

-- Composite: project + selected model
CREATE INDEX idx_p040_mc_project_sel       ON pack040_mv.mv_model_comparisons(project_id, comparison_name)
    WHERE is_selected = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_mc_updated
    BEFORE UPDATE ON pack040_mv.mv_model_comparisons
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_regression_params ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_model_diagnostics ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_baseline_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_model_comparisons ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_bl_tenant_isolation
    ON pack040_mv.mv_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_bl_service_bypass
    ON pack040_mv.mv_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_rp_tenant_isolation
    ON pack040_mv.mv_regression_params
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_rp_service_bypass
    ON pack040_mv.mv_regression_params
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_md_tenant_isolation
    ON pack040_mv.mv_model_diagnostics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_md_service_bypass
    ON pack040_mv.mv_model_diagnostics
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_bd_tenant_isolation
    ON pack040_mv.mv_baseline_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_bd_service_bypass
    ON pack040_mv.mv_baseline_data
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_mc_tenant_isolation
    ON pack040_mv.mv_model_comparisons
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_mc_service_bypass
    ON pack040_mv.mv_model_comparisons
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_regression_params TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_model_diagnostics TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_baseline_data TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_model_comparisons TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_baselines IS
    'Energy baseline models with regression coefficients, ASHRAE 14 statistics (CVRMSE, NMBE, R-squared), and validation status.';
COMMENT ON TABLE pack040_mv.mv_regression_params IS
    'Detailed regression parameters with standard errors, t-statistics, p-values, confidence intervals, and VIF for each baseline model.';
COMMENT ON TABLE pack040_mv.mv_model_diagnostics IS
    'Model diagnostic tests including residual analysis, autocorrelation, heteroscedasticity, normality, and influential observations.';
COMMENT ON TABLE pack040_mv.mv_baseline_data IS
    'Raw and processed baseline data points with dependent/independent variable values, residuals, and data quality flags.';
COMMENT ON TABLE pack040_mv.mv_model_comparisons IS
    'Side-by-side candidate model comparison with ranking, scoring, and selection rationale for automated model selection.';

COMMENT ON COLUMN pack040_mv.mv_baselines.model_type IS 'Regression model type: OLS, 3P_COOLING, 3P_HEATING, 4P, 5P, TOWT, MULTIVARIATE.';
COMMENT ON COLUMN pack040_mv.mv_baselines.cvrmse_pct IS 'Coefficient of Variation of RMSE (%) - ASHRAE 14 key metric.';
COMMENT ON COLUMN pack040_mv.mv_baselines.nmbe_pct IS 'Normalized Mean Bias Error (%) - ASHRAE 14 key metric.';
COMMENT ON COLUMN pack040_mv.mv_baselines.durbin_watson IS 'Durbin-Watson statistic for autocorrelation (range 0-4, 2=no autocorrelation).';
COMMENT ON COLUMN pack040_mv.mv_baselines.change_point_1 IS 'First change-point temperature for 3P/4P/5P models.';
COMMENT ON COLUMN pack040_mv.mv_baselines.change_point_2 IS 'Second change-point temperature for 5P models.';
COMMENT ON COLUMN pack040_mv.mv_baselines.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_regression_params.vif IS 'Variance Inflation Factor for multicollinearity detection (VIF > 10 indicates multicollinearity).';
COMMENT ON COLUMN pack040_mv.mv_regression_params.is_significant IS 'Whether the parameter is statistically significant at the configured significance level.';

COMMENT ON COLUMN pack040_mv.mv_model_diagnostics.diagnostic_type IS 'Category of diagnostic test: RESIDUAL_ANALYSIS, AUTOCORRELATION, HETEROSCEDASTICITY, etc.';
COMMENT ON COLUMN pack040_mv.mv_model_diagnostics.max_cooks_distance IS 'Maximum Cook''s distance among all data points (>4/n considered influential).';

COMMENT ON COLUMN pack040_mv.mv_baseline_data.standardized_residual IS 'Residual divided by standard error - values > 2 may indicate outliers.';
COMMENT ON COLUMN pack040_mv.mv_baseline_data.is_influential IS 'Whether this point is identified as influential by Cook''s distance or leverage.';

COMMENT ON COLUMN pack040_mv.mv_model_comparisons.overall_score IS 'Composite score (0-100) combining ASHRAE 14 criteria for automated model ranking.';
COMMENT ON COLUMN pack040_mv.mv_model_comparisons.is_selected IS 'Whether this candidate was selected as the best model for the project.';
