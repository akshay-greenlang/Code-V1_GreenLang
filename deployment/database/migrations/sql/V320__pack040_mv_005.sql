-- =============================================================================
-- V320: PACK-040 M&V Pack - Uncertainty Analysis
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for uncertainty quantification per ASHRAE Guideline 14-2014.
-- Covers measurement uncertainty (meter accuracy, CT/PT errors), model
-- uncertainty (regression standard error, prediction intervals), sampling
-- uncertainty (Option A key parameter measurement), combined uncertainty
-- propagation, and fractional savings uncertainty (FSU) records.
--
-- Tables (5):
--   1. pack040_mv.mv_measurement_uncertainty
--   2. pack040_mv.mv_model_uncertainty
--   3. pack040_mv.mv_sampling_uncertainty
--   4. pack040_mv.mv_combined_uncertainty
--   5. pack040_mv.mv_fsu_records
--
-- Previous: V319__pack040_mv_004.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_measurement_uncertainty
-- =============================================================================
-- Measurement uncertainty from metering equipment including meter accuracy
-- class, current/potential transformer errors, calibration drift, and
-- data acquisition system errors. Quantifies the uncertainty contribution
-- from physical measurement to the overall savings uncertainty.

CREATE TABLE pack040_mv.mv_measurement_uncertainty (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    savings_period_id           UUID            REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE SET NULL,
    meter_id                    UUID,
    meter_name                  VARCHAR(255),
    -- Meter specifications
    accuracy_class              VARCHAR(20),
    meter_accuracy_pct          NUMERIC(6,4)    NOT NULL DEFAULT 0.5,
    ct_ratio                    NUMERIC(10,2),
    ct_accuracy_pct             NUMERIC(6,4),
    pt_ratio                    NUMERIC(10,2),
    pt_accuracy_pct             NUMERIC(6,4),
    -- Calibration uncertainty
    last_calibration_date       DATE,
    calibration_drift_pct       NUMERIC(6,4),
    calibration_uncertainty_pct NUMERIC(6,4),
    -- Data acquisition
    daq_resolution_pct          NUMERIC(6,4),
    daq_linearity_pct           NUMERIC(6,4),
    daq_repeatability_pct       NUMERIC(6,4),
    -- Environmental effects
    temperature_effect_pct      NUMERIC(6,4),
    humidity_effect_pct         NUMERIC(6,4),
    -- Combined measurement uncertainty
    combined_measurement_unc_pct NUMERIC(8,4)   NOT NULL,
    measurement_unc_kwh         NUMERIC(18,3),
    -- Calculation details
    calculation_method          VARCHAR(50)     NOT NULL DEFAULT 'RSS',
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 95.0,
    coverage_factor             NUMERIC(5,3)    NOT NULL DEFAULT 2.0,
    expanded_unc_pct            NUMERIC(8,4),
    expanded_unc_kwh            NUMERIC(18,3),
    -- Supporting data
    uncertainty_budget          JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_mu_accuracy_class CHECK (
        accuracy_class IS NULL OR accuracy_class IN (
            'CLASS_0_1', 'CLASS_0_2', 'CLASS_0_5', 'CLASS_1',
            'CLASS_2', 'CLASS_3', 'REVENUE_GRADE', 'UTILITY_GRADE',
            'MONITORING_GRADE', 'INDICATIVE'
        )
    ),
    CONSTRAINT chk_p040_mu_meter_acc CHECK (
        meter_accuracy_pct >= 0 AND meter_accuracy_pct <= 100
    ),
    CONSTRAINT chk_p040_mu_ct_acc CHECK (
        ct_accuracy_pct IS NULL OR (ct_accuracy_pct >= 0 AND ct_accuracy_pct <= 100)
    ),
    CONSTRAINT chk_p040_mu_pt_acc CHECK (
        pt_accuracy_pct IS NULL OR (pt_accuracy_pct >= 0 AND pt_accuracy_pct <= 100)
    ),
    CONSTRAINT chk_p040_mu_combined CHECK (
        combined_measurement_unc_pct >= 0
    ),
    CONSTRAINT chk_p040_mu_calc_method CHECK (
        calculation_method IN ('RSS', 'LINEAR', 'MONTE_CARLO', 'GUM')
    ),
    CONSTRAINT chk_p040_mu_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.9
    ),
    CONSTRAINT chk_p040_mu_coverage CHECK (
        coverage_factor > 0 AND coverage_factor <= 5
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_mu_tenant            ON pack040_mv.mv_measurement_uncertainty(tenant_id);
CREATE INDEX idx_p040_mu_project           ON pack040_mv.mv_measurement_uncertainty(project_id);
CREATE INDEX idx_p040_mu_period            ON pack040_mv.mv_measurement_uncertainty(savings_period_id);
CREATE INDEX idx_p040_mu_meter             ON pack040_mv.mv_measurement_uncertainty(meter_id);
CREATE INDEX idx_p040_mu_accuracy          ON pack040_mv.mv_measurement_uncertainty(accuracy_class);
CREATE INDEX idx_p040_mu_created           ON pack040_mv.mv_measurement_uncertainty(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_mu_updated
    BEFORE UPDATE ON pack040_mv.mv_measurement_uncertainty
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_model_uncertainty
-- =============================================================================
-- Model uncertainty arising from regression model limitations. Quantifies the
-- uncertainty in baseline predictions due to regression standard error,
-- prediction interval width, model misspecification, and extrapolation
-- beyond the range of baseline data.

CREATE TABLE pack040_mv.mv_model_uncertainty (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    savings_period_id           UUID            REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE SET NULL,
    -- Regression model uncertainty
    standard_error_estimate     NUMERIC(18,6)   NOT NULL,
    cvrmse_pct                  NUMERIC(8,4)    NOT NULL,
    nmbe_pct                    NUMERIC(8,4),
    r_squared                   NUMERIC(8,6),
    degrees_of_freedom          INTEGER         NOT NULL,
    -- Prediction uncertainty for reporting period
    num_reporting_points        INTEGER         NOT NULL,
    t_statistic                 NUMERIC(8,4)    NOT NULL,
    prediction_se               NUMERIC(18,6),
    prediction_interval_lower   NUMERIC(18,3),
    prediction_interval_upper   NUMERIC(18,3),
    -- Extrapolation check
    is_extrapolating            BOOLEAN         NOT NULL DEFAULT false,
    max_baseline_temp_f         NUMERIC(8,3),
    min_baseline_temp_f         NUMERIC(8,3),
    max_reporting_temp_f        NUMERIC(8,3),
    min_reporting_temp_f        NUMERIC(8,3),
    extrapolation_pct           NUMERIC(8,4),
    -- Model uncertainty value
    model_unc_pct               NUMERIC(8,4)    NOT NULL,
    model_unc_kwh               NUMERIC(18,3),
    -- ASHRAE 14 specific
    autocorrelation_adjustment  NUMERIC(6,4),
    effective_n                 NUMERIC(10,3),
    -- Calculation
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    calculation_formula         TEXT,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_mou_se CHECK (
        standard_error_estimate >= 0
    ),
    CONSTRAINT chk_p040_mou_cvrmse CHECK (
        cvrmse_pct >= 0
    ),
    CONSTRAINT chk_p040_mou_r_squared CHECK (
        r_squared IS NULL OR (r_squared >= 0 AND r_squared <= 1)
    ),
    CONSTRAINT chk_p040_mou_dof CHECK (
        degrees_of_freedom >= 1
    ),
    CONSTRAINT chk_p040_mou_n_reporting CHECK (
        num_reporting_points >= 1
    ),
    CONSTRAINT chk_p040_mou_t_stat CHECK (
        t_statistic > 0
    ),
    CONSTRAINT chk_p040_mou_model_unc CHECK (
        model_unc_pct >= 0
    ),
    CONSTRAINT chk_p040_mou_extrap CHECK (
        extrapolation_pct IS NULL OR extrapolation_pct >= 0
    ),
    CONSTRAINT chk_p040_mou_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.9
    ),
    CONSTRAINT chk_p040_mou_pred_interval CHECK (
        prediction_interval_lower IS NULL OR prediction_interval_upper IS NULL OR
        prediction_interval_lower <= prediction_interval_upper
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_mou_tenant           ON pack040_mv.mv_model_uncertainty(tenant_id);
CREATE INDEX idx_p040_mou_project          ON pack040_mv.mv_model_uncertainty(project_id);
CREATE INDEX idx_p040_mou_baseline         ON pack040_mv.mv_model_uncertainty(baseline_id);
CREATE INDEX idx_p040_mou_period           ON pack040_mv.mv_model_uncertainty(savings_period_id);
CREATE INDEX idx_p040_mou_extrap           ON pack040_mv.mv_model_uncertainty(is_extrapolating) WHERE is_extrapolating = true;
CREATE INDEX idx_p040_mou_created          ON pack040_mv.mv_model_uncertainty(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_mou_updated
    BEFORE UPDATE ON pack040_mv.mv_model_uncertainty
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_sampling_uncertainty
-- =============================================================================
-- Sampling uncertainty for IPMVP Option A key parameter measurement where
-- not all units are measured directly. Quantifies the uncertainty due to
-- extrapolating sample measurements to the full population of affected
-- equipment using t-distribution statistics.

CREATE TABLE pack040_mv.mv_sampling_uncertainty (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    savings_period_id           UUID            REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE SET NULL,
    -- Population
    population_description      TEXT            NOT NULL,
    population_size             INTEGER         NOT NULL,
    sample_size                 INTEGER         NOT NULL,
    sampling_method             VARCHAR(50)     NOT NULL DEFAULT 'SIMPLE_RANDOM',
    -- Sample statistics
    sample_mean                 NUMERIC(18,6)   NOT NULL,
    sample_std_dev              NUMERIC(18,6)   NOT NULL,
    coefficient_of_variation    NUMERIC(8,4)    NOT NULL,
    sample_min                  NUMERIC(18,6),
    sample_max                  NUMERIC(18,6),
    -- Confidence interval
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    precision_pct               NUMERIC(8,4)    NOT NULL,
    t_statistic                 NUMERIC(8,4)    NOT NULL,
    degrees_of_freedom          INTEGER         NOT NULL,
    margin_of_error             NUMERIC(18,6),
    confidence_lower            NUMERIC(18,6),
    confidence_upper            NUMERIC(18,6),
    -- Required sample size
    required_sample_size        INTEGER,
    achieved_precision_pct      NUMERIC(8,4),
    meets_precision_target      BOOLEAN,
    -- Sampling uncertainty
    sampling_unc_pct            NUMERIC(8,4)    NOT NULL,
    sampling_unc_kwh            NUMERIC(18,3),
    -- Stratification (if stratified sampling)
    is_stratified               BOOLEAN         NOT NULL DEFAULT false,
    num_strata                  INTEGER,
    strata_details              JSONB           DEFAULT '[]',
    -- Finite population correction
    fpc_applied                 BOOLEAN         NOT NULL DEFAULT false,
    fpc_factor                  NUMERIC(8,6),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_su_population CHECK (
        population_size >= 1
    ),
    CONSTRAINT chk_p040_su_sample CHECK (
        sample_size >= 1 AND sample_size <= population_size
    ),
    CONSTRAINT chk_p040_su_method CHECK (
        sampling_method IN (
            'SIMPLE_RANDOM', 'STRATIFIED', 'SYSTEMATIC', 'CLUSTER',
            'CENSUS', 'CONVENIENCE', 'QUOTA'
        )
    ),
    CONSTRAINT chk_p040_su_cv CHECK (
        coefficient_of_variation >= 0
    ),
    CONSTRAINT chk_p040_su_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.9
    ),
    CONSTRAINT chk_p040_su_precision CHECK (
        precision_pct > 0 AND precision_pct <= 100
    ),
    CONSTRAINT chk_p040_su_t_stat CHECK (
        t_statistic > 0
    ),
    CONSTRAINT chk_p040_su_dof CHECK (
        degrees_of_freedom >= 1
    ),
    CONSTRAINT chk_p040_su_sampling_unc CHECK (
        sampling_unc_pct >= 0
    ),
    CONSTRAINT chk_p040_su_fpc CHECK (
        fpc_factor IS NULL OR (fpc_factor > 0 AND fpc_factor <= 1)
    ),
    CONSTRAINT chk_p040_su_strata CHECK (
        num_strata IS NULL OR num_strata >= 2
    ),
    CONSTRAINT chk_p040_su_req_sample CHECK (
        required_sample_size IS NULL OR required_sample_size >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_su_tenant            ON pack040_mv.mv_sampling_uncertainty(tenant_id);
CREATE INDEX idx_p040_su_project           ON pack040_mv.mv_sampling_uncertainty(project_id);
CREATE INDEX idx_p040_su_ecm               ON pack040_mv.mv_sampling_uncertainty(ecm_id);
CREATE INDEX idx_p040_su_period            ON pack040_mv.mv_sampling_uncertainty(savings_period_id);
CREATE INDEX idx_p040_su_method            ON pack040_mv.mv_sampling_uncertainty(sampling_method);
CREATE INDEX idx_p040_su_meets             ON pack040_mv.mv_sampling_uncertainty(meets_precision_target);
CREATE INDEX idx_p040_su_created           ON pack040_mv.mv_sampling_uncertainty(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_su_updated
    BEFORE UPDATE ON pack040_mv.mv_sampling_uncertainty
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_combined_uncertainty
-- =============================================================================
-- Combined uncertainty propagation merging measurement, model, and sampling
-- uncertainty components using root-sum-square (RSS) method per ASHRAE 14.
-- Produces the total savings uncertainty for each reporting period.

CREATE TABLE pack040_mv.mv_combined_uncertainty (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    savings_period_id           UUID            NOT NULL REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE CASCADE,
    ecm_id                      UUID            REFERENCES pack040_mv.mv_ecms(id) ON DELETE SET NULL,
    -- Component references
    measurement_unc_id          UUID            REFERENCES pack040_mv.mv_measurement_uncertainty(id) ON DELETE SET NULL,
    model_unc_id                UUID            REFERENCES pack040_mv.mv_model_uncertainty(id) ON DELETE SET NULL,
    sampling_unc_id             UUID            REFERENCES pack040_mv.mv_sampling_uncertainty(id) ON DELETE SET NULL,
    -- Component values (kWh)
    measurement_unc_kwh         NUMERIC(18,3)   NOT NULL DEFAULT 0,
    model_unc_kwh               NUMERIC(18,3)   NOT NULL DEFAULT 0,
    sampling_unc_kwh            NUMERIC(18,3)   NOT NULL DEFAULT 0,
    nonroutine_adj_unc_kwh      NUMERIC(18,3)   DEFAULT 0,
    -- Component values (%)
    measurement_unc_pct         NUMERIC(8,4)    NOT NULL DEFAULT 0,
    model_unc_pct               NUMERIC(8,4)    NOT NULL DEFAULT 0,
    sampling_unc_pct            NUMERIC(8,4)    NOT NULL DEFAULT 0,
    nonroutine_adj_unc_pct      NUMERIC(8,4)    DEFAULT 0,
    -- Combined result
    propagation_method          VARCHAR(30)     NOT NULL DEFAULT 'RSS',
    combined_unc_kwh            NUMERIC(18,3)   NOT NULL,
    combined_unc_pct            NUMERIC(8,4)    NOT NULL,
    -- Savings context
    total_savings_kwh           NUMERIC(18,3)   NOT NULL,
    savings_minus_unc_kwh       NUMERIC(18,3),
    savings_plus_unc_kwh        NUMERIC(18,3),
    -- Confidence
    confidence_level_pct        NUMERIC(5,2)    NOT NULL DEFAULT 90.0,
    coverage_factor             NUMERIC(5,3)    NOT NULL DEFAULT 1.645,
    -- Dominant component
    dominant_component          VARCHAR(30),
    dominant_component_pct      NUMERIC(8,4),
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_cu_propagation CHECK (
        propagation_method IN ('RSS', 'LINEAR', 'MONTE_CARLO')
    ),
    CONSTRAINT chk_p040_cu_combined_pct CHECK (
        combined_unc_pct >= 0
    ),
    CONSTRAINT chk_p040_cu_combined_kwh CHECK (
        combined_unc_kwh >= 0
    ),
    CONSTRAINT chk_p040_cu_confidence CHECK (
        confidence_level_pct >= 50 AND confidence_level_pct <= 99.9
    ),
    CONSTRAINT chk_p040_cu_coverage CHECK (
        coverage_factor > 0 AND coverage_factor <= 5
    ),
    CONSTRAINT chk_p040_cu_dominant CHECK (
        dominant_component IS NULL OR dominant_component IN (
            'MEASUREMENT', 'MODEL', 'SAMPLING', 'NONROUTINE_ADJUSTMENT'
        )
    ),
    CONSTRAINT chk_p040_cu_bounds CHECK (
        savings_minus_unc_kwh IS NULL OR savings_plus_unc_kwh IS NULL OR
        savings_minus_unc_kwh <= savings_plus_unc_kwh
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_cu_tenant            ON pack040_mv.mv_combined_uncertainty(tenant_id);
CREATE INDEX idx_p040_cu_project           ON pack040_mv.mv_combined_uncertainty(project_id);
CREATE INDEX idx_p040_cu_period            ON pack040_mv.mv_combined_uncertainty(savings_period_id);
CREATE INDEX idx_p040_cu_ecm               ON pack040_mv.mv_combined_uncertainty(ecm_id);
CREATE INDEX idx_p040_cu_dominant          ON pack040_mv.mv_combined_uncertainty(dominant_component);
CREATE INDEX idx_p040_cu_created           ON pack040_mv.mv_combined_uncertainty(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_cu_updated
    BEFORE UPDATE ON pack040_mv.mv_combined_uncertainty
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_fsu_records
-- =============================================================================
-- Fractional Savings Uncertainty (FSU) records per ASHRAE Guideline 14.
-- FSU is the key metric determining whether savings are statistically
-- significant. ASHRAE 14 requires FSU < 50% at 68% confidence for savings
-- to be considered verified.

CREATE TABLE pack040_mv.mv_fsu_records (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    savings_period_id           UUID            NOT NULL REFERENCES pack040_mv.mv_savings_periods(id) ON DELETE CASCADE,
    combined_unc_id             UUID            REFERENCES pack040_mv.mv_combined_uncertainty(id) ON DELETE SET NULL,
    -- FSU calculation inputs
    total_savings_kwh           NUMERIC(18,3)   NOT NULL,
    baseline_energy_kwh         NUMERIC(18,3)   NOT NULL,
    actual_energy_kwh           NUMERIC(18,3)   NOT NULL,
    savings_fraction            NUMERIC(8,6)    NOT NULL,
    -- FSU at different confidence levels
    fsu_68_pct                  NUMERIC(8,4)    NOT NULL,
    fsu_80_pct                  NUMERIC(8,4),
    fsu_90_pct                  NUMERIC(8,4),
    fsu_95_pct                  NUMERIC(8,4),
    -- ASHRAE 14 compliance
    ashrae_14_threshold_pct     NUMERIC(6,2)    NOT NULL DEFAULT 50.0,
    ashrae_14_confidence_pct    NUMERIC(5,2)    NOT NULL DEFAULT 68.0,
    passes_ashrae_14            BOOLEAN         NOT NULL,
    -- Minimum detectable savings
    min_detectable_savings_kwh  NUMERIC(18,3),
    min_detectable_savings_pct  NUMERIC(8,4),
    savings_exceeds_mds         BOOLEAN,
    -- t-test for significance
    t_statistic_savings         NUMERIC(8,4),
    t_critical                  NUMERIC(8,4),
    savings_significant         BOOLEAN,
    -- Relative precision
    relative_precision_pct      NUMERIC(8,4),
    absolute_precision_kwh      NUMERIC(18,3),
    -- Context
    ipmvp_option                VARCHAR(10),
    data_granularity            VARCHAR(20),
    num_baseline_points         INTEGER,
    num_reporting_points        INTEGER,
    -- Recommendations
    improvement_recommendations JSONB           DEFAULT '[]',
    sensitivity_analysis        JSONB           DEFAULT '{}',
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_fsu_savings_frac CHECK (
        savings_fraction >= -1 AND savings_fraction <= 1
    ),
    CONSTRAINT chk_p040_fsu_68 CHECK (
        fsu_68_pct >= 0
    ),
    CONSTRAINT chk_p040_fsu_90 CHECK (
        fsu_90_pct IS NULL OR fsu_90_pct >= 0
    ),
    CONSTRAINT chk_p040_fsu_threshold CHECK (
        ashrae_14_threshold_pct > 0 AND ashrae_14_threshold_pct <= 200
    ),
    CONSTRAINT chk_p040_fsu_ashrae_conf CHECK (
        ashrae_14_confidence_pct >= 50 AND ashrae_14_confidence_pct <= 99.9
    ),
    CONSTRAINT chk_p040_fsu_baseline CHECK (
        baseline_energy_kwh >= 0
    ),
    CONSTRAINT chk_p040_fsu_actual CHECK (
        actual_energy_kwh >= 0
    ),
    CONSTRAINT chk_p040_fsu_mds CHECK (
        min_detectable_savings_kwh IS NULL OR min_detectable_savings_kwh >= 0
    ),
    CONSTRAINT chk_p040_fsu_option CHECK (
        ipmvp_option IS NULL OR ipmvp_option IN (
            'OPTION_A', 'OPTION_B', 'OPTION_C', 'OPTION_D'
        )
    ),
    CONSTRAINT chk_p040_fsu_granularity CHECK (
        data_granularity IS NULL OR data_granularity IN (
            'HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'BILLING_PERIOD'
        )
    ),
    CONSTRAINT uq_p040_fsu_project_period UNIQUE (project_id, savings_period_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_fsu_tenant           ON pack040_mv.mv_fsu_records(tenant_id);
CREATE INDEX idx_p040_fsu_project          ON pack040_mv.mv_fsu_records(project_id);
CREATE INDEX idx_p040_fsu_period           ON pack040_mv.mv_fsu_records(savings_period_id);
CREATE INDEX idx_p040_fsu_combined         ON pack040_mv.mv_fsu_records(combined_unc_id);
CREATE INDEX idx_p040_fsu_passes           ON pack040_mv.mv_fsu_records(passes_ashrae_14);
CREATE INDEX idx_p040_fsu_significant      ON pack040_mv.mv_fsu_records(savings_significant);
CREATE INDEX idx_p040_fsu_option           ON pack040_mv.mv_fsu_records(ipmvp_option);
CREATE INDEX idx_p040_fsu_created          ON pack040_mv.mv_fsu_records(created_at DESC);

-- Composite: project + passing FSU records
CREATE INDEX idx_p040_fsu_project_pass     ON pack040_mv.mv_fsu_records(project_id, savings_period_id)
    WHERE passes_ashrae_14 = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_fsu_updated
    BEFORE UPDATE ON pack040_mv.mv_fsu_records
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_measurement_uncertainty ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_model_uncertainty ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_sampling_uncertainty ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_combined_uncertainty ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_fsu_records ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_mu_tenant_isolation
    ON pack040_mv.mv_measurement_uncertainty
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_mu_service_bypass
    ON pack040_mv.mv_measurement_uncertainty
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_mou_tenant_isolation
    ON pack040_mv.mv_model_uncertainty
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_mou_service_bypass
    ON pack040_mv.mv_model_uncertainty
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_su_tenant_isolation
    ON pack040_mv.mv_sampling_uncertainty
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_su_service_bypass
    ON pack040_mv.mv_sampling_uncertainty
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_cu_tenant_isolation
    ON pack040_mv.mv_combined_uncertainty
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_cu_service_bypass
    ON pack040_mv.mv_combined_uncertainty
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_fsu_tenant_isolation
    ON pack040_mv.mv_fsu_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_fsu_service_bypass
    ON pack040_mv.mv_fsu_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_measurement_uncertainty TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_model_uncertainty TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_sampling_uncertainty TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_combined_uncertainty TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_fsu_records TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_measurement_uncertainty IS
    'Measurement uncertainty from metering equipment: meter accuracy, CT/PT errors, calibration drift, and DAQ system errors.';
COMMENT ON TABLE pack040_mv.mv_model_uncertainty IS
    'Regression model uncertainty: standard error, prediction intervals, extrapolation checks, and autocorrelation adjustments.';
COMMENT ON TABLE pack040_mv.mv_sampling_uncertainty IS
    'Sampling uncertainty for IPMVP Option A key parameter measurement with sample size calculations and confidence intervals.';
COMMENT ON TABLE pack040_mv.mv_combined_uncertainty IS
    'Combined uncertainty propagation using RSS method merging measurement, model, and sampling components per ASHRAE 14.';
COMMENT ON TABLE pack040_mv.mv_fsu_records IS
    'Fractional Savings Uncertainty (FSU) per ASHRAE 14 with minimum detectable savings and significance testing.';

COMMENT ON COLUMN pack040_mv.mv_measurement_uncertainty.combined_measurement_unc_pct IS 'Root-sum-square of all measurement uncertainty components as percentage of measured value.';
COMMENT ON COLUMN pack040_mv.mv_measurement_uncertainty.coverage_factor IS 'Multiplier for expanding standard uncertainty to desired confidence level (k=2 for ~95%).';
COMMENT ON COLUMN pack040_mv.mv_measurement_uncertainty.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_model_uncertainty.cvrmse_pct IS 'Coefficient of Variation of RMSE used in ASHRAE 14 uncertainty formula.';
COMMENT ON COLUMN pack040_mv.mv_model_uncertainty.effective_n IS 'Effective sample size adjusted for autocorrelation (reduced from actual n).';
COMMENT ON COLUMN pack040_mv.mv_model_uncertainty.is_extrapolating IS 'Whether reporting period conditions exceed baseline data range (increases uncertainty).';

COMMENT ON COLUMN pack040_mv.mv_sampling_uncertainty.coefficient_of_variation IS 'Sample standard deviation / sample mean - measures relative variability in sample.';
COMMENT ON COLUMN pack040_mv.mv_sampling_uncertainty.fpc_factor IS 'Finite population correction factor = sqrt((N-n)/(N-1)), applied when n/N > 0.05.';

COMMENT ON COLUMN pack040_mv.mv_combined_uncertainty.dominant_component IS 'Uncertainty component contributing the largest share to combined uncertainty.';

COMMENT ON COLUMN pack040_mv.mv_fsu_records.fsu_68_pct IS 'Fractional Savings Uncertainty at 68% confidence (1-sigma) - primary ASHRAE 14 metric.';
COMMENT ON COLUMN pack040_mv.mv_fsu_records.min_detectable_savings_kwh IS 'Minimum savings that can be detected above noise at the specified confidence level.';
COMMENT ON COLUMN pack040_mv.mv_fsu_records.passes_ashrae_14 IS 'Whether FSU at 68% confidence is less than 50% threshold per ASHRAE 14.';
