-- =============================================================================
-- V310: PACK-039 Energy Monitoring Pack - EnPI Tracking
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    005 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates Energy Performance Indicator (EnPI) tracking tables aligned
-- with ISO 50001 and ISO 50006 standards. Includes EnPI definitions with
-- boundary and normalization variables, calculated EnPI values over time,
-- energy baselines with adjustment factors, CUSUM (Cumulative Sum)
-- tracking for performance monitoring, and regression models for
-- energy-production relationships.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_enpi_definitions
--   2. pack039_energy_monitoring.em_enpi_values
--   3. pack039_energy_monitoring.em_energy_baselines
--   4. pack039_energy_monitoring.em_cusum_tracking
--   5. pack039_energy_monitoring.em_regression_models
--
-- Previous: V309__pack039_energy_monitoring_004.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_enpi_definitions
-- =============================================================================
-- Definitions of Energy Performance Indicators per ISO 50006. Each EnPI
-- is a quantitative value or measure of energy performance defined by
-- the organization. EnPIs can be simple ratios (energy per unit of
-- production), regression-based models, or composite indicators that
-- account for multiple relevant variables (weather, occupancy, production).

CREATE TABLE pack039_energy_monitoring.em_enpi_definitions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    enpi_name               VARCHAR(255)    NOT NULL,
    enpi_code               VARCHAR(50)     NOT NULL,
    enpi_type               VARCHAR(30)     NOT NULL DEFAULT 'RATIO',
    enpi_category           VARCHAR(50)     NOT NULL DEFAULT 'FACILITY',
    description             TEXT,
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    normalizing_variable    VARCHAR(100),
    normalizing_unit        VARCHAR(50),
    output_unit             VARCHAR(50)     NOT NULL DEFAULT 'kWh/unit',
    meter_ids               UUID[]          NOT NULL DEFAULT '{}',
    boundary_description    TEXT,
    formula_expression      TEXT,
    relevant_variables      JSONB           DEFAULT '[]',
    weighting_factors       JSONB           DEFAULT '{}',
    target_value            NUMERIC(15,6),
    target_direction        VARCHAR(10)     NOT NULL DEFAULT 'LOWER',
    warning_threshold_pct   NUMERIC(7,4),
    critical_threshold_pct  NUMERIC(7,4),
    baseline_id             UUID,
    regression_model_id     UUID,
    calculation_frequency   VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    aggregation_method      VARCHAR(20)     NOT NULL DEFAULT 'SUM',
    data_quality_min_pct    NUMERIC(5,2)    DEFAULT 90.0,
    iso_50001_scope         VARCHAR(50),
    significant_energy_use  BOOLEAN         NOT NULL DEFAULT false,
    is_primary_enpi         BOOLEAN         NOT NULL DEFAULT false,
    is_published            BOOLEAN         NOT NULL DEFAULT false,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    effective_from          DATE            NOT NULL DEFAULT CURRENT_DATE,
    effective_to            DATE,
    last_calculated_at      TIMESTAMPTZ,
    last_calculated_value   NUMERIC(15,6),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_ed_type CHECK (
        enpi_type IN (
            'RATIO', 'REGRESSION', 'ABSOLUTE', 'INDEXED',
            'SPECIFIC', 'COMPOSITE', 'MODELED', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ed_category CHECK (
        enpi_category IN (
            'FACILITY', 'BUILDING', 'PROCESS', 'EQUIPMENT',
            'DEPARTMENT', 'PRODUCT', 'SITE', 'PORTFOLIO', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_ed_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'COMPRESSED_AIR', 'TOTAL_ENERGY', 'PRIMARY_ENERGY',
            'RENEWABLE', 'FOSSIL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_ed_direction CHECK (
        target_direction IN ('LOWER', 'HIGHER', 'TARGET')
    ),
    CONSTRAINT chk_p039_ed_frequency CHECK (
        calculation_frequency IN (
            'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL', 'ON_DEMAND'
        )
    ),
    CONSTRAINT chk_p039_ed_aggregation CHECK (
        aggregation_method IN ('SUM', 'AVERAGE', 'WEIGHTED_AVERAGE', 'MAX', 'MIN', 'MEDIAN')
    ),
    CONSTRAINT chk_p039_ed_quality_min CHECK (
        data_quality_min_pct IS NULL OR (data_quality_min_pct >= 0 AND data_quality_min_pct <= 100)
    ),
    CONSTRAINT chk_p039_ed_threshold_warn CHECK (
        warning_threshold_pct IS NULL OR warning_threshold_pct > 0
    ),
    CONSTRAINT chk_p039_ed_threshold_crit CHECK (
        critical_threshold_pct IS NULL OR critical_threshold_pct > 0
    ),
    CONSTRAINT chk_p039_ed_dates CHECK (
        effective_to IS NULL OR effective_from <= effective_to
    ),
    CONSTRAINT uq_p039_ed_tenant_code UNIQUE (tenant_id, enpi_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ed_tenant          ON pack039_energy_monitoring.em_enpi_definitions(tenant_id);
CREATE INDEX idx_p039_ed_facility        ON pack039_energy_monitoring.em_enpi_definitions(facility_id);
CREATE INDEX idx_p039_ed_code            ON pack039_energy_monitoring.em_enpi_definitions(enpi_code);
CREATE INDEX idx_p039_ed_type            ON pack039_energy_monitoring.em_enpi_definitions(enpi_type);
CREATE INDEX idx_p039_ed_category        ON pack039_energy_monitoring.em_enpi_definitions(enpi_category);
CREATE INDEX idx_p039_ed_energy_type     ON pack039_energy_monitoring.em_enpi_definitions(energy_type);
CREATE INDEX idx_p039_ed_active          ON pack039_energy_monitoring.em_enpi_definitions(is_active) WHERE is_active = true;
CREATE INDEX idx_p039_ed_primary         ON pack039_energy_monitoring.em_enpi_definitions(is_primary_enpi) WHERE is_primary_enpi = true;
CREATE INDEX idx_p039_ed_seu             ON pack039_energy_monitoring.em_enpi_definitions(significant_energy_use) WHERE significant_energy_use = true;
CREATE INDEX idx_p039_ed_frequency       ON pack039_energy_monitoring.em_enpi_definitions(calculation_frequency);
CREATE INDEX idx_p039_ed_baseline        ON pack039_energy_monitoring.em_enpi_definitions(baseline_id);
CREATE INDEX idx_p039_ed_created         ON pack039_energy_monitoring.em_enpi_definitions(created_at DESC);
CREATE INDEX idx_p039_ed_meters          ON pack039_energy_monitoring.em_enpi_definitions USING GIN(meter_ids);
CREATE INDEX idx_p039_ed_variables       ON pack039_energy_monitoring.em_enpi_definitions USING GIN(relevant_variables);

-- Composite: active facility EnPIs
CREATE INDEX idx_p039_ed_fac_active      ON pack039_energy_monitoring.em_enpi_definitions(facility_id, enpi_category)
    WHERE is_active = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ed_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_enpi_definitions
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_enpi_values
-- =============================================================================
-- Calculated EnPI values over time. Each row represents the EnPI value
-- for a specific period, including the raw energy consumption, normalizing
-- variable values, and the resulting EnPI. Tracks performance against
-- baseline and target values for continuous improvement monitoring.

CREATE TABLE pack039_energy_monitoring.em_enpi_values (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enpi_id                 UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_enpi_definitions(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    enpi_value              NUMERIC(15,6)   NOT NULL,
    energy_consumed         NUMERIC(15,3)   NOT NULL,
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    normalizing_value       NUMERIC(15,3),
    normalizing_unit        VARCHAR(50),
    baseline_enpi_value     NUMERIC(15,6),
    target_enpi_value       NUMERIC(15,6),
    improvement_pct         NUMERIC(10,4),
    improvement_absolute    NUMERIC(15,6),
    energy_savings_kwh      NUMERIC(15,3),
    energy_savings_pct      NUMERIC(10,4),
    performance_status      VARCHAR(20)     NOT NULL DEFAULT 'ON_TARGET',
    regression_predicted    NUMERIC(15,6),
    regression_residual     NUMERIC(15,6),
    cusum_value             NUMERIC(15,6),
    relevant_variable_values JSONB          DEFAULT '{}',
    weather_hdd             NUMERIC(10,2),
    weather_cdd             NUMERIC(10,2),
    production_output       NUMERIC(15,3),
    production_unit         VARCHAR(50),
    occupancy_pct           NUMERIC(5,2),
    operating_hours         NUMERIC(8,2),
    data_quality_pct        NUMERIC(5,2),
    data_completeness_pct   NUMERIC(5,2),
    adjustment_factors      JSONB           DEFAULT '{}',
    notes                   TEXT,
    is_approved             BOOLEAN         NOT NULL DEFAULT false,
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ev_period_type CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p039_ev_performance CHECK (
        performance_status IN (
            'ON_TARGET', 'ABOVE_TARGET', 'BELOW_TARGET',
            'WARNING', 'CRITICAL', 'INSUFFICIENT_DATA'
        )
    ),
    CONSTRAINT chk_p039_ev_energy CHECK (
        energy_consumed >= 0
    ),
    CONSTRAINT chk_p039_ev_quality CHECK (
        data_quality_pct IS NULL OR (data_quality_pct >= 0 AND data_quality_pct <= 100)
    ),
    CONSTRAINT chk_p039_ev_completeness CHECK (
        data_completeness_pct IS NULL OR (data_completeness_pct >= 0 AND data_completeness_pct <= 100)
    ),
    CONSTRAINT chk_p039_ev_occupancy CHECK (
        occupancy_pct IS NULL OR (occupancy_pct >= 0 AND occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p039_ev_operating CHECK (
        operating_hours IS NULL OR operating_hours >= 0
    ),
    CONSTRAINT chk_p039_ev_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT uq_p039_ev_enpi_period UNIQUE (enpi_id, period_type, period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ev_enpi            ON pack039_energy_monitoring.em_enpi_values(enpi_id);
CREATE INDEX idx_p039_ev_tenant          ON pack039_energy_monitoring.em_enpi_values(tenant_id);
CREATE INDEX idx_p039_ev_period_type     ON pack039_energy_monitoring.em_enpi_values(period_type);
CREATE INDEX idx_p039_ev_period_start    ON pack039_energy_monitoring.em_enpi_values(period_start DESC);
CREATE INDEX idx_p039_ev_performance     ON pack039_energy_monitoring.em_enpi_values(performance_status);
CREATE INDEX idx_p039_ev_value           ON pack039_energy_monitoring.em_enpi_values(enpi_value);
CREATE INDEX idx_p039_ev_improvement     ON pack039_energy_monitoring.em_enpi_values(improvement_pct);
CREATE INDEX idx_p039_ev_approved        ON pack039_energy_monitoring.em_enpi_values(is_approved) WHERE is_approved = false;
CREATE INDEX idx_p039_ev_created         ON pack039_energy_monitoring.em_enpi_values(created_at DESC);
CREATE INDEX idx_p039_ev_variables       ON pack039_energy_monitoring.em_enpi_values USING GIN(relevant_variable_values);
CREATE INDEX idx_p039_ev_adjustments     ON pack039_energy_monitoring.em_enpi_values USING GIN(adjustment_factors);

-- Composite: monthly EnPI values for trending charts
CREATE INDEX idx_p039_ev_enpi_monthly    ON pack039_energy_monitoring.em_enpi_values(enpi_id, period_start DESC)
    WHERE period_type = 'MONTHLY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ev_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_enpi_values
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_energy_baselines
-- =============================================================================
-- Energy baselines per ISO 50001/50006 representing the quantitative
-- reference for comparing energy performance. Baselines include period
-- definitions, adjustment methodology, static factors, and non-routine
-- adjustment records. Baselines may be fixed or periodically updated
-- and serve as the denominator in EnPI improvement calculations.

CREATE TABLE pack039_energy_monitoring.em_energy_baselines (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    baseline_name           VARCHAR(255)    NOT NULL,
    baseline_code           VARCHAR(50)     NOT NULL,
    baseline_type           VARCHAR(30)     NOT NULL DEFAULT 'FIXED',
    energy_type             VARCHAR(50)     NOT NULL DEFAULT 'ELECTRICITY',
    baseline_period_start   DATE            NOT NULL,
    baseline_period_end     DATE            NOT NULL,
    total_energy_consumed   NUMERIC(15,3)   NOT NULL,
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    total_cost              NUMERIC(12,2),
    cost_currency           VARCHAR(3)      DEFAULT 'USD',
    normalizing_variable    VARCHAR(100),
    normalizing_value_total NUMERIC(15,3),
    normalizing_unit        VARCHAR(50),
    baseline_enpi_value     NUMERIC(15,6),
    enpi_unit               VARCHAR(50),
    regression_equation     TEXT,
    regression_r_squared    NUMERIC(5,4),
    regression_coefficients JSONB           DEFAULT '{}',
    static_factors          JSONB           DEFAULT '[]',
    non_routine_adjustments JSONB           DEFAULT '[]',
    adjustment_methodology  TEXT,
    weather_hdd_total       NUMERIC(10,2),
    weather_cdd_total       NUMERIC(10,2),
    weather_station_id      VARCHAR(50),
    production_total        NUMERIC(15,3),
    production_unit         VARCHAR(50),
    operating_hours_total   NUMERIC(10,2),
    floor_area_m2           NUMERIC(12,2),
    occupancy_avg_pct       NUMERIC(5,2),
    monthly_breakdown       JSONB           DEFAULT '[]',
    is_current              BOOLEAN         NOT NULL DEFAULT true,
    approval_status         VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    reason_for_revision     TEXT,
    previous_baseline_id    UUID,
    revision_number         INTEGER         NOT NULL DEFAULT 1,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_eb_type CHECK (
        baseline_type IN (
            'FIXED', 'ROLLING', 'ADJUSTED', 'WEATHER_NORMALIZED',
            'REGRESSION', 'MULTIVARIATE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_eb_energy_type CHECK (
        energy_type IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'CHILLED_WATER',
            'HOT_WATER', 'TOTAL_ENERGY', 'PRIMARY_ENERGY', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_eb_approval CHECK (
        approval_status IN ('DRAFT', 'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'SUPERSEDED')
    ),
    CONSTRAINT chk_p039_eb_energy CHECK (
        total_energy_consumed >= 0
    ),
    CONSTRAINT chk_p039_eb_r_squared CHECK (
        regression_r_squared IS NULL OR (regression_r_squared >= 0 AND regression_r_squared <= 1)
    ),
    CONSTRAINT chk_p039_eb_occupancy CHECK (
        occupancy_avg_pct IS NULL OR (occupancy_avg_pct >= 0 AND occupancy_avg_pct <= 100)
    ),
    CONSTRAINT chk_p039_eb_floor CHECK (
        floor_area_m2 IS NULL OR floor_area_m2 > 0
    ),
    CONSTRAINT chk_p039_eb_revision CHECK (
        revision_number >= 1
    ),
    CONSTRAINT chk_p039_eb_dates CHECK (
        baseline_period_start <= baseline_period_end
    ),
    CONSTRAINT uq_p039_eb_tenant_code UNIQUE (tenant_id, baseline_code, revision_number)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_eb_tenant          ON pack039_energy_monitoring.em_energy_baselines(tenant_id);
CREATE INDEX idx_p039_eb_facility        ON pack039_energy_monitoring.em_energy_baselines(facility_id);
CREATE INDEX idx_p039_eb_code            ON pack039_energy_monitoring.em_energy_baselines(baseline_code);
CREATE INDEX idx_p039_eb_type            ON pack039_energy_monitoring.em_energy_baselines(baseline_type);
CREATE INDEX idx_p039_eb_energy_type     ON pack039_energy_monitoring.em_energy_baselines(energy_type);
CREATE INDEX idx_p039_eb_current         ON pack039_energy_monitoring.em_energy_baselines(is_current) WHERE is_current = true;
CREATE INDEX idx_p039_eb_approval        ON pack039_energy_monitoring.em_energy_baselines(approval_status);
CREATE INDEX idx_p039_eb_period          ON pack039_energy_monitoring.em_energy_baselines(baseline_period_start, baseline_period_end);
CREATE INDEX idx_p039_eb_created         ON pack039_energy_monitoring.em_energy_baselines(created_at DESC);
CREATE INDEX idx_p039_eb_monthly         ON pack039_energy_monitoring.em_energy_baselines USING GIN(monthly_breakdown);
CREATE INDEX idx_p039_eb_coefficients    ON pack039_energy_monitoring.em_energy_baselines USING GIN(regression_coefficients);

-- Composite: current approved baselines by facility
CREATE INDEX idx_p039_eb_fac_current     ON pack039_energy_monitoring.em_energy_baselines(facility_id, energy_type)
    WHERE is_current = true AND approval_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_eb_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_energy_baselines
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_cusum_tracking
-- =============================================================================
-- Cumulative Sum (CUSUM) tracking for energy performance monitoring per
-- ISO 50006. CUSUM plots show the cumulative difference between actual
-- and expected (baseline) energy consumption over time. A downward trend
-- indicates improving performance; an upward trend indicates degradation.
-- Used to detect sustained performance changes and trigger investigations.

CREATE TABLE pack039_energy_monitoring.em_cusum_tracking (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enpi_id                 UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_enpi_definitions(id) ON DELETE CASCADE,
    baseline_id             UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_energy_baselines(id),
    tenant_id               UUID            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    period_date             DATE            NOT NULL,
    actual_energy           NUMERIC(15,3)   NOT NULL,
    expected_energy         NUMERIC(15,3)   NOT NULL,
    period_difference       NUMERIC(15,3)   NOT NULL,
    cumulative_sum          NUMERIC(18,3)   NOT NULL,
    cumulative_savings_kwh  NUMERIC(18,3)   NOT NULL DEFAULT 0,
    cumulative_savings_pct  NUMERIC(10,4),
    cumulative_cost_savings NUMERIC(15,2),
    energy_unit             VARCHAR(20)     NOT NULL DEFAULT 'kWh',
    cost_currency           VARCHAR(3)      DEFAULT 'USD',
    trend_direction         VARCHAR(20)     NOT NULL DEFAULT 'STABLE',
    trend_change_detected   BOOLEAN         NOT NULL DEFAULT false,
    trend_change_date       DATE,
    control_limit_upper     NUMERIC(18,3),
    control_limit_lower     NUMERIC(18,3),
    is_out_of_control       BOOLEAN         NOT NULL DEFAULT false,
    relevant_variable_values JSONB          DEFAULT '{}',
    adjustment_applied      BOOLEAN         NOT NULL DEFAULT false,
    adjustment_value        NUMERIC(15,3),
    adjustment_reason       TEXT,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ct_period_type CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY')
    ),
    CONSTRAINT chk_p039_ct_trend CHECK (
        trend_direction IN ('IMPROVING', 'STABLE', 'DEGRADING', 'VOLATILE')
    ),
    CONSTRAINT chk_p039_ct_energy CHECK (
        actual_energy >= 0 AND expected_energy >= 0
    ),
    CONSTRAINT uq_p039_ct_enpi_baseline_period UNIQUE (enpi_id, baseline_id, period_type, period_date)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ct_enpi            ON pack039_energy_monitoring.em_cusum_tracking(enpi_id);
CREATE INDEX idx_p039_ct_baseline        ON pack039_energy_monitoring.em_cusum_tracking(baseline_id);
CREATE INDEX idx_p039_ct_tenant          ON pack039_energy_monitoring.em_cusum_tracking(tenant_id);
CREATE INDEX idx_p039_ct_period_type     ON pack039_energy_monitoring.em_cusum_tracking(period_type);
CREATE INDEX idx_p039_ct_period_date     ON pack039_energy_monitoring.em_cusum_tracking(period_date DESC);
CREATE INDEX idx_p039_ct_trend           ON pack039_energy_monitoring.em_cusum_tracking(trend_direction);
CREATE INDEX idx_p039_ct_ooc             ON pack039_energy_monitoring.em_cusum_tracking(is_out_of_control) WHERE is_out_of_control = true;
CREATE INDEX idx_p039_ct_change          ON pack039_energy_monitoring.em_cusum_tracking(trend_change_detected) WHERE trend_change_detected = true;
CREATE INDEX idx_p039_ct_created         ON pack039_energy_monitoring.em_cusum_tracking(created_at DESC);

-- Composite: monthly CUSUM by EnPI for charting
CREATE INDEX idx_p039_ct_enpi_monthly    ON pack039_energy_monitoring.em_cusum_tracking(enpi_id, period_date DESC)
    WHERE period_type = 'MONTHLY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ct_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_cusum_tracking
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_regression_models
-- =============================================================================
-- Statistical regression models for energy-variable relationships used
-- in EnPI calculation and baseline adjustment. Models capture the
-- mathematical relationship between energy consumption and relevant
-- variables (temperature, production, occupancy) per ISO 50006. Supports
-- linear, multivariate, and polynomial regression with goodness-of-fit
-- metrics and validation statistics.

CREATE TABLE pack039_energy_monitoring.em_regression_models (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    enpi_id                 UUID            REFERENCES pack039_energy_monitoring.em_enpi_definitions(id),
    baseline_id             UUID            REFERENCES pack039_energy_monitoring.em_energy_baselines(id),
    model_name              VARCHAR(255)    NOT NULL,
    model_code              VARCHAR(50)     NOT NULL,
    model_type              VARCHAR(30)     NOT NULL DEFAULT 'LINEAR',
    dependent_variable      VARCHAR(100)    NOT NULL DEFAULT 'energy_consumption',
    dependent_unit          VARCHAR(30)     NOT NULL DEFAULT 'kWh',
    independent_variables   JSONB           NOT NULL DEFAULT '[]',
    equation_text           TEXT            NOT NULL,
    intercept               NUMERIC(18,6)   NOT NULL,
    coefficients            JSONB           NOT NULL DEFAULT '{}',
    r_squared               NUMERIC(7,5)    NOT NULL,
    adjusted_r_squared      NUMERIC(7,5),
    rmse                    NUMERIC(15,3),
    mae                     NUMERIC(15,3),
    mape_pct                NUMERIC(8,4),
    cv_rmse_pct             NUMERIC(8,4),
    f_statistic             NUMERIC(12,4),
    p_value                 NUMERIC(10,8),
    degrees_of_freedom      INTEGER,
    sample_size             INTEGER         NOT NULL,
    training_period_start   DATE            NOT NULL,
    training_period_end     DATE            NOT NULL,
    validation_r_squared    NUMERIC(7,5),
    validation_rmse         NUMERIC(15,3),
    validation_mape_pct     NUMERIC(8,4),
    cross_validation_folds  INTEGER,
    residual_normality_test VARCHAR(10),
    residual_autocorrelation NUMERIC(5,4),
    durbin_watson           NUMERIC(5,4),
    variable_ranges         JSONB           DEFAULT '{}',
    model_assumptions       JSONB           DEFAULT '[]',
    outliers_removed        INTEGER         DEFAULT 0,
    model_status            VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    is_current              BOOLEAN         NOT NULL DEFAULT true,
    version                 INTEGER         NOT NULL DEFAULT 1,
    previous_model_id       UUID,
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_rm_type CHECK (
        model_type IN (
            'LINEAR', 'MULTIVARIATE', 'POLYNOMIAL', 'CHANGE_POINT',
            'PIECEWISE', 'EXPONENTIAL', 'LOGARITHMIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_rm_r_squared CHECK (
        r_squared >= 0 AND r_squared <= 1
    ),
    CONSTRAINT chk_p039_rm_adj_r_squared CHECK (
        adjusted_r_squared IS NULL OR (adjusted_r_squared >= -1 AND adjusted_r_squared <= 1)
    ),
    CONSTRAINT chk_p039_rm_mape CHECK (
        mape_pct IS NULL OR mape_pct >= 0
    ),
    CONSTRAINT chk_p039_rm_cv_rmse CHECK (
        cv_rmse_pct IS NULL OR cv_rmse_pct >= 0
    ),
    CONSTRAINT chk_p039_rm_sample CHECK (
        sample_size >= 1
    ),
    CONSTRAINT chk_p039_rm_normality CHECK (
        residual_normality_test IS NULL OR residual_normality_test IN ('PASS', 'FAIL', 'MARGINAL')
    ),
    CONSTRAINT chk_p039_rm_dw CHECK (
        durbin_watson IS NULL OR (durbin_watson >= 0 AND durbin_watson <= 4)
    ),
    CONSTRAINT chk_p039_rm_status CHECK (
        model_status IN ('DRAFT', 'VALIDATED', 'APPROVED', 'REJECTED', 'SUPERSEDED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p039_rm_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p039_rm_dates CHECK (
        training_period_start <= training_period_end
    ),
    CONSTRAINT uq_p039_rm_tenant_code_version UNIQUE (tenant_id, model_code, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_rm_tenant          ON pack039_energy_monitoring.em_regression_models(tenant_id);
CREATE INDEX idx_p039_rm_facility        ON pack039_energy_monitoring.em_regression_models(facility_id);
CREATE INDEX idx_p039_rm_enpi            ON pack039_energy_monitoring.em_regression_models(enpi_id);
CREATE INDEX idx_p039_rm_baseline        ON pack039_energy_monitoring.em_regression_models(baseline_id);
CREATE INDEX idx_p039_rm_code            ON pack039_energy_monitoring.em_regression_models(model_code);
CREATE INDEX idx_p039_rm_type            ON pack039_energy_monitoring.em_regression_models(model_type);
CREATE INDEX idx_p039_rm_r_squared       ON pack039_energy_monitoring.em_regression_models(r_squared DESC);
CREATE INDEX idx_p039_rm_status          ON pack039_energy_monitoring.em_regression_models(model_status);
CREATE INDEX idx_p039_rm_current         ON pack039_energy_monitoring.em_regression_models(is_current) WHERE is_current = true;
CREATE INDEX idx_p039_rm_created         ON pack039_energy_monitoring.em_regression_models(created_at DESC);
CREATE INDEX idx_p039_rm_coefficients    ON pack039_energy_monitoring.em_regression_models USING GIN(coefficients);
CREATE INDEX idx_p039_rm_variables       ON pack039_energy_monitoring.em_regression_models USING GIN(independent_variables);

-- Composite: current approved models by facility
CREATE INDEX idx_p039_rm_fac_approved    ON pack039_energy_monitoring.em_regression_models(facility_id, model_type)
    WHERE is_current = true AND model_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_rm_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_regression_models
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_enpi_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_enpi_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_energy_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_cusum_tracking ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_regression_models ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_ed_tenant_isolation
    ON pack039_energy_monitoring.em_enpi_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ed_service_bypass
    ON pack039_energy_monitoring.em_enpi_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ev_tenant_isolation
    ON pack039_energy_monitoring.em_enpi_values
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ev_service_bypass
    ON pack039_energy_monitoring.em_enpi_values
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_eb_tenant_isolation
    ON pack039_energy_monitoring.em_energy_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_eb_service_bypass
    ON pack039_energy_monitoring.em_energy_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ct_tenant_isolation
    ON pack039_energy_monitoring.em_cusum_tracking
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ct_service_bypass
    ON pack039_energy_monitoring.em_cusum_tracking
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_rm_tenant_isolation
    ON pack039_energy_monitoring.em_regression_models
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_rm_service_bypass
    ON pack039_energy_monitoring.em_regression_models
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_enpi_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_enpi_values TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_energy_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_cusum_tracking TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_regression_models TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_enpi_definitions IS
    'Energy Performance Indicator definitions per ISO 50006 with boundary, normalization variables, targets, and calculation configuration.';
COMMENT ON TABLE pack039_energy_monitoring.em_enpi_values IS
    'Calculated EnPI values over time with energy consumption, normalizing variables, baseline comparison, and improvement tracking.';
COMMENT ON TABLE pack039_energy_monitoring.em_energy_baselines IS
    'Energy baselines per ISO 50001/50006 with period definitions, adjustment methodology, regression parameters, and approval workflow.';
COMMENT ON TABLE pack039_energy_monitoring.em_cusum_tracking IS
    'CUSUM (Cumulative Sum) tracking for energy performance monitoring showing cumulative difference from baseline over time.';
COMMENT ON TABLE pack039_energy_monitoring.em_regression_models IS
    'Statistical regression models for energy-variable relationships with goodness-of-fit metrics and cross-validation results.';

COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_definitions.enpi_type IS 'EnPI methodology: RATIO (simple), REGRESSION (model-based), ABSOLUTE, INDEXED, SPECIFIC, COMPOSITE.';
COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_definitions.significant_energy_use IS 'ISO 50001 SEU flag: whether this EnPI tracks a Significant Energy Use.';
COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_definitions.relevant_variables IS 'JSON array of variables affecting energy performance: [{name, unit, source, importance}].';
COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_definitions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_values.improvement_pct IS 'Percentage improvement compared to baseline EnPI. Negative values indicate degradation.';
COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_values.cusum_value IS 'Running cumulative sum value at this period for CUSUM chart rendering.';
COMMENT ON COLUMN pack039_energy_monitoring.em_enpi_values.performance_status IS 'Performance assessment: ON_TARGET, ABOVE_TARGET, BELOW_TARGET, WARNING, CRITICAL.';

COMMENT ON COLUMN pack039_energy_monitoring.em_energy_baselines.static_factors IS 'ISO 50006 static factors: conditions affecting energy that do not routinely change. JSON array.';
COMMENT ON COLUMN pack039_energy_monitoring.em_energy_baselines.non_routine_adjustments IS 'ISO 50006 non-routine adjustments: one-time changes affecting baseline comparison. JSON array.';

COMMENT ON COLUMN pack039_energy_monitoring.em_cusum_tracking.cumulative_sum IS 'Running cumulative sum of (actual - expected) energy. Negative = savings, Positive = excess.';
COMMENT ON COLUMN pack039_energy_monitoring.em_cusum_tracking.is_out_of_control IS 'Whether CUSUM exceeds control limits, indicating statistically significant performance change.';

COMMENT ON COLUMN pack039_energy_monitoring.em_regression_models.cv_rmse_pct IS 'Coefficient of Variation of RMSE (CV-RMSE%). ASHRAE Guideline 14 requires <25% for monthly models.';
COMMENT ON COLUMN pack039_energy_monitoring.em_regression_models.durbin_watson IS 'Durbin-Watson statistic for residual autocorrelation (0-4). Values near 2 indicate no autocorrelation.';
