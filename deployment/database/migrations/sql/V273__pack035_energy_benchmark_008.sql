-- =============================================================================
-- V273: PACK-035 Energy Benchmark Pack - Trend Analysis Tables
-- =============================================================================
-- Pack:         PACK-035 (Energy Benchmark Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Time-series trend analysis for energy performance monitoring. Includes
-- rolling EUI data points, Statistical Process Control (SPC) charts,
-- performance alerts, and step-change detection for identifying
-- significant shifts in energy consumption patterns.
--
-- Tables (4):
--   1. pack035_energy_benchmark.trend_data_points
--   2. pack035_energy_benchmark.spc_control_charts
--   3. pack035_energy_benchmark.performance_alerts
--   4. pack035_energy_benchmark.step_changes
--
-- Previous: V272__pack035_energy_benchmark_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack035_energy_benchmark.trend_data_points
-- =============================================================================
-- Time-series data points for energy performance trend analysis.
-- Stores rolling 12-month EUI, weather-normalised EUI, CUSUM
-- (cumulative sum), and EWMA (exponentially weighted moving average)
-- values for each facility at periodic intervals.

CREATE TABLE pack035_energy_benchmark.trend_data_points (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    data_date               DATE            NOT NULL,
    period_type             VARCHAR(20)     DEFAULT 'MONTHLY',
    -- EUI values
    rolling_12m_eui         DECIMAL(10, 4),
    weather_normalised_eui  DECIMAL(10, 4),
    site_eui                DECIMAL(10, 4),
    -- Statistical indicators
    cusum_value             DECIMAL(14, 4),
    cusum_upper             DECIMAL(14, 4),
    cusum_lower             DECIMAL(14, 4),
    ewma_value              DECIMAL(10, 4),
    ewma_lambda             DECIMAL(4, 3)   DEFAULT 0.200,
    -- Consumption data
    period_energy_kwh       DECIMAL(14, 4),
    expected_energy_kwh     DECIMAL(14, 4),
    residual_kwh            DECIMAL(14, 4),
    -- Alert flags
    is_alert                BOOLEAN         DEFAULT false,
    alert_type              VARCHAR(30),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_tdp_period CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY')
    ),
    CONSTRAINT chk_p035_tdp_rolling CHECK (
        rolling_12m_eui IS NULL OR rolling_12m_eui >= 0
    ),
    CONSTRAINT chk_p035_tdp_weather CHECK (
        weather_normalised_eui IS NULL OR weather_normalised_eui >= 0
    ),
    CONSTRAINT chk_p035_tdp_site CHECK (
        site_eui IS NULL OR site_eui >= 0
    ),
    CONSTRAINT chk_p035_tdp_lambda CHECK (
        ewma_lambda IS NULL OR (ewma_lambda > 0 AND ewma_lambda <= 1)
    ),
    CONSTRAINT chk_p035_tdp_alert_type CHECK (
        alert_type IS NULL OR alert_type IN (
            'CUSUM_BREACH', 'EWMA_BREACH', 'SPIKE', 'TREND_SHIFT',
            'SEASONAL_ANOMALY', 'DATA_GAP', 'REGRESSION_DRIFT'
        )
    ),
    CONSTRAINT uq_p035_tdp_facility_date UNIQUE (facility_id, data_date, period_type)
);

-- Indexes
CREATE INDEX idx_p035_tdp_facility       ON pack035_energy_benchmark.trend_data_points(facility_id);
CREATE INDEX idx_p035_tdp_tenant         ON pack035_energy_benchmark.trend_data_points(tenant_id);
CREATE INDEX idx_p035_tdp_date           ON pack035_energy_benchmark.trend_data_points(data_date DESC);
CREATE INDEX idx_p035_tdp_period         ON pack035_energy_benchmark.trend_data_points(period_type);
CREATE INDEX idx_p035_tdp_alert          ON pack035_energy_benchmark.trend_data_points(is_alert);
CREATE INDEX idx_p035_tdp_fac_date       ON pack035_energy_benchmark.trend_data_points(facility_id, data_date DESC);
CREATE INDEX idx_p035_tdp_cusum          ON pack035_energy_benchmark.trend_data_points(cusum_value);

-- =============================================================================
-- Table 2: pack035_energy_benchmark.spc_control_charts
-- =============================================================================
-- Statistical Process Control chart parameters per facility. Defines
-- control limits (UCL/LCL) for monitoring energy consumption against
-- a stable baseline period. Supports Shewhart (I-MR), CUSUM, and
-- EWMA chart types.

CREATE TABLE pack035_energy_benchmark.spc_control_charts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    chart_type              VARCHAR(20)     NOT NULL,
    chart_name              VARCHAR(255),
    metric_name             VARCHAR(50)     NOT NULL DEFAULT 'EUI',
    -- Control limits
    center_line             DECIMAL(10, 4)  NOT NULL,
    ucl_warning             DECIMAL(10, 4),
    ucl_control             DECIMAL(10, 4)  NOT NULL,
    lcl_warning             DECIMAL(10, 4),
    lcl_control             DECIMAL(10, 4)  NOT NULL,
    sigma                   DECIMAL(10, 4)  NOT NULL,
    -- CUSUM parameters
    cusum_k                 DECIMAL(8, 4),
    cusum_h                 DECIMAL(8, 4),
    -- EWMA parameters
    ewma_lambda             DECIMAL(4, 3),
    ewma_l                  DECIMAL(6, 3),
    -- Baseline
    baseline_start          DATE            NOT NULL,
    baseline_end            DATE            NOT NULL,
    n_baseline_points       INTEGER,
    -- Status
    is_active               BOOLEAN         DEFAULT true,
    last_recalculated       TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_spc_type CHECK (
        chart_type IN ('SHEWHART', 'CUSUM', 'EWMA', 'I_MR', 'XBAR_R')
    ),
    CONSTRAINT chk_p035_spc_sigma CHECK (
        sigma > 0
    ),
    CONSTRAINT chk_p035_spc_limits CHECK (
        lcl_control <= center_line AND center_line <= ucl_control
    ),
    CONSTRAINT chk_p035_spc_warning CHECK (
        ucl_warning IS NULL OR lcl_warning IS NULL
        OR (lcl_warning <= center_line AND center_line <= ucl_warning)
    ),
    CONSTRAINT chk_p035_spc_baseline CHECK (
        baseline_start < baseline_end
    )
);

-- Indexes
CREATE INDEX idx_p035_spc_facility       ON pack035_energy_benchmark.spc_control_charts(facility_id);
CREATE INDEX idx_p035_spc_tenant         ON pack035_energy_benchmark.spc_control_charts(tenant_id);
CREATE INDEX idx_p035_spc_type           ON pack035_energy_benchmark.spc_control_charts(chart_type);
CREATE INDEX idx_p035_spc_active         ON pack035_energy_benchmark.spc_control_charts(is_active);
CREATE INDEX idx_p035_spc_fac_type       ON pack035_energy_benchmark.spc_control_charts(facility_id, chart_type);

-- =============================================================================
-- Table 3: pack035_energy_benchmark.performance_alerts
-- =============================================================================
-- Alert records generated when energy performance deviates from expected
-- patterns. Includes alert type, severity, deviation metrics, and
-- acknowledgement tracking for operational response.

CREATE TABLE pack035_energy_benchmark.performance_alerts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    alert_date              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    alert_type              VARCHAR(50)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
    metric_name             VARCHAR(50)     NOT NULL,
    expected_value          DECIMAL(14, 4),
    actual_value            DECIMAL(14, 4),
    deviation_pct           DECIMAL(8, 4),
    deviation_sigma         DECIMAL(8, 4),
    message                 TEXT            NOT NULL,
    recommendation          TEXT,
    -- Alert source
    source_chart_id         UUID            REFERENCES pack035_energy_benchmark.spc_control_charts(id) ON DELETE SET NULL,
    source_model_id         UUID            REFERENCES pack035_energy_benchmark.regression_models(id) ON DELETE SET NULL,
    -- Status tracking
    acknowledged            BOOLEAN         DEFAULT false,
    acknowledged_by         UUID,
    acknowledged_at         TIMESTAMPTZ,
    resolved                BOOLEAN         DEFAULT false,
    resolved_by             UUID,
    resolved_at             TIMESTAMPTZ,
    resolution_notes        TEXT,
    -- Metadata
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_pa_type CHECK (
        alert_type IN (
            'EUI_SPIKE', 'EUI_TREND_UP', 'EUI_TREND_DOWN', 'CUSUM_BREACH',
            'EWMA_BREACH', 'CONTROL_LIMIT_BREACH', 'WARNING_LIMIT_BREACH',
            'SEASONAL_ANOMALY', 'BASELOAD_SHIFT', 'WEATHER_DEVIATION',
            'DATA_QUALITY', 'DATA_GAP', 'RATING_EXPIRY', 'STRANDING_RISK',
            'TARGET_DEVIATION', 'BENCHMARK_REGRESSION'
        )
    ),
    CONSTRAINT chk_p035_pa_severity CHECK (
        severity IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')
    ),
    CONSTRAINT chk_p035_pa_ack CHECK (
        (acknowledged = false AND acknowledged_by IS NULL AND acknowledged_at IS NULL) OR
        (acknowledged = true)
    ),
    CONSTRAINT chk_p035_pa_resolved CHECK (
        (resolved = false AND resolved_by IS NULL AND resolved_at IS NULL) OR
        (resolved = true)
    )
);

-- Indexes
CREATE INDEX idx_p035_pa_facility        ON pack035_energy_benchmark.performance_alerts(facility_id);
CREATE INDEX idx_p035_pa_tenant          ON pack035_energy_benchmark.performance_alerts(tenant_id);
CREATE INDEX idx_p035_pa_date            ON pack035_energy_benchmark.performance_alerts(alert_date DESC);
CREATE INDEX idx_p035_pa_type            ON pack035_energy_benchmark.performance_alerts(alert_type);
CREATE INDEX idx_p035_pa_severity        ON pack035_energy_benchmark.performance_alerts(severity);
CREATE INDEX idx_p035_pa_ack             ON pack035_energy_benchmark.performance_alerts(acknowledged);
CREATE INDEX idx_p035_pa_resolved        ON pack035_energy_benchmark.performance_alerts(resolved);
CREATE INDEX idx_p035_pa_fac_unack       ON pack035_energy_benchmark.performance_alerts(facility_id)
    WHERE acknowledged = false;

-- =============================================================================
-- Table 4: pack035_energy_benchmark.step_changes
-- =============================================================================
-- Detected step changes (level shifts) in energy consumption that
-- indicate significant events such as equipment changes, operational
-- modifications, or building envelope upgrades.

CREATE TABLE pack035_energy_benchmark.step_changes (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack035_energy_benchmark.facility_profiles(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    detected_date           DATE            NOT NULL,
    effective_date          DATE,
    change_type             VARCHAR(30)     NOT NULL,
    direction               VARCHAR(10)     NOT NULL,
    -- Magnitude
    magnitude_kwh_m2        DECIMAL(10, 4),
    magnitude_pct           DECIMAL(8, 4),
    magnitude_kwh_annual    DECIMAL(14, 4),
    cost_impact_eur         DECIMAL(14, 4),
    co2_impact_kg           DECIMAL(14, 4),
    -- Context
    probable_cause          TEXT,
    cause_category          VARCHAR(50),
    confirmed               BOOLEAN         DEFAULT false,
    confirmed_by            UUID,
    confirmed_at            TIMESTAMPTZ,
    linked_measure_id       UUID,
    notes                   TEXT,
    -- Statistical
    pre_change_mean         DECIMAL(10, 4),
    post_change_mean        DECIMAL(10, 4),
    statistical_significance DECIMAL(6, 4),
    detection_method        VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p035_sc_type CHECK (
        change_type IN (
            'EQUIPMENT_CHANGE', 'OPERATIONAL_CHANGE', 'OCCUPANCY_CHANGE',
            'BUILDING_MODIFICATION', 'METERING_CHANGE', 'TARIFF_CHANGE',
            'WEATHER_EVENT', 'ECM_IMPLEMENTATION', 'UNKNOWN', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_sc_direction CHECK (
        direction IN ('INCREASE', 'DECREASE')
    ),
    CONSTRAINT chk_p035_sc_cause_cat CHECK (
        cause_category IS NULL OR cause_category IN (
            'HVAC', 'LIGHTING', 'PLUG_LOADS', 'PROCESS', 'ENVELOPE',
            'CONTROLS', 'OCCUPANCY', 'WEATHER', 'METERING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p035_sc_significance CHECK (
        statistical_significance IS NULL OR (statistical_significance >= 0 AND statistical_significance <= 1)
    ),
    CONSTRAINT chk_p035_sc_detection CHECK (
        detection_method IS NULL OR detection_method IN (
            'CUSUM', 'PETTITT', 'BUISHAND', 'SEQUENTIAL_T_TEST',
            'BAYESIAN_CHANGE_POINT', 'MANUAL', 'OTHER'
        )
    )
);

-- Indexes
CREATE INDEX idx_p035_sc_facility        ON pack035_energy_benchmark.step_changes(facility_id);
CREATE INDEX idx_p035_sc_tenant          ON pack035_energy_benchmark.step_changes(tenant_id);
CREATE INDEX idx_p035_sc_detected        ON pack035_energy_benchmark.step_changes(detected_date DESC);
CREATE INDEX idx_p035_sc_type            ON pack035_energy_benchmark.step_changes(change_type);
CREATE INDEX idx_p035_sc_direction       ON pack035_energy_benchmark.step_changes(direction);
CREATE INDEX idx_p035_sc_confirmed       ON pack035_energy_benchmark.step_changes(confirmed);
CREATE INDEX idx_p035_sc_cause           ON pack035_energy_benchmark.step_changes(cause_category);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack035_energy_benchmark.trend_data_points ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.spc_control_charts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.performance_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack035_energy_benchmark.step_changes ENABLE ROW LEVEL SECURITY;

CREATE POLICY p035_tdp_tenant_isolation ON pack035_energy_benchmark.trend_data_points
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_tdp_service_bypass ON pack035_energy_benchmark.trend_data_points
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_spc_tenant_isolation ON pack035_energy_benchmark.spc_control_charts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_spc_service_bypass ON pack035_energy_benchmark.spc_control_charts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_pa_tenant_isolation ON pack035_energy_benchmark.performance_alerts
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_pa_service_bypass ON pack035_energy_benchmark.performance_alerts
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p035_sc_tenant_isolation ON pack035_energy_benchmark.step_changes
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p035_sc_service_bypass ON pack035_energy_benchmark.step_changes
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.trend_data_points TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.spc_control_charts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.performance_alerts TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack035_energy_benchmark.step_changes TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack035_energy_benchmark.trend_data_points IS
    'Time-series data points for energy performance trend analysis: rolling EUI, CUSUM, EWMA indicators.';
COMMENT ON TABLE pack035_energy_benchmark.spc_control_charts IS
    'SPC control chart parameters per facility: Shewhart, CUSUM, EWMA charts with control limits.';
COMMENT ON TABLE pack035_energy_benchmark.performance_alerts IS
    'Alert records for energy performance deviations with severity, deviation metrics, and acknowledgement tracking.';
COMMENT ON TABLE pack035_energy_benchmark.step_changes IS
    'Detected step changes in energy consumption indicating equipment changes, operational modifications, or ECM impacts.';

COMMENT ON COLUMN pack035_energy_benchmark.trend_data_points.cusum_value IS
    'Cumulative Sum (CUSUM) value tracking cumulative deviation from baseline mean. Breach of threshold signals persistent shift.';
COMMENT ON COLUMN pack035_energy_benchmark.trend_data_points.ewma_value IS
    'Exponentially Weighted Moving Average value. Smoothed indicator with configurable lambda (0.2 typical).';
COMMENT ON COLUMN pack035_energy_benchmark.spc_control_charts.cusum_k IS
    'CUSUM allowance parameter k (typically 0.5 * sigma). Deviations less than k are not accumulated.';
COMMENT ON COLUMN pack035_energy_benchmark.spc_control_charts.cusum_h IS
    'CUSUM decision interval h (typically 4-5 * sigma). Alarm when CUSUM exceeds h.';
