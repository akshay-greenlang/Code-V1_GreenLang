-- =============================================================================
-- V260: PACK-034 ISO 50001 Energy Management System - CUSUM Tables
-- =============================================================================
-- Pack:         PACK-034 (ISO 50001 Energy Management System Pack)
-- Migration:    005 of 010
-- Date:         March 2026
--
-- Creates Cumulative Sum (CUSUM) monitoring tables for continuous energy
-- performance tracking. CUSUM charts detect small sustained shifts in energy
-- consumption that periodic comparisons may miss, per IPMVP guidelines.
--
-- Tables (3):
--   1. pack034_iso50001.cusum_monitors
--   2. pack034_iso50001.cusum_data_points
--   3. pack034_iso50001.cusum_alerts
--
-- Previous: V259__pack034_iso50001_004.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack034_iso50001.cusum_monitors
-- =============================================================================
-- CUSUM monitoring configurations linking a baseline model to an EnPI for
-- continuous performance tracking with configurable alert thresholds.

CREATE TABLE pack034_iso50001.cusum_monitors (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    enms_id                     UUID            NOT NULL REFERENCES pack034_iso50001.energy_management_systems(id) ON DELETE CASCADE,
    monitor_name                VARCHAR(500)    NOT NULL,
    baseline_id                 UUID            REFERENCES pack034_iso50001.energy_baselines(id) ON DELETE SET NULL,
    enpi_id                     UUID            REFERENCES pack034_iso50001.energy_performance_indicators(id) ON DELETE SET NULL,
    alert_threshold             DECIMAL(12,4)   NOT NULL,
    monitoring_interval         VARCHAR(20)     NOT NULL DEFAULT 'monthly',
    status                      VARCHAR(20)     NOT NULL DEFAULT 'active',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_cm_interval CHECK (
        monitoring_interval IN ('daily', 'weekly', 'monthly')
    ),
    CONSTRAINT chk_p034_cm_status CHECK (
        status IN ('active', 'paused', 'completed')
    ),
    CONSTRAINT chk_p034_cm_threshold CHECK (
        alert_threshold > 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_cm_enms            ON pack034_iso50001.cusum_monitors(enms_id);
CREATE INDEX idx_p034_cm_baseline        ON pack034_iso50001.cusum_monitors(baseline_id);
CREATE INDEX idx_p034_cm_enpi            ON pack034_iso50001.cusum_monitors(enpi_id);
CREATE INDEX idx_p034_cm_status          ON pack034_iso50001.cusum_monitors(status);
CREATE INDEX idx_p034_cm_interval        ON pack034_iso50001.cusum_monitors(monitoring_interval);
CREATE INDEX idx_p034_cm_created         ON pack034_iso50001.cusum_monitors(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_cm_updated
    BEFORE UPDATE ON pack034_iso50001.cusum_monitors
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack034_iso50001.cusum_data_points
-- =============================================================================
-- Individual CUSUM data points recording actual vs. expected consumption,
-- running cumulative sum, and control limits for each monitoring period.

CREATE TABLE pack034_iso50001.cusum_data_points (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    monitor_id                  UUID            NOT NULL REFERENCES pack034_iso50001.cusum_monitors(id) ON DELETE CASCADE,
    period_date                 DATE            NOT NULL,
    actual_consumption          DECIMAL(18,4)   NOT NULL,
    expected_consumption        DECIMAL(18,4)   NOT NULL,
    difference                  DECIMAL(18,4)   NOT NULL,
    cumulative_sum              DECIMAL(18,4)   NOT NULL,
    upper_limit                 DECIMAL(18,4),
    lower_limit                 DECIMAL(18,4),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_cdp_actual CHECK (
        actual_consumption >= 0
    ),
    CONSTRAINT chk_p034_cdp_expected CHECK (
        expected_consumption >= 0
    ),
    CONSTRAINT chk_p034_cdp_limits CHECK (
        lower_limit IS NULL OR upper_limit IS NULL OR lower_limit <= upper_limit
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_cdp_monitor        ON pack034_iso50001.cusum_data_points(monitor_id);
CREATE INDEX idx_p034_cdp_date           ON pack034_iso50001.cusum_data_points(period_date DESC);
CREATE INDEX idx_p034_cdp_cusum          ON pack034_iso50001.cusum_data_points(cumulative_sum);
CREATE INDEX idx_p034_cdp_created        ON pack034_iso50001.cusum_data_points(created_at DESC);
CREATE UNIQUE INDEX idx_p034_cdp_monitor_date
    ON pack034_iso50001.cusum_data_points(monitor_id, period_date);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_cdp_updated
    BEFORE UPDATE ON pack034_iso50001.cusum_data_points
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack034_iso50001.cusum_alerts
-- =============================================================================
-- Alerts generated when CUSUM values exceed control limits, indicating
-- performance degradation, improvement, or trend changes requiring
-- investigation and root cause analysis.

CREATE TABLE pack034_iso50001.cusum_alerts (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    monitor_id                  UUID            NOT NULL REFERENCES pack034_iso50001.cusum_monitors(id) ON DELETE CASCADE,
    alert_date                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    alert_type                  VARCHAR(30)     NOT NULL,
    cumulative_value            DECIMAL(18,4)   NOT NULL,
    threshold_exceeded          DECIMAL(18,4)   NOT NULL,
    acknowledged                BOOLEAN         NOT NULL DEFAULT FALSE,
    acknowledged_by             UUID,
    root_cause                  TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p034_ca_type CHECK (
        alert_type IN ('performance_degradation', 'performance_improvement', 'trend_change')
    ),
    CONSTRAINT chk_p034_ca_ack CHECK (
        acknowledged = FALSE OR acknowledged_by IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p034_ca_monitor         ON pack034_iso50001.cusum_alerts(monitor_id);
CREATE INDEX idx_p034_ca_date            ON pack034_iso50001.cusum_alerts(alert_date DESC);
CREATE INDEX idx_p034_ca_type            ON pack034_iso50001.cusum_alerts(alert_type);
CREATE INDEX idx_p034_ca_ack             ON pack034_iso50001.cusum_alerts(acknowledged);
CREATE INDEX idx_p034_ca_created         ON pack034_iso50001.cusum_alerts(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p034_ca_updated
    BEFORE UPDATE ON pack034_iso50001.cusum_alerts
    FOR EACH ROW EXECUTE FUNCTION pack034_iso50001.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack034_iso50001.cusum_monitors ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.cusum_data_points ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack034_iso50001.cusum_alerts ENABLE ROW LEVEL SECURITY;

CREATE POLICY p034_cm_tenant_isolation
    ON pack034_iso50001.cusum_monitors
    USING (enms_id IN (
        SELECT id FROM pack034_iso50001.energy_management_systems
        WHERE organization_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p034_cm_service_bypass
    ON pack034_iso50001.cusum_monitors
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_cdp_tenant_isolation
    ON pack034_iso50001.cusum_data_points
    USING (monitor_id IN (
        SELECT id FROM pack034_iso50001.cusum_monitors
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_cdp_service_bypass
    ON pack034_iso50001.cusum_data_points
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p034_ca_tenant_isolation
    ON pack034_iso50001.cusum_alerts
    USING (monitor_id IN (
        SELECT id FROM pack034_iso50001.cusum_monitors
        WHERE enms_id IN (
            SELECT id FROM pack034_iso50001.energy_management_systems
            WHERE organization_id = current_setting('app.current_tenant')::UUID
        )
    ));
CREATE POLICY p034_ca_service_bypass
    ON pack034_iso50001.cusum_alerts
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.cusum_monitors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.cusum_data_points TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack034_iso50001.cusum_alerts TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack034_iso50001.cusum_monitors IS
    'CUSUM monitoring configurations for continuous energy performance tracking with configurable alert thresholds and intervals.';

COMMENT ON TABLE pack034_iso50001.cusum_data_points IS
    'Individual CUSUM data points with actual vs. expected consumption, running cumulative sum, and control limits.';

COMMENT ON TABLE pack034_iso50001.cusum_alerts IS
    'Alerts generated when CUSUM values exceed control limits, requiring investigation and root cause analysis.';

COMMENT ON COLUMN pack034_iso50001.cusum_monitors.alert_threshold IS
    'CUSUM threshold value that triggers an alert when exceeded (in energy units, e.g., kWh).';
COMMENT ON COLUMN pack034_iso50001.cusum_monitors.monitoring_interval IS
    'Frequency of CUSUM calculations: daily, weekly, or monthly.';
COMMENT ON COLUMN pack034_iso50001.cusum_data_points.difference IS
    'Actual minus expected consumption for this period. Positive = over-consuming.';
COMMENT ON COLUMN pack034_iso50001.cusum_data_points.cumulative_sum IS
    'Running cumulative sum of differences. Rising trend = sustained over-consumption.';
COMMENT ON COLUMN pack034_iso50001.cusum_data_points.upper_limit IS
    'Upper control limit for the CUSUM chart. Exceedance triggers degradation alert.';
COMMENT ON COLUMN pack034_iso50001.cusum_data_points.lower_limit IS
    'Lower control limit for the CUSUM chart. Below this indicates sustained improvement.';
COMMENT ON COLUMN pack034_iso50001.cusum_alerts.alert_type IS
    'Type of alert: performance_degradation (over-consuming), performance_improvement (under-consuming), trend_change (shift detected).';
COMMENT ON COLUMN pack034_iso50001.cusum_alerts.root_cause IS
    'Root cause analysis text documenting why the alert was triggered.';
