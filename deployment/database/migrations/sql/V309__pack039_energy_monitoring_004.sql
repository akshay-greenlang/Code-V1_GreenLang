-- =============================================================================
-- V309: PACK-039 Energy Monitoring Pack - Anomaly Detection
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates anomaly detection tables for identifying unusual energy
-- consumption patterns. Includes detected anomalies with classification
-- and severity, configurable detection rules, baseline profiles for
-- comparison, investigation workflow records, and pre-computed anomaly
-- statistics for dashboard rendering and trending.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_anomalies
--   2. pack039_energy_monitoring.em_anomaly_rules
--   3. pack039_energy_monitoring.em_anomaly_baselines
--   4. pack039_energy_monitoring.em_investigation_records
--   5. pack039_energy_monitoring.em_anomaly_stats
--
-- Previous: V308__pack039_energy_monitoring_003.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_anomalies
-- =============================================================================
-- Detected energy consumption anomalies with classification, severity,
-- impact estimation, and lifecycle tracking. Each anomaly represents
-- a deviation from expected behavior that warrants investigation.
-- Anomalies progress through a lifecycle: DETECTED -> INVESTIGATING ->
-- CONFIRMED/FALSE_POSITIVE -> RESOLVED/CLOSED.

CREATE TABLE pack039_energy_monitoring.em_anomalies (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    anomaly_rule_id         UUID,
    baseline_id             UUID,
    anomaly_code            VARCHAR(50)     NOT NULL,
    anomaly_type            VARCHAR(50)     NOT NULL DEFAULT 'CONSUMPTION_SPIKE',
    anomaly_category        VARCHAR(50)     NOT NULL DEFAULT 'CONSUMPTION',
    severity                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    detection_method        VARCHAR(50)     NOT NULL DEFAULT 'THRESHOLD',
    anomaly_status          VARCHAR(30)     NOT NULL DEFAULT 'DETECTED',
    detected_at             TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    anomaly_start           TIMESTAMPTZ     NOT NULL,
    anomaly_end             TIMESTAMPTZ,
    duration_minutes        INTEGER,
    expected_value          NUMERIC(18,6),
    actual_value            NUMERIC(18,6)   NOT NULL,
    deviation_value         NUMERIC(18,6),
    deviation_pct           NUMERIC(10,4),
    confidence_score        NUMERIC(5,2)    NOT NULL DEFAULT 50.0,
    estimated_energy_impact_kwh NUMERIC(15,3),
    estimated_cost_impact   NUMERIC(12,2),
    cost_currency           VARCHAR(3)      DEFAULT 'USD',
    affected_intervals      INTEGER         DEFAULT 0,
    related_anomaly_ids     UUID[]          DEFAULT '{}',
    root_cause              VARCHAR(100),
    root_cause_category     VARCHAR(50),
    assigned_to             UUID,
    assigned_at             TIMESTAMPTZ,
    acknowledged_at         TIMESTAMPTZ,
    acknowledged_by         UUID,
    resolved_at             TIMESTAMPTZ,
    resolved_by             UUID,
    resolution_notes        TEXT,
    resolution_action       VARCHAR(50),
    is_false_positive       BOOLEAN         NOT NULL DEFAULT false,
    is_recurring            BOOLEAN         NOT NULL DEFAULT false,
    recurrence_pattern      VARCHAR(50),
    tags                    JSONB           DEFAULT '[]',
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_an_type CHECK (
        anomaly_type IN (
            'CONSUMPTION_SPIKE', 'CONSUMPTION_DROP', 'BASELOAD_SHIFT',
            'OFF_HOURS_USAGE', 'WEEKEND_ANOMALY', 'HOLIDAY_ANOMALY',
            'WEATHER_DEVIATION', 'OCCUPANCY_MISMATCH', 'EQUIPMENT_MALFUNCTION',
            'METER_FAULT', 'DEMAND_SPIKE', 'POWER_FACTOR_ANOMALY',
            'PHASE_IMBALANCE', 'HARMONIC_DISTORTION', 'VOLTAGE_ANOMALY',
            'SEASONAL_DEVIATION', 'TREND_BREAK', 'PATTERN_CHANGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_an_category CHECK (
        anomaly_category IN (
            'CONSUMPTION', 'DEMAND', 'POWER_QUALITY', 'EQUIPMENT',
            'METER', 'WEATHER', 'OCCUPANCY', 'BEHAVIORAL', 'SYSTEM'
        )
    ),
    CONSTRAINT chk_p039_an_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p039_an_detection CHECK (
        detection_method IN (
            'THRESHOLD', 'STATISTICAL', 'MACHINE_LEARNING', 'RULE_BASED',
            'PATTERN_MATCHING', 'REGRESSION', 'CUSUM', 'EWMA',
            'ISOLATION_FOREST', 'DBSCAN', 'MANUAL', 'COMPOSITE'
        )
    ),
    CONSTRAINT chk_p039_an_status CHECK (
        anomaly_status IN (
            'DETECTED', 'ACKNOWLEDGED', 'INVESTIGATING', 'CONFIRMED',
            'FALSE_POSITIVE', 'RESOLVED', 'CLOSED', 'DEFERRED', 'RECURRING'
        )
    ),
    CONSTRAINT chk_p039_an_confidence CHECK (
        confidence_score >= 0 AND confidence_score <= 100
    ),
    CONSTRAINT chk_p039_an_duration CHECK (
        duration_minutes IS NULL OR duration_minutes >= 0
    ),
    CONSTRAINT chk_p039_an_intervals CHECK (
        affected_intervals IS NULL OR affected_intervals >= 0
    ),
    CONSTRAINT chk_p039_an_root_cause_cat CHECK (
        root_cause_category IS NULL OR root_cause_category IN (
            'EQUIPMENT', 'OPERATIONAL', 'WEATHER', 'OCCUPANCY',
            'MAINTENANCE', 'METER', 'NETWORK', 'BEHAVIORAL',
            'PROCESS_CHANGE', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p039_an_resolution CHECK (
        resolution_action IS NULL OR resolution_action IN (
            'EQUIPMENT_REPAIR', 'SCHEDULE_ADJUSTMENT', 'CONTROL_TUNING',
            'METER_REPLACEMENT', 'PROCESS_CHANGE', 'BEHAVIOR_CHANGE',
            'ACCEPTED', 'NO_ACTION', 'MONITORING', 'ESCALATED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_an_recurrence CHECK (
        recurrence_pattern IS NULL OR recurrence_pattern IN (
            'DAILY', 'WEEKLY', 'MONTHLY', 'SEASONAL', 'RANDOM', 'EVENT_DRIVEN'
        )
    ),
    CONSTRAINT chk_p039_an_dates CHECK (
        anomaly_end IS NULL OR anomaly_start <= anomaly_end
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_an_meter           ON pack039_energy_monitoring.em_anomalies(meter_id);
CREATE INDEX idx_p039_an_tenant          ON pack039_energy_monitoring.em_anomalies(tenant_id);
CREATE INDEX idx_p039_an_rule            ON pack039_energy_monitoring.em_anomalies(anomaly_rule_id);
CREATE INDEX idx_p039_an_baseline        ON pack039_energy_monitoring.em_anomalies(baseline_id);
CREATE INDEX idx_p039_an_code            ON pack039_energy_monitoring.em_anomalies(anomaly_code);
CREATE INDEX idx_p039_an_type            ON pack039_energy_monitoring.em_anomalies(anomaly_type);
CREATE INDEX idx_p039_an_category        ON pack039_energy_monitoring.em_anomalies(anomaly_category);
CREATE INDEX idx_p039_an_severity        ON pack039_energy_monitoring.em_anomalies(severity);
CREATE INDEX idx_p039_an_status          ON pack039_energy_monitoring.em_anomalies(anomaly_status);
CREATE INDEX idx_p039_an_detected        ON pack039_energy_monitoring.em_anomalies(detected_at DESC);
CREATE INDEX idx_p039_an_start           ON pack039_energy_monitoring.em_anomalies(anomaly_start DESC);
CREATE INDEX idx_p039_an_confidence      ON pack039_energy_monitoring.em_anomalies(confidence_score DESC);
CREATE INDEX idx_p039_an_assigned        ON pack039_energy_monitoring.em_anomalies(assigned_to) WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_p039_an_false_pos       ON pack039_energy_monitoring.em_anomalies(is_false_positive) WHERE is_false_positive = true;
CREATE INDEX idx_p039_an_recurring       ON pack039_energy_monitoring.em_anomalies(is_recurring) WHERE is_recurring = true;
CREATE INDEX idx_p039_an_created         ON pack039_energy_monitoring.em_anomalies(created_at DESC);
CREATE INDEX idx_p039_an_tags            ON pack039_energy_monitoring.em_anomalies USING GIN(tags);
CREATE INDEX idx_p039_an_details         ON pack039_energy_monitoring.em_anomalies USING GIN(details);
CREATE INDEX idx_p039_an_related         ON pack039_energy_monitoring.em_anomalies USING GIN(related_anomaly_ids);

-- Composite: open anomalies by severity for operations dashboard
CREATE INDEX idx_p039_an_open_severity   ON pack039_energy_monitoring.em_anomalies(severity, detected_at DESC)
    WHERE anomaly_status IN ('DETECTED', 'ACKNOWLEDGED', 'INVESTIGATING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_an_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_anomalies
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_anomaly_rules
-- =============================================================================
-- Configurable rules for anomaly detection engines. Each rule defines
-- detection parameters, thresholds, applicable schedules, and sensitivity
-- settings. Rules can be meter-specific or applied globally across all
-- meters of a given type or energy commodity.

CREATE TABLE pack039_energy_monitoring.em_anomaly_rules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    rule_name               VARCHAR(255)    NOT NULL,
    rule_code               VARCHAR(50)     NOT NULL,
    detection_method        VARCHAR(50)     NOT NULL DEFAULT 'THRESHOLD',
    anomaly_type_detected   VARCHAR(50)     NOT NULL DEFAULT 'CONSUMPTION_SPIKE',
    severity_default        VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    description             TEXT,
    applies_to_meter_types  VARCHAR(50)[]   DEFAULT '{}',
    applies_to_energy_types VARCHAR(50)[]   DEFAULT '{}',
    applies_to_meter_ids    UUID[],
    threshold_upper_pct     NUMERIC(10,4),
    threshold_lower_pct     NUMERIC(10,4),
    threshold_absolute_upper NUMERIC(18,6),
    threshold_absolute_lower NUMERIC(18,6),
    comparison_window_hours INTEGER         DEFAULT 24,
    comparison_day_type     VARCHAR(20)     DEFAULT 'SAME_DAY_TYPE',
    baseline_period_days    INTEGER         DEFAULT 90,
    min_deviation_kwh       NUMERIC(12,3),
    sensitivity             VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    min_confidence_pct      NUMERIC(5,2)    NOT NULL DEFAULT 70.0,
    min_duration_minutes    INTEGER         DEFAULT 0,
    cooldown_minutes        INTEGER         DEFAULT 60,
    max_alerts_per_day      INTEGER         DEFAULT 10,
    active_hours_start      TIME,
    active_hours_end        TIME,
    active_days             INTEGER[]       DEFAULT '{1,2,3,4,5,6,7}',
    exclude_holidays        BOOLEAN         NOT NULL DEFAULT false,
    exclude_weekends        BOOLEAN         NOT NULL DEFAULT false,
    auto_acknowledge        BOOLEAN         NOT NULL DEFAULT false,
    trigger_alarm           BOOLEAN         NOT NULL DEFAULT true,
    alarm_definition_id     UUID,
    notification_channels   JSONB           DEFAULT '[]',
    statistical_params      JSONB           DEFAULT '{}',
    ml_model_id             VARCHAR(100),
    ml_model_version        VARCHAR(50),
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    execution_order         INTEGER         NOT NULL DEFAULT 100,
    last_triggered_at       TIMESTAMPTZ,
    trigger_count           BIGINT          NOT NULL DEFAULT 0,
    false_positive_rate     NUMERIC(5,2),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ar_detection CHECK (
        detection_method IN (
            'THRESHOLD', 'STATISTICAL', 'MACHINE_LEARNING', 'RULE_BASED',
            'PATTERN_MATCHING', 'REGRESSION', 'CUSUM', 'EWMA',
            'ISOLATION_FOREST', 'DBSCAN', 'COMPOSITE'
        )
    ),
    CONSTRAINT chk_p039_ar_anomaly_type CHECK (
        anomaly_type_detected IN (
            'CONSUMPTION_SPIKE', 'CONSUMPTION_DROP', 'BASELOAD_SHIFT',
            'OFF_HOURS_USAGE', 'WEEKEND_ANOMALY', 'HOLIDAY_ANOMALY',
            'WEATHER_DEVIATION', 'EQUIPMENT_MALFUNCTION', 'METER_FAULT',
            'DEMAND_SPIKE', 'POWER_FACTOR_ANOMALY', 'PHASE_IMBALANCE',
            'SEASONAL_DEVIATION', 'TREND_BREAK', 'PATTERN_CHANGE', 'ANY'
        )
    ),
    CONSTRAINT chk_p039_ar_severity CHECK (
        severity_default IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p039_ar_sensitivity CHECK (
        sensitivity IN ('VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    ),
    CONSTRAINT chk_p039_ar_comparison CHECK (
        comparison_day_type IS NULL OR comparison_day_type IN (
            'SAME_DAY_TYPE', 'SAME_WEEKDAY', 'PREVIOUS_DAY', 'PREVIOUS_WEEK',
            'ROLLING_AVERAGE', 'SEASONAL', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ar_confidence CHECK (
        min_confidence_pct >= 0 AND min_confidence_pct <= 100
    ),
    CONSTRAINT chk_p039_ar_cooldown CHECK (
        cooldown_minutes IS NULL OR cooldown_minutes >= 0
    ),
    CONSTRAINT chk_p039_ar_max_alerts CHECK (
        max_alerts_per_day IS NULL OR (max_alerts_per_day >= 1 AND max_alerts_per_day <= 1000)
    ),
    CONSTRAINT chk_p039_ar_order CHECK (
        execution_order >= 1 AND execution_order <= 9999
    ),
    CONSTRAINT chk_p039_ar_fp_rate CHECK (
        false_positive_rate IS NULL OR (false_positive_rate >= 0 AND false_positive_rate <= 100)
    ),
    CONSTRAINT chk_p039_ar_counts CHECK (
        trigger_count >= 0
    ),
    CONSTRAINT uq_p039_ar_tenant_code UNIQUE (tenant_id, rule_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ar_tenant          ON pack039_energy_monitoring.em_anomaly_rules(tenant_id);
CREATE INDEX idx_p039_ar_code            ON pack039_energy_monitoring.em_anomaly_rules(rule_code);
CREATE INDEX idx_p039_ar_detection       ON pack039_energy_monitoring.em_anomaly_rules(detection_method);
CREATE INDEX idx_p039_ar_anomaly_type    ON pack039_energy_monitoring.em_anomaly_rules(anomaly_type_detected);
CREATE INDEX idx_p039_ar_severity        ON pack039_energy_monitoring.em_anomaly_rules(severity_default);
CREATE INDEX idx_p039_ar_sensitivity     ON pack039_energy_monitoring.em_anomaly_rules(sensitivity);
CREATE INDEX idx_p039_ar_enabled         ON pack039_energy_monitoring.em_anomaly_rules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_ar_order           ON pack039_energy_monitoring.em_anomaly_rules(execution_order);
CREATE INDEX idx_p039_ar_created         ON pack039_energy_monitoring.em_anomaly_rules(created_at DESC);
CREATE INDEX idx_p039_ar_meter_types     ON pack039_energy_monitoring.em_anomaly_rules USING GIN(applies_to_meter_types);
CREATE INDEX idx_p039_ar_energy_types    ON pack039_energy_monitoring.em_anomaly_rules USING GIN(applies_to_energy_types);
CREATE INDEX idx_p039_ar_stats           ON pack039_energy_monitoring.em_anomaly_rules USING GIN(statistical_params);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ar_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_anomaly_rules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_anomaly_baselines
-- =============================================================================
-- Baseline energy profiles used as reference for anomaly detection. Each
-- baseline represents expected consumption patterns for a meter under
-- normal operating conditions, segmented by day type, season, and
-- operating mode. Baselines are built from historical data and updated
-- periodically to reflect changing conditions.

CREATE TABLE pack039_energy_monitoring.em_anomaly_baselines (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    baseline_name           VARCHAR(255)    NOT NULL,
    baseline_type           VARCHAR(30)     NOT NULL DEFAULT 'ROLLING',
    day_type                VARCHAR(20)     NOT NULL DEFAULT 'ALL',
    season                  VARCHAR(20)     NOT NULL DEFAULT 'ALL',
    operating_mode          VARCHAR(30)     DEFAULT 'NORMAL',
    training_period_start   DATE            NOT NULL,
    training_period_end     DATE            NOT NULL,
    training_sample_days    INTEGER         NOT NULL DEFAULT 0,
    hourly_profile          JSONB           NOT NULL DEFAULT '[]',
    daily_total_avg         NUMERIC(15,3),
    daily_total_stddev      NUMERIC(15,3),
    daily_peak_avg          NUMERIC(12,3),
    daily_peak_stddev       NUMERIC(12,3),
    daily_baseload_avg      NUMERIC(12,3),
    daily_baseload_stddev   NUMERIC(12,3),
    load_factor_avg         NUMERIC(5,4),
    load_factor_stddev      NUMERIC(5,4),
    temperature_correlation NUMERIC(5,4),
    temperature_base_c      NUMERIC(6,2),
    hdd_coefficient         NUMERIC(10,4),
    cdd_coefficient         NUMERIC(10,4),
    percentile_5            NUMERIC(15,3),
    percentile_25           NUMERIC(15,3),
    percentile_50           NUMERIC(15,3),
    percentile_75           NUMERIC(15,3),
    percentile_95           NUMERIC(15,3),
    model_r_squared         NUMERIC(5,4),
    model_rmse              NUMERIC(12,3),
    model_mape_pct          NUMERIC(8,4),
    is_current              BOOLEAN         NOT NULL DEFAULT true,
    last_updated_reason     VARCHAR(50),
    refresh_schedule_days   INTEGER         DEFAULT 30,
    next_refresh_date       DATE,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ab_type CHECK (
        baseline_type IN (
            'ROLLING', 'FIXED', 'SEASONAL', 'REGRESSION',
            'CLUSTERING', 'COMPOSITE', 'MANUAL'
        )
    ),
    CONSTRAINT chk_p039_ab_day_type CHECK (
        day_type IN ('ALL', 'WEEKDAY', 'WEEKEND', 'HOLIDAY', 'MONDAY',
                     'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY',
                     'SATURDAY', 'SUNDAY')
    ),
    CONSTRAINT chk_p039_ab_season CHECK (
        season IN ('ALL', 'SUMMER', 'WINTER', 'SPRING', 'FALL', 'SHOULDER')
    ),
    CONSTRAINT chk_p039_ab_mode CHECK (
        operating_mode IS NULL OR operating_mode IN (
            'NORMAL', 'REDUCED', 'STANDBY', 'SHUTDOWN', 'STARTUP',
            'PEAK', 'MAINTENANCE', 'EMERGENCY'
        )
    ),
    CONSTRAINT chk_p039_ab_training_dates CHECK (
        training_period_start <= training_period_end
    ),
    CONSTRAINT chk_p039_ab_sample_days CHECK (
        training_sample_days >= 0
    ),
    CONSTRAINT chk_p039_ab_r_squared CHECK (
        model_r_squared IS NULL OR (model_r_squared >= 0 AND model_r_squared <= 1)
    ),
    CONSTRAINT chk_p039_ab_mape CHECK (
        model_mape_pct IS NULL OR model_mape_pct >= 0
    ),
    CONSTRAINT chk_p039_ab_refresh CHECK (
        refresh_schedule_days IS NULL OR (refresh_schedule_days >= 1 AND refresh_schedule_days <= 365)
    ),
    CONSTRAINT chk_p039_ab_load_factor CHECK (
        load_factor_avg IS NULL OR (load_factor_avg >= 0 AND load_factor_avg <= 1)
    ),
    CONSTRAINT chk_p039_ab_temp_corr CHECK (
        temperature_correlation IS NULL OR (temperature_correlation >= -1 AND temperature_correlation <= 1)
    ),
    CONSTRAINT uq_p039_ab_meter_daytype_season UNIQUE (meter_id, baseline_type, day_type, season, operating_mode)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ab_meter           ON pack039_energy_monitoring.em_anomaly_baselines(meter_id);
CREATE INDEX idx_p039_ab_tenant          ON pack039_energy_monitoring.em_anomaly_baselines(tenant_id);
CREATE INDEX idx_p039_ab_type            ON pack039_energy_monitoring.em_anomaly_baselines(baseline_type);
CREATE INDEX idx_p039_ab_day_type        ON pack039_energy_monitoring.em_anomaly_baselines(day_type);
CREATE INDEX idx_p039_ab_season          ON pack039_energy_monitoring.em_anomaly_baselines(season);
CREATE INDEX idx_p039_ab_current         ON pack039_energy_monitoring.em_anomaly_baselines(is_current) WHERE is_current = true;
CREATE INDEX idx_p039_ab_next_refresh    ON pack039_energy_monitoring.em_anomaly_baselines(next_refresh_date);
CREATE INDEX idx_p039_ab_created         ON pack039_energy_monitoring.em_anomaly_baselines(created_at DESC);
CREATE INDEX idx_p039_ab_profile         ON pack039_energy_monitoring.em_anomaly_baselines USING GIN(hourly_profile);

-- Composite: current baselines by meter for detection queries
CREATE INDEX idx_p039_ab_meter_current   ON pack039_energy_monitoring.em_anomaly_baselines(meter_id, day_type, season)
    WHERE is_current = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ab_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_anomaly_baselines
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_investigation_records
-- =============================================================================
-- Investigation workflow for confirmed anomalies. Tracks the investigation
-- process from assignment through root cause analysis to resolution.
-- Includes findings, evidence references, corrective actions, and
-- follow-up tracking. Supports the complete anomaly lifecycle from
-- detection through remediation.

CREATE TABLE pack039_energy_monitoring.em_investigation_records (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    anomaly_id              UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_anomalies(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    investigation_code      VARCHAR(50)     NOT NULL,
    investigation_status    VARCHAR(30)     NOT NULL DEFAULT 'OPEN',
    priority                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    assigned_to             UUID,
    assigned_at             TIMESTAMPTZ,
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    due_date                DATE,
    investigation_type      VARCHAR(30)     NOT NULL DEFAULT 'STANDARD',
    root_cause_identified   BOOLEAN         NOT NULL DEFAULT false,
    root_cause              TEXT,
    root_cause_category     VARCHAR(50),
    root_cause_equipment_id UUID,
    findings_summary        TEXT,
    evidence_document_ids   UUID[]          DEFAULT '{}',
    corrective_action       TEXT,
    preventive_action       TEXT,
    action_due_date         DATE,
    action_completed        BOOLEAN         NOT NULL DEFAULT false,
    action_completed_at     TIMESTAMPTZ,
    energy_saved_kwh        NUMERIC(15,3),
    cost_saved              NUMERIC(12,2),
    cost_of_investigation   NUMERIC(12,2),
    follow_up_required      BOOLEAN         NOT NULL DEFAULT false,
    follow_up_date          DATE,
    follow_up_notes         TEXT,
    follow_up_completed     BOOLEAN         NOT NULL DEFAULT false,
    notes                   TEXT,
    timeline                JSONB           DEFAULT '[]',
    attachments             JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ir_status CHECK (
        investigation_status IN (
            'OPEN', 'IN_PROGRESS', 'PENDING_INFO', 'ROOT_CAUSE_FOUND',
            'ACTION_REQUIRED', 'ACTION_IN_PROGRESS', 'COMPLETED',
            'CLOSED', 'CANCELLED', 'DEFERRED'
        )
    ),
    CONSTRAINT chk_p039_ir_priority CHECK (
        priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'URGENT')
    ),
    CONSTRAINT chk_p039_ir_type CHECK (
        investigation_type IN (
            'STANDARD', 'RAPID', 'DEEP_DIVE', 'RECURRING', 'COMPLIANCE', 'POST_INCIDENT'
        )
    ),
    CONSTRAINT chk_p039_ir_root_cat CHECK (
        root_cause_category IS NULL OR root_cause_category IN (
            'EQUIPMENT', 'OPERATIONAL', 'WEATHER', 'OCCUPANCY',
            'MAINTENANCE', 'METER', 'NETWORK', 'BEHAVIORAL',
            'PROCESS_CHANGE', 'DESIGN', 'EXTERNAL', 'UNKNOWN'
        )
    ),
    CONSTRAINT chk_p039_ir_energy_saved CHECK (
        energy_saved_kwh IS NULL OR energy_saved_kwh >= 0
    ),
    CONSTRAINT chk_p039_ir_cost_saved CHECK (
        cost_saved IS NULL OR cost_saved >= 0
    ),
    CONSTRAINT chk_p039_ir_cost_inv CHECK (
        cost_of_investigation IS NULL OR cost_of_investigation >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ir_anomaly         ON pack039_energy_monitoring.em_investigation_records(anomaly_id);
CREATE INDEX idx_p039_ir_tenant          ON pack039_energy_monitoring.em_investigation_records(tenant_id);
CREATE INDEX idx_p039_ir_code            ON pack039_energy_monitoring.em_investigation_records(investigation_code);
CREATE INDEX idx_p039_ir_status          ON pack039_energy_monitoring.em_investigation_records(investigation_status);
CREATE INDEX idx_p039_ir_priority        ON pack039_energy_monitoring.em_investigation_records(priority);
CREATE INDEX idx_p039_ir_assigned        ON pack039_energy_monitoring.em_investigation_records(assigned_to) WHERE assigned_to IS NOT NULL;
CREATE INDEX idx_p039_ir_due_date        ON pack039_energy_monitoring.em_investigation_records(due_date);
CREATE INDEX idx_p039_ir_root_cat        ON pack039_energy_monitoring.em_investigation_records(root_cause_category);
CREATE INDEX idx_p039_ir_follow_up       ON pack039_energy_monitoring.em_investigation_records(follow_up_date) WHERE follow_up_required = true AND follow_up_completed = false;
CREATE INDEX idx_p039_ir_action_due      ON pack039_energy_monitoring.em_investigation_records(action_due_date) WHERE action_completed = false;
CREATE INDEX idx_p039_ir_created         ON pack039_energy_monitoring.em_investigation_records(created_at DESC);
CREATE INDEX idx_p039_ir_timeline        ON pack039_energy_monitoring.em_investigation_records USING GIN(timeline);

-- Composite: open investigations by priority for workqueue
CREATE INDEX idx_p039_ir_open_priority   ON pack039_energy_monitoring.em_investigation_records(priority, due_date)
    WHERE investigation_status IN ('OPEN', 'IN_PROGRESS', 'PENDING_INFO', 'ACTION_REQUIRED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ir_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_investigation_records
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_anomaly_stats
-- =============================================================================
-- Pre-computed anomaly statistics by meter, period, and category for
-- dashboard rendering, trend analysis, and KPI calculation. Aggregated
-- periodically from the anomalies table to provide fast access to
-- anomaly metrics without real-time aggregation queries.

CREATE TABLE pack039_energy_monitoring.em_anomaly_stats (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID            REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    stats_period_type       VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    anomaly_category        VARCHAR(50),
    total_anomalies         INTEGER         NOT NULL DEFAULT 0,
    critical_count          INTEGER         NOT NULL DEFAULT 0,
    high_count              INTEGER         NOT NULL DEFAULT 0,
    medium_count            INTEGER         NOT NULL DEFAULT 0,
    low_count               INTEGER         NOT NULL DEFAULT 0,
    confirmed_count         INTEGER         NOT NULL DEFAULT 0,
    false_positive_count    INTEGER         NOT NULL DEFAULT 0,
    resolved_count          INTEGER         NOT NULL DEFAULT 0,
    open_count              INTEGER         NOT NULL DEFAULT 0,
    recurring_count         INTEGER         NOT NULL DEFAULT 0,
    avg_resolution_hours    NUMERIC(10,2),
    median_resolution_hours NUMERIC(10,2),
    total_energy_impact_kwh NUMERIC(15,3)   DEFAULT 0,
    total_cost_impact       NUMERIC(12,2)   DEFAULT 0,
    total_energy_saved_kwh  NUMERIC(15,3)   DEFAULT 0,
    total_cost_saved        NUMERIC(12,2)   DEFAULT 0,
    avg_confidence_score    NUMERIC(5,2),
    false_positive_rate_pct NUMERIC(7,4),
    detection_methods_used  JSONB           DEFAULT '{}',
    top_root_causes         JSONB           DEFAULT '[]',
    trend_direction         VARCHAR(20),
    trend_pct_change        NUMERIC(10,4),
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ast_period_type CHECK (
        stats_period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p039_ast_counts CHECK (
        total_anomalies >= 0 AND critical_count >= 0 AND high_count >= 0 AND
        medium_count >= 0 AND low_count >= 0 AND confirmed_count >= 0 AND
        false_positive_count >= 0 AND resolved_count >= 0 AND open_count >= 0 AND
        recurring_count >= 0
    ),
    CONSTRAINT chk_p039_ast_resolution CHECK (
        avg_resolution_hours IS NULL OR avg_resolution_hours >= 0
    ),
    CONSTRAINT chk_p039_ast_fp_rate CHECK (
        false_positive_rate_pct IS NULL OR (false_positive_rate_pct >= 0 AND false_positive_rate_pct <= 100)
    ),
    CONSTRAINT chk_p039_ast_trend CHECK (
        trend_direction IS NULL OR trend_direction IN (
            'IMPROVING', 'STABLE', 'DEGRADING', 'INSUFFICIENT_DATA'
        )
    ),
    CONSTRAINT chk_p039_ast_dates CHECK (
        period_start <= period_end
    ),
    CONSTRAINT uq_p039_ast_meter_period_cat UNIQUE (meter_id, stats_period_type, period_start, anomaly_category)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ast_meter          ON pack039_energy_monitoring.em_anomaly_stats(meter_id);
CREATE INDEX idx_p039_ast_tenant         ON pack039_energy_monitoring.em_anomaly_stats(tenant_id);
CREATE INDEX idx_p039_ast_period_type    ON pack039_energy_monitoring.em_anomaly_stats(stats_period_type);
CREATE INDEX idx_p039_ast_period_start   ON pack039_energy_monitoring.em_anomaly_stats(period_start DESC);
CREATE INDEX idx_p039_ast_category       ON pack039_energy_monitoring.em_anomaly_stats(anomaly_category);
CREATE INDEX idx_p039_ast_total          ON pack039_energy_monitoring.em_anomaly_stats(total_anomalies DESC);
CREATE INDEX idx_p039_ast_trend          ON pack039_energy_monitoring.em_anomaly_stats(trend_direction);
CREATE INDEX idx_p039_ast_created        ON pack039_energy_monitoring.em_anomaly_stats(created_at DESC);

-- Composite: monthly stats by meter for trending
CREATE INDEX idx_p039_ast_meter_monthly  ON pack039_energy_monitoring.em_anomaly_stats(meter_id, period_start DESC)
    WHERE stats_period_type = 'MONTHLY';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ast_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_anomaly_stats
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_anomaly_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_anomaly_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_investigation_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_anomaly_stats ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_an_tenant_isolation
    ON pack039_energy_monitoring.em_anomalies
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_an_service_bypass
    ON pack039_energy_monitoring.em_anomalies
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ar_tenant_isolation
    ON pack039_energy_monitoring.em_anomaly_rules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ar_service_bypass
    ON pack039_energy_monitoring.em_anomaly_rules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ab_tenant_isolation
    ON pack039_energy_monitoring.em_anomaly_baselines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ab_service_bypass
    ON pack039_energy_monitoring.em_anomaly_baselines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ir_tenant_isolation
    ON pack039_energy_monitoring.em_investigation_records
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ir_service_bypass
    ON pack039_energy_monitoring.em_investigation_records
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ast_tenant_isolation
    ON pack039_energy_monitoring.em_anomaly_stats
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ast_service_bypass
    ON pack039_energy_monitoring.em_anomaly_stats
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_anomalies TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_anomaly_rules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_anomaly_baselines TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_investigation_records TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_anomaly_stats TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_anomalies IS
    'Detected energy consumption anomalies with classification, severity, lifecycle tracking, root cause analysis, and impact estimation.';
COMMENT ON TABLE pack039_energy_monitoring.em_anomaly_rules IS
    'Configurable anomaly detection rules with thresholds, statistical parameters, sensitivity settings, and notification configuration.';
COMMENT ON TABLE pack039_energy_monitoring.em_anomaly_baselines IS
    'Baseline energy profiles for anomaly detection, segmented by day type, season, and operating mode with statistical parameters.';
COMMENT ON TABLE pack039_energy_monitoring.em_investigation_records IS
    'Investigation workflow records for anomalies from assignment through root cause analysis to resolution and follow-up.';
COMMENT ON TABLE pack039_energy_monitoring.em_anomaly_stats IS
    'Pre-computed anomaly statistics by meter, period, and category for dashboard rendering and trend analysis.';

COMMENT ON COLUMN pack039_energy_monitoring.em_anomalies.anomaly_type IS 'Classification: CONSUMPTION_SPIKE, BASELOAD_SHIFT, OFF_HOURS_USAGE, EQUIPMENT_MALFUNCTION, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomalies.detection_method IS 'Algorithm used: THRESHOLD, STATISTICAL, MACHINE_LEARNING, CUSUM, EWMA, ISOLATION_FOREST, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomalies.confidence_score IS 'Detection confidence (0-100). Higher scores indicate stronger anomaly evidence.';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomalies.estimated_energy_impact_kwh IS 'Estimated excess or deficit energy in kWh attributable to the anomaly.';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomalies.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_rules.sensitivity IS 'Detection sensitivity: VERY_LOW (few alerts), LOW, MEDIUM, HIGH, VERY_HIGH (many alerts).';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_rules.cooldown_minutes IS 'Minimum minutes between consecutive alerts from the same rule to prevent alert fatigue.';

COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_baselines.hourly_profile IS 'JSON array of 24 hourly expected values: [{hour, mean, stddev, min, max, p5, p95}].';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_baselines.temperature_correlation IS 'Pearson correlation coefficient between energy consumption and temperature (-1 to 1).';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_baselines.hdd_coefficient IS 'Heating degree-day regression coefficient (kWh per HDD). Used for weather-normalized anomaly detection.';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_baselines.cdd_coefficient IS 'Cooling degree-day regression coefficient (kWh per CDD). Used for weather-normalized anomaly detection.';

COMMENT ON COLUMN pack039_energy_monitoring.em_investigation_records.investigation_status IS 'Lifecycle: OPEN -> IN_PROGRESS -> ROOT_CAUSE_FOUND -> ACTION_REQUIRED -> COMPLETED -> CLOSED.';
COMMENT ON COLUMN pack039_energy_monitoring.em_investigation_records.timeline IS 'JSON array of investigation events: [{timestamp, action, actor, notes}].';

COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_stats.false_positive_rate_pct IS 'Percentage of anomalies confirmed as false positives in the period (lower is better).';
COMMENT ON COLUMN pack039_energy_monitoring.em_anomaly_stats.trend_direction IS 'Anomaly trend: IMPROVING (fewer), STABLE, DEGRADING (more), INSUFFICIENT_DATA.';
