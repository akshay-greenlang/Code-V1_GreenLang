-- =============================================================================
-- V313: PACK-039 Energy Monitoring Pack - Alarm Management
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    008 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates alarm management tables for real-time energy monitoring alerts.
-- Includes alarm definitions with configurable thresholds and conditions,
-- alarm event records with lifecycle tracking, acknowledgment workflow,
-- suppression rules for maintenance windows, and escalation configuration
-- for multi-tier notification routing.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_alarm_definitions
--   2. pack039_energy_monitoring.em_alarm_events
--   3. pack039_energy_monitoring.em_alarm_acknowledgments
--   4. pack039_energy_monitoring.em_suppression_rules
--   5. pack039_energy_monitoring.em_escalation_configs
--
-- Previous: V312__pack039_energy_monitoring_007.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_alarm_definitions
-- =============================================================================
-- Configurable alarm definitions for energy monitoring conditions.
-- Each alarm defines a trigger condition (threshold, rate-of-change,
-- state change), severity level, notification channels, and suppression
-- windows. Alarms are the real-time operational alerting layer,
-- distinct from anomaly detection (analytical) and budget alerts (financial).

CREATE TABLE pack039_energy_monitoring.em_alarm_definitions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    meter_id                UUID            REFERENCES pack039_energy_monitoring.em_meters(id) ON DELETE SET NULL,
    alarm_name              VARCHAR(255)    NOT NULL,
    alarm_code              VARCHAR(50)     NOT NULL,
    alarm_category          VARCHAR(50)     NOT NULL DEFAULT 'CONSUMPTION',
    alarm_type              VARCHAR(30)     NOT NULL DEFAULT 'THRESHOLD',
    severity                VARCHAR(20)     NOT NULL DEFAULT 'WARNING',
    priority                INTEGER         NOT NULL DEFAULT 3,
    description             TEXT,
    condition_expression    TEXT            NOT NULL,
    condition_type          VARCHAR(30)     NOT NULL DEFAULT 'GREATER_THAN',
    threshold_value         NUMERIC(18,6),
    threshold_unit          VARCHAR(30),
    hysteresis_value        NUMERIC(18,6),
    hysteresis_pct          NUMERIC(7,4),
    deadband_value          NUMERIC(18,6),
    sustained_duration_minutes INTEGER     DEFAULT 0,
    evaluation_interval_minutes INTEGER    NOT NULL DEFAULT 15,
    comparison_method       VARCHAR(30),
    comparison_reference    NUMERIC(18,6),
    rate_of_change_limit    NUMERIC(12,4),
    rate_of_change_period_minutes INTEGER,
    applies_to_energy_types VARCHAR(50)[]   DEFAULT '{ELECTRICITY}',
    applies_to_tariff_periods VARCHAR(30)[] DEFAULT '{}',
    active_schedule_start   TIME,
    active_schedule_end     TIME,
    active_days             INTEGER[]       DEFAULT '{1,2,3,4,5,6,7}',
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    notification_channels   JSONB           DEFAULT '[]',
    notification_recipients JSONB           DEFAULT '[]',
    notification_template   VARCHAR(100),
    auto_acknowledge_minutes INTEGER,
    auto_close_minutes      INTEGER,
    requires_investigation  BOOLEAN         NOT NULL DEFAULT false,
    linked_anomaly_rule_id  UUID,
    max_active_per_day      INTEGER         DEFAULT 100,
    cooldown_minutes        INTEGER         DEFAULT 30,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    alarm_status            VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    total_triggers          BIGINT          NOT NULL DEFAULT 0,
    total_false_alarms      BIGINT          NOT NULL DEFAULT 0,
    last_triggered_at       TIMESTAMPTZ,
    last_cleared_at         TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_ad_category CHECK (
        alarm_category IN (
            'CONSUMPTION', 'DEMAND', 'POWER_QUALITY', 'EQUIPMENT',
            'METER', 'COMMUNICATION', 'DATA_QUALITY', 'COST',
            'ENVIRONMENTAL', 'COMPLIANCE', 'SYSTEM', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ad_type CHECK (
        alarm_type IN (
            'THRESHOLD', 'RATE_OF_CHANGE', 'STATE_CHANGE',
            'DEVIATION', 'COMPARISON', 'ABSENCE', 'PATTERN',
            'COMPOSITE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_ad_severity CHECK (
        severity IN ('INFO', 'WARNING', 'ALARM', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p039_ad_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p039_ad_condition_type CHECK (
        condition_type IN (
            'GREATER_THAN', 'LESS_THAN', 'EQUAL_TO', 'NOT_EQUAL',
            'BETWEEN', 'OUTSIDE_RANGE', 'RATE_EXCEEDS',
            'STUCK_VALUE', 'MISSING_DATA', 'STATE_CHANGE',
            'CUSTOM_EXPRESSION'
        )
    ),
    CONSTRAINT chk_p039_ad_comparison CHECK (
        comparison_method IS NULL OR comparison_method IN (
            'ABSOLUTE', 'PERCENTAGE', 'BASELINE', 'PREVIOUS_PERIOD',
            'SAME_TIME_YESTERDAY', 'SAME_TIME_LAST_WEEK', 'AVERAGE'
        )
    ),
    CONSTRAINT chk_p039_ad_alarm_status CHECK (
        alarm_status IN ('ACTIVE', 'DISABLED', 'MAINTENANCE', 'TESTING', 'RETIRED')
    ),
    CONSTRAINT chk_p039_ad_sustained CHECK (
        sustained_duration_minutes IS NULL OR sustained_duration_minutes >= 0
    ),
    CONSTRAINT chk_p039_ad_eval_interval CHECK (
        evaluation_interval_minutes >= 1 AND evaluation_interval_minutes <= 1440
    ),
    CONSTRAINT chk_p039_ad_cooldown CHECK (
        cooldown_minutes IS NULL OR cooldown_minutes >= 0
    ),
    CONSTRAINT chk_p039_ad_max_active CHECK (
        max_active_per_day IS NULL OR (max_active_per_day >= 1 AND max_active_per_day <= 10000)
    ),
    CONSTRAINT chk_p039_ad_auto_ack CHECK (
        auto_acknowledge_minutes IS NULL OR auto_acknowledge_minutes > 0
    ),
    CONSTRAINT chk_p039_ad_auto_close CHECK (
        auto_close_minutes IS NULL OR auto_close_minutes > 0
    ),
    CONSTRAINT chk_p039_ad_counts CHECK (
        total_triggers >= 0 AND total_false_alarms >= 0
    ),
    CONSTRAINT uq_p039_ad_tenant_code UNIQUE (tenant_id, alarm_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ad_tenant          ON pack039_energy_monitoring.em_alarm_definitions(tenant_id);
CREATE INDEX idx_p039_ad_facility        ON pack039_energy_monitoring.em_alarm_definitions(facility_id);
CREATE INDEX idx_p039_ad_meter           ON pack039_energy_monitoring.em_alarm_definitions(meter_id);
CREATE INDEX idx_p039_ad_code            ON pack039_energy_monitoring.em_alarm_definitions(alarm_code);
CREATE INDEX idx_p039_ad_category        ON pack039_energy_monitoring.em_alarm_definitions(alarm_category);
CREATE INDEX idx_p039_ad_type            ON pack039_energy_monitoring.em_alarm_definitions(alarm_type);
CREATE INDEX idx_p039_ad_severity        ON pack039_energy_monitoring.em_alarm_definitions(severity);
CREATE INDEX idx_p039_ad_priority        ON pack039_energy_monitoring.em_alarm_definitions(priority);
CREATE INDEX idx_p039_ad_enabled         ON pack039_energy_monitoring.em_alarm_definitions(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_ad_status          ON pack039_energy_monitoring.em_alarm_definitions(alarm_status);
CREATE INDEX idx_p039_ad_last_trigger    ON pack039_energy_monitoring.em_alarm_definitions(last_triggered_at DESC);
CREATE INDEX idx_p039_ad_created         ON pack039_energy_monitoring.em_alarm_definitions(created_at DESC);
CREATE INDEX idx_p039_ad_recipients      ON pack039_energy_monitoring.em_alarm_definitions USING GIN(notification_recipients);
CREATE INDEX idx_p039_ad_energy_types    ON pack039_energy_monitoring.em_alarm_definitions USING GIN(applies_to_energy_types);

-- Composite: active enabled alarms by meter for evaluation
CREATE INDEX idx_p039_ad_meter_active    ON pack039_energy_monitoring.em_alarm_definitions(meter_id, priority)
    WHERE is_enabled = true AND alarm_status = 'ACTIVE';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ad_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_alarm_definitions
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_alarm_events
-- =============================================================================
-- Individual alarm event instances triggered by alarm definitions. Each
-- event represents a single alarm occurrence with trigger details,
-- duration, severity, and resolution tracking. Events progress through
-- a lifecycle: TRIGGERED -> ACKNOWLEDGED -> RESOLVED/CLEARED.

CREATE TABLE pack039_energy_monitoring.em_alarm_events (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    alarm_definition_id     UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_alarm_definitions(id) ON DELETE CASCADE,
    meter_id                UUID            REFERENCES pack039_energy_monitoring.em_meters(id),
    tenant_id               UUID            NOT NULL,
    event_code              VARCHAR(100)    NOT NULL,
    triggered_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    cleared_at              TIMESTAMPTZ,
    duration_minutes        INTEGER,
    severity                VARCHAR(20)     NOT NULL,
    priority                INTEGER         NOT NULL DEFAULT 3,
    event_status            VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    trigger_value           NUMERIC(18,6),
    threshold_value         NUMERIC(18,6),
    deviation_value         NUMERIC(18,6),
    deviation_pct           NUMERIC(10,4),
    trigger_condition       TEXT,
    data_timestamp          TIMESTAMPTZ,
    affected_meters         UUID[]          DEFAULT '{}',
    estimated_energy_impact_kwh NUMERIC(15,3),
    estimated_cost_impact   NUMERIC(12,2),
    is_suppressed           BOOLEAN         NOT NULL DEFAULT false,
    suppression_rule_id     UUID,
    suppression_reason      TEXT,
    is_false_alarm          BOOLEAN         NOT NULL DEFAULT false,
    notification_sent       BOOLEAN         NOT NULL DEFAULT false,
    notification_sent_at    TIMESTAMPTZ,
    notification_channels_used JSONB        DEFAULT '[]',
    escalation_level        INTEGER         NOT NULL DEFAULT 0,
    last_escalated_at       TIMESTAMPTZ,
    root_cause              TEXT,
    corrective_action       TEXT,
    tags                    JSONB           DEFAULT '[]',
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ae_severity CHECK (
        severity IN ('INFO', 'WARNING', 'ALARM', 'CRITICAL', 'EMERGENCY')
    ),
    CONSTRAINT chk_p039_ae_priority CHECK (
        priority >= 1 AND priority <= 5
    ),
    CONSTRAINT chk_p039_ae_status CHECK (
        event_status IN (
            'ACTIVE', 'ACKNOWLEDGED', 'INVESTIGATING', 'RESOLVED',
            'CLEARED', 'SUPPRESSED', 'AUTO_CLOSED', 'EXPIRED'
        )
    ),
    CONSTRAINT chk_p039_ae_duration CHECK (
        duration_minutes IS NULL OR duration_minutes >= 0
    ),
    CONSTRAINT chk_p039_ae_escalation CHECK (
        escalation_level >= 0 AND escalation_level <= 10
    ),
    CONSTRAINT chk_p039_ae_dates CHECK (
        cleared_at IS NULL OR triggered_at <= cleared_at
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ae_alarm_def       ON pack039_energy_monitoring.em_alarm_events(alarm_definition_id);
CREATE INDEX idx_p039_ae_meter           ON pack039_energy_monitoring.em_alarm_events(meter_id);
CREATE INDEX idx_p039_ae_tenant          ON pack039_energy_monitoring.em_alarm_events(tenant_id);
CREATE INDEX idx_p039_ae_event_code      ON pack039_energy_monitoring.em_alarm_events(event_code);
CREATE INDEX idx_p039_ae_triggered       ON pack039_energy_monitoring.em_alarm_events(triggered_at DESC);
CREATE INDEX idx_p039_ae_severity        ON pack039_energy_monitoring.em_alarm_events(severity);
CREATE INDEX idx_p039_ae_priority        ON pack039_energy_monitoring.em_alarm_events(priority);
CREATE INDEX idx_p039_ae_status          ON pack039_energy_monitoring.em_alarm_events(event_status);
CREATE INDEX idx_p039_ae_suppressed      ON pack039_energy_monitoring.em_alarm_events(is_suppressed) WHERE is_suppressed = true;
CREATE INDEX idx_p039_ae_false_alarm     ON pack039_energy_monitoring.em_alarm_events(is_false_alarm) WHERE is_false_alarm = true;
CREATE INDEX idx_p039_ae_notification    ON pack039_energy_monitoring.em_alarm_events(notification_sent) WHERE notification_sent = false;
CREATE INDEX idx_p039_ae_escalation      ON pack039_energy_monitoring.em_alarm_events(escalation_level) WHERE escalation_level > 0;
CREATE INDEX idx_p039_ae_created         ON pack039_energy_monitoring.em_alarm_events(created_at DESC);
CREATE INDEX idx_p039_ae_tags            ON pack039_energy_monitoring.em_alarm_events USING GIN(tags);
CREATE INDEX idx_p039_ae_details         ON pack039_energy_monitoring.em_alarm_events USING GIN(details);
CREATE INDEX idx_p039_ae_affected        ON pack039_energy_monitoring.em_alarm_events USING GIN(affected_meters);

-- Composite: active alarms by severity for operations dashboard
CREATE INDEX idx_p039_ae_active_sev      ON pack039_energy_monitoring.em_alarm_events(severity, priority, triggered_at DESC)
    WHERE event_status IN ('ACTIVE', 'ACKNOWLEDGED', 'INVESTIGATING');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ae_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_alarm_events
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_alarm_acknowledgments
-- =============================================================================
-- Acknowledgment records for alarm events. Tracks who acknowledged each
-- alarm, when, with what notes, and any initial assessment. Multiple
-- acknowledgments are allowed (e.g., operator ack, supervisor ack).
-- Supports shift-based acknowledgment workflows.

CREATE TABLE pack039_energy_monitoring.em_alarm_acknowledgments (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    alarm_event_id          UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_alarm_events(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    acknowledged_by         UUID            NOT NULL,
    acknowledged_at         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    acknowledgment_type     VARCHAR(30)     NOT NULL DEFAULT 'OPERATOR',
    response_time_minutes   INTEGER,
    initial_assessment      TEXT,
    action_taken            VARCHAR(100),
    action_description      TEXT,
    is_false_alarm          BOOLEAN         NOT NULL DEFAULT false,
    escalation_requested    BOOLEAN         NOT NULL DEFAULT false,
    escalation_reason       TEXT,
    shift_id                VARCHAR(50),
    shift_name              VARCHAR(100),
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ack_type CHECK (
        acknowledgment_type IN (
            'OPERATOR', 'SUPERVISOR', 'ENGINEER', 'MANAGER',
            'SYSTEM', 'AUTO', 'ESCALATION'
        )
    ),
    CONSTRAINT chk_p039_ack_response CHECK (
        response_time_minutes IS NULL OR response_time_minutes >= 0
    ),
    CONSTRAINT chk_p039_ack_action CHECK (
        action_taken IS NULL OR action_taken IN (
            'INVESTIGATING', 'DISPATCHED', 'REMOTE_FIX', 'WORK_ORDER',
            'DEFERRED', 'NO_ACTION', 'FALSE_ALARM', 'MONITORING',
            'CONTACTED_VENDOR', 'ADJUSTED_SETPOINT', 'OTHER'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ack_event          ON pack039_energy_monitoring.em_alarm_acknowledgments(alarm_event_id);
CREATE INDEX idx_p039_ack_tenant         ON pack039_energy_monitoring.em_alarm_acknowledgments(tenant_id);
CREATE INDEX idx_p039_ack_by             ON pack039_energy_monitoring.em_alarm_acknowledgments(acknowledged_by);
CREATE INDEX idx_p039_ack_at             ON pack039_energy_monitoring.em_alarm_acknowledgments(acknowledged_at DESC);
CREATE INDEX idx_p039_ack_type           ON pack039_energy_monitoring.em_alarm_acknowledgments(acknowledgment_type);
CREATE INDEX idx_p039_ack_action         ON pack039_energy_monitoring.em_alarm_acknowledgments(action_taken);
CREATE INDEX idx_p039_ack_response       ON pack039_energy_monitoring.em_alarm_acknowledgments(response_time_minutes);
CREATE INDEX idx_p039_ack_shift          ON pack039_energy_monitoring.em_alarm_acknowledgments(shift_id);
CREATE INDEX idx_p039_ack_false          ON pack039_energy_monitoring.em_alarm_acknowledgments(is_false_alarm) WHERE is_false_alarm = true;
CREATE INDEX idx_p039_ack_created        ON pack039_energy_monitoring.em_alarm_acknowledgments(created_at DESC);

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_suppression_rules
-- =============================================================================
-- Rules for suppressing alarms during planned events such as maintenance
-- windows, testing periods, equipment shutdowns, or known operational
-- changes. Suppressed alarms are still recorded but not notified,
-- reducing alert fatigue while maintaining the audit trail.

CREATE TABLE pack039_energy_monitoring.em_suppression_rules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    rule_name               VARCHAR(255)    NOT NULL,
    rule_code               VARCHAR(50)     NOT NULL,
    suppression_type        VARCHAR(30)     NOT NULL DEFAULT 'MAINTENANCE',
    applies_to_alarm_ids    UUID[]          DEFAULT '{}',
    applies_to_alarm_categories VARCHAR(50)[] DEFAULT '{}',
    applies_to_meter_ids    UUID[]          DEFAULT '{}',
    applies_to_severities   VARCHAR(20)[]   DEFAULT '{}',
    suppression_start       TIMESTAMPTZ     NOT NULL,
    suppression_end         TIMESTAMPTZ     NOT NULL,
    recurrence_pattern      VARCHAR(30),
    recurrence_config       JSONB           DEFAULT '{}',
    reason                  TEXT            NOT NULL,
    requested_by            UUID            NOT NULL,
    approved_by             UUID,
    approved_at             TIMESTAMPTZ,
    work_order_reference    VARCHAR(100),
    max_suppressed_events   INTEGER,
    suppressed_event_count  INTEGER         NOT NULL DEFAULT 0,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    is_approved             BOOLEAN         NOT NULL DEFAULT false,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_sr_type CHECK (
        suppression_type IN (
            'MAINTENANCE', 'TESTING', 'COMMISSIONING', 'SHUTDOWN',
            'STARTUP', 'CONSTRUCTION', 'KNOWN_ISSUE', 'WEATHER',
            'HOLIDAY', 'SCHEDULE', 'MANUAL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p039_sr_recurrence CHECK (
        recurrence_pattern IS NULL OR recurrence_pattern IN (
            'NONE', 'DAILY', 'WEEKLY', 'MONTHLY', 'YEARLY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_sr_dates CHECK (
        suppression_start < suppression_end
    ),
    CONSTRAINT chk_p039_sr_max_events CHECK (
        max_suppressed_events IS NULL OR max_suppressed_events > 0
    ),
    CONSTRAINT chk_p039_sr_event_count CHECK (
        suppressed_event_count >= 0
    ),
    CONSTRAINT uq_p039_sr_tenant_code UNIQUE (tenant_id, rule_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_sr_tenant          ON pack039_energy_monitoring.em_suppression_rules(tenant_id);
CREATE INDEX idx_p039_sr_facility        ON pack039_energy_monitoring.em_suppression_rules(facility_id);
CREATE INDEX idx_p039_sr_code            ON pack039_energy_monitoring.em_suppression_rules(rule_code);
CREATE INDEX idx_p039_sr_type            ON pack039_energy_monitoring.em_suppression_rules(suppression_type);
CREATE INDEX idx_p039_sr_start           ON pack039_energy_monitoring.em_suppression_rules(suppression_start);
CREATE INDEX idx_p039_sr_end             ON pack039_energy_monitoring.em_suppression_rules(suppression_end);
CREATE INDEX idx_p039_sr_active          ON pack039_energy_monitoring.em_suppression_rules(is_active) WHERE is_active = true;
CREATE INDEX idx_p039_sr_approved        ON pack039_energy_monitoring.em_suppression_rules(is_approved) WHERE is_approved = false;
CREATE INDEX idx_p039_sr_created         ON pack039_energy_monitoring.em_suppression_rules(created_at DESC);
CREATE INDEX idx_p039_sr_alarm_ids       ON pack039_energy_monitoring.em_suppression_rules USING GIN(applies_to_alarm_ids);
CREATE INDEX idx_p039_sr_meter_ids       ON pack039_energy_monitoring.em_suppression_rules USING GIN(applies_to_meter_ids);
CREATE INDEX idx_p039_sr_categories      ON pack039_energy_monitoring.em_suppression_rules USING GIN(applies_to_alarm_categories);

-- Composite: currently active suppressions for alarm evaluation
CREATE INDEX idx_p039_sr_current_active  ON pack039_energy_monitoring.em_suppression_rules(suppression_start, suppression_end)
    WHERE is_active = true AND is_approved = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_sr_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_suppression_rules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_escalation_configs
-- =============================================================================
-- Multi-tier escalation configuration for alarm notifications. Defines
-- escalation paths based on alarm severity, priority, and response time
-- targets. Each tier specifies recipients, notification channels, and
-- the escalation delay before moving to the next tier.

CREATE TABLE pack039_energy_monitoring.em_escalation_configs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    config_name             VARCHAR(255)    NOT NULL,
    config_code             VARCHAR(50)     NOT NULL,
    applies_to_severities   VARCHAR(20)[]   NOT NULL DEFAULT '{WARNING,ALARM,CRITICAL,EMERGENCY}',
    applies_to_priorities   INTEGER[]       DEFAULT '{1,2,3,4,5}',
    applies_to_categories   VARCHAR(50)[]   DEFAULT '{}',
    applies_to_alarm_ids    UUID[]          DEFAULT '{}',
    escalation_tiers        JSONB           NOT NULL DEFAULT '[]',
    max_tiers               INTEGER         NOT NULL DEFAULT 3,
    response_time_target_minutes INTEGER    NOT NULL DEFAULT 30,
    acknowledgment_required BOOLEAN         NOT NULL DEFAULT true,
    auto_escalate           BOOLEAN         NOT NULL DEFAULT true,
    auto_escalate_after_minutes INTEGER     NOT NULL DEFAULT 60,
    business_hours_only     BOOLEAN         NOT NULL DEFAULT false,
    business_hours_start    TIME            DEFAULT '08:00',
    business_hours_end      TIME            DEFAULT '18:00',
    business_days           INTEGER[]       DEFAULT '{1,2,3,4,5}',
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    on_call_schedule_id     VARCHAR(100),
    on_call_rotation        JSONB           DEFAULT '{}',
    notification_template_id VARCHAR(100),
    include_alarm_details   BOOLEAN         NOT NULL DEFAULT true,
    include_meter_data      BOOLEAN         NOT NULL DEFAULT true,
    include_trend_chart     BOOLEAN         NOT NULL DEFAULT false,
    sla_response_minutes    INTEGER,
    sla_resolution_minutes  INTEGER,
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ec_max_tiers CHECK (
        max_tiers >= 1 AND max_tiers <= 10
    ),
    CONSTRAINT chk_p039_ec_response CHECK (
        response_time_target_minutes >= 1
    ),
    CONSTRAINT chk_p039_ec_auto_esc CHECK (
        auto_escalate_after_minutes >= 1
    ),
    CONSTRAINT chk_p039_ec_sla_response CHECK (
        sla_response_minutes IS NULL OR sla_response_minutes > 0
    ),
    CONSTRAINT chk_p039_ec_sla_resolution CHECK (
        sla_resolution_minutes IS NULL OR sla_resolution_minutes > 0
    ),
    CONSTRAINT uq_p039_ec_tenant_code UNIQUE (tenant_id, config_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ec_tenant          ON pack039_energy_monitoring.em_escalation_configs(tenant_id);
CREATE INDEX idx_p039_ec_facility        ON pack039_energy_monitoring.em_escalation_configs(facility_id);
CREATE INDEX idx_p039_ec_code            ON pack039_energy_monitoring.em_escalation_configs(config_code);
CREATE INDEX idx_p039_ec_enabled         ON pack039_energy_monitoring.em_escalation_configs(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_ec_created         ON pack039_energy_monitoring.em_escalation_configs(created_at DESC);
CREATE INDEX idx_p039_ec_severities      ON pack039_energy_monitoring.em_escalation_configs USING GIN(applies_to_severities);
CREATE INDEX idx_p039_ec_priorities      ON pack039_energy_monitoring.em_escalation_configs USING GIN(applies_to_priorities);
CREATE INDEX idx_p039_ec_categories      ON pack039_energy_monitoring.em_escalation_configs USING GIN(applies_to_categories);
CREATE INDEX idx_p039_ec_tiers           ON pack039_energy_monitoring.em_escalation_configs USING GIN(escalation_tiers);
CREATE INDEX idx_p039_ec_alarm_ids       ON pack039_energy_monitoring.em_escalation_configs USING GIN(applies_to_alarm_ids);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_ec_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_escalation_configs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_alarm_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_alarm_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_alarm_acknowledgments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_suppression_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_escalation_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_ad_tenant_isolation
    ON pack039_energy_monitoring.em_alarm_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ad_service_bypass
    ON pack039_energy_monitoring.em_alarm_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ae_tenant_isolation
    ON pack039_energy_monitoring.em_alarm_events
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ae_service_bypass
    ON pack039_energy_monitoring.em_alarm_events
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ack_tenant_isolation
    ON pack039_energy_monitoring.em_alarm_acknowledgments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ack_service_bypass
    ON pack039_energy_monitoring.em_alarm_acknowledgments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_sr_tenant_isolation
    ON pack039_energy_monitoring.em_suppression_rules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_sr_service_bypass
    ON pack039_energy_monitoring.em_suppression_rules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ec_tenant_isolation
    ON pack039_energy_monitoring.em_escalation_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ec_service_bypass
    ON pack039_energy_monitoring.em_escalation_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_alarm_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON pack039_energy_monitoring.em_alarm_events TO PUBLIC;
GRANT SELECT, INSERT ON pack039_energy_monitoring.em_alarm_acknowledgments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_suppression_rules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_escalation_configs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_alarm_definitions IS
    'Configurable alarm definitions for real-time energy monitoring with trigger conditions, severity, and notification routing.';
COMMENT ON TABLE pack039_energy_monitoring.em_alarm_events IS
    'Individual alarm event instances with lifecycle tracking, suppression status, and escalation levels.';
COMMENT ON TABLE pack039_energy_monitoring.em_alarm_acknowledgments IS
    'Alarm acknowledgment records with operator response, initial assessment, and action tracking.';
COMMENT ON TABLE pack039_energy_monitoring.em_suppression_rules IS
    'Alarm suppression rules for maintenance windows, testing, and known operational events.';
COMMENT ON TABLE pack039_energy_monitoring.em_escalation_configs IS
    'Multi-tier escalation configuration for alarm notifications with response time targets and SLA tracking.';

COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_definitions.severity IS 'Alarm severity: INFO, WARNING, ALARM, CRITICAL, EMERGENCY (ISA-18.2 aligned).';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_definitions.sustained_duration_minutes IS 'Time in minutes the condition must persist before triggering the alarm.';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_definitions.hysteresis_value IS 'Deadband for alarm clearing. Alarm clears when value crosses threshold minus hysteresis.';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_definitions.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_events.event_status IS 'Event lifecycle: ACTIVE -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED/CLEARED.';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_events.escalation_level IS 'Current escalation tier (0 = not escalated, 1 = first escalation, etc.).';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_events.is_suppressed IS 'Whether this event was suppressed by a suppression rule (logged but not notified).';

COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_acknowledgments.response_time_minutes IS 'Time between alarm trigger and acknowledgment in minutes. Key SLA metric.';
COMMENT ON COLUMN pack039_energy_monitoring.em_alarm_acknowledgments.action_taken IS 'Initial action: INVESTIGATING, DISPATCHED, REMOTE_FIX, WORK_ORDER, etc.';

COMMENT ON COLUMN pack039_energy_monitoring.em_suppression_rules.suppression_type IS 'Reason: MAINTENANCE, TESTING, COMMISSIONING, SHUTDOWN, KNOWN_ISSUE, etc.';

COMMENT ON COLUMN pack039_energy_monitoring.em_escalation_configs.escalation_tiers IS 'JSON array of tiers: [{tier, delay_minutes, recipients, channels, sla_minutes}].';
COMMENT ON COLUMN pack039_energy_monitoring.em_escalation_configs.on_call_rotation IS 'JSON on-call schedule: {type, rotation_period, members, overrides}.';
