-- =============================================================================
-- V315: PACK-039 Energy Monitoring Pack - Views, Indexes, Audit Trail, Seed Data
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Final migration: pack-level audit trail, materialized views for
-- dashboards, real-time operational view, composite indexes for common
-- query patterns, and seed data for meter types, energy types, EnPI
-- benchmarks, alarm templates, and report templates.
--
-- Tables (1):
--   1. pack039_energy_monitoring.pack039_audit_trail
--
-- Materialized Views (3):
--   1. pack039_energy_monitoring.mv_meter_data_summary
--   2. pack039_energy_monitoring.mv_enpi_performance_summary
--   3. pack039_energy_monitoring.mv_cost_allocation_summary
--
-- Views (1):
--   1. pack039_energy_monitoring.v_energy_monitoring_dashboard
--
-- Previous: V314__pack039_energy_monitoring_009.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.pack039_audit_trail
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.pack039_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_id                UUID,
    tenant_id               UUID,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID,
    actor                   TEXT            NOT NULL,
    actor_role              VARCHAR(50),
    ip_address              VARCHAR(45),
    old_values              JSONB,
    new_values              JSONB,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_p039_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'CONFIGURE',
                   'ALERT', 'ACKNOWLEDGE', 'RESOLVE', 'SUPPRESS',
                   'ESCALATE', 'CALIBRATE', 'VALIDATE', 'CORRECT',
                   'ALLOCATE', 'BILL', 'RECONCILE', 'FORECAST',
                   'BUDGET', 'SCHEDULE', 'GENERATE_REPORT')
    )
);

CREATE INDEX idx_p039_trail_meter        ON pack039_energy_monitoring.pack039_audit_trail(meter_id);
CREATE INDEX idx_p039_trail_tenant       ON pack039_energy_monitoring.pack039_audit_trail(tenant_id);
CREATE INDEX idx_p039_trail_action       ON pack039_energy_monitoring.pack039_audit_trail(action);
CREATE INDEX idx_p039_trail_entity       ON pack039_energy_monitoring.pack039_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p039_trail_actor        ON pack039_energy_monitoring.pack039_audit_trail(actor);
CREATE INDEX idx_p039_trail_created      ON pack039_energy_monitoring.pack039_audit_trail(created_at DESC);
CREATE INDEX idx_p039_trail_details      ON pack039_energy_monitoring.pack039_audit_trail USING GIN(details);

ALTER TABLE pack039_energy_monitoring.pack039_audit_trail ENABLE ROW LEVEL SECURITY;
CREATE POLICY p039_trail_tenant_isolation ON pack039_energy_monitoring.pack039_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_trail_service_bypass ON pack039_energy_monitoring.pack039_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Materialized View 1: mv_meter_data_summary
-- =============================================================================
-- Per-meter data summary with latest readings, quality scores, alarm
-- counts, and completeness metrics for the monitoring overview dashboard.

CREATE MATERIALIZED VIEW pack039_energy_monitoring.mv_meter_data_summary AS
SELECT
    m.id AS meter_id,
    m.tenant_id,
    m.facility_id,
    m.meter_name,
    m.meter_type,
    m.meter_category,
    m.energy_type,
    m.meter_status,
    m.communication_protocol,
    m.is_revenue_grade,
    m.is_virtual,
    m.building_name,
    m.data_quality_score,
    -- Latest interval data
    (SELECT id.value FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id ORDER BY id.timestamp DESC LIMIT 1) AS latest_reading_value,
    (SELECT id.timestamp FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id ORDER BY id.timestamp DESC LIMIT 1) AS latest_reading_timestamp,
    (SELECT id.demand_kw FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id ORDER BY id.timestamp DESC LIMIT 1) AS latest_demand_kw,
    -- 24h totals
    (SELECT COALESCE(SUM(id.value), 0) FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id AND id.timestamp >= NOW() - INTERVAL '24 hours') AS energy_24h_kwh,
    (SELECT COALESCE(MAX(id.demand_kw), 0) FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id AND id.timestamp >= NOW() - INTERVAL '24 hours') AS peak_demand_24h_kw,
    -- 30d totals
    (SELECT COALESCE(SUM(id.value), 0) FROM pack039_energy_monitoring.em_interval_data id
     WHERE id.meter_id = m.id AND id.timestamp >= NOW() - INTERVAL '30 days') AS energy_30d_kwh,
    -- Data quality (latest daily)
    (SELECT qs.overall_score FROM pack039_energy_monitoring.em_quality_scores qs
     WHERE qs.meter_id = m.id AND qs.score_period_type = 'DAILY'
     ORDER BY qs.period_start DESC LIMIT 1) AS latest_quality_score,
    (SELECT qs.quality_grade FROM pack039_energy_monitoring.em_quality_scores qs
     WHERE qs.meter_id = m.id AND qs.score_period_type = 'DAILY'
     ORDER BY qs.period_start DESC LIMIT 1) AS latest_quality_grade,
    (SELECT qs.completeness_score FROM pack039_energy_monitoring.em_quality_scores qs
     WHERE qs.meter_id = m.id AND qs.score_period_type = 'DAILY'
     ORDER BY qs.period_start DESC LIMIT 1) AS latest_completeness_score,
    -- Completeness (latest daily)
    (SELECT cl.completeness_pct FROM pack039_energy_monitoring.em_completeness_logs cl
     WHERE cl.meter_id = m.id AND cl.check_period_type = 'DAILY'
     ORDER BY cl.period_start DESC LIMIT 1) AS latest_completeness_pct,
    -- Active anomalies
    (SELECT COUNT(*) FROM pack039_energy_monitoring.em_anomalies an
     WHERE an.meter_id = m.id AND an.anomaly_status IN ('DETECTED', 'ACKNOWLEDGED', 'INVESTIGATING')) AS active_anomaly_count,
    -- Active alarms
    (SELECT COUNT(*) FROM pack039_energy_monitoring.em_alarm_events ae
     WHERE ae.meter_id = m.id AND ae.event_status IN ('ACTIVE', 'ACKNOWLEDGED')) AS active_alarm_count,
    -- Acquisition status
    (SELECT sch.last_poll_status FROM pack039_energy_monitoring.em_acquisition_schedules sch
     WHERE sch.meter_id = m.id AND sch.is_enabled = true
     ORDER BY sch.last_poll_at DESC LIMIT 1) AS last_poll_status,
    (SELECT sch.last_poll_at FROM pack039_energy_monitoring.em_acquisition_schedules sch
     WHERE sch.meter_id = m.id AND sch.is_enabled = true
     ORDER BY sch.last_poll_at DESC LIMIT 1) AS last_poll_at,
    (SELECT sch.consecutive_failures FROM pack039_energy_monitoring.em_acquisition_schedules sch
     WHERE sch.meter_id = m.id AND sch.is_enabled = true
     ORDER BY sch.last_poll_at DESC LIMIT 1) AS consecutive_poll_failures,
    -- Channel count
    (SELECT COUNT(*) FROM pack039_energy_monitoring.em_meter_channels mc
     WHERE mc.meter_id = m.id AND mc.is_enabled = true) AS active_channel_count,
    -- Calibration status
    (SELECT cr.next_due_date FROM pack039_energy_monitoring.em_calibration_records cr
     WHERE cr.meter_id = m.id AND cr.pass_fail = 'PASS'
     ORDER BY cr.calibration_date DESC LIMIT 1) AS next_calibration_due
FROM pack039_energy_monitoring.em_meters m
WITH NO DATA;

CREATE UNIQUE INDEX idx_p039_mv_mds_meter ON pack039_energy_monitoring.mv_meter_data_summary(meter_id);
CREATE INDEX idx_p039_mv_mds_tenant ON pack039_energy_monitoring.mv_meter_data_summary(tenant_id);
CREATE INDEX idx_p039_mv_mds_facility ON pack039_energy_monitoring.mv_meter_data_summary(facility_id);
CREATE INDEX idx_p039_mv_mds_type ON pack039_energy_monitoring.mv_meter_data_summary(meter_type);
CREATE INDEX idx_p039_mv_mds_category ON pack039_energy_monitoring.mv_meter_data_summary(meter_category);
CREATE INDEX idx_p039_mv_mds_status ON pack039_energy_monitoring.mv_meter_data_summary(meter_status);
CREATE INDEX idx_p039_mv_mds_energy_type ON pack039_energy_monitoring.mv_meter_data_summary(energy_type);
CREATE INDEX idx_p039_mv_mds_building ON pack039_energy_monitoring.mv_meter_data_summary(building_name);

-- =============================================================================
-- Materialized View 2: mv_enpi_performance_summary
-- =============================================================================
-- EnPI performance overview by facility with latest values, improvement
-- trends, baseline comparison, and CUSUM status for management dashboards.

CREATE MATERIALIZED VIEW pack039_energy_monitoring.mv_enpi_performance_summary AS
SELECT
    ed.id AS enpi_id,
    ed.tenant_id,
    ed.facility_id,
    ed.enpi_name,
    ed.enpi_code,
    ed.enpi_type,
    ed.enpi_category,
    ed.energy_type,
    ed.output_unit,
    ed.target_value,
    ed.target_direction,
    ed.is_primary_enpi,
    ed.significant_energy_use,
    ed.is_active,
    -- Latest value
    (SELECT ev.enpi_value FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
     ORDER BY ev.period_start DESC LIMIT 1) AS latest_monthly_value,
    (SELECT ev.period_start FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
     ORDER BY ev.period_start DESC LIMIT 1) AS latest_period_start,
    (SELECT ev.improvement_pct FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
     ORDER BY ev.period_start DESC LIMIT 1) AS latest_improvement_pct,
    (SELECT ev.energy_savings_kwh FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
     ORDER BY ev.period_start DESC LIMIT 1) AS latest_savings_kwh,
    (SELECT ev.performance_status FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
     ORDER BY ev.period_start DESC LIMIT 1) AS latest_performance_status,
    -- YTD improvement
    (SELECT AVG(ev.improvement_pct) FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
       AND ev.period_start >= DATE_TRUNC('year', CURRENT_DATE)) AS ytd_avg_improvement_pct,
    (SELECT COALESCE(SUM(ev.energy_savings_kwh), 0) FROM pack039_energy_monitoring.em_enpi_values ev
     WHERE ev.enpi_id = ed.id AND ev.period_type = 'MONTHLY'
       AND ev.period_start >= DATE_TRUNC('year', CURRENT_DATE)) AS ytd_energy_savings_kwh,
    -- CUSUM latest
    (SELECT ct.cumulative_sum FROM pack039_energy_monitoring.em_cusum_tracking ct
     WHERE ct.enpi_id = ed.id AND ct.period_type = 'MONTHLY'
     ORDER BY ct.period_date DESC LIMIT 1) AS latest_cusum_value,
    (SELECT ct.trend_direction FROM pack039_energy_monitoring.em_cusum_tracking ct
     WHERE ct.enpi_id = ed.id AND ct.period_type = 'MONTHLY'
     ORDER BY ct.period_date DESC LIMIT 1) AS cusum_trend_direction,
    (SELECT ct.is_out_of_control FROM pack039_energy_monitoring.em_cusum_tracking ct
     WHERE ct.enpi_id = ed.id AND ct.period_type = 'MONTHLY'
     ORDER BY ct.period_date DESC LIMIT 1) AS cusum_out_of_control,
    -- Baseline info
    (SELECT eb.baseline_name FROM pack039_energy_monitoring.em_energy_baselines eb
     WHERE eb.id = ed.baseline_id) AS baseline_name,
    (SELECT eb.baseline_enpi_value FROM pack039_energy_monitoring.em_energy_baselines eb
     WHERE eb.id = ed.baseline_id) AS baseline_enpi_value,
    -- Regression model quality
    (SELECT rm.r_squared FROM pack039_energy_monitoring.em_regression_models rm
     WHERE rm.enpi_id = ed.id AND rm.is_current = true
     ORDER BY rm.r_squared DESC LIMIT 1) AS model_r_squared
FROM pack039_energy_monitoring.em_enpi_definitions ed
WHERE ed.is_active = true
WITH NO DATA;

CREATE UNIQUE INDEX idx_p039_mv_eps_enpi ON pack039_energy_monitoring.mv_enpi_performance_summary(enpi_id);
CREATE INDEX idx_p039_mv_eps_tenant ON pack039_energy_monitoring.mv_enpi_performance_summary(tenant_id);
CREATE INDEX idx_p039_mv_eps_facility ON pack039_energy_monitoring.mv_enpi_performance_summary(facility_id);
CREATE INDEX idx_p039_mv_eps_type ON pack039_energy_monitoring.mv_enpi_performance_summary(enpi_type);
CREATE INDEX idx_p039_mv_eps_category ON pack039_energy_monitoring.mv_enpi_performance_summary(enpi_category);
CREATE INDEX idx_p039_mv_eps_primary ON pack039_energy_monitoring.mv_enpi_performance_summary(is_primary_enpi);
CREATE INDEX idx_p039_mv_eps_status ON pack039_energy_monitoring.mv_enpi_performance_summary(latest_performance_status);

-- =============================================================================
-- Materialized View 3: mv_cost_allocation_summary
-- =============================================================================
-- Cost allocation summary by tenant, facility, and account for financial
-- dashboards. Aggregates billing period costs, energy consumption, and
-- reconciliation status across all accounts.

CREATE MATERIALIZED VIEW pack039_energy_monitoring.mv_cost_allocation_summary AS
SELECT
    ta.id AS account_id,
    ta.tenant_id,
    ta.facility_id,
    ta.account_name,
    ta.account_code,
    ta.account_type,
    ta.allocation_method,
    ta.account_status,
    -- Latest billing period
    (SELECT ca.billing_period_start FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id ORDER BY ca.billing_period_start DESC LIMIT 1) AS latest_period_start,
    (SELECT ca.energy_consumed_kwh FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id ORDER BY ca.billing_period_start DESC LIMIT 1) AS latest_energy_kwh,
    (SELECT ca.total_charge FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id ORDER BY ca.billing_period_start DESC LIMIT 1) AS latest_total_charge,
    (SELECT ca.cost_per_kwh FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id ORDER BY ca.billing_period_start DESC LIMIT 1) AS latest_cost_per_kwh,
    -- YTD totals
    (SELECT COALESCE(SUM(ca.energy_consumed_kwh), 0) FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id AND ca.billing_period_start >= DATE_TRUNC('year', CURRENT_DATE)) AS ytd_energy_kwh,
    (SELECT COALESCE(SUM(ca.total_charge), 0) FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id AND ca.billing_period_start >= DATE_TRUNC('year', CURRENT_DATE)) AS ytd_total_cost,
    (SELECT COALESCE(SUM(ca.demand_charge), 0) FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id AND ca.billing_period_start >= DATE_TRUNC('year', CURRENT_DATE)) AS ytd_demand_charges,
    -- 12-month totals
    (SELECT COALESCE(SUM(ca.energy_consumed_kwh), 0) FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id AND ca.billing_period_start >= NOW() - INTERVAL '12 months') AS energy_12m_kwh,
    (SELECT COALESCE(SUM(ca.total_charge), 0) FROM pack039_energy_monitoring.em_cost_allocations ca
     WHERE ca.account_id = ta.id AND ca.billing_period_start >= NOW() - INTERVAL '12 months') AS cost_12m_total,
    -- Billing status
    (SELECT COUNT(*) FROM pack039_energy_monitoring.em_billing_records br
     WHERE br.account_id = ta.id AND br.payment_status IN ('PENDING', 'SENT', 'OVERDUE')) AS outstanding_invoices,
    (SELECT COALESCE(SUM(br.amount_due), 0) FROM pack039_energy_monitoring.em_billing_records br
     WHERE br.account_id = ta.id AND br.payment_status IN ('PENDING', 'SENT', 'OVERDUE')) AS outstanding_amount,
    -- Budget status
    (SELECT bg.budget_status FROM pack039_energy_monitoring.em_budgets bg
     WHERE bg.account_id = ta.id AND bg.budget_status = 'ACTIVE'
     ORDER BY bg.fiscal_year DESC LIMIT 1) AS budget_status,
    (SELECT bg.ytd_variance_cost_pct FROM pack039_energy_monitoring.em_budgets bg
     WHERE bg.account_id = ta.id AND bg.budget_status = 'ACTIVE'
     ORDER BY bg.fiscal_year DESC LIMIT 1) AS budget_variance_pct
FROM pack039_energy_monitoring.em_tenant_accounts ta
WHERE ta.account_status = 'ACTIVE'
WITH NO DATA;

CREATE UNIQUE INDEX idx_p039_mv_cas_account ON pack039_energy_monitoring.mv_cost_allocation_summary(account_id);
CREATE INDEX idx_p039_mv_cas_tenant ON pack039_energy_monitoring.mv_cost_allocation_summary(tenant_id);
CREATE INDEX idx_p039_mv_cas_facility ON pack039_energy_monitoring.mv_cost_allocation_summary(facility_id);
CREATE INDEX idx_p039_mv_cas_type ON pack039_energy_monitoring.mv_cost_allocation_summary(account_type);
CREATE INDEX idx_p039_mv_cas_status ON pack039_energy_monitoring.mv_cost_allocation_summary(account_status);

-- =============================================================================
-- View: v_energy_monitoring_dashboard
-- =============================================================================
-- Real-time operations dashboard combining meter status, latest readings,
-- active alarms, anomaly counts, data quality, and budget compliance.

CREATE OR REPLACE VIEW pack039_energy_monitoring.v_energy_monitoring_dashboard AS
SELECT
    m.id AS meter_id,
    m.tenant_id,
    m.facility_id,
    m.meter_name,
    m.meter_type,
    m.energy_type,
    m.meter_status,
    m.building_name,
    m.is_revenue_grade,
    -- Latest reading
    latest_id.value AS latest_value,
    latest_id.timestamp AS latest_timestamp,
    latest_id.demand_kw AS latest_demand_kw,
    latest_id.power_factor AS latest_power_factor,
    latest_id.data_quality AS latest_data_quality,
    -- Acquisition health
    latest_sch.last_poll_status,
    latest_sch.last_poll_at,
    latest_sch.consecutive_failures AS poll_failures,
    -- Quality summary
    latest_qs.overall_score AS quality_score,
    latest_qs.quality_grade,
    latest_qs.completeness_score,
    -- Active alerts
    alarms.active_count AS active_alarms,
    alarms.critical_count AS critical_alarms,
    -- Active anomalies
    anomalies.open_count AS open_anomalies,
    anomalies.high_severity_count AS high_severity_anomalies,
    -- EnPI latest
    enpi.latest_enpi_value,
    enpi.performance_status AS enpi_status,
    enpi.improvement_pct AS enpi_improvement_pct,
    -- Calibration
    cal.next_calibration_due,
    cal.days_until_calibration
FROM pack039_energy_monitoring.em_meters m
LEFT JOIN LATERAL (
    SELECT value, timestamp, demand_kw, power_factor, data_quality
    FROM pack039_energy_monitoring.em_interval_data
    WHERE meter_id = m.id
    ORDER BY timestamp DESC
    LIMIT 1
) latest_id ON TRUE
LEFT JOIN LATERAL (
    SELECT last_poll_status, last_poll_at, consecutive_failures
    FROM pack039_energy_monitoring.em_acquisition_schedules
    WHERE meter_id = m.id AND is_enabled = true
    ORDER BY last_poll_at DESC
    LIMIT 1
) latest_sch ON TRUE
LEFT JOIN LATERAL (
    SELECT overall_score, quality_grade, completeness_score
    FROM pack039_energy_monitoring.em_quality_scores
    WHERE meter_id = m.id AND score_period_type = 'DAILY'
    ORDER BY period_start DESC
    LIMIT 1
) latest_qs ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS active_count,
           COUNT(*) FILTER (WHERE severity IN ('CRITICAL', 'EMERGENCY')) AS critical_count
    FROM pack039_energy_monitoring.em_alarm_events
    WHERE meter_id = m.id AND event_status IN ('ACTIVE', 'ACKNOWLEDGED')
) alarms ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS open_count,
           COUNT(*) FILTER (WHERE severity IN ('HIGH', 'CRITICAL')) AS high_severity_count
    FROM pack039_energy_monitoring.em_anomalies
    WHERE meter_id = m.id AND anomaly_status IN ('DETECTED', 'ACKNOWLEDGED', 'INVESTIGATING')
) anomalies ON TRUE
LEFT JOIN LATERAL (
    SELECT ev.enpi_value AS latest_enpi_value,
           ev.performance_status,
           ev.improvement_pct
    FROM pack039_energy_monitoring.em_enpi_values ev
    JOIN pack039_energy_monitoring.em_enpi_definitions ed ON ev.enpi_id = ed.id
    WHERE ed.meter_ids @> ARRAY[m.id]
      AND ed.is_primary_enpi = true AND ed.is_active = true
      AND ev.period_type = 'MONTHLY'
    ORDER BY ev.period_start DESC
    LIMIT 1
) enpi ON TRUE
LEFT JOIN LATERAL (
    SELECT cr.next_due_date AS next_calibration_due,
           (cr.next_due_date - CURRENT_DATE) AS days_until_calibration
    FROM pack039_energy_monitoring.em_calibration_records cr
    WHERE cr.meter_id = m.id AND cr.pass_fail = 'PASS'
    ORDER BY cr.calibration_date DESC
    LIMIT 1
) cal ON TRUE
WHERE m.meter_status IN ('ACTIVE', 'MAINTENANCE');

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Interval data by meter + date range for time-series charts
CREATE INDEX idx_p039_id_meter_range     ON pack039_energy_monitoring.em_interval_data(meter_id, timestamp DESC)
    WHERE data_quality IN ('RAW', 'VALIDATED', 'CORRECTED');

-- Anomalies by meter + open for operations
CREATE INDEX idx_p039_an_meter_open      ON pack039_energy_monitoring.em_anomalies(meter_id, detected_at DESC)
    WHERE anomaly_status IN ('DETECTED', 'ACKNOWLEDGED', 'INVESTIGATING');

-- Cost allocations by facility for facility-level summaries
CREATE INDEX idx_p039_ca_fac_period      ON pack039_energy_monitoring.em_cost_allocations(facility_id, billing_period_start DESC);

-- EnPI values by facility for benchmarking
CREATE INDEX idx_p039_ev_fac_period      ON pack039_energy_monitoring.em_enpi_values(enpi_id, period_start DESC)
    WHERE is_approved = true;

-- Budget periods with over-budget status for alerts
CREATE INDEX idx_p039_bp_over_budget     ON pack039_energy_monitoring.em_budget_periods(budget_id, period_start DESC)
    WHERE variance_status IN ('OVER_BUDGET', 'CRITICAL');

-- Alarm events by facility for facility-level alarm summary
CREATE INDEX idx_p039_ae_alarm_active    ON pack039_energy_monitoring.em_alarm_events(tenant_id, severity, triggered_at DESC)
    WHERE event_status IN ('ACTIVE', 'ACKNOWLEDGED');

-- Completeness logs below SLA threshold
CREATE INDEX idx_p039_cl_below_sla       ON pack039_energy_monitoring.em_completeness_logs(meter_id, period_start DESC)
    WHERE meets_threshold = false;

-- Billing records overdue for collections
CREATE INDEX idx_p039_br_overdue_coll    ON pack039_energy_monitoring.em_billing_records(tenant_id, due_date)
    WHERE payment_status = 'OVERDUE';

-- =============================================================================
-- Grants
-- =============================================================================
GRANT SELECT, INSERT ON pack039_energy_monitoring.pack039_audit_trail TO PUBLIC;
GRANT SELECT ON pack039_energy_monitoring.mv_meter_data_summary TO PUBLIC;
GRANT SELECT ON pack039_energy_monitoring.mv_enpi_performance_summary TO PUBLIC;
GRANT SELECT ON pack039_energy_monitoring.mv_cost_allocation_summary TO PUBLIC;
GRANT SELECT ON pack039_energy_monitoring.v_energy_monitoring_dashboard TO PUBLIC;

-- =============================================================================
-- Seed Data: Meter Type Reference
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.em_meter_type_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    meter_type              VARCHAR(50)     NOT NULL,
    meter_type_name         VARCHAR(100)    NOT NULL,
    description             TEXT,
    typical_accuracy_class  VARCHAR(20),
    typical_protocols       VARCHAR(50)[]   DEFAULT '{}',
    typical_interval_minutes INTEGER,
    supports_bidirectional  BOOLEAN         NOT NULL DEFAULT false,
    supports_power_quality  BOOLEAN         NOT NULL DEFAULT false,
    typical_cost_range      VARCHAR(50),
    best_application        VARCHAR(200),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p039_mtr_type UNIQUE (meter_type)
);

INSERT INTO pack039_energy_monitoring.em_meter_type_reference (meter_type, meter_type_name, description, typical_accuracy_class, typical_protocols, typical_interval_minutes, supports_bidirectional, supports_power_quality, typical_cost_range, best_application) VALUES
('INTERVAL', 'Interval Meter', 'Standard utility interval meter recording energy consumption at fixed intervals. Most common meter type for commercial and industrial facilities.', 'CLASS_0_5', '{MODBUS_TCP,DLMS_COSEM}', 15, false, false, '$500-$2,000', 'Main utility metering, demand monitoring'),
('SMART_METER', 'Smart Meter (AMI)', 'Advanced Metering Infrastructure meter with two-way communication. Deployed by utilities for automated meter reading and demand response.', 'CLASS_0_5', '{DLMS_COSEM,MQTT,HTTP_REST}', 15, true, false, '$200-$500', 'Utility revenue metering, AMI deployment'),
('SCADA', 'SCADA Meter Point', 'Measurement point within a SCADA or BMS system. Provides real-time power monitoring with high-resolution data.', 'CLASS_0_2', '{MODBUS_TCP,OPC_UA,BACNET_IP}', 1, true, true, '$1,000-$5,000', 'Industrial process monitoring, power quality'),
('PULSE', 'Pulse Output Meter', 'Meter generating pulse outputs proportional to energy consumption. Each pulse represents a fixed quantity of energy.', 'CLASS_1', '{PULSE_COUNTER,MODBUS_RTU}', 15, false, false, '$100-$500', 'Sub-metering, retrofit installations'),
('CT_LOGGER', 'CT Data Logger', 'Current transformer-based data logger for non-invasive power monitoring. Clamps around conductors without disconnecting circuits.', 'CLASS_1', '{MODBUS_TCP,HTTP_REST,MQTT}', 5, true, true, '$200-$1,500', 'Non-invasive sub-metering, temporary monitoring'),
('POWER_ANALYZER', 'Power Quality Analyzer', 'High-accuracy power measurement device capturing voltage, current, harmonics, and power quality metrics.', 'CLASS_0_1', '{MODBUS_TCP,OPC_UA,HTTP_REST}', 1, true, true, '$2,000-$10,000', 'Power quality analysis, harmonic monitoring'),
('IOT_SENSOR', 'IoT Energy Sensor', 'Low-cost IoT sensor for distributed energy monitoring. Wireless communication with cloud data aggregation.', 'CLASS_2', '{MQTT,HTTP_REST}', 5, false, false, '$50-$300', 'Distributed monitoring, equipment-level metering'),
('MANUAL_READ', 'Manual Reading', 'Manual meter reading captured by operators during periodic inspections. No automated data acquisition.', 'CLASS_2', '{MANUAL,CSV_IMPORT}', 60, false, false, '$50-$200', 'Legacy meters, small facilities, backup verification'),
('VIRTUAL', 'Virtual Meter', 'Calculated meter derived from mathematical operations on physical meter data. No physical hardware.', NULL, '{}', 15, true, false, 'N/A', 'Net metering, building aggregation, loss calculation'),
('UTILITY_AMI', 'Utility AMI Meter', 'Utility-owned advanced metering infrastructure meter. Data accessed via utility API or green button data.', 'REVENUE_GRADE', '{API,HTTP_REST}', 15, true, false, 'Utility-owned', 'Revenue metering, utility billing data'),
('DATA_LOGGER', 'Standalone Data Logger', 'Independent data recording device with local storage and periodic upload. Battery-backed for reliability.', 'CLASS_1', '{CSV_IMPORT,MODBUS_RTU,HTTP_REST}', 15, false, false, '$300-$1,000', 'Remote sites, temporary monitoring campaigns'),
('BMS_POINT', 'BMS Integration Point', 'Energy measurement point integrated from Building Management System. Data accessed via BACnet or API.', 'CLASS_1', '{BACNET_IP,BACNET_MSTP,OPC_UA}', 5, false, true, 'Included in BMS', 'HVAC energy monitoring, building automation');

GRANT SELECT ON pack039_energy_monitoring.em_meter_type_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Energy Type Reference
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.em_energy_type_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    energy_type             VARCHAR(50)     NOT NULL,
    energy_type_name        VARCHAR(100)    NOT NULL,
    description             TEXT,
    default_unit            VARCHAR(20)     NOT NULL,
    conversion_to_kwh       NUMERIC(15,6),
    emission_factor_kg_co2_per_kwh NUMERIC(10,6),
    typical_cost_per_unit   NUMERIC(10,4),
    cost_currency           VARCHAR(3)      DEFAULT 'USD',
    is_renewable            BOOLEAN         NOT NULL DEFAULT false,
    primary_energy_factor   NUMERIC(5,3)    NOT NULL DEFAULT 1.0,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p039_etr_type UNIQUE (energy_type)
);

INSERT INTO pack039_energy_monitoring.em_energy_type_reference (energy_type, energy_type_name, description, default_unit, conversion_to_kwh, emission_factor_kg_co2_per_kwh, typical_cost_per_unit, is_renewable, primary_energy_factor) VALUES
('ELECTRICITY', 'Electricity', 'Grid-supplied electrical energy. Most commonly metered energy type for commercial and industrial facilities.', 'kWh', 1.0, 0.417, 0.1200, false, 2.500),
('NATURAL_GAS', 'Natural Gas', 'Pipeline-delivered natural gas for heating, cooking, and industrial processes. Measured in therms, CCF, or MCF.', 'therms', 29.3001, 0.181, 1.2000, false, 1.100),
('STEAM', 'Steam', 'District or on-site generated steam for space heating, process heating, and humidification.', 'tonnes_steam', 694.4444, 0.230, 25.0000, false, 1.200),
('CHILLED_WATER', 'Chilled Water', 'District or on-site chilled water for space cooling and process cooling.', 'ton_hours', 3.5169, 0.500, 0.1500, false, 1.500),
('HOT_WATER', 'Hot Water', 'District or on-site hot water for space heating and domestic hot water.', 'kWh', 1.0, 0.200, 0.0800, false, 1.100),
('COMPRESSED_AIR', 'Compressed Air', 'Compressed air system energy for pneumatic tools, actuators, and industrial processes.', 'Nm3', 0.1100, 0.417, 0.0300, false, 2.500),
('DIESEL', 'Diesel Fuel', 'Diesel fuel for generators, vehicles, and heating. Measured in liters or gallons.', 'liters', 10.7000, 2.689, 1.5000, false, 1.000),
('PROPANE', 'Propane (LPG)', 'Liquefied petroleum gas for heating, cooking, and emergency generation.', 'liters', 7.0800, 1.536, 0.8000, false, 1.000),
('FUEL_OIL', 'Fuel Oil', 'Heavy fuel oil for heating and industrial boilers. Measured in liters or gallons.', 'liters', 11.3000, 2.960, 1.2000, false, 1.000),
('DISTRICT_HEATING', 'District Heating', 'Centralized district heating system delivering hot water or steam from a utility provider.', 'kWh', 1.0, 0.200, 0.0900, false, 1.100),
('DISTRICT_COOLING', 'District Cooling', 'Centralized district cooling system delivering chilled water from a utility provider.', 'kWh', 1.0, 0.450, 0.1100, false, 1.300),
('SOLAR_THERMAL', 'Solar Thermal', 'Solar thermal energy collection for water heating and space heating. Renewable energy source.', 'kWh', 1.0, 0.000, 0.0000, true, 1.000),
('BIOMASS', 'Biomass', 'Biomass energy from wood chips, pellets, or agricultural waste for heating and power generation.', 'kWh', 1.0, 0.015, 0.0600, true, 1.000),
('HYDROGEN', 'Hydrogen', 'Hydrogen fuel for fuel cells, industrial processes, and heating. Measured in kg.', 'kg', 33.3000, 0.000, 5.0000, true, 1.000);

GRANT SELECT ON pack039_energy_monitoring.em_energy_type_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: EnPI Benchmark Reference
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.em_enpi_benchmark_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_type           VARCHAR(50)     NOT NULL,
    facility_type_name      VARCHAR(100)    NOT NULL,
    enpi_name               VARCHAR(100)    NOT NULL,
    enpi_unit               VARCHAR(50)     NOT NULL,
    benchmark_median        NUMERIC(15,6)   NOT NULL,
    benchmark_25th_pct      NUMERIC(15,6),
    benchmark_75th_pct      NUMERIC(15,6),
    benchmark_best_in_class NUMERIC(15,6),
    energy_star_target      NUMERIC(15,6),
    source                  VARCHAR(200),
    country_code            CHAR(2)         DEFAULT 'US',
    climate_zone            VARCHAR(20),
    year                    INTEGER         DEFAULT 2024,
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p039_ebr_facility_enpi UNIQUE (facility_type, enpi_name, country_code, climate_zone)
);

INSERT INTO pack039_energy_monitoring.em_enpi_benchmark_reference (facility_type, facility_type_name, enpi_name, enpi_unit, benchmark_median, benchmark_25th_pct, benchmark_75th_pct, benchmark_best_in_class, energy_star_target, source, climate_zone) VALUES
('OFFICE', 'Office Building', 'Energy Use Intensity', 'kWh/m2/year', 200.0, 150.0, 270.0, 90.0, 130.0, 'CBECS 2018, Energy Star', 'MIXED_HUMID'),
('OFFICE', 'Office Building', 'Energy Use Intensity', 'kWh/m2/year', 180.0, 130.0, 240.0, 80.0, 115.0, 'CBECS 2018, Energy Star', 'MARINE'),
('RETAIL', 'Retail Store', 'Energy Use Intensity', 'kWh/m2/year', 250.0, 180.0, 340.0, 120.0, 170.0, 'CBECS 2018', 'MIXED_HUMID'),
('WAREHOUSE', 'Warehouse', 'Energy Use Intensity', 'kWh/m2/year', 80.0, 50.0, 130.0, 30.0, 55.0, 'CBECS 2018', 'MIXED_HUMID'),
('MANUFACTURING', 'Manufacturing Facility', 'Energy per Production Unit', 'kWh/unit', 15.0, 8.0, 25.0, 5.0, NULL, 'Industry Average', NULL),
('DATA_CENTER', 'Data Center', 'PUE (Power Usage Effectiveness)', 'ratio', 1.58, 1.40, 1.80, 1.10, 1.30, 'Uptime Institute 2024', NULL),
('HOSPITAL', 'Hospital', 'Energy Use Intensity', 'kWh/m2/year', 550.0, 400.0, 700.0, 300.0, 380.0, 'CBECS 2018, Energy Star', 'MIXED_HUMID'),
('UNIVERSITY', 'University Campus', 'Energy Use Intensity', 'kWh/m2/year', 280.0, 200.0, 370.0, 140.0, 190.0, 'CBECS 2018', 'MIXED_HUMID'),
('HOTEL', 'Hotel', 'Energy per Occupied Room Night', 'kWh/room-night', 60.0, 40.0, 85.0, 25.0, 38.0, 'Energy Star', 'MIXED_HUMID'),
('COLD_STORAGE', 'Cold Storage', 'Energy per Volume', 'kWh/m3/year', 180.0, 120.0, 250.0, 80.0, NULL, 'Industry Average', NULL),
('SUPERMARKET', 'Supermarket', 'Energy per Sales Area', 'kWh/m2/year', 550.0, 400.0, 700.0, 280.0, 370.0, 'Energy Star', 'MIXED_HUMID'),
('SCHOOL', 'K-12 School', 'Energy Use Intensity', 'kWh/m2/year', 150.0, 100.0, 210.0, 65.0, 95.0, 'Energy Star', 'MIXED_HUMID');

GRANT SELECT ON pack039_energy_monitoring.em_enpi_benchmark_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Alarm Template Reference
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.em_alarm_template_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    template_code           VARCHAR(50)     NOT NULL,
    template_name           VARCHAR(255)    NOT NULL,
    alarm_category          VARCHAR(50)     NOT NULL,
    alarm_type              VARCHAR(30)     NOT NULL,
    severity                VARCHAR(20)     NOT NULL,
    condition_type          VARCHAR(30)     NOT NULL,
    description             TEXT,
    default_threshold       NUMERIC(18,6),
    threshold_unit          VARCHAR(30),
    default_sustained_minutes INTEGER,
    default_cooldown_minutes INTEGER,
    recommended_for         VARCHAR(200),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p039_atr_code UNIQUE (template_code)
);

INSERT INTO pack039_energy_monitoring.em_alarm_template_reference (template_code, template_name, alarm_category, alarm_type, severity, condition_type, description, default_threshold, threshold_unit, default_sustained_minutes, default_cooldown_minutes, recommended_for) VALUES
('ALM_DEMAND_SPIKE', 'Demand Spike Alert', 'DEMAND', 'THRESHOLD', 'ALARM', 'GREATER_THAN', 'Triggers when instantaneous demand exceeds a configured threshold. Used for peak demand management.', NULL, 'kW', 15, 60, 'All commercial/industrial facilities'),
('ALM_BASELOAD_HIGH', 'Elevated Baseload', 'CONSUMPTION', 'DEVIATION', 'WARNING', 'GREATER_THAN', 'Triggers when off-hours baseload exceeds normal levels by a configured percentage.', 20.0, 'percent', 60, 120, 'Office buildings, retail stores'),
('ALM_COMM_FAILURE', 'Communication Failure', 'COMMUNICATION', 'ABSENCE', 'CRITICAL', 'MISSING_DATA', 'Triggers when no data is received from a meter for a configured period.', 30.0, 'minutes', 0, 30, 'All meters'),
('ALM_DATA_QUALITY', 'Data Quality Degradation', 'DATA_QUALITY', 'THRESHOLD', 'WARNING', 'LESS_THAN', 'Triggers when data quality score drops below threshold.', 80.0, 'percent', 0, 60, 'Revenue-grade meters'),
('ALM_POWER_FACTOR', 'Low Power Factor', 'POWER_QUALITY', 'THRESHOLD', 'WARNING', 'LESS_THAN', 'Triggers when power factor drops below utility penalty threshold.', 0.90, 'ratio', 15, 60, 'Industrial facilities, large motors'),
('ALM_VOLTAGE_SAG', 'Voltage Sag', 'POWER_QUALITY', 'THRESHOLD', 'ALARM', 'LESS_THAN', 'Triggers when voltage drops below acceptable range.', 0.90, 'pu', 1, 30, 'Sensitive equipment, data centers'),
('ALM_OVERCONSUMPTION', 'Budget Overconsumption', 'CONSUMPTION', 'DEVIATION', 'WARNING', 'GREATER_THAN', 'Triggers when daily consumption exceeds budget by configured percentage.', 15.0, 'percent', 0, 1440, 'Budget-tracked facilities'),
('ALM_METER_OFFLINE', 'Meter Offline', 'METER', 'STATE_CHANGE', 'CRITICAL', 'STATE_CHANGE', 'Triggers when meter transitions to OFFLINE or FAULT status.', NULL, NULL, 0, 60, 'All active meters'),
('ALM_STUCK_VALUE', 'Stuck Meter Value', 'METER', 'PATTERN', 'WARNING', 'STUCK_VALUE', 'Triggers when meter reports identical values for extended period indicating a potential fault.', 60.0, 'minutes', 60, 120, 'All interval meters'),
('ALM_CAL_DUE', 'Calibration Due', 'METER', 'THRESHOLD', 'INFO', 'LESS_THAN', 'Triggers when meter calibration due date is within configured days.', 30.0, 'days', 0, 1440, 'Revenue-grade meters');

GRANT SELECT ON pack039_energy_monitoring.em_alarm_template_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Report Template Reference
-- =============================================================================
CREATE TABLE pack039_energy_monitoring.em_report_template_reference (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    template_code           VARCHAR(50)     NOT NULL,
    template_name           VARCHAR(255)    NOT NULL,
    report_type             VARCHAR(50)     NOT NULL,
    report_category         VARCHAR(30)     NOT NULL,
    description             TEXT,
    default_format          VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    default_frequency       VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    typical_sections        VARCHAR(100)[]  DEFAULT '{}',
    target_audience         VARCHAR(30)     NOT NULL,
    estimated_pages         INTEGER,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p039_rtr_code UNIQUE (template_code)
);

INSERT INTO pack039_energy_monitoring.em_report_template_reference (template_code, template_name, report_type, report_category, description, default_format, default_frequency, typical_sections, target_audience, estimated_pages) VALUES
('RPT_MONTHLY_ENERGY', 'Monthly Energy Summary', 'MONTHLY_SUMMARY', 'OPERATIONAL', 'Comprehensive monthly energy consumption summary with demand profiles, cost breakdown, and year-over-year comparison.', 'PDF', 'MONTHLY', '{consumption_summary,demand_analysis,cost_breakdown,yoy_comparison,meter_status}', 'OPERATIONS', 8),
('RPT_ENPI_TRACKING', 'EnPI Performance Report', 'ENPI_REPORT', 'COMPLIANCE', 'ISO 50001 EnPI tracking report with baseline comparison, CUSUM charts, and regression model performance.', 'PDF', 'MONTHLY', '{enpi_summary,baseline_comparison,cusum_chart,regression_analysis,improvement_actions}', 'MANAGEMENT', 12),
('RPT_VARIANCE_ANALYSIS', 'Budget Variance Analysis', 'VARIANCE_REPORT', 'FINANCIAL', 'Budget vs actual variance report with weather normalization and factor decomposition.', 'EXCEL', 'MONTHLY', '{variance_summary,factor_decomposition,weather_analysis,forecast_update,recommendations}', 'FINANCE', 6),
('RPT_COST_ALLOCATION', 'Cost Allocation Statement', 'COST_ALLOCATION', 'FINANCIAL', 'Tenant cost allocation statement with energy charges, demand charges, and common area distribution.', 'PDF', 'MONTHLY', '{allocation_summary,charge_breakdown,usage_comparison,reconciliation_status}', 'FINANCE', 4),
('RPT_ALARM_SUMMARY', 'Alarm Summary Report', 'ALARM_SUMMARY', 'OPERATIONAL', 'Weekly alarm event summary with response time metrics, recurring alarms, and false alarm analysis.', 'PDF', 'WEEKLY', '{alarm_summary,severity_breakdown,response_metrics,recurring_alarms,suppressed_events}', 'OPERATIONS', 5),
('RPT_DATA_QUALITY', 'Data Quality Report', 'DATA_QUALITY', 'COMPLIANCE', 'Data quality assessment with completeness metrics, validation results, and correction audit trail.', 'PDF', 'MONTHLY', '{quality_scores,completeness_analysis,validation_summary,correction_audit,sla_compliance}', 'ENGINEERING', 6),
('RPT_EXECUTIVE_DASH', 'Executive Energy Dashboard', 'EXECUTIVE_DASHBOARD', 'EXECUTIVE', 'High-level executive summary with portfolio energy KPIs, cost trends, and sustainability progress.', 'PDF', 'MONTHLY', '{kpi_scorecard,cost_trend,consumption_trend,sustainability_progress,anomaly_highlights}', 'MANAGEMENT', 3),
('RPT_ANOMALY_DIGEST', 'Anomaly Investigation Digest', 'ANOMALY_REPORT', 'ENGINEERING', 'Anomaly detection summary with investigation status, root cause analysis, and energy impact assessment.', 'PDF', 'WEEKLY', '{anomaly_summary,investigation_status,root_cause_analysis,energy_impact,recommendations}', 'ENGINEERING', 7),
('RPT_BENCHMARK', 'Energy Benchmarking Report', 'BENCHMARKING', 'SUSTAINABILITY', 'Facility benchmarking report comparing energy performance against industry standards and peer buildings.', 'PDF', 'QUARTERLY', '{eui_comparison,peer_ranking,enpi_benchmarks,improvement_potential,action_plan}', 'SUSTAINABILITY', 10),
('RPT_BILLING_STMT', 'Tenant Billing Statement', 'BILLING_REPORT', 'FINANCIAL', 'Detailed tenant billing statement with consumption breakdown, rate application, and payment history.', 'PDF', 'MONTHLY', '{billing_summary,consumption_detail,rate_application,payment_history,usage_chart}', 'TENANT', 3);

GRANT SELECT ON pack039_energy_monitoring.em_report_template_reference TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.pack039_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-039 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack039_energy_monitoring.mv_meter_data_summary IS
    'Per-meter data summary with latest readings, quality scores, alarm counts, and acquisition health. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack039_energy_monitoring.mv_meter_data_summary;';
COMMENT ON MATERIALIZED VIEW pack039_energy_monitoring.mv_enpi_performance_summary IS
    'EnPI performance overview by facility with latest values, improvement trends, and CUSUM status. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack039_energy_monitoring.mv_enpi_performance_summary;';
COMMENT ON MATERIALIZED VIEW pack039_energy_monitoring.mv_cost_allocation_summary IS
    'Cost allocation summary by account with YTD costs, outstanding invoices, and budget variance. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack039_energy_monitoring.mv_cost_allocation_summary;';
COMMENT ON VIEW pack039_energy_monitoring.v_energy_monitoring_dashboard IS
    'Real-time operations dashboard combining meter status, latest readings, active alarms, anomalies, EnPI, and calibration status.';

COMMENT ON TABLE pack039_energy_monitoring.em_meter_type_reference IS
    'Reference table for meter types with typical specifications, accuracy classes, and recommended applications.';
COMMENT ON TABLE pack039_energy_monitoring.em_energy_type_reference IS
    'Reference table for energy types with conversion factors, emission factors, and primary energy factors.';
COMMENT ON TABLE pack039_energy_monitoring.em_enpi_benchmark_reference IS
    'EnPI benchmark reference values by facility type, climate zone, and country for performance comparison.';
COMMENT ON TABLE pack039_energy_monitoring.em_alarm_template_reference IS
    'Alarm definition templates with recommended thresholds and configurations for common monitoring scenarios.';
COMMENT ON TABLE pack039_energy_monitoring.em_report_template_reference IS
    'Report template catalog with standard report types, sections, and target audiences.';
