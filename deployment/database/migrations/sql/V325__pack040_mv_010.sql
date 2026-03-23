-- =============================================================================
-- V325: PACK-040 M&V Pack - Views, Indexes, Audit Trail, Seed Data
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    010 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Final migration: pack-level audit trail, materialized views for
-- dashboards, real-time operational view, composite indexes for common
-- query patterns, and seed data for IPMVP options, ASHRAE 14 criteria,
-- compliance frameworks, meter accuracy classes, and baseline model types.
--
-- Tables (1):
--   1. pack040_mv.pack040_audit_trail
--
-- Materialized Views (3):
--   1. pack040_mv.mv_project_savings_summary
--   2. pack040_mv.mv_baseline_model_summary
--   3. pack040_mv.mv_compliance_status_summary
--
-- Views (1):
--   1. pack040_mv.v_mv_dashboard
--
-- Previous: V324__pack040_mv_009.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.pack040_audit_trail
-- =============================================================================
CREATE TABLE pack040_mv.pack040_audit_trail (
    audit_trail_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id                  UUID,
    tenant_id                   UUID,
    action                      VARCHAR(50)     NOT NULL,
    entity_type                 VARCHAR(100)    NOT NULL,
    entity_id                   UUID,
    actor                       TEXT            NOT NULL,
    actor_role                  VARCHAR(50),
    ip_address                  VARCHAR(45),
    old_values                  JSONB,
    new_values                  JSONB,
    details                     JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_p040_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'CONFIGURE',
                   'GENERATE_REPORT', 'RUN_REGRESSION', 'VALIDATE_MODEL',
                   'CALCULATE_SAVINGS', 'CALCULATE_UNCERTAINTY',
                   'EVALUATE_OPTION', 'SELECT_OPTION', 'CHECK_COMPLIANCE',
                   'TRACK_PERSISTENCE', 'DETECT_DEGRADATION',
                   'TRIGGER_RECOMMISSIONING', 'CALIBRATE_METER',
                   'ADJUST_BASELINE', 'SCHEDULE_REPORT')
    )
);

CREATE INDEX idx_p040_trail_project        ON pack040_mv.pack040_audit_trail(project_id);
CREATE INDEX idx_p040_trail_tenant         ON pack040_mv.pack040_audit_trail(tenant_id);
CREATE INDEX idx_p040_trail_action         ON pack040_mv.pack040_audit_trail(action);
CREATE INDEX idx_p040_trail_entity         ON pack040_mv.pack040_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p040_trail_actor          ON pack040_mv.pack040_audit_trail(actor);
CREATE INDEX idx_p040_trail_created        ON pack040_mv.pack040_audit_trail(created_at DESC);
CREATE INDEX idx_p040_trail_details        ON pack040_mv.pack040_audit_trail USING GIN(details);

ALTER TABLE pack040_mv.pack040_audit_trail ENABLE ROW LEVEL SECURITY;
CREATE POLICY p040_trail_tenant_isolation ON pack040_mv.pack040_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_trail_service_bypass ON pack040_mv.pack040_audit_trail
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Materialized View 1: mv_project_savings_summary
-- =============================================================================
-- Per-project savings overview with latest savings, cumulative totals,
-- uncertainty, persistence status, and guarantee performance.

CREATE MATERIALIZED VIEW pack040_mv.mv_project_savings_summary AS
SELECT
    p.id AS project_id,
    p.tenant_id,
    p.facility_id,
    p.project_name,
    p.project_code,
    p.project_type,
    p.project_status,
    p.ipmvp_option,
    p.compliance_framework,
    p.energy_type,
    p.baseline_period_start,
    p.baseline_period_end,
    p.reporting_period_start,
    p.reporting_period_end,
    p.guaranteed_savings_kwh,
    p.guaranteed_cost_savings,
    -- Latest savings period
    (SELECT sp.avoided_energy_kwh FROM pack040_mv.mv_savings_periods sp
     WHERE sp.project_id = p.id AND sp.period_status IN ('APPROVED', 'FINAL')
     ORDER BY sp.period_start DESC LIMIT 1) AS latest_avoided_energy_kwh,
    (SELECT sp.savings_pct FROM pack040_mv.mv_savings_periods sp
     WHERE sp.project_id = p.id AND sp.period_status IN ('APPROVED', 'FINAL')
     ORDER BY sp.period_start DESC LIMIT 1) AS latest_savings_pct,
    (SELECT sp.period_start FROM pack040_mv.mv_savings_periods sp
     WHERE sp.project_id = p.id AND sp.period_status IN ('APPROVED', 'FINAL')
     ORDER BY sp.period_start DESC LIMIT 1) AS latest_period_start,
    -- Cumulative savings
    (SELECT cs.ltd_avoided_energy_kwh FROM pack040_mv.mv_cumulative_savings cs
     WHERE cs.project_id = p.id ORDER BY cs.as_of_date DESC LIMIT 1) AS ltd_avoided_energy_kwh,
    (SELECT cs.ltd_total_cost_savings FROM pack040_mv.mv_cumulative_savings cs
     WHERE cs.project_id = p.id ORDER BY cs.as_of_date DESC LIMIT 1) AS ltd_total_cost_savings,
    (SELECT cs.guaranteed_pct_achieved FROM pack040_mv.mv_cumulative_savings cs
     WHERE cs.project_id = p.id ORDER BY cs.as_of_date DESC LIMIT 1) AS guaranteed_pct_achieved,
    (SELECT cs.simple_payback_achieved FROM pack040_mv.mv_cumulative_savings cs
     WHERE cs.project_id = p.id ORDER BY cs.as_of_date DESC LIMIT 1) AS payback_achieved,
    -- Latest FSU
    (SELECT f.fsu_68_pct FROM pack040_mv.mv_fsu_records f
     WHERE f.project_id = p.id ORDER BY f.created_at DESC LIMIT 1) AS latest_fsu_68_pct,
    (SELECT f.passes_ashrae_14 FROM pack040_mv.mv_fsu_records f
     WHERE f.project_id = p.id ORDER BY f.created_at DESC LIMIT 1) AS passes_ashrae_14,
    (SELECT f.savings_significant FROM pack040_mv.mv_fsu_records f
     WHERE f.project_id = p.id ORDER BY f.created_at DESC LIMIT 1) AS savings_significant,
    -- Persistence
    (SELECT pr.persistence_factor FROM pack040_mv.mv_persistence_records pr
     WHERE pr.project_id = p.id ORDER BY pr.tracking_year DESC LIMIT 1) AS latest_persistence_factor,
    (SELECT pr.persistence_status FROM pack040_mv.mv_persistence_records pr
     WHERE pr.project_id = p.id ORDER BY pr.tracking_year DESC LIMIT 1) AS persistence_status,
    (SELECT pr.trend_direction FROM pack040_mv.mv_persistence_records pr
     WHERE pr.project_id = p.id ORDER BY pr.tracking_year DESC LIMIT 1) AS persistence_trend,
    -- ECM count
    (SELECT COUNT(*) FROM pack040_mv.mv_project_ecm_map pem
     WHERE pem.project_id = p.id) AS total_ecm_count,
    (SELECT COUNT(*) FROM pack040_mv.mv_project_ecm_map pem
     WHERE pem.project_id = p.id AND pem.verification_status = 'VERIFIED') AS verified_ecm_count,
    -- Active alerts
    (SELECT COUNT(*) FROM pack040_mv.mv_persistence_alerts pa
     WHERE pa.project_id = p.id AND pa.alert_status IN ('ACTIVE', 'ACKNOWLEDGED')) AS active_alert_count,
    -- Compliance
    (SELECT cc.overall_status FROM pack040_mv.mv_compliance_checks cc
     WHERE cc.project_id = p.id ORDER BY cc.check_date DESC LIMIT 1) AS compliance_status,
    (SELECT cc.compliance_pct FROM pack040_mv.mv_compliance_checks cc
     WHERE cc.project_id = p.id ORDER BY cc.check_date DESC LIMIT 1) AS compliance_pct
FROM pack040_mv.mv_projects p
WITH NO DATA;

CREATE UNIQUE INDEX idx_p040_mv_pss_project ON pack040_mv.mv_project_savings_summary(project_id);
CREATE INDEX idx_p040_mv_pss_tenant ON pack040_mv.mv_project_savings_summary(tenant_id);
CREATE INDEX idx_p040_mv_pss_facility ON pack040_mv.mv_project_savings_summary(facility_id);
CREATE INDEX idx_p040_mv_pss_status ON pack040_mv.mv_project_savings_summary(project_status);
CREATE INDEX idx_p040_mv_pss_option ON pack040_mv.mv_project_savings_summary(ipmvp_option);
CREATE INDEX idx_p040_mv_pss_framework ON pack040_mv.mv_project_savings_summary(compliance_framework);
CREATE INDEX idx_p040_mv_pss_persistence ON pack040_mv.mv_project_savings_summary(persistence_status);

-- =============================================================================
-- Materialized View 2: mv_baseline_model_summary
-- =============================================================================
-- Summary of all current baseline models with regression statistics,
-- validation status, and model quality indicators.

CREATE MATERIALIZED VIEW pack040_mv.mv_baseline_model_summary AS
SELECT
    bl.id AS baseline_id,
    bl.tenant_id,
    bl.project_id,
    p.project_name,
    p.facility_id,
    bl.baseline_name,
    bl.baseline_version,
    bl.model_type,
    bl.data_granularity,
    bl.baseline_period_start,
    bl.baseline_period_end,
    bl.num_data_points,
    bl.num_excluded_points,
    bl.independent_variables,
    -- Key statistics
    bl.r_squared,
    bl.adjusted_r_squared,
    bl.cvrmse_pct,
    bl.nmbe_pct,
    bl.f_statistic,
    bl.f_p_value,
    bl.durbin_watson,
    bl.intercept,
    bl.change_point_1,
    bl.change_point_2,
    bl.balance_point_heating_f,
    bl.balance_point_cooling_f,
    -- Validation
    bl.validation_status,
    bl.passes_cvrmse,
    bl.passes_nmbe,
    bl.passes_r_squared,
    bl.passes_all_criteria,
    -- Diagnostics summary
    (SELECT COUNT(*) FROM pack040_mv.mv_model_diagnostics md
     WHERE md.baseline_id = bl.id AND md.result = 'FAIL') AS failed_diagnostic_count,
    (SELECT COUNT(*) FROM pack040_mv.mv_model_diagnostics md
     WHERE md.baseline_id = bl.id AND md.result = 'WARNING') AS warning_diagnostic_count,
    -- Regression parameter count
    (SELECT COUNT(*) FROM pack040_mv.mv_regression_params rp
     WHERE rp.baseline_id = bl.id) AS param_count,
    (SELECT COUNT(*) FROM pack040_mv.mv_regression_params rp
     WHERE rp.baseline_id = bl.id AND rp.is_significant = true) AS significant_param_count,
    -- Data quality
    (SELECT COUNT(*) FROM pack040_mv.mv_baseline_data bd
     WHERE bd.baseline_id = bl.id AND bd.is_excluded = true) AS excluded_point_count,
    (SELECT COUNT(*) FROM pack040_mv.mv_baseline_data bd
     WHERE bd.baseline_id = bl.id AND bd.is_influential = true) AS influential_point_count,
    -- Model comparison rank
    (SELECT mc.candidate_rank FROM pack040_mv.mv_model_comparisons mc
     WHERE mc.candidate_baseline_id = bl.id AND mc.is_selected = true
     LIMIT 1) AS comparison_rank,
    (SELECT mc.overall_score FROM pack040_mv.mv_model_comparisons mc
     WHERE mc.candidate_baseline_id = bl.id AND mc.is_selected = true
     LIMIT 1) AS comparison_score,
    bl.model_equation,
    bl.approved_by,
    bl.approved_at
FROM pack040_mv.mv_baselines bl
JOIN pack040_mv.mv_projects p ON bl.project_id = p.id
WHERE bl.is_current = true
WITH NO DATA;

CREATE UNIQUE INDEX idx_p040_mv_bms_baseline ON pack040_mv.mv_baseline_model_summary(baseline_id);
CREATE INDEX idx_p040_mv_bms_tenant ON pack040_mv.mv_baseline_model_summary(tenant_id);
CREATE INDEX idx_p040_mv_bms_project ON pack040_mv.mv_baseline_model_summary(project_id);
CREATE INDEX idx_p040_mv_bms_model ON pack040_mv.mv_baseline_model_summary(model_type);
CREATE INDEX idx_p040_mv_bms_valid ON pack040_mv.mv_baseline_model_summary(validation_status);
CREATE INDEX idx_p040_mv_bms_passes ON pack040_mv.mv_baseline_model_summary(passes_all_criteria);
CREATE INDEX idx_p040_mv_bms_granularity ON pack040_mv.mv_baseline_model_summary(data_granularity);

-- =============================================================================
-- Materialized View 3: mv_compliance_status_summary
-- =============================================================================
-- Compliance status overview by project and framework for compliance
-- dashboards and executive reporting.

CREATE MATERIALIZED VIEW pack040_mv.mv_compliance_status_summary AS
SELECT
    cc.id AS check_id,
    cc.tenant_id,
    cc.project_id,
    p.project_name,
    p.project_code,
    p.facility_id,
    cc.compliance_framework,
    cc.framework_version,
    cc.check_date,
    cc.total_requirements,
    cc.requirements_met,
    cc.requirements_not_met,
    cc.requirements_not_applicable,
    cc.requirements_with_warnings,
    cc.compliance_pct,
    cc.overall_status,
    cc.cvrmse_check,
    cc.nmbe_check,
    cc.r_squared_check,
    cc.fsu_check,
    cc.data_completeness_check,
    cc.documentation_check,
    -- Findings summary
    (SELECT COUNT(*) FROM pack040_mv.mv_compliance_findings cf
     WHERE cf.compliance_check_id = cc.id AND cf.finding_status = 'NON_COMPLIANT') AS non_compliant_findings,
    (SELECT COUNT(*) FROM pack040_mv.mv_compliance_findings cf
     WHERE cf.compliance_check_id = cc.id AND cf.finding_severity = 'CRITICAL') AS critical_findings,
    (SELECT COUNT(*) FROM pack040_mv.mv_compliance_findings cf
     WHERE cf.compliance_check_id = cc.id AND cf.corrective_action_required = true
     AND cf.corrective_action_status NOT IN ('COMPLETED', 'VERIFIED')) AS open_corrective_actions,
    -- Report reference
    (SELECT ro.id FROM pack040_mv.mv_report_outputs ro
     WHERE ro.id = cc.report_output_id LIMIT 1) AS report_id,
    (SELECT ro.review_status FROM pack040_mv.mv_report_outputs ro
     WHERE ro.id = cc.report_output_id LIMIT 1) AS report_review_status,
    cc.checked_by,
    cc.reviewed_by
FROM pack040_mv.mv_compliance_checks cc
JOIN pack040_mv.mv_projects p ON cc.project_id = p.id
WHERE cc.check_date = (
    SELECT MAX(cc2.check_date)
    FROM pack040_mv.mv_compliance_checks cc2
    WHERE cc2.project_id = cc.project_id AND cc2.compliance_framework = cc.compliance_framework
)
WITH NO DATA;

CREATE UNIQUE INDEX idx_p040_mv_css_check ON pack040_mv.mv_compliance_status_summary(check_id);
CREATE INDEX idx_p040_mv_css_tenant ON pack040_mv.mv_compliance_status_summary(tenant_id);
CREATE INDEX idx_p040_mv_css_project ON pack040_mv.mv_compliance_status_summary(project_id);
CREATE INDEX idx_p040_mv_css_framework ON pack040_mv.mv_compliance_status_summary(compliance_framework);
CREATE INDEX idx_p040_mv_css_status ON pack040_mv.mv_compliance_status_summary(overall_status);
CREATE INDEX idx_p040_mv_css_date ON pack040_mv.mv_compliance_status_summary(check_date DESC);

-- =============================================================================
-- View: v_mv_dashboard
-- =============================================================================
-- Real-time M&V operations dashboard combining project status, latest
-- savings, persistence, compliance, and active alerts.

CREATE OR REPLACE VIEW pack040_mv.v_mv_dashboard AS
SELECT
    p.id AS project_id,
    p.tenant_id,
    p.facility_id,
    p.project_name,
    p.project_code,
    p.project_type,
    p.project_status,
    p.ipmvp_option,
    p.compliance_framework,
    p.energy_type,
    p.guaranteed_savings_kwh,
    -- Baseline info
    latest_bl.baseline_name,
    latest_bl.model_type,
    latest_bl.r_squared,
    latest_bl.cvrmse_pct,
    latest_bl.nmbe_pct,
    latest_bl.validation_status AS baseline_validation,
    -- Latest savings
    latest_sp.avoided_energy_kwh AS latest_savings_kwh,
    latest_sp.savings_pct AS latest_savings_pct,
    latest_sp.period_start AS latest_savings_period,
    latest_sp.period_status AS savings_status,
    -- Latest cost savings
    latest_cs.total_cost_savings AS latest_cost_savings,
    latest_cs.currency_code,
    -- Cumulative
    latest_cum.ltd_avoided_energy_kwh,
    latest_cum.ltd_total_cost_savings,
    latest_cum.guaranteed_pct_achieved,
    latest_cum.simple_payback_achieved,
    -- Uncertainty
    latest_fsu.fsu_68_pct,
    latest_fsu.passes_ashrae_14,
    latest_fsu.savings_significant,
    -- Persistence
    latest_pr.persistence_factor,
    latest_pr.persistence_status,
    latest_pr.trend_direction,
    -- Active alerts
    alerts.active_count AS active_alerts,
    alerts.critical_count AS critical_alerts,
    -- Compliance
    latest_cc.overall_status AS compliance_status,
    latest_cc.compliance_pct,
    -- Guarantee
    latest_pg.meets_guarantee,
    latest_pg.performance_ratio,
    latest_pg.settlement_status
FROM pack040_mv.mv_projects p
LEFT JOIN LATERAL (
    SELECT baseline_name, model_type, r_squared, cvrmse_pct, nmbe_pct, validation_status
    FROM pack040_mv.mv_baselines
    WHERE project_id = p.id AND is_current = true
    ORDER BY baseline_version DESC
    LIMIT 1
) latest_bl ON TRUE
LEFT JOIN LATERAL (
    SELECT avoided_energy_kwh, savings_pct, period_start, period_status
    FROM pack040_mv.mv_savings_periods
    WHERE project_id = p.id
    ORDER BY period_start DESC
    LIMIT 1
) latest_sp ON TRUE
LEFT JOIN LATERAL (
    SELECT total_cost_savings, currency_code
    FROM pack040_mv.mv_cost_savings
    WHERE project_id = p.id
    ORDER BY created_at DESC
    LIMIT 1
) latest_cs ON TRUE
LEFT JOIN LATERAL (
    SELECT ltd_avoided_energy_kwh, ltd_total_cost_savings, guaranteed_pct_achieved, simple_payback_achieved
    FROM pack040_mv.mv_cumulative_savings
    WHERE project_id = p.id
    ORDER BY as_of_date DESC
    LIMIT 1
) latest_cum ON TRUE
LEFT JOIN LATERAL (
    SELECT fsu_68_pct, passes_ashrae_14, savings_significant
    FROM pack040_mv.mv_fsu_records
    WHERE project_id = p.id
    ORDER BY created_at DESC
    LIMIT 1
) latest_fsu ON TRUE
LEFT JOIN LATERAL (
    SELECT persistence_factor, persistence_status, trend_direction
    FROM pack040_mv.mv_persistence_records
    WHERE project_id = p.id
    ORDER BY tracking_year DESC
    LIMIT 1
) latest_pr ON TRUE
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS active_count,
           COUNT(*) FILTER (WHERE alert_severity IN ('CRITICAL', 'EMERGENCY')) AS critical_count
    FROM pack040_mv.mv_persistence_alerts
    WHERE project_id = p.id AND alert_status IN ('ACTIVE', 'ACKNOWLEDGED')
) alerts ON TRUE
LEFT JOIN LATERAL (
    SELECT overall_status, compliance_pct
    FROM pack040_mv.mv_compliance_checks
    WHERE project_id = p.id
    ORDER BY check_date DESC
    LIMIT 1
) latest_cc ON TRUE
LEFT JOIN LATERAL (
    SELECT meets_guarantee, performance_ratio, settlement_status
    FROM pack040_mv.mv_performance_guarantees
    WHERE project_id = p.id
    ORDER BY contract_year DESC
    LIMIT 1
) latest_pg ON TRUE
WHERE p.project_status NOT IN ('CANCELLED', 'ARCHIVED');

-- =============================================================================
-- Additional Composite Indexes for Common Query Patterns
-- =============================================================================

-- Savings periods by project + date range for time-series charts
CREATE INDEX idx_p040_sp_project_range     ON pack040_mv.mv_savings_periods(project_id, period_start DESC)
    WHERE period_status IN ('CALCULATED', 'APPROVED', 'FINAL');

-- Baselines by project + current for model selection
CREATE INDEX idx_p040_bl_project_models    ON pack040_mv.mv_baselines(project_id, model_type, r_squared DESC)
    WHERE is_current = true;

-- Persistence records by project + recent for trend analysis
CREATE INDEX idx_p040_pr_project_trend     ON pack040_mv.mv_persistence_records(project_id, tracking_year DESC, persistence_factor);

-- Compliance checks by project + framework for compliance dashboard
CREATE INDEX idx_p040_cc_project_fw        ON pack040_mv.mv_compliance_checks(project_id, compliance_framework, check_date DESC);

-- Performance guarantees by project + unmet for risk dashboard
CREATE INDEX idx_p040_pg_project_unmet     ON pack040_mv.mv_performance_guarantees(project_id, contract_year DESC)
    WHERE meets_guarantee = false;

-- Report outputs by project + recent for report history
CREATE INDEX idx_p040_ro_project_recent    ON pack040_mv.mv_report_outputs(project_id, report_type, generated_at DESC);

-- Degradation analysis with recommissioning needed
CREATE INDEX idx_p040_da_project_recom     ON pack040_mv.mv_degradation_analysis(project_id, analysis_date DESC)
    WHERE recommissioning_recommended = true;

-- Adjustment summaries by project for savings calculation pipeline
CREATE INDEX idx_p040_as_project_period    ON pack040_mv.mv_adjustment_summaries(project_id, reporting_period_start DESC)
    WHERE summary_status = 'APPROVED';

-- =============================================================================
-- Grants
-- =============================================================================
GRANT SELECT, INSERT ON pack040_mv.pack040_audit_trail TO PUBLIC;
GRANT SELECT ON pack040_mv.mv_project_savings_summary TO PUBLIC;
GRANT SELECT ON pack040_mv.mv_baseline_model_summary TO PUBLIC;
GRANT SELECT ON pack040_mv.mv_compliance_status_summary TO PUBLIC;
GRANT SELECT ON pack040_mv.v_mv_dashboard TO PUBLIC;

-- =============================================================================
-- Seed Data: IPMVP Option Reference
-- =============================================================================
CREATE TABLE pack040_mv.mv_ipmvp_option_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    option_code                 VARCHAR(10)     NOT NULL,
    option_name                 VARCHAR(100)    NOT NULL,
    full_name                   VARCHAR(255)    NOT NULL,
    description                 TEXT,
    measurement_boundary        VARCHAR(50)     NOT NULL,
    parameter_measurement       VARCHAR(50)     NOT NULL,
    uses_stipulated_values      BOOLEAN         NOT NULL,
    typical_accuracy            VARCHAR(20)     NOT NULL,
    typical_cost                VARCHAR(20)     NOT NULL,
    typical_complexity          VARCHAR(20)     NOT NULL,
    best_for                    TEXT,
    not_suitable_for            TEXT,
    savings_equation            TEXT,
    key_requirements            JSONB           DEFAULT '[]',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p040_ior_code UNIQUE (option_code)
);

INSERT INTO pack040_mv.mv_ipmvp_option_reference (option_code, option_name, full_name, description, measurement_boundary, parameter_measurement, uses_stipulated_values, typical_accuracy, typical_cost, typical_complexity, best_for, not_suitable_for, savings_equation, key_requirements) VALUES
('OPTION_A', 'Option A', 'Retrofit Isolation: Key Parameter Measurement', 'Savings determined by field measurement of the key performance parameter(s) which define the energy use of the ECM''s affected system(s). Parameters not selected for measurement are estimated. Savings are determined from engineering calculations using measured and estimated values.', 'RETROFIT_ISOLATION', 'KEY_PARAMETER', true, 'MEDIUM', 'LOW', 'LOW', 'Single-system retrofits where one key parameter dominates savings (e.g., lighting hours, motor load). Best when stipulated parameters have low sensitivity to savings.', 'Complex systems with multiple interacting parameters. Cases where stipulated parameters have high sensitivity to total savings.', 'Savings = (Baseline kW stipulated - Post kW measured) * Operating hours stipulated', '["Key parameter must be measured continuously or with representative spot measurements","Stipulated values must be documented with engineering justification","Sensitivity analysis required for stipulated parameters","Uncertainty from stipulation must be quantified"]'),
('OPTION_B', 'Option B', 'Retrofit Isolation: All Parameter Measurement', 'Savings determined by field measurement of ALL parameters needed to determine the energy use of the ECM''s affected system(s). No stipulated values are used. Provides the most accurate retrofit isolation approach.', 'RETROFIT_ISOLATION', 'ALL_PARAMETERS', false, 'HIGH', 'MEDIUM', 'MEDIUM', 'Complex system retrofits where all parameters can be measured (e.g., chiller retrofit, VFD on pump). Preferred when high accuracy is required.', 'Whole-building measures where isolation is impractical. Large populations of similar measures where census measurement is too expensive.', 'Savings = Baseline Energy measured - Post Energy measured (continuous metering)', '["All energy-determining parameters must be measured","Short-term or continuous measurement required","Meter accuracy must be documented","Pre and post measurements at same operating conditions preferred"]'),
('OPTION_C', 'Option C', 'Whole Facility', 'Savings determined by measuring whole-facility energy use with utility meters or sub-meters. Regression analysis of whole-facility energy data is used to create a baseline model. Routine and non-routine adjustments account for changes between periods.', 'WHOLE_FACILITY', 'UTILITY_METERS', false, 'MEDIUM', 'LOW', 'MEDIUM', 'Multiple ECMs implemented simultaneously. Building-wide measures (BAS optimization, commissioning). Cases where ECM affects >10% of whole-facility energy. Projects using existing utility meters.', 'ECMs affecting <10% of whole-facility energy (savings lost in noise). Single-system retrofits where isolation is straightforward and more accurate.', 'Savings = (Adjusted Baseline Energy - Reporting Period Energy) +/- Non-Routine Adjustments', '["Minimum 12 months baseline data (monthly) or equivalent daily/hourly","ASHRAE 14 model validation: CVRMSE <25% monthly, NMBE +/-5%","Non-routine adjustments must be documented and quantified","Weather and production normalization required if applicable"]'),
('OPTION_D', 'Option D', 'Calibrated Simulation', 'Savings determined through simulation of the energy use of a facility or sub-system. The simulation model must be calibrated to actual utility data per ASHRAE 14 criteria. Used when baseline cannot be measured directly.', 'WHOLE_FACILITY', 'CALIBRATED_SIMULATION', false, 'HIGH', 'HIGH', 'HIGH', 'New construction where no pre-retrofit baseline exists. Complex interactive effects that cannot be separated. Deep energy retrofits changing multiple building systems. Design-phase M&V planning.', 'Simple single-measure retrofits where direct measurement is easier. Projects without simulation expertise or tools. Cases where calibration data is insufficient.', 'Savings = Calibrated Simulation Baseline - Calibrated Simulation Post (or Measured Post)', '["Simulation model must be calibrated to ASHRAE 14 criteria","CVRMSE <25% monthly for calibrated model vs. actual data","Independent review of simulation inputs and assumptions","Model must capture all significant energy end uses"]');

GRANT SELECT ON pack040_mv.mv_ipmvp_option_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: ASHRAE 14 Criteria Reference
-- =============================================================================
CREATE TABLE pack040_mv.mv_ashrae14_criteria_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    criterion_code              VARCHAR(50)     NOT NULL,
    criterion_name              VARCHAR(255)    NOT NULL,
    description                 TEXT,
    data_granularity            VARCHAR(20)     NOT NULL,
    threshold_value             NUMERIC(10,4)   NOT NULL,
    threshold_unit              VARCHAR(20)     NOT NULL,
    threshold_direction         VARCHAR(10)     NOT NULL,
    is_mandatory                BOOLEAN         NOT NULL DEFAULT true,
    reference_section           VARCHAR(50),
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p040_a14_code_gran UNIQUE (criterion_code, data_granularity)
);

INSERT INTO pack040_mv.mv_ashrae14_criteria_reference (criterion_code, criterion_name, description, data_granularity, threshold_value, threshold_unit, threshold_direction, is_mandatory, reference_section, notes) VALUES
('CVRMSE', 'Coefficient of Variation of RMSE', 'Ratio of root mean square error to mean of measured data, expressed as percentage. Primary accuracy metric for regression model validation.', 'MONTHLY', 25.0, 'percent', 'BELOW', true, 'Section 5.3.2.4', 'Monthly data models must have CVRMSE < 25%'),
('CVRMSE', 'Coefficient of Variation of RMSE', 'Ratio of root mean square error to mean of measured data, expressed as percentage. Primary accuracy metric for regression model validation.', 'DAILY', 30.0, 'percent', 'BELOW', true, 'Section 5.3.2.4', 'Daily data models must have CVRMSE < 30%'),
('CVRMSE', 'Coefficient of Variation of RMSE', 'Ratio of root mean square error to mean of measured data, expressed as percentage. Primary accuracy metric for regression model validation.', 'HOURLY', 30.0, 'percent', 'BELOW', true, 'Section 5.3.2.4', 'Hourly data models must have CVRMSE < 30%'),
('NMBE', 'Normalized Mean Bias Error', 'Mean difference between predicted and measured values normalized by mean of measured data. Indicates systematic over/under prediction.', 'MONTHLY', 5.0, 'percent', 'WITHIN', true, 'Section 5.3.2.4', 'Monthly NMBE must be within +/-5%'),
('NMBE', 'Normalized Mean Bias Error', 'Mean difference between predicted and measured values normalized by mean of measured data. Indicates systematic over/under prediction.', 'DAILY', 10.0, 'percent', 'WITHIN', true, 'Section 5.3.2.4', 'Daily NMBE must be within +/-10%'),
('NMBE', 'Normalized Mean Bias Error', 'Mean difference between predicted and measured values normalized by mean of measured data. Indicates systematic over/under prediction.', 'HOURLY', 10.0, 'percent', 'WITHIN', true, 'Section 5.3.2.4', 'Hourly NMBE must be within +/-10%'),
('R_SQUARED', 'Coefficient of Determination', 'Proportion of variance in dependent variable explained by independent variables. Higher values indicate better model fit.', 'MONTHLY', 0.70, 'ratio', 'ABOVE', true, 'Section 5.3.2.4', 'Monthly models should have R-squared > 0.70'),
('R_SQUARED', 'Coefficient of Determination', 'Proportion of variance in dependent variable explained by independent variables. Higher values indicate better model fit.', 'DAILY', 0.50, 'ratio', 'ABOVE', true, 'Section 5.3.2.4', 'Daily models should have R-squared > 0.50'),
('FSU_68', 'Fractional Savings Uncertainty at 68%', 'Ratio of savings uncertainty to total savings at 68% confidence (1-sigma). Primary ASHRAE 14 uncertainty metric.', 'MONTHLY', 50.0, 'percent', 'BELOW', true, 'Section 5.3.3', 'FSU at 68% confidence must be < 50% for savings to be verified'),
('MIN_DATA_POINTS', 'Minimum Baseline Data Points', 'Minimum number of data points required for baseline regression model development.', 'MONTHLY', 12.0, 'count', 'ABOVE', true, 'Section 5.3.2.2', 'Minimum 12 monthly data points (1 year) required for baseline'),
('MIN_DATA_POINTS', 'Minimum Baseline Data Points', 'Minimum number of data points required for baseline regression model development.', 'DAILY', 365.0, 'count', 'ABOVE', false, 'Section 5.3.2.2', 'Minimum 365 daily data points (1 year) recommended for baseline'),
('DW_STATISTIC', 'Durbin-Watson Statistic', 'Test for autocorrelation in regression residuals. Values near 2 indicate no autocorrelation.', 'MONTHLY', 2.0, 'ratio', 'NEAR', false, 'Section 5.3.2.4', 'DW values 1.5-2.5 generally acceptable; <1.0 or >3.0 indicates significant autocorrelation');

GRANT SELECT ON pack040_mv.mv_ashrae14_criteria_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Compliance Framework Reference
-- =============================================================================
CREATE TABLE pack040_mv.mv_compliance_framework_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    framework_code              VARCHAR(50)     NOT NULL,
    framework_name              VARCHAR(255)    NOT NULL,
    full_name                   VARCHAR(500)    NOT NULL,
    version                     VARCHAR(20),
    issuing_body                VARCHAR(255),
    description                 TEXT,
    jurisdiction                VARCHAR(50)     NOT NULL DEFAULT 'INTERNATIONAL',
    num_requirements            INTEGER,
    key_requirements            JSONB           DEFAULT '[]',
    url                         VARCHAR(500),
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p040_cfr_code UNIQUE (framework_code)
);

INSERT INTO pack040_mv.mv_compliance_framework_reference (framework_code, framework_name, full_name, version, issuing_body, description, jurisdiction, num_requirements, key_requirements, url) VALUES
('IPMVP', 'IPMVP Core Concepts', 'International Performance Measurement and Verification Protocol - Core Concepts', '2022', 'Efficiency Valuation Organization (EVO)', 'Primary international M&V protocol defining Options A-D, baseline development, adjustments, and savings calculation methodology.', 'INTERNATIONAL', 45, '["Define measurement boundary","Select appropriate IPMVP option","Develop baseline model","Document routine and non-routine adjustments","Calculate savings with uncertainty","Maintain audit trail"]', 'https://evo-world.org/en/products-services-mainmenu-en/protocols/ipmvp'),
('ASHRAE_14', 'ASHRAE Guideline 14', 'ASHRAE Guideline 14-2014: Measurement of Energy, Demand, and Water Savings', '2014', 'American Society of Heating, Refrigerating and Air-Conditioning Engineers', 'Defines statistical requirements for M&V including regression model validation criteria (CVRMSE, NMBE, R-squared), uncertainty quantification, and fractional savings uncertainty.', 'US', 35, '["CVRMSE < 25% monthly / 30% daily","NMBE within +/-5% monthly / +/-10% daily","R-squared > 0.70 monthly / 0.50 daily","FSU < 50% at 68% confidence","Minimum 12 months baseline data","Durbin-Watson autocorrelation check"]', 'https://www.ashrae.org/technical-resources/bookstore/guideline-14-2014'),
('ISO_50015', 'ISO 50015', 'ISO 50015:2014 - Measurement and Verification of Energy Performance of Organizations', '2014', 'International Organization for Standardization', 'Defines M&V framework aligned with IPMVP including M&V plan requirements, baseline, reporting period, adjustments, and reporting. Designed for ISO 50001 integration.', 'INTERNATIONAL', 30, '["Develop M&V plan per Clause 5","Define measurement boundary per Clause 6","Establish baseline per Clause 7","Calculate energy performance improvement per Clause 8","Report results per Clause 9"]', 'https://www.iso.org/standard/60043.html'),
('FEMP_4_0', 'FEMP M&V Guidelines', 'Federal Energy Management Program M&V Guidelines Version 4.0', '4.0', 'U.S. Department of Energy', 'M&V requirements for US federal energy savings performance contracts (ESPCs) and utility energy service contracts (UESCs). Aligned with IPMVP.', 'US', 40, '["IPMVP-compliant M&V plan","Risk assessment for savings uncertainty","Annual M&V reporting","Performance guarantee tracking","Independent M&V review for large projects"]', 'https://www.energy.gov/eere/femp/mv-guidelines'),
('EU_EED', 'EU EED Article 7', 'EU Energy Efficiency Directive 2023/1791 - Article 7 Energy Savings Obligation', '2023', 'European Parliament and Council', 'Energy savings calculation and verification methodology for EU member state energy efficiency obligation schemes.', 'EU', 25, '["Bottom-up savings calculation","Additionality demonstration","Materiality verification","Double counting prevention","Lifetime savings estimation"]', 'https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficiency-targets-directive-and-rules/energy-efficiency-directive_en'),
('EU_EPC', 'EU EPC Directive', 'EU Energy Performance Contract Directive 2012/27/EU Article 18', '2012', 'European Parliament and Council', 'Requirements for energy performance contracting including savings verification, quality assurance, and dispute resolution.', 'EU', 20, '["Independent savings verification","Quality assurance procedures","Minimum savings guarantee","Dispute resolution mechanism","Annual reporting requirements"]', 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32012L0027');

GRANT SELECT ON pack040_mv.mv_compliance_framework_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Meter Accuracy Class Reference
-- =============================================================================
CREATE TABLE pack040_mv.mv_meter_accuracy_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    accuracy_class              VARCHAR(20)     NOT NULL,
    accuracy_name               VARCHAR(100)    NOT NULL,
    accuracy_pct                NUMERIC(6,4)    NOT NULL,
    description                 TEXT,
    standard                    VARCHAR(100),
    typical_applications        TEXT,
    suitable_for_mv             BOOLEAN         NOT NULL DEFAULT true,
    suitable_for_revenue        BOOLEAN         NOT NULL DEFAULT false,
    typical_cost_range          VARCHAR(50),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p040_mar_class UNIQUE (accuracy_class)
);

INSERT INTO pack040_mv.mv_meter_accuracy_reference (accuracy_class, accuracy_name, accuracy_pct, description, standard, typical_applications, suitable_for_mv, suitable_for_revenue, typical_cost_range) VALUES
('CLASS_0_1', 'Class 0.1 (Laboratory)', 0.1, 'Highest accuracy class for laboratory reference standards and calibration equipment.', 'IEC 62053-22', 'Calibration laboratories, reference standards', true, true, '$5,000-$20,000'),
('CLASS_0_2', 'Class 0.2 (Precision)', 0.2, 'High accuracy for precision metering, power quality analysis, and reference checks.', 'IEC 62053-22', 'Power quality analyzers, reference meters, SCADA', true, true, '$2,000-$10,000'),
('CLASS_0_5', 'Class 0.5 (Revenue)', 0.5, 'Standard revenue-grade accuracy for utility billing and financial settlement.', 'IEC 62053-22 / ANSI C12.20', 'Utility revenue meters, main facility meters, AMI', true, true, '$500-$3,000'),
('CLASS_1', 'Class 1 (Commercial)', 1.0, 'Standard commercial metering accuracy for sub-metering and M&V applications.', 'IEC 62053-21', 'Sub-meters, M&V meters, tenant meters, CT loggers', true, false, '$200-$1,500'),
('CLASS_2', 'Class 2 (Monitoring)', 2.0, 'Monitoring-grade accuracy for indicative energy tracking and trend analysis.', 'IEC 62053-21', 'IoT sensors, plug-load monitors, temporary monitoring', true, false, '$50-$500'),
('CLASS_3', 'Class 3 (Indicative)', 3.0, 'Indicative accuracy for rough energy estimates and screening purposes.', 'IEC 62053-21', 'Portable loggers, spot measurements, screening audits', false, false, '$30-$200'),
('REVENUE_GRADE', 'Revenue Grade', 0.2, 'Meets revenue-grade accuracy per ANSI C12.20 Class 0.2 for utility billing.', 'ANSI C12.20', 'Utility billing, performance contracts, ESCO guarantees', true, true, '$1,000-$5,000'),
('UTILITY_GRADE', 'Utility Grade', 0.5, 'Meets utility-grade accuracy per ANSI C12.20 Class 0.5 for general commercial metering.', 'ANSI C12.20', 'Facility main meters, utility check meters', true, true, '$500-$2,000'),
('MONITORING_GRADE', 'Monitoring Grade', 1.5, 'Adequate for monitoring and M&V trending but not for financial settlement.', 'Industry Standard', 'Energy monitoring, M&V sub-meters, BMS integration', true, false, '$100-$800'),
('INDICATIVE', 'Indicative', 5.0, 'Indicative-only accuracy for rough estimates, not suitable for M&V savings verification.', 'N/A', 'Screening audits, portable spot checks, rough estimates', false, false, '$20-$100');

GRANT SELECT ON pack040_mv.mv_meter_accuracy_reference TO PUBLIC;

-- =============================================================================
-- Seed Data: Baseline Model Type Reference
-- =============================================================================
CREATE TABLE pack040_mv.mv_baseline_model_type_reference (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    model_code                  VARCHAR(30)     NOT NULL,
    model_name                  VARCHAR(100)    NOT NULL,
    full_name                   VARCHAR(255)    NOT NULL,
    description                 TEXT,
    equation_form               TEXT,
    num_parameters              VARCHAR(20)     NOT NULL,
    independent_variables       TEXT,
    best_for                    TEXT,
    limitations                 TEXT,
    min_data_points_monthly     INTEGER,
    min_data_points_daily       INTEGER,
    supports_change_point       BOOLEAN         NOT NULL DEFAULT false,
    supports_weather_norm       BOOLEAN         NOT NULL DEFAULT true,
    complexity                  VARCHAR(20)     NOT NULL,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_p040_bmtr_code UNIQUE (model_code)
);

INSERT INTO pack040_mv.mv_baseline_model_type_reference (model_code, model_name, full_name, description, equation_form, num_parameters, independent_variables, best_for, limitations, min_data_points_monthly, min_data_points_daily, supports_change_point, supports_weather_norm, complexity) VALUES
('OLS', 'OLS', 'Ordinary Least Squares Linear Regression', 'Standard multivariate linear regression using ordinary least squares estimation. Assumes linear relationship between energy and independent variables.', 'E = a + b1*X1 + b2*X2 + ... + bn*Xn', '2-10', 'Temperature, production, occupancy, operating hours, daylight hours', 'Simple linear relationships, single dominant variable, non-weather-dependent facilities', 'Cannot model non-linear relationships, change-points, or HVAC switching behavior', 12, 90, false, true, 'LOW'),
('3P_COOLING', '3P Cooling', 'Three-Parameter Cooling Change-Point Model', 'Piecewise linear model with a flat segment below the change-point and a positive slope above. Models cooling-dominant buildings where energy increases with temperature above a balance point.', 'E = a + b * max(0, T - Tcp)', '3', 'Temperature (outdoor dry-bulb)', 'Cooling-dominant commercial buildings (offices, retail), facilities with minimal heating', 'Only models one change-point, does not capture heating behavior', 12, 90, true, true, 'MEDIUM'),
('3P_HEATING', '3P Heating', 'Three-Parameter Heating Change-Point Model', 'Piecewise linear model with a flat segment above the change-point and a positive slope below. Models heating-dominant buildings where energy increases with decreasing temperature.', 'E = a + b * max(0, Thp - T)', '3', 'Temperature (outdoor dry-bulb)', 'Heating-dominant buildings, gas/steam heating, cold climate facilities', 'Only models one change-point, does not capture cooling behavior', 12, 90, true, true, 'MEDIUM'),
('4P', '4P', 'Four-Parameter Change-Point Model', 'Combined heating and cooling model with a single change-point temperature. Flat segment around the balance point with heating slope below and cooling slope above.', 'E = a + bh*max(0,Tcp-T) + bc*max(0,T-Tcp)', '4', 'Temperature (outdoor dry-bulb)', 'Buildings with both heating and cooling but similar balance points', 'Single change-point may not capture different heating and cooling balance points', 12, 120, true, true, 'MEDIUM'),
('5P', '5P', 'Five-Parameter Change-Point Model', 'Full heating and cooling model with separate change-point temperatures for heating and cooling. Flat segment between the two change-points with independent heating and cooling slopes.', 'E = a + bh*max(0,Thp-T) + bc*max(0,T-Tcp)', '5', 'Temperature (outdoor dry-bulb)', 'Buildings with distinct heating and cooling modes, different balance points, mixed-use facilities', 'Requires sufficient data in all three regimes (heating, base, cooling). More data needed for reliable fit.', 12, 180, true, true, 'HIGH'),
('TOWT', 'TOWT', 'Time-of-Week and Temperature Model', 'Combined schedule and temperature model using indicator variables for each hour-of-week and temperature bins. Captures both schedule and weather effects simultaneously.', 'E = f(TOW_indicators, T_bins)', '50-200', 'Hour-of-week, temperature bins, holidays', 'Sub-daily data (hourly/daily), buildings with strong schedule patterns, complex HVAC schedules', 'Requires hourly or sub-hourly data, many parameters, risk of overfitting with short baselines', 12, 365, false, true, 'HIGH'),
('MULTIVARIATE', 'Multivariate', 'Multivariate Linear Regression', 'Linear regression with multiple independent variables beyond temperature (production, occupancy, weather, etc.). May include interaction terms.', 'E = a + b1*T + b2*Production + b3*Occupancy + ...', '3-15', 'Temperature, production volume, occupancy, operating hours, other process variables', 'Industrial facilities with production-dependent energy, complex commercial buildings', 'Multicollinearity risk with correlated variables, requires VIF checks', 12, 90, false, true, 'MEDIUM'),
('MEAN', 'Mean', 'Simple Average Baseline', 'Baseline defined as the mean consumption during the baseline period. No regression modeling. Used when no independent variables explain consumption variance.', 'E = mean(E_baseline)', '1', 'None', 'Base-load processes with constant energy use regardless of conditions, Option A/B isolation', 'Cannot normalize for weather, production, or other variables. Very limited applicability.', 6, 30, false, false, 'LOW'),
('DEGREE_DAY', 'Degree Day', 'Degree-Day Regression Model', 'Simple regression of energy against heating degree-days (HDD) and/or cooling degree-days (CDD). Uses pre-calculated degree-days rather than raw temperature.', 'E = a + b_h*HDD + b_c*CDD', '2-3', 'HDD, CDD (with fixed or optimized balance points)', 'Utility bill analysis, quick screening, monthly weather normalization', 'Less accurate than change-point models, assumes linear degree-day relationship', 12, 30, false, true, 'LOW');

GRANT SELECT ON pack040_mv.mv_baseline_model_type_reference TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.pack040_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-040 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack040_mv.mv_project_savings_summary IS
    'Per-project savings overview with latest savings, cumulative totals, uncertainty, persistence, and guarantee status. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack040_mv.mv_project_savings_summary;';
COMMENT ON MATERIALIZED VIEW pack040_mv.mv_baseline_model_summary IS
    'Current baseline model summary with regression statistics, validation status, and diagnostics quality. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack040_mv.mv_baseline_model_summary;';
COMMENT ON MATERIALIZED VIEW pack040_mv.mv_compliance_status_summary IS
    'Latest compliance check results per project per framework for compliance dashboards. Refresh: REFRESH MATERIALIZED VIEW CONCURRENTLY pack040_mv.mv_compliance_status_summary;';
COMMENT ON VIEW pack040_mv.v_mv_dashboard IS
    'Real-time M&V operations dashboard combining project status, savings, persistence, compliance, alerts, and guarantee performance.';

COMMENT ON TABLE pack040_mv.mv_ipmvp_option_reference IS
    'Reference table for IPMVP Options A-D with descriptions, applicability, accuracy, cost, and key requirements.';
COMMENT ON TABLE pack040_mv.mv_ashrae14_criteria_reference IS
    'ASHRAE Guideline 14-2014 statistical criteria for baseline model validation by data granularity.';
COMMENT ON TABLE pack040_mv.mv_compliance_framework_reference IS
    'Reference table for M&V compliance frameworks (IPMVP, ASHRAE 14, ISO 50015, FEMP, EU EED, EU EPC).';
COMMENT ON TABLE pack040_mv.mv_meter_accuracy_reference IS
    'Meter accuracy class reference with specifications, standards, and M&V suitability per IEC/ANSI standards.';
COMMENT ON TABLE pack040_mv.mv_baseline_model_type_reference IS
    'Baseline regression model type reference with equations, applicability, and data requirements.';
