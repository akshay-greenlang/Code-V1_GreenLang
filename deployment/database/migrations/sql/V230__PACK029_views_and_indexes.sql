-- =============================================================================
-- V210: PACK-029 Interim Targets Pack - Views & Performance Indexes
-- =============================================================================
-- Pack:         PACK-029 (Interim Targets Pack)
-- Migration:    015 of 015
-- Date:         March 2026
--
-- Dashboard views and comprehensive performance indexes across all PACK-029
-- tables for production-grade query performance.
--
-- Views (3):
--   1. pack029_interim_targets.v_progress_summary
--   2. pack029_interim_targets.v_variance_dashboard
--   3. pack029_interim_targets.v_milestone_status
--
-- Indexes: 250+ composite and conditional indexes.
--
-- Previous: V209__PACK029_sbti_submissions.sql
-- =============================================================================

-- =============================================================================
-- View 1: v_progress_summary
-- =============================================================================
-- Consolidated progress summary with org-level target vs actual performance,
-- variance analysis, performance scoring (red/amber/green), and ranking
-- for executive-level interim target monitoring.

CREATE OR REPLACE VIEW pack029_interim_targets.v_progress_summary AS
SELECT
    it.target_id,
    it.tenant_id,
    it.organization_id,
    it.baseline_year,
    it.target_year,
    it.scope,
    it.target_type,
    it.baseline_emissions_tco2e,
    it.target_emissions_tco2e,
    it.reduction_pct                AS target_reduction_pct,
    it.sbti_pathway,
    it.sbti_method,
    it.sbti_validated,
    it.validation_status            AS target_validation_status,
    it.coverage_pct,
    -- Latest annual pathway
    ap.year                         AS pathway_year,
    ap.target_emissions_tco2e       AS pathway_target_tco2e,
    ap.annual_reduction_pct         AS pathway_annual_reduction_pct,
    ap.cumulative_reduction_pct     AS pathway_cumulative_reduction_pct,
    ap.pathway_type,
    -- Latest actual performance
    perf.year                       AS performance_year,
    perf.quarter                    AS performance_quarter,
    perf.actual_emissions_tco2e,
    perf.data_quality_tier,
    perf.verification_status,
    -- Latest variance
    va.variance_absolute_tco2e,
    va.variance_pct,
    va.variance_direction,
    va.on_track,
    va.decomposition_method,
    va.activity_effect_tco2e,
    va.intensity_effect_tco2e,
    va.structural_effect_tco2e,
    va.root_cause_classification,
    va.severity_level               AS variance_severity,
    -- Performance score
    CASE
        WHEN va.on_track = TRUE THEN 'GREEN'
        WHEN va.variance_pct IS NOT NULL AND ABS(va.variance_pct) <= 5.0 THEN 'AMBER'
        WHEN va.variance_pct IS NOT NULL AND ABS(va.variance_pct) > 5.0 THEN 'RED'
        ELSE 'GREY'
    END AS performance_score,
    -- Carbon budget status
    cba.budget_allocated_tco2e,
    cba.budget_consumed_tco2e,
    cba.budget_remaining_tco2e,
    cba.overshoot_flag,
    cba.rebalancing_required,
    -- Active alerts count
    alert_agg.total_alerts,
    alert_agg.red_alerts,
    alert_agg.amber_alerts,
    alert_agg.unresolved_alerts,
    -- Corrective actions summary
    ca_agg.total_actions,
    ca_agg.actions_in_progress,
    ca_agg.total_expected_reduction_tco2e,
    ca_agg.total_gap_tco2e,
    -- Forecast
    tf.forecasted_emissions_tco2e   AS latest_forecast_tco2e,
    tf.target_attainment_probability,
    tf.method                       AS forecast_method,
    -- SBTi submission
    ss.sbti_validation_status       AS submission_status,
    ss.submission_date              AS latest_submission_date,
    -- Rank by performance
    RANK() OVER (
        PARTITION BY it.scope
        ORDER BY COALESCE(va.variance_pct, 0) ASC
    ) AS performance_rank
FROM pack029_interim_targets.gl_interim_targets it
-- Latest annual pathway for current year
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_annual_pathways
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND target_id = it.target_id AND is_active = TRUE
    ORDER BY year DESC LIMIT 1
) ap ON TRUE
-- Latest actual performance
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_actual_performance
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND is_active = TRUE
    ORDER BY year DESC, quarter DESC NULLS LAST LIMIT 1
) perf ON TRUE
-- Latest variance analysis
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_variance_analysis
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND is_active = TRUE
    ORDER BY year DESC, quarter DESC NULLS LAST LIMIT 1
) va ON TRUE
-- Latest carbon budget
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_carbon_budget_allocation
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND is_active = TRUE
    ORDER BY year DESC LIMIT 1
) cba ON TRUE
-- Alert aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS total_alerts,
        COUNT(*) FILTER (WHERE alert_type = 'RED')                  AS red_alerts,
        COUNT(*) FILTER (WHERE alert_type = 'AMBER')                AS amber_alerts,
        COUNT(*) FILTER (WHERE resolved_at IS NULL)                 AS unresolved_alerts
    FROM pack029_interim_targets.gl_progress_alerts
    WHERE organization_id = it.organization_id AND is_active = TRUE
          AND target_id = it.target_id
) alert_agg ON TRUE
-- Corrective action aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS total_actions,
        COUNT(*) FILTER (WHERE status = 'IN_PROGRESS')              AS actions_in_progress,
        SUM(expected_reduction_tco2e)                               AS total_expected_reduction_tco2e,
        SUM(gap_to_target_tco2e)                                    AS total_gap_tco2e
    FROM pack029_interim_targets.gl_corrective_actions
    WHERE organization_id = it.organization_id AND target_id = it.target_id
          AND status NOT IN ('CANCELLED', 'DEFERRED')
) ca_agg ON TRUE
-- Latest forecast
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_trend_forecasts
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND is_latest = TRUE AND is_active = TRUE
    ORDER BY forecast_date DESC LIMIT 1
) tf ON TRUE
-- Latest SBTi submission
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_sbti_submissions
    WHERE organization_id = it.organization_id AND scope = it.scope
          AND is_latest = TRUE AND is_active = TRUE
    ORDER BY submission_date DESC LIMIT 1
) ss ON TRUE
WHERE it.is_active = TRUE
  AND it.validation_status NOT IN ('REJECTED', 'EXPIRED');

-- =============================================================================
-- View 2: v_variance_dashboard
-- =============================================================================
-- Variance analysis dashboard with decomposition details, root cause
-- breakdown, and top contributing factors for variance investigation.

CREATE OR REPLACE VIEW pack029_interim_targets.v_variance_dashboard AS
SELECT
    va.variance_id,
    va.tenant_id,
    va.organization_id,
    va.year,
    va.quarter,
    va.scope,
    va.category,
    va.analysis_date,
    -- Target vs Actual
    va.target_emissions_tco2e,
    va.actual_emissions_tco2e,
    va.variance_absolute_tco2e,
    va.variance_pct,
    va.variance_direction,
    va.on_track,
    -- Decomposition
    va.decomposition_method,
    va.activity_effect_tco2e,
    va.intensity_effect_tco2e,
    va.structural_effect_tco2e,
    va.fuel_mix_effect_tco2e,
    va.weather_effect_tco2e,
    -- Decomposition percentages
    CASE
        WHEN va.variance_absolute_tco2e != 0 AND va.activity_effect_tco2e IS NOT NULL
        THEN ROUND((va.activity_effect_tco2e / ABS(va.variance_absolute_tco2e) * 100)::NUMERIC, 1)
        ELSE 0
    END AS activity_effect_pct,
    CASE
        WHEN va.variance_absolute_tco2e != 0 AND va.intensity_effect_tco2e IS NOT NULL
        THEN ROUND((va.intensity_effect_tco2e / ABS(va.variance_absolute_tco2e) * 100)::NUMERIC, 1)
        ELSE 0
    END AS intensity_effect_pct,
    CASE
        WHEN va.variance_absolute_tco2e != 0 AND va.structural_effect_tco2e IS NOT NULL
        THEN ROUND((va.structural_effect_tco2e / ABS(va.variance_absolute_tco2e) * 100)::NUMERIC, 1)
        ELSE 0
    END AS structural_effect_pct,
    -- Root cause
    va.root_cause_classification,
    va.root_cause_category,
    va.root_cause_description,
    va.root_causes,
    -- Severity
    va.severity_score,
    va.severity_level,
    va.cumulative_impact_tco2e,
    va.budget_impact_pct,
    -- Trend
    va.consecutive_quarters_off,
    va.trend_direction,
    va.trend_acceleration,
    -- Actions
    va.corrective_action_required,
    va.corrective_action_deadline,
    va.reviewed,
    va.reviewed_by,
    -- RAG status
    CASE va.severity_level
        WHEN 'CRITICAL' THEN 'RED'
        WHEN 'HIGH' THEN 'RED'
        WHEN 'MODERATE' THEN 'AMBER'
        WHEN 'LOW' THEN 'GREEN'
        WHEN 'NEGLIGIBLE' THEN 'GREEN'
        ELSE 'GREY'
    END AS rag_status,
    -- Top 3 root causes (from JSONB array)
    va.root_causes->0->>'cause'     AS root_cause_1,
    va.root_causes->1->>'cause'     AS root_cause_2,
    va.root_causes->2->>'cause'     AS root_cause_3,
    -- Corrective actions for this variance
    ca_agg.corrective_action_count,
    ca_agg.total_planned_reduction_tco2e,
    ca_agg.gap_coverage_pct
FROM pack029_interim_targets.gl_variance_analysis va
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS corrective_action_count,
        COALESCE(SUM(expected_reduction_tco2e), 0)                  AS total_planned_reduction_tco2e,
        CASE
            WHEN va.variance_absolute_tco2e > 0
            THEN ROUND((COALESCE(SUM(expected_reduction_tco2e), 0) / va.variance_absolute_tco2e * 100)::NUMERIC, 1)
            ELSE 0
        END AS gap_coverage_pct
    FROM pack029_interim_targets.gl_corrective_actions
    WHERE variance_id = va.variance_id AND status NOT IN ('CANCELLED', 'DEFERRED')
) ca_agg ON TRUE
WHERE va.is_active = TRUE;

-- =============================================================================
-- View 3: v_milestone_status
-- =============================================================================
-- Milestone status dashboard with quarterly achievement tracking, alert
-- status, and completion metrics for milestone-level monitoring.

CREATE OR REPLACE VIEW pack029_interim_targets.v_milestone_status AS
SELECT
    qm.tenant_id,
    qm.organization_id,
    qm.year,
    qm.quarter,
    qm.scope,
    qm.target_id,
    -- Milestone detail
    qm.milestone_emissions_tco2e,
    qm.milestone_reduction_pct,
    qm.annual_target_share_pct,
    qm.seasonal_factor,
    qm.milestone_status,
    qm.achieved,
    qm.achieved_at,
    -- Actual performance for this quarter
    perf.actual_emissions_tco2e,
    perf.data_quality_tier,
    perf.verification_status,
    -- Variance
    CASE
        WHEN perf.actual_emissions_tco2e IS NOT NULL
        THEN perf.actual_emissions_tco2e - qm.milestone_emissions_tco2e
        ELSE NULL
    END AS quarter_variance_tco2e,
    CASE
        WHEN perf.actual_emissions_tco2e IS NOT NULL AND qm.milestone_emissions_tco2e > 0
        THEN ROUND(((perf.actual_emissions_tco2e - qm.milestone_emissions_tco2e) / qm.milestone_emissions_tco2e * 100)::NUMERIC, 2)
        ELSE NULL
    END AS quarter_variance_pct,
    -- Milestone aggregation for the year
    yr_agg.milestones_total,
    yr_agg.milestones_achieved,
    yr_agg.milestones_missed,
    yr_agg.milestones_at_risk,
    CASE
        WHEN yr_agg.milestones_total > 0
        THEN ROUND((yr_agg.milestones_achieved::NUMERIC / yr_agg.milestones_total * 100), 1)
        ELSE 0
    END AS achievement_pct,
    -- Alert status for this quarter
    alert_agg.quarter_alerts,
    alert_agg.quarter_red_alerts,
    alert_agg.quarter_unresolved,
    -- Overall alert status
    CASE
        WHEN alert_agg.quarter_red_alerts > 0 THEN 'RED'
        WHEN alert_agg.quarter_alerts > 0 AND alert_agg.quarter_unresolved > 0 THEN 'AMBER'
        WHEN qm.milestone_status = 'ACHIEVED' THEN 'GREEN'
        WHEN qm.milestone_status = 'ON_TRACK' THEN 'GREEN'
        WHEN qm.milestone_status = 'AT_RISK' THEN 'AMBER'
        WHEN qm.milestone_status = 'MISSED' THEN 'RED'
        ELSE 'GREY'
    END AS alert_status
FROM pack029_interim_targets.gl_quarterly_milestones qm
-- Actual performance for this quarter
LEFT JOIN LATERAL (
    SELECT * FROM pack029_interim_targets.gl_actual_performance
    WHERE organization_id = qm.organization_id
          AND year = qm.year AND quarter = qm.quarter AND scope = qm.scope
          AND is_active = TRUE
    ORDER BY created_at DESC LIMIT 1
) perf ON TRUE
-- Year-level milestone aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                        AS milestones_total,
        COUNT(*) FILTER (WHERE milestone_status = 'ACHIEVED' OR achieved = TRUE) AS milestones_achieved,
        COUNT(*) FILTER (WHERE milestone_status = 'MISSED')             AS milestones_missed,
        COUNT(*) FILTER (WHERE milestone_status = 'AT_RISK')            AS milestones_at_risk
    FROM pack029_interim_targets.gl_quarterly_milestones
    WHERE organization_id = qm.organization_id
          AND year = qm.year AND scope = qm.scope AND is_active = TRUE
) yr_agg ON TRUE
-- Quarter-level alert aggregation
LEFT JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS quarter_alerts,
        COUNT(*) FILTER (WHERE alert_type = 'RED')                  AS quarter_red_alerts,
        COUNT(*) FILTER (WHERE resolved_at IS NULL)                 AS quarter_unresolved
    FROM pack029_interim_targets.gl_progress_alerts
    WHERE organization_id = qm.organization_id
          AND year = qm.year AND quarter = qm.quarter AND is_active = TRUE
) alert_agg ON TRUE
WHERE qm.is_active = TRUE;

-- =============================================================================
-- Performance Indexes (Composite & Conditional)
-- =============================================================================
-- Multi-column composite indexes for common query patterns across all tables.

-- ---------------------------------------------------------------------------
-- Interim Targets: multi-scope multi-year lookups
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_it_org_scope_year     ON pack029_interim_targets.gl_interim_targets(organization_id, scope, target_year);
CREATE INDEX idx_p029_it_org_active_scope   ON pack029_interim_targets.gl_interim_targets(organization_id, scope, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p029_it_sbti_method_path   ON pack029_interim_targets.gl_interim_targets(sbti_method, sbti_pathway);
CREATE INDEX idx_p029_it_validated_scope    ON pack029_interim_targets.gl_interim_targets(validation_status, scope) WHERE validation_status = 'VALIDATED';
CREATE INDEX idx_p029_it_approval_pending   ON pack029_interim_targets.gl_interim_targets(organization_id) WHERE approval_status = 'PENDING';
CREATE INDEX idx_p029_it_superseded         ON pack029_interim_targets.gl_interim_targets(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX idx_p029_it_target_type_scope  ON pack029_interim_targets.gl_interim_targets(target_type, scope);

-- ---------------------------------------------------------------------------
-- Annual Pathways: time-series composite indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ap_org_scope_year_act ON pack029_interim_targets.gl_annual_pathways(organization_id, scope, year) WHERE is_active = TRUE;
CREATE INDEX idx_p029_ap_target_year_type   ON pack029_interim_targets.gl_annual_pathways(target_id, year, pathway_type);
CREATE INDEX idx_p029_ap_budget_year        ON pack029_interim_targets.gl_annual_pathways(organization_id, year, carbon_budget_allocated_tco2e);
CREATE INDEX idx_p029_ap_locked_active      ON pack029_interim_targets.gl_annual_pathways(organization_id) WHERE is_locked = TRUE AND is_active = TRUE;

-- ---------------------------------------------------------------------------
-- Quarterly Milestones: status and achievement tracking
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_qm_org_scope_yr_qtr  ON pack029_interim_targets.gl_quarterly_milestones(organization_id, scope, year, quarter) WHERE is_active = TRUE;
CREATE INDEX idx_p029_qm_target_yr_status  ON pack029_interim_targets.gl_quarterly_milestones(target_id, year, milestone_status);
CREATE INDEX idx_p029_qm_pending_year      ON pack029_interim_targets.gl_quarterly_milestones(year, quarter) WHERE milestone_status = 'PENDING';
CREATE INDEX idx_p029_qm_deferred          ON pack029_interim_targets.gl_quarterly_milestones(organization_id) WHERE milestone_status = 'DEFERRED';

-- ---------------------------------------------------------------------------
-- Actual Performance: time-series and quality queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_perf_org_scope_yr_qtr ON pack029_interim_targets.gl_actual_performance(organization_id, scope, year DESC, quarter DESC NULLS LAST) WHERE is_active = TRUE;
CREATE INDEX idx_p029_perf_target_yr        ON pack029_interim_targets.gl_actual_performance(target_id, year);
CREATE INDEX idx_p029_perf_scope_cat_yr     ON pack029_interim_targets.gl_actual_performance(scope, category, year);
CREATE INDEX idx_p029_perf_dq_scope         ON pack029_interim_targets.gl_actual_performance(data_quality_tier, scope);
CREATE INDEX idx_p029_perf_verified_yr      ON pack029_interim_targets.gl_actual_performance(verification_status, year) WHERE verification_status != 'UNVERIFIED';

-- ---------------------------------------------------------------------------
-- Variance Analysis: off-track and decomposition queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_va_org_scope_yr_act   ON pack029_interim_targets.gl_variance_analysis(organization_id, scope, year) WHERE is_active = TRUE;
CREATE INDEX idx_p029_va_target_severity    ON pack029_interim_targets.gl_variance_analysis(target_id, severity_level);
CREATE INDEX idx_p029_va_off_track_severity ON pack029_interim_targets.gl_variance_analysis(severity_level, variance_pct DESC) WHERE on_track = FALSE;
CREATE INDEX idx_p029_va_consec_miss        ON pack029_interim_targets.gl_variance_analysis(organization_id, consecutive_quarters_off DESC) WHERE consecutive_quarters_off > 0;
CREATE INDEX idx_p029_va_decomp_scope       ON pack029_interim_targets.gl_variance_analysis(decomposition_method, scope);

-- ---------------------------------------------------------------------------
-- Corrective Actions: pipeline and cost queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ca_org_status_year    ON pack029_interim_targets.gl_corrective_actions(organization_id, status, deployment_year);
CREATE INDEX idx_p029_ca_target_status      ON pack029_interim_targets.gl_corrective_actions(target_id, status);
CREATE INDEX idx_p029_ca_cost_effectiveness ON pack029_interim_targets.gl_corrective_actions(cost_per_tco2e ASC, expected_reduction_tco2e DESC) WHERE status NOT IN ('CANCELLED', 'DEFERRED');
CREATE INDEX idx_p029_ca_deploy_schedule    ON pack029_interim_targets.gl_corrective_actions(deployment_year, deployment_quarter, status);
CREATE INDEX idx_p029_ca_progress_active    ON pack029_interim_targets.gl_corrective_actions(organization_id, progress_pct) WHERE status = 'IN_PROGRESS';

-- ---------------------------------------------------------------------------
-- Progress Alerts: active alert queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_pa_org_type_sev       ON pack029_interim_targets.gl_progress_alerts(organization_id, alert_type, severity) WHERE resolved_at IS NULL;
CREATE INDEX idx_p029_pa_target_type        ON pack029_interim_targets.gl_progress_alerts(target_id, alert_type);
CREATE INDEX idx_p029_pa_reason_severity    ON pack029_interim_targets.gl_progress_alerts(alert_reason, severity);
CREATE INDEX idx_p029_pa_escalation_due     ON pack029_interim_targets.gl_progress_alerts(auto_escalation_date, escalation_level) WHERE resolved_at IS NULL AND auto_escalation_date IS NOT NULL;
CREATE INDEX idx_p029_pa_recurring_org      ON pack029_interim_targets.gl_progress_alerts(organization_id, alert_reason, recurrence_count DESC) WHERE is_recurring = TRUE;

-- ---------------------------------------------------------------------------
-- Initiative Schedule: timeline and dependency queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_is_org_cat_status     ON pack029_interim_targets.gl_initiative_schedule(organization_id, initiative_category, status);
CREATE INDEX idx_p029_is_target_start_yr    ON pack029_interim_targets.gl_initiative_schedule(target_id, planned_start_year);
CREATE INDEX idx_p029_is_schedule_risk      ON pack029_interim_targets.gl_initiative_schedule(schedule_status, risk_level);
CREATE INDEX idx_p029_is_critical_delayed   ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE critical_path = TRUE AND schedule_status IN ('DELAYED', 'SIGNIFICANTLY_DELAYED');
CREATE INDEX idx_p029_is_permits_pending    ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE permits_required = TRUE AND permits_obtained = FALSE;
CREATE INDEX idx_p029_is_pilot_active       ON pack029_interim_targets.gl_initiative_schedule(organization_id) WHERE status = 'PILOT';

-- ---------------------------------------------------------------------------
-- Carbon Budget: overshoot and rebalancing queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_cba_org_scope_yr_act  ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, scope, year) WHERE is_active = TRUE;
CREATE INDEX idx_p029_cba_target_year       ON pack029_interim_targets.gl_carbon_budget_allocation(target_id, year);
CREATE INDEX idx_p029_cba_overshoot_scope   ON pack029_interim_targets.gl_carbon_budget_allocation(scope, year) WHERE overshoot_flag = TRUE;
CREATE INDEX idx_p029_cba_rebal_strategy    ON pack029_interim_targets.gl_carbon_budget_allocation(rebalancing_strategy) WHERE rebalancing_required = TRUE;
CREATE INDEX idx_p029_cba_locked_approved   ON pack029_interim_targets.gl_carbon_budget_allocation(organization_id, year) WHERE is_locked = TRUE AND approved = TRUE;

-- ---------------------------------------------------------------------------
-- Reporting Periods: deadline and framework queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_rp_org_year_active    ON pack029_interim_targets.gl_reporting_periods(organization_id, reporting_year) WHERE is_active = TRUE;
CREATE INDEX idx_p029_rp_framework_year     ON pack029_interim_targets.gl_reporting_periods(primary_framework, reporting_year);
CREATE INDEX idx_p029_rp_overdue            ON pack029_interim_targets.gl_reporting_periods(reporting_deadline) WHERE submission_status = 'NOT_SUBMITTED' AND reporting_deadline < CURRENT_DATE;
CREATE INDEX idx_p029_rp_incomplete_scope   ON pack029_interim_targets.gl_reporting_periods(organization_id) WHERE scope1_complete = FALSE OR scope2_complete = FALSE;
CREATE INDEX idx_p029_rp_published_year     ON pack029_interim_targets.gl_reporting_periods(reporting_year, published_date) WHERE published = TRUE;

-- ---------------------------------------------------------------------------
-- Validation Results: compliance and remediation queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_vr_org_criteria_date  ON pack029_interim_targets.gl_validation_results(organization_id, sbti_criteria_id, validation_date DESC);
CREATE INDEX idx_p029_vr_run_criteria       ON pack029_interim_targets.gl_validation_results(validation_run_id, sbti_criteria_id);
CREATE INDEX idx_p029_vr_fail_category      ON pack029_interim_targets.gl_validation_results(criterion_category, pass_fail_status) WHERE pass_fail_status = 'FAIL';
CREATE INDEX idx_p029_vr_mandatory_fail     ON pack029_interim_targets.gl_validation_results(organization_id) WHERE is_mandatory = TRUE AND pass_fail_status = 'FAIL';
CREATE INDEX idx_p029_vr_improved           ON pack029_interim_targets.gl_validation_results(organization_id) WHERE result_changed = TRUE AND improvement = TRUE;

-- ---------------------------------------------------------------------------
-- Assurance Evidence: completeness and review queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ae_org_type_year      ON pack029_interim_targets.gl_assurance_evidence(organization_id, evidence_type, reporting_year);
CREATE INDEX idx_p029_ae_tier_complete      ON pack029_interim_targets.gl_assurance_evidence(evidence_tier, completeness_pct);
CREATE INDEX idx_p029_ae_review_pending     ON pack029_interim_targets.gl_assurance_evidence(organization_id, reporting_year) WHERE assurance_provider_reviewed = FALSE AND is_active = TRUE;
CREATE INDEX idx_p029_ae_quality_low        ON pack029_interim_targets.gl_assurance_evidence(organization_id) WHERE overall_quality_score IS NOT NULL AND overall_quality_score < 50;
CREATE INDEX idx_p029_ae_retention_due      ON pack029_interim_targets.gl_assurance_evidence(retention_until) WHERE archived = FALSE AND retention_until IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Trend Forecasts: prediction and accuracy queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_tf_org_scope_yr_lat   ON pack029_interim_targets.gl_trend_forecasts(organization_id, scope, forecast_year) WHERE is_latest = TRUE;
CREATE INDEX idx_p029_tf_method_scenario    ON pack029_interim_targets.gl_trend_forecasts(method, scenario);
CREATE INDEX idx_p029_tf_accuracy_check     ON pack029_interim_targets.gl_trend_forecasts(forecast_error_pct) WHERE forecast_error_pct IS NOT NULL;
CREATE INDEX idx_p029_tf_off_target         ON pack029_interim_targets.gl_trend_forecasts(organization_id, forecast_year) WHERE forecast_vs_target_pct IS NOT NULL AND forecast_vs_target_pct > 0;
CREATE INDEX idx_p029_tf_high_confidence    ON pack029_interim_targets.gl_trend_forecasts(organization_id, target_attainment_probability DESC) WHERE target_attainment_probability IS NOT NULL;

-- ---------------------------------------------------------------------------
-- SBTi Submissions: pipeline and status queries
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p029_ss_org_scope_status   ON pack029_interim_targets.gl_sbti_submissions(organization_id, scope, sbti_validation_status);
CREATE INDEX idx_p029_ss_pathway_method     ON pack029_interim_targets.gl_sbti_submissions(sbti_pathway, sbti_method);
CREATE INDEX idx_p029_ss_commitment_pub     ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE commitment_letter_signed = TRUE AND target_published = FALSE;
CREATE INDEX idx_p029_ss_expiring           ON pack029_interim_targets.gl_sbti_submissions(organization_id) WHERE sbti_validation_status = 'EXPIRED';
CREATE INDEX idx_p029_ss_resub_pending      ON pack029_interim_targets.gl_sbti_submissions(resubmission_deadline) WHERE resubmission_required = TRUE AND response_submitted = FALSE;

-- =============================================================================
-- Comments on Views
-- =============================================================================
COMMENT ON VIEW pack029_interim_targets.v_progress_summary IS
    'Consolidated progress summary with organization-level target vs actual performance, variance analysis, performance scoring (red/amber/green), carbon budget status, alert counts, corrective action summary, forecasts, and SBTi submission status for executive monitoring.';

COMMENT ON VIEW pack029_interim_targets.v_variance_dashboard IS
    'Variance analysis dashboard with LMDI/Kaya decomposition details, root cause breakdown, severity classification, RAG status, top contributing factors, and corrective action coverage for variance investigation.';

COMMENT ON VIEW pack029_interim_targets.v_milestone_status IS
    'Quarterly milestone status dashboard with achievement tracking, actual vs target comparison, year-level completion metrics, alert status aggregation, and RAG classification for milestone-level monitoring.';
